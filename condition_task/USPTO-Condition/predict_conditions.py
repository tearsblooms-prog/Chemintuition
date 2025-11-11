import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
import json
import sys
import re
import random
import math

# --- PyG Imports ---
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter

# --- RDKit Import ---
try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("RDKit not found. Please install it: pip install rdkit-pypi")
    sys.exit(1)

try:
    from model import AutoregressiveModel
    from data_utils import ReactionConditionTokenizer
    from ChemSReactMPNN import DualChannelMPNN
except ImportError as e:
    print(f"Error: Required local files are missing.")
    print(f"ImportError: {e}")
    sys.exit(1)

# --- Input/Output Files ---
INPUT_CSV_PATH = 'data/demo_reaction.csv'
OUTPUT_CSV_PATH = 'data/demo_reaction_output.csv'

# --- Model Weight Paths ---
MODEL_WEIGHTS_DIR = 'results'  # Results directory from training script
MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, 'model_best.pth')
MPNN_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, 'mpnn_model_best.pth')


EMBEDDING_DIM = 256
TRANSFORMER_LAYERS = 6
TRANSFORMER_HEADS = 8
MAX_TOKEN_SEQ_LEN = 640  #

# ReactionMPNN
MPNN_HIDDEN_FEATS = 128
MPNN_NUM_STEP_MESSAGE_PASSING = 3
MPNN_READOUT_FEATS = 512

# --- Atom Feature Configuration ---
ONE_HOT_ATOMIC_NUM_SIZE = 35
NUM_HYBRIDIZATION_TYPES = 6
ATOM_FEATURE_SIZE = (
        ONE_HOT_ATOMIC_NUM_SIZE +
        1 +  # Degree
        1 +  # Formal charge
        1 +  # Num radical electrons
        NUM_HYBRIDIZATION_TYPES +
        1 +  # Is aromatic
        1 +  # Total num Hs
        1  # Is potential reaction center
)

# --- Bond Feature Configuration ---
NUM_BOND_TYPES = 4
EDGE_FEATURE_SIZE = NUM_BOND_TYPES

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


def is_valid_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    if re.search(r'%\d{3,}', smiles):
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def get_atom_features(atom: Chem.Atom) -> torch.Tensor:
    vec = torch.zeros(ATOM_FEATURE_SIZE, dtype=torch.float)
    idx = 0
    atomic_num = atom.GetAtomicNum()
    if 0 < atomic_num <= ONE_HOT_ATOMIC_NUM_SIZE:
        vec[atomic_num - 1] = 1.0
    idx += ONE_HOT_ATOMIC_NUM_SIZE
    vec[idx] = float(atom.GetDegree())
    idx += 1
    charge = float(atom.GetFormalCharge())
    vec[idx] = charge
    idx += 1
    rad_electrons = float(atom.GetNumRadicalElectrons())
    vec[idx] = rad_electrons
    idx += 1
    hybrid_map = {
        Chem.rdchem.HybridizationType.SP: 0, Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2, Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
    }
    hybrid_idx = hybrid_map.get(atom.GetHybridization(), NUM_HYBRIDIZATION_TYPES - 1)
    vec[idx + hybrid_idx] = 1.0
    idx += NUM_HYBRIDIZATION_TYPES
    is_aromatic = float(atom.GetIsAromatic())
    vec[idx] = is_aromatic
    idx += 1
    vec[idx] = float(atom.GetTotalNumHs())
    idx += 1
    is_potential_center = 0.0
    if atomic_num not in [1, 6]:
        is_potential_center = 1.0
    elif charge != 0:
        is_potential_center = 1.0
    elif rad_electrons != 0:
        is_potential_center = 1.0
    elif is_aromatic > 0:
        is_potential_center = 1.0
    vec[idx] = is_potential_center
    idx += 1
    if idx != ATOM_FEATURE_SIZE:
        raise ValueError(f"Atom feature vector length mismatch: expected {ATOM_FEATURE_SIZE}, got {idx}")
    return vec


def get_bond_features(bond: Chem.Bond) -> torch.Tensor:
    bond_type = bond.GetBondType()
    feat = torch.zeros(EDGE_FEATURE_SIZE, dtype=torch.float)
    if bond_type in BOND_TYPE_MAP:
        feat[BOND_TYPE_MAP[bond_type]] = 1.0
    return feat


def mol_to_atomic_graph(smiles: str) -> Data:
    empty_graph = Data(x=torch.zeros((0, ATOM_FEATURE_SIZE)),
                       edge_index=torch.zeros((2, 0), dtype=torch.long),
                       edge_attr=torch.zeros((0, EDGE_FEATURE_SIZE)),
                       num_nodes=0)
    try:
        if not is_valid_smiles(smiles):
            return empty_graph
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return empty_graph
        atom_feats_list = [get_atom_features(atom) for atom in mol.GetAtoms()]
        if not atom_feats_list:
            return empty_graph
        x = torch.stack(atom_feats_list, dim=0)
        edges = []
        edge_feats_list = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = get_bond_features(bond)
            edges.extend([[i, j], [j, i]])
            edge_feats_list.extend([bond_feat, bond_feat])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges \
            else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.stack(edge_feats_list, dim=0) if edge_feats_list \
            else torch.zeros((0, EDGE_FEATURE_SIZE), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=mol.GetNumAtoms())
    except Exception:
        return empty_graph

@torch.no_grad()
def predict_conditions(model, mpnn_model, r1_graph, r2_graph, p_graphs, tokenizer, device, max_len):
    # 1. Set models to evaluation mode
    model.eval()
    mpnn_model.eval()

    # 2. Prepare graph data
    r1_batch = Batch.from_data_list([r1_graph]).to(device)
    r2_batch = Batch.from_data_list([r2_graph]).to(device)

    # Products can be multiple
    p_batch = Batch.from_data_list(p_graphs).to(device)
    # All product graphs belong to this sample (index 0)
    p_graph_idx_tensor = torch.zeros(len(p_graphs), dtype=torch.long, device=device)

    # 3. Extract graph features via MPNN
    aggregated_r1_feats, _ = mpnn_model(r1_batch)
    aggregated_r2_feats, _ = mpnn_model(r2_batch)
    p_graph_feats, _ = mpnn_model(p_batch)

    # 4. Aggregate graph features
    num_samples = 1
    aggregated_p_feats = torch.zeros(num_samples, p_graph_feats.size(1), device=device)
    if p_graph_feats.numel() > 0:
        # Use scatter_sum to aggregate features of all product graphs
        scatter_result = scatter(p_graph_feats, p_graph_idx_tensor, dim=0, reduce='sum')
        aggregated_p_feats[:scatter_result.size(0)] = scatter_result

    aggregated_r_feats = aggregated_r1_feats + aggregated_r2_feats

    # 5. Assemble the final condition vector
    # [B, MPNN_READOUT_FEATS * 2]
    condition_vector = torch.cat([aggregated_r_feats, aggregated_p_feats], dim=1)

    # 6. Autoregressive Generation
    input_ids = [tokenizer.bos_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_len):
        # Pass the current sequence and condition vector
        logits = model(input_tensor, condition_vector)

        # Get logits from the last time step
        last_logits = logits[:, -1, :]

        # Greedy decoding
        next_token_id = torch.argmax(last_logits, dim=-1).item()

        # Check if the end-of-sequence token is generated
        if next_token_id == tokenizer.eos_id:
            break

        # Add the newly generated token to the sequence
        input_ids.append(next_token_id)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Add EOS (if not present at the end of the loop)
    if input_ids[-1] != tokenizer.eos_id:
        input_ids.append(tokenizer.eos_id)

    return input_ids

# --- Fingerprint Structure Constants ---
MAX_STAGES = 2
MAX_STEPS_PER_STAGE = 10
TOTAL_POTENTIAL_STEPS = MAX_STAGES * MAX_STEPS_PER_STAGE  # 20

NUM_REAGENTS = 5
NUM_CATALYSTS = 4
NUM_SOLVENTS = 4
NUM_TEMP_TIME_SEGMENTS = 4
NUM_VALUES_PER_SEGMENT = 4  # T_start, T_end, duration_min, is_reflux
NUM_OTHER_CONDITIONS = 2

VALUES_PER_STEP = 1 + NUM_REAGENTS + NUM_CATALYSTS + NUM_SOLVENTS + \
                  (NUM_TEMP_TIME_SEGMENTS * NUM_VALUES_PER_SEGMENT) + \
                  NUM_OTHER_CONDITIONS  # 32

# --- Dictionary Path Constants ---
DATA_DIR = 'data'
CHEM_DICT_FILE = os.path.join(DATA_DIR, 'chemical_to_id.json')
ID_CHEM_DICT_FILE = os.path.join(DATA_DIR, 'id_to_chemical.json')


def load_dictionaries():
    """Load dictionaries from JSON files. Create empty ones if not found."""
    if not os.path.exists(DATA_DIR):
        print(f"Warning: Data directory '{DATA_DIR}' not found. Creating it.")
        os.makedirs(DATA_DIR)

    chemical_to_id = {}
    id_to_chemical = {}

    try:
        if os.path.exists(CHEM_DICT_FILE):
            with open(CHEM_DICT_FILE, 'r', encoding='utf-8') as f:
                chemical_to_id = json.load(f)
        if os.path.exists(ID_CHEM_DICT_FILE):
            with open(ID_CHEM_DICT_FILE, 'r', encoding='utf-8') as f:
                # Keys must be int for correct loading
                id_to_chemical = {int(k): v for k, v in json.load(f).items()}
    except Exception as e:
        print(f"Warning: Failed to load chemical dictionaries: {e}")
    except Exception as e:
        print(f"Warning: Failed to load other condition dictionaries: {e}")

    return chemical_to_id, id_to_chemical


def fingerprint_to_structured_condition(fingerprint, id_to_chemical):
    """Decode numerical fingerprint back to structured condition string."""
    if not isinstance(fingerprint, list):
        try:
            fingerprint = json.loads(fingerprint)
        except (json.JSONDecodeError, TypeError):
            fingerprint = [float(x) for x in str(fingerprint).strip('[]').split(',') if x.strip()]

    decoded_steps = []
    for global_step_idx in range(TOTAL_POTENTIAL_STEPS):
        step_offset = global_step_idx * VALUES_PER_STEP

        # Check the "is_active" flag
        if step_offset < len(fingerprint) and fingerprint[step_offset] == 1.0:
            stage_num = (global_step_idx // MAX_STEPS_PER_STAGE) + 1
            step_in_stage_num = (global_step_idx % MAX_STEPS_PER_STAGE) + 1
            step_label = f"Step{stage_num}.{step_in_stage_num}:"

            parts = [step_label]
            current_offset = step_offset + 1

            # --- Reagents ---
            r_ids = [int(x) for x in fingerprint[current_offset: current_offset + NUM_REAGENTS]]
            current_offset += NUM_REAGENTS
            if any(r_id > 0 for r_id in r_ids):
                parts.append("Reagents:" + "#".join(
                    [id_to_chemical.get(str(r_id), f"ID_{r_id}_NOT_FOUND") for r_id in r_ids if r_id > 0]))

            # --- Catalysts ---
            c_ids = [int(x) for x in fingerprint[current_offset: current_offset + NUM_CATALYSTS]]
            current_offset += NUM_CATALYSTS
            if any(c_id > 0 for c_id in c_ids):
                parts.append("Catalysts:" + "#".join(
                    [id_to_chemical.get(str(c_id), f"ID_{c_id}_NOT_FOUND") for c_id in c_ids if c_id > 0]))

            # --- Solvents ---
            s_ids = [int(x) for x in fingerprint[current_offset: current_offset + NUM_SOLVENTS]]
            current_offset += NUM_SOLVENTS
            if any(s_id > 0 for s_id in s_ids):
                parts.append("Solvents:" + "#".join(
                    [id_to_chemical.get(str(s_id), f"ID_{s_id}_NOT_FOUND") for s_id in s_ids if s_id > 0]))

            step_desc = parts[0] + " "
            component_strings = [p.strip() for p in parts[1:] if p.strip()]
            step_desc += "; ".join(component_strings)
            decoded_steps.append(step_desc)

    return "\n".join(decoded_steps)


def main():
    print("--- Starting Reaction Condition Prediction ---")

    # 1. Check input file
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found '{INPUT_CSV_PATH}'")
        print("Please create a predict_input.csv file with columns: 'reactant1_smiles', 'reactant2_smiles', 'product_smiles'.")
        # Create an example input file
        example_data = {
            'reactant1_smiles': ['CC(=O)Cl', 'CCO'],
            'reactant2_smiles': ['NCCN', ''],
            'product_smiles': ['CC(=O)NCCN', 'CCOC(C)=O']
        }
        try:
            pd.DataFrame(example_data).to_csv(INPUT_CSV_PATH, index=False)
            print(f"Example input file created: '{INPUT_CSV_PATH}'")
        except Exception as e:
            print(f"Failed to create example file: {e}")
        return

    # 2. Check model files
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MPNN_MODEL_PATH):
        print(f"Error: Model weights not found.")
        print(f"Please ensure the following files exist in the '{MODEL_WEIGHTS_DIR}' directory:")
        print(f" - {os.path.basename(MODEL_PATH)}")
        print(f" - {os.path.basename(MPNN_MODEL_PATH)}")
        return

    # 3. Set device and Tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    tokenizer = ReactionConditionTokenizer()
    VOCAB_SIZE = len(tokenizer.vocab)
    DEFAULT_SMILES = 'C'  # Default SMILES used during training

    print("--- Loading decoding dictionaries ---")
    chemical_to_id, id_to_chemical = load_dictionaries()

    if not id_to_chemical:
        print(f"Warning: Chemical decoding dictionary (e.g., '{ID_CHEM_DICT_FILE}') is empty or not found.")
        print("Structured decoding results may show 'ID_NOT_FOUND'.")
        print(f"Please ensure the dictionary files exist in the '{DATA_DIR}' directory.")
    str_id_to_chemical = {str(k): v for k, v in id_to_chemical.items()}
    print("Dictionaries loaded.")

    # 5. Initialize models
    print("Initializing models...")
    mpnn_model = DualChannelMPNN(
        node_in_feats=ATOM_FEATURE_SIZE,
        edge_in_feats=EDGE_FEATURE_SIZE,
        hidden_feats=MPNN_HIDDEN_FEATS,
        num_step_message_passing=MPNN_NUM_STEP_MESSAGE_PASSING,
        readout_feats=MPNN_READOUT_FEATS
    ).to(device)

    model = AutoregressiveModel(
        vocab_size=VOCAB_SIZE, model_dim=EMBEDDING_DIM,
        num_layers=TRANSFORMER_LAYERS, num_heads=TRANSFORMER_HEADS,
        condition_dim=MPNN_READOUT_FEATS * 2,  # reactants + products
        max_seq_len=MAX_TOKEN_SEQ_LEN
    ).to(device)

    # 6. Load weights
    print("Loading model weights...")
    try:
        mpnn_model.load_state_dict(torch.load(MPNN_MODEL_PATH, map_location=device))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        print("Please ensure model hyperparameters match those used during training.")
        return

    # 7. Load and process input data
    print(f"Reading input data: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    # Ensure required columns exist
    required_cols = ['reactant1_smiles', 'reactant2_smiles', 'product_smiles']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input CSV must contain the following columns: {required_cols}")
        return

    # Handle missing reactant2_smiles
    df['reactant2_smiles'] = df['reactant2_smiles'].fillna('')

    # 8. Loop through and execute predictions
    results_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        try:
            # Reactant 1
            r1_smiles = row['reactant1_smiles']
            r1_graph = mol_to_atomic_graph(r1_smiles)
            if r1_graph.num_nodes == 0:
                r1_graph = mol_to_atomic_graph(DEFAULT_SMILES)

            # Reactant 2
            r2_smiles = row['reactant2_smiles']
            if not r2_smiles or pd.isna(r2_smiles):
                r2_smiles = DEFAULT_SMILES
            r2_graph = mol_to_atomic_graph(r2_smiles)
            if r2_graph.num_nodes == 0:
                r2_graph = mol_to_atomic_graph(DEFAULT_SMILES)

            # Products (can be multiple, separated by '.')
            p_smiles_list = str(row['product_smiles']).split('.')
            product_graphs = [g for g in [mol_to_atomic_graph(s) for s in p_smiles_list if s] if g.num_nodes > 0]
            if not product_graphs:
                product_graphs = [mol_to_atomic_graph(DEFAULT_SMILES)]

            # --- b. Execute prediction (Generate Token IDs) ---
            predicted_token_ids = predict_conditions(
                model, mpnn_model, r1_graph, r2_graph, product_graphs,
                tokenizer, device, MAX_TOKEN_SEQ_LEN
            )

            # --- c. Decode results (Token ID -> Raw Token String) ---
            decoded_tokens = [tokenizer.id_to_token.get(tid, '[UNK]') for tid in predicted_token_ids]

            # --- d. Convert back to fingerprint (Token ID -> 640d Vector) ---
            predicted_fingerprint_tensor = tokenizer.tokens_to_fingerprint(predicted_token_ids)
            predicted_fingerprint_list = predicted_fingerprint_tensor.cpu().numpy().tolist()
            fingerprint_json = json.dumps(predicted_fingerprint_list)

            try:
                decoded_structure = fingerprint_to_structured_condition(
                    predicted_fingerprint_list,
                    str_id_to_chemical,
                )
            except Exception as decode_e:
                decoded_structure = f"Error during structured decoding: {decode_e}"

            # --- f. Store results ---
            new_row = row.to_dict()
            new_row['predicted_fingerprint_json'] = fingerprint_json
            new_row['decoded_structured_condition'] = decoded_structure  # (e.g., "Step1.1: Reagents:X; ...")
            results_list.append(new_row)

        except Exception as e:
            print(f"Error processing row {row.name}: {e}")
            new_row = row.to_dict()
            new_row['predicted_fingerprint_json'] = f"Error: {e}"
            new_row['decoded_condition_tokens'] = f"Error: {e}"
            new_row['decoded_structured_condition'] = f"Error: {e}"
            results_list.append(new_row)

    # 9. Save output
    print("\nPrediction complete.")
    output_df = pd.DataFrame(results_list)
    try:
        output_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"Results successfully saved to: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"Failed to save output file: {e}")


if __name__ == "__main__":
    main()