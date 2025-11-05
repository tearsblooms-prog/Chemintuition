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

# --- PyG Imports ---
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter

# --- Local Imports ---
from data_utils import ReactionConditionTokenizer
from model import AutoregressiveModel
from DualChannelMPNN import DualChannelMPNN

# --- RDKit Import ---
from rdkit import Chem


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

# --- Configuration ---
CSV_FILE_PATH = 'data/Reaxys_total_syn_condition_fingerprint.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Random Seed ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 2e-4
MPNN_LEARNING_RATE = 1e-5
MPNN_READOUT_FEATS = 512
MAX_TOKEN_SEQ_LEN = 640
EMBEDDING_DIM = 256
TRANSFORMER_LAYERS = 6
TRANSFORMER_HEADS = 8
EARLY_STOPPING_PATIENCE = 10

MPNN_HIDDEN_FEATS = 64
MPNN_NUM_STEP_MESSAGE_PASSING = 3

# --- Setup & Tokenizer ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
tokenizer = ReactionConditionTokenizer()
VOCAB_SIZE = len(tokenizer.vocab)
print(f"Tokenizer initialized with vocab size: {VOCAB_SIZE}")

class MultiMolReactionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.default_smiles = 'C'

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        reactant_smiles_list = [s for s in str(row['reactants']).split('.') if s]
        smiles_r1, smiles_r2 = self.default_smiles, self.default_smiles

        if len(reactant_smiles_list) == 1:
            smiles_r1 = reactant_smiles_list[0]
        elif len(reactant_smiles_list) == 2:
            smiles_r1 = reactant_smiles_list[0]
            smiles_r2 = reactant_smiles_list[1]
        elif len(reactant_smiles_list) > 2:
            smiles_r1 = reactant_smiles_list[0]
            smiles_r2 = '.'.join(reactant_smiles_list[1:])

        reactant_graph_1 = mol_to_atomic_graph(smiles_r1)
        reactant_graph_2 = mol_to_atomic_graph(smiles_r2)

        if reactant_graph_1.num_nodes == 0:
            reactant_graph_1 = mol_to_atomic_graph(self.default_smiles)
        if reactant_graph_2.num_nodes == 0:
            reactant_graph_2 = mol_to_atomic_graph(self.default_smiles)

        product_smiles_list = str(row['products']).split('.')
        product_graphs = [g for g in [mol_to_atomic_graph(s) for s in product_smiles_list if s] if g.num_nodes > 0]
        if not product_graphs:
            product_graphs = [mol_to_atomic_graph(self.default_smiles)]

        cond_fingerprint_640d = torch.tensor(json.loads(row['condition_fingerprint']), dtype=torch.float)
        token_ids = self.tokenizer.fingerprint_to_tokens(cond_fingerprint_640d)
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]

        return reactant_graph_1, reactant_graph_2, product_graphs, torch.as_tensor(token_ids, dtype=torch.long)


print("Loading preprocessed dataset...")
full_df = pd.read_csv(CSV_FILE_PATH)
train_df = full_df[full_df['dataset'] == 'train'].reset_index(drop=True)
val_df = full_df[full_df['dataset'] == 'val'].reset_index(drop=True)
test_df = full_df[full_df['dataset'] == 'test'].reset_index(drop=True)
train_dataset = MultiMolReactionDataset(train_df, tokenizer, MAX_TOKEN_SEQ_LEN)
val_dataset = MultiMolReactionDataset(val_df, tokenizer, MAX_TOKEN_SEQ_LEN)
test_dataset = MultiMolReactionDataset(test_df, tokenizer, MAX_TOKEN_SEQ_LEN)
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

def collate_fn(batch):
    r_graphs_1_b, r_graphs_2_b, p_graphs_b, tokens_b = zip(*batch)

    r1_batch = Batch.from_data_list(list(r_graphs_1_b))
    r2_batch = Batch.from_data_list(list(r_graphs_2_b))

    p_graphs_flat = []
    p_graph_to_sample_idx = []

    for i, sample_graphs in enumerate(p_graphs_b):
        p_graphs_flat.extend(sample_graphs)
        p_graph_to_sample_idx.extend([i] * len(sample_graphs))

    p_batch = Batch.from_data_list(p_graphs_flat)
    p_graph_idx_tensor = torch.tensor(p_graph_to_sample_idx, dtype=torch.long)

    tokens_padded = pad_sequence(tokens_b, batch_first=True, padding_value=tokenizer.pad_id)

    return r1_batch, r2_batch, p_batch, p_graph_idx_tensor, tokens_padded


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print("Initializing DualChannelMPNN...")
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
    condition_dim=MPNN_READOUT_FEATS * 2, max_seq_len=MAX_TOKEN_SEQ_LEN
).to(device)

optimizer = torch.optim.AdamW([
    {'params': model.parameters(), 'lr': LEARNING_RATE},
    {'params': mpnn_model.parameters(), 'lr': MPNN_LEARNING_RATE}
])
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

@torch.no_grad()
def evaluate(model, mpnn_model, val_loader, criterion, device):
    model.eval()
    mpnn_model.eval()
    total_loss = 0.0
    pbar = tqdm(val_loader, desc="Evaluating on validation set")
    for r1_batch, r2_batch, p_batch, p_graph_idx, tokens_batch in pbar:
        r1_batch, r2_batch, p_batch = r1_batch.to(device), r2_batch.to(device), p_batch.to(device)
        p_graph_idx, tokens_batch = p_graph_idx.to(device), tokens_batch.to(device)

        aggregated_r1_feats, _ = mpnn_model(r1_batch)
        aggregated_r2_feats, _ = mpnn_model(r2_batch)
        p_graph_feats, _ = mpnn_model(p_batch)

        num_samples = tokens_batch.size(0)
        aggregated_p_feats = torch.zeros(num_samples, p_graph_feats.size(1), device=device)
        if p_graph_feats.numel() > 0:
            scatter_result = scatter(p_graph_feats, p_graph_idx, dim=0, reduce='sum')
            aggregated_p_feats[:scatter_result.size(0)] = scatter_result

        aggregated_r_feats = aggregated_r1_feats + aggregated_r2_feats
        graph_contexts_batch = torch.cat([aggregated_r_feats, aggregated_p_feats], dim=1)

        input_ids = tokens_batch[:, :-1]
        target_ids = tokens_batch[:, 1:]
        logits = model(input_ids, graph_contexts_batch)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_ids.reshape(-1))
        total_loss += loss.item()
        pbar.set_postfix(val_loss=loss.item())
    return total_loss / len(val_loader)


@torch.no_grad()
def run_test_and_generate_table(model, mpnn_model, test_loader, tokenizer, device):
    print("\n" + "=" * 80)
    print(" " * 15 + "Generating Final Performance Table on Test Set")
    print("=" * 80)

    FEATURES_TO_EVALUATE = {
        'c1': {'type': 'discrete', 'token_rel_pos': 5},
        's1': {'type': 'discrete', 'token_rel_pos': 9},
        'r1': {'type': 'discrete', 'token_rel_pos': 0},
    }
    TEMP_TOKEN_REL_POS = 13

    num_metrics = 5  # Top-1, 3, 5, 10, 15
    results = {name: {'hits': np.zeros(num_metrics), 'count': 0} for name in FEATURES_TO_EVALUATE}
    results['Overall (c1s1r1)'] = {'hits': np.zeros(num_metrics), 'count': 0}

    total_temp_abs_error = 0.0
    temp_count = 0

    model.eval()
    mpnn_model.eval()
    special_tokens = {tokenizer.pad_id, tokenizer.eos_id, tokenizer.bos_id, tokenizer.sep_id}

    for r1_batch, r2_batch, p_batch, p_graph_idx, tokens_batch in tqdm(test_loader, desc="Final Test Evaluation"):
        r1_batch, r2_batch, p_batch = r1_batch.to(device), r2_batch.to(device), p_batch.to(device)
        p_graph_idx, tokens_batch = p_graph_idx.to(device), tokens_batch.to(device)

        aggregated_r1_feats, _ = mpnn_model(r1_batch)
        aggregated_r2_feats, _ = mpnn_model(r2_batch)
        p_graph_feats, _ = mpnn_model(p_batch)

        num_samples = tokens_batch.size(0)
        aggregated_p_feats = torch.zeros(num_samples, p_graph_feats.size(1), device=device)
        if p_graph_feats.numel() > 0:
            scatter_result = scatter(p_graph_feats, p_graph_idx, dim=0, reduce='sum')
            aggregated_p_feats[:scatter_result.size(0)] = scatter_result

        aggregated_r_feats = aggregated_r1_feats + aggregated_r2_feats
        graph_contexts_batch = torch.cat([aggregated_r_feats, aggregated_p_feats], dim=1)

        input_ids = tokens_batch[:, :-1]
        target_ids = tokens_batch[:, 1:]
        logits = model(input_ids, graph_contexts_batch)
        k_max = 15

        for i in range(tokens_batch.shape[0]):
            sample_logits, sample_targets = logits[i], target_ids[i]
            try:
                sep_indices = [j for j, tok in enumerate(input_ids[i].cpu().tolist()) if tok == tokenizer.sep_id]
                step_starts = [0] + [idx + 1 for idx in sep_indices]
                if not step_starts: continue
                step_start_logit_idx = step_starts[0]
            except (ValueError, IndexError):
                continue

            # --- Calculate the Top-k accuracy rates of c1, s1 and r1 ---
            for name, props in FEATURES_TO_EVALUATE.items():
                token_abs_idx = step_start_logit_idx + props['token_rel_pos']
                if token_abs_idx >= len(sample_targets): continue

                target_token = sample_targets[token_abs_idx].item()
                if target_token in special_tokens: continue

                results[name]['count'] += 1
                feature_logits = sample_logits[token_abs_idx]
                _, topk_preds = torch.topk(feature_logits, k=k_max)
                topk_preds = topk_preds.cpu().tolist()

                if target_token in topk_preds[:1]: results[name]['hits'][0] += 1
                if target_token in topk_preds[:3]: results[name]['hits'][1] += 1
                if target_token in topk_preds[:5]: results[name]['hits'][2] += 1
                if target_token in topk_preds[:10]: results[name]['hits'][3] += 1
                if target_token in topk_preds[:15]: results[name]['hits'][4] += 1

            # --- Calculate the accuracy rate of Overall (c1s1r1) ---
            is_sample_valid_for_overall = True
            all_conditions_correct = np.ones(num_metrics, dtype=bool)
            for name, props in FEATURES_TO_EVALUATE.items():
                token_abs_idx = step_start_logit_idx + props['token_rel_pos']
                if token_abs_idx >= len(sample_targets):
                    is_sample_valid_for_overall = False;
                    break

                target_token = sample_targets[token_abs_idx].item()
                if target_token in special_tokens:
                    is_sample_valid_for_overall = False;
                    break

                feature_logits = sample_logits[token_abs_idx]
                _, topk_preds = torch.topk(feature_logits, k=k_max)
                topk_preds = topk_preds.cpu().tolist()

                if target_token not in topk_preds[:1]:  all_conditions_correct[0] = False
                if target_token not in topk_preds[:3]:  all_conditions_correct[1] = False
                if target_token not in topk_preds[:5]:  all_conditions_correct[2] = False
                if target_token not in topk_preds[:10]: all_conditions_correct[3] = False
                if target_token not in topk_preds[:15]: all_conditions_correct[4] = False

            if is_sample_valid_for_overall:
                results['Overall (c1s1r1)']['count'] += 1
                results['Overall (c1s1r1)']['hits'] += all_conditions_correct

            # --- 计算温度 MAE ---
            temp_token_abs_idx = step_start_logit_idx + TEMP_TOKEN_REL_POS
            if temp_token_abs_idx < len(sample_targets):
                target_temp_token_id = sample_targets[temp_token_abs_idx].item()
                if target_temp_token_id not in special_tokens:
                    pred_temp_token_id = torch.argmax(sample_logits[temp_token_abs_idx]).item()
                    target_temp_token_str = tokenizer.id_to_token.get(target_temp_token_id)
                    pred_temp_token_str = tokenizer.id_to_token.get(pred_temp_token_id)
                    if target_temp_token_str and pred_temp_token_str:
                        target_temp_val = tokenizer._undiscretize_temp(target_temp_token_str)
                        pred_temp_val = tokenizer._undiscretize_temp(pred_temp_token_str)
                        if target_temp_val != -999.0 and pred_temp_val != -999.0:
                            total_temp_abs_error += abs(target_temp_val - pred_temp_val)
                            temp_count += 1

    df_data = []
    order = ['c1', 's1', 'r1', 'Overall (c1s1r1)']
    for name in order:
        res = results[name]
        row = {'Conditions': name}
        if res['count'] > 0:
            acc = res['hits'] / res['count']
            row.update({'Top-1': f"{acc[0]:.4f}", 'Top-3': f"{acc[1]:.4f}", 'Top-5': f"{acc[2]:.4f}",
                        'Top-10': f"{acc[3]:.4f}", 'Top-15': f"{acc[4]:.4f}"})
        else:
            row.update({'Top-1': 'N/A', 'Top-3': 'N/A', 'Top-5': 'N/A', 'Top-10': 'N/A', 'Top-15': 'N/A'})
        df_data.append(row)

    temp_mae = total_temp_abs_error / temp_count if temp_count > 0 else 'N/A'
    mae_row = {'Conditions': 'Temp MAE (°C)'}
    if isinstance(temp_mae, float):
        mae_row.update({'Top-1': f"{temp_mae:.4f}", 'Top-3': '', 'Top-5': '', 'Top-10': '', 'Top-15': ''})
    else:
        mae_row.update({'Top-1': 'N/A', 'Top-3': '', 'Top-5': '', 'Top-10': '', 'Top-15': ''})
    df_data.append(mae_row)

    df = pd.DataFrame(df_data)
    print("\nChemical context condition performance")
    print("-" * 80)
    print(df.to_markdown(index=False))
    print("-" * 80 + "\n")
    table_path = os.path.join(RESULTS_DIR, 'performance_table_styled.md')
    with open(table_path, 'w') as f:
        f.write("# Chemical context condition performance\n\n")
        f.write(df.to_markdown(index=False))
    print(f"Performance table saved to {table_path}")

if __name__ == '__main__':
    best_val_loss = float('inf')
    early_stopping_counter = 0
    print("\n--- Starting Training ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        mpnn_model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for step, (r1_batch, r2_batch, p_batch, p_graph_idx, tokens) in enumerate(pbar):
            optimizer.zero_grad()
            r1_batch, r2_batch, p_batch = r1_batch.to(device), r2_batch.to(device), p_batch.to(device)
            p_graph_idx, tokens = p_graph_idx.to(device), tokens.to(device)

            aggregated_r1_feats, _ = mpnn_model(r1_batch)
            aggregated_r2_feats, _ = mpnn_model(r2_batch)
            p_graph_feats, _ = mpnn_model(p_batch)

            num_samples = tokens.size(0)
            aggregated_p_feats = torch.zeros(num_samples, p_graph_feats.size(1), device=device)
            if p_graph_feats.numel() > 0:
                scatter_result = scatter(p_graph_feats, p_graph_idx, dim=0, reduce='sum')
                aggregated_p_feats[:scatter_result.size(0)] = scatter_result

            aggregated_r_feats = aggregated_r1_feats + aggregated_r2_feats
            graph_contexts = torch.cat([aggregated_r_feats, aggregated_p_feats], dim=1)

            input_ids, target_ids = tokens[:, :-1], tokens[:, 1:]
            logits = model(input_ids, graph_contexts)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_ids.reshape(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Stopping training at Epoch {epoch}, Step {step} due to NaN/Inf loss.")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mpnn_model.parameters(), 1.0)  # 对 MPNN 也进行梯度裁剪
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix(train_loss=loss.item())

        if 'loss' in locals() and (torch.isnan(loss) or torch.isinf(loss)): break

        avg_val_loss = evaluate(model, mpnn_model, val_loader, criterion, device)
        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"  Average Training Loss: {total_train_loss / len(train_loader):.4f}")
        print(f"  Average Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best model found! Saving 'model_best.pth' with validation loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'model_best.pth'))
            torch.save(mpnn_model.state_dict(), os.path.join(RESULTS_DIR, 'mpnn_model_best.pth'))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"--- Early stopping triggered after {epoch} epochs. ---")
                break

        if epoch % 20 == 0 or epoch == EPOCHS:
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f'model_epoch_{epoch}.pth'))
            torch.save(mpnn_model.state_dict(), os.path.join(RESULTS_DIR, f'mpnn_model_epoch_{epoch}.pth'))


    print("--- Training complete. ---")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")

    print("\nLoading best model for final test evaluation...")
    try:
        model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'model_best.pth')))
        mpnn_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'mpnn_model_best.pth')))
        print("Best model weights loaded successfully.")
    except FileNotFoundError:
        print("Warning: Best model not found. Using the model from the final epoch for testing.")

    run_test_and_generate_table(model, mpnn_model, test_loader, tokenizer, device)