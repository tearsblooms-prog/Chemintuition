import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm
import re
import os
import json


# --- Atom Feature Configuration ---
ONE_HOT_ATOMIC_NUM_SIZE = 35
NUM_HYBRIDIZATION_TYPES = 6
ATOM_FEATURE_SIZE = (
        ONE_HOT_ATOMIC_NUM_SIZE + 1 + 1 + 1 +
        NUM_HYBRIDIZATION_TYPES + 1 + 1 + 1
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
    if not isinstance(smiles, str) or not smiles.strip(): return False
    if re.search(r'%\d{3,}', smiles): return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    vec = np.zeros(ATOM_FEATURE_SIZE, dtype=np.float32)
    idx, atomic_num = 0, atom.GetAtomicNum()
    if 0 < atomic_num <= ONE_HOT_ATOMIC_NUM_SIZE: vec[atomic_num - 1] = 1.0
    idx += ONE_HOT_ATOMIC_NUM_SIZE;
    vec[idx] = float(atom.GetDegree());
    idx += 1
    vec[idx] = float(atom.GetFormalCharge());
    idx += 1;
    vec[idx] = float(atom.GetNumRadicalElectrons());
    idx += 1
    hybrid_map = {Chem.rdchem.HybridizationType.SP: 0, Chem.rdchem.HybridizationType.SP2: 1,
                  Chem.rdchem.HybridizationType.SP3: 2, Chem.rdchem.HybridizationType.SP3D: 3,
                  Chem.rdchem.HybridizationType.SP3D2: 4}
    hybrid_idx = hybrid_map.get(atom.GetHybridization(), NUM_HYBRIDIZATION_TYPES - 1);
    vec[idx + hybrid_idx] = 1.0;
    idx += NUM_HYBRIDIZATION_TYPES
    vec[idx] = float(atom.GetIsAromatic());
    idx += 1;
    vec[idx] = float(atom.GetTotalNumHs());
    idx += 1
    is_potential_center = 1.0 if (atomic_num not in [1,
                                                     6] or atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0 or atom.GetIsAromatic()) else 0.0
    vec[idx] = is_potential_center
    return vec


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    feat = np.zeros(EDGE_FEATURE_SIZE, dtype=np.float32)
    bond_type = bond.GetBondType()
    if bond_type in BOND_TYPE_MAP: feat[BOND_TYPE_MAP[bond_type]] = 1.0
    return feat


def mol_to_numpy_graphs(smiles: str):
    empty_graph = {'x': np.zeros((0, ATOM_FEATURE_SIZE), dtype=np.float32),
                   'edge_index': np.zeros((2, 0), dtype=np.int64),
                   'edge_attr': np.zeros((0, EDGE_FEATURE_SIZE), dtype=np.float32)}
    if not is_valid_smiles(smiles): return empty_graph
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return empty_graph
        atom_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
        if not atom_feats: return empty_graph
        x = np.stack(atom_feats, axis=0)
        edges, edge_feats = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = get_bond_features(bond)
            edges.extend([[i, j], [j, i]]);
            edge_feats.extend([bond_feat, bond_feat])
        edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.stack(edge_feats, axis=0) if edge_feats else np.zeros((0, EDGE_FEATURE_SIZE), dtype=np.float32)
        return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}
    except Exception:
        return empty_graph

def main():
    # --- Configuration ---
    source_csv_path = 'suzuki_miyaura_smiles_with_fingerprint.csv'
    id_to_chem_json_path = 'data/suzuki_id_to_chemical.json'
    cleaned_csv_path = 'suzuki_miyaura_cleaned.csv'
    graphs_npz_path = 'preprocessed_graphs.npz'

    print("Starting preprocessing...")

    graphs_to_save = {}

    # --- 1. Process id_to_chemical.json ---
    if not os.path.exists(id_to_chem_json_path):
        print(f"Error: condition chemical file '{id_to_chem_json_path}' not found.")
        return

    with open(id_to_chem_json_path, 'r') as f:
        id_to_smiles_map = json.load(f)

    print(f"Processing {len(id_to_smiles_map)} condition chemicals from '{id_to_chem_json_path}'...")
    for chem_id, smiles in tqdm(id_to_smiles_map.items(), desc="Processing condition chemicals"):
        graph = mol_to_numpy_graphs(smiles)
        graphs_to_save[f'chem_x_{chem_id}'] = graph['x']
        graphs_to_save[f'chem_ei_{chem_id}'] = graph['edge_index']
        graphs_to_save[f'chem_ea_{chem_id}'] = graph['edge_attr']

    # --- 2. Process main dataset CSV file ---
    if not os.path.exists(source_csv_path):
        print(f"Error: source file '{source_csv_path}' not found.")
        return

    print(f"\nCleaning and processing main dataset: '{source_csv_path}'")
    raw_df = pd.read_csv(source_csv_path)

    def check_row_data(row):
        if not is_valid_smiles(row['reactant1_smiles']): return False
        r2_s = row.get('reactant2_smiles')
        if pd.notna(r2_s) and str(r2_s).strip() and not is_valid_smiles(str(r2_s)): return False
        p_s = str(row['product_smiles'])
        if pd.isna(p_s) or not any(is_valid_smiles(s) for s in p_s.split(';') if s.strip()): return False
        return True

    valid_mask = raw_df.apply(check_row_data, axis=1)
    cleaned_df = raw_df[valid_mask].reset_index(drop=True)

    print(f"Data cleaning complete. Valid entries: {len(cleaned_df)} rows. Saving to: {cleaned_csv_path}")
    cleaned_df.to_csv(cleaned_csv_path, index=False)

    print(f"Processing main reaction molecules from '{cleaned_csv_path}'...")
    for i in tqdm(range(len(cleaned_df)), desc="Processing main reaction molecules"):
        row = cleaned_df.iloc[i]

        r1_graph = mol_to_numpy_graphs(row['reactant1_smiles'])
        graphs_to_save[f'r1_x_{i}'] = r1_graph['x']
        graphs_to_save[f'r1_ei_{i}'] = r1_graph['edge_index']
        graphs_to_save[f'r1_ea_{i}'] = r1_graph['edge_attr']

        r2_smiles = str(row.get('reactant2_smiles', "")) if pd.notna(row.get('reactant2_smiles')) else ""
        r2_graph = mol_to_numpy_graphs(r2_smiles)
        graphs_to_save[f'r2_x_{i}'] = r2_graph['x']
        graphs_to_save[f'r2_ei_{i}'] = r2_graph['edge_index']
        graphs_to_save[f'r2_ea_{i}'] = r2_graph['edge_attr']

        p_graph = {}
        prods = [s.strip() for s in str(row['product_smiles']).split(';') if s.strip()]
        for p_smiles in prods:
            p_graph_candidate = mol_to_numpy_graphs(p_smiles)
            if p_graph_candidate['x'].shape[0] > 0:
                p_graph = p_graph_candidate
                break
        if not p_graph:
            p_graph = mol_to_numpy_graphs(prods[0]) if prods else mol_to_numpy_graphs("")
        graphs_to_save[f'p_x_{i}'] = p_graph['x']
        graphs_to_save[f'p_ei_{i}'] = p_graph['edge_index']
        graphs_to_save[f'p_ea_{i}'] = p_graph['edge_attr']

    # --- 3. Save all graph data ---
    print(f"\nSaving all graph data ({len(graphs_to_save) // 3} molecules) to: {graphs_npz_path}")
    np.savez_compressed(graphs_npz_path, **graphs_to_save)
    print("Preprocessing complete!")


if __name__ == '__main__':
    main()
