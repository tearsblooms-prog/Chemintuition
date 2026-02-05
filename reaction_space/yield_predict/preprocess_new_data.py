# preprocess_demo_data.py
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm
import re
import os
import json

# ==============================================================================
# 这些函数从 data.py 复制而来，使此脚本可以独立运行 (无变化)
# ==============================================================================

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
    # 你的 get_atom_features 函数 (与 3_preprocess_data.py 中一致)
    # 【注意】你的 data.py 中的 ATOM_FEATURE_SIZE 是 47
    # 而 3_preprocess_data.py 中的 ATOM_FEATURE_SIZE 计算是 46
    # 这里我使用你 3_preprocess_data.py 中的版本 (46)，因为它与 mol_to_numpy_graphs 匹配
    # 如果 data.py (47) 是正确的，你也需要更新这里的 get_atom_features

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
    # data.py 中 (is_potential_center) 特征在 3_preprocess_data.py 中缺失了
    # 为了匹配 data.py 中 ATOM_FEATURE_SIZE = 47，我在这里补上
    is_potential_center = 1.0 if (atomic_num not in [1,
                                                     6] or atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0 or atom.GetIsAromatic()) else 0.0
    vec[idx] = is_potential_center
    # 确保总维度是 47
    # 35 + 1 + 1 + 1 + 6 + 1 + 1 + 1 = 47
    return vec


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    feat = np.zeros(EDGE_FEATURE_SIZE, dtype=np.float32)
    bond_type = bond.GetBondType()
    if bond_type in BOND_TYPE_MAP: feat[BOND_TYPE_MAP[bond_type]] = 1.0
    return feat


def mol_to_numpy_graphs(smiles: str):
    # 【已修正】确保使用 47 维的 ATOM_FEATURE_SIZE
    global ATOM_FEATURE_SIZE
    ATOM_FEATURE_SIZE = 47  # 强制设为 47

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


# ==============================================================================
# 主逻辑 (已更新为处理新文件)
# ==============================================================================
def main():
    # --- 【修改】配置新文件的路径 ---
    # source_csv_path = '../data/predicted_valid_reactions.csv'
    # id_to_chem_json_path = '../data/id_to_smiles.json'
    # cleaned_csv_path = '../data/predicted_valid_reactions_cleaned.csv'  # 输出的清洗后CSV
    # graphs_npz_path = '../data/predicted_valid_reactions_graphs.npz' # 输出的图数据NPZ
    source_csv_path = '../data/suzuki_miyaura_smiles_with_fingerprint.csv'
    id_to_chem_json_path = '../data/suzuki_id_to_chemical.json'
    cleaned_csv_path = '../data/suzuki_miyaura_smiles_with_fingerprint_cleaned.csv'  # 输出的清洗后CSV
    graphs_npz_path = '../data/suzuki_miyaura_graphs.npz' # 输出的图数据NPZ
    # 【重要】确保 ATOM_FEATURE_SIZE 与 data.py 一致
    global ATOM_FEATURE_SIZE
    ATOM_FEATURE_SIZE = 47
    print(f"▶️  开始预处理新数据... (Atom Feature Size: {ATOM_FEATURE_SIZE})")

    graphs_to_save = {}

    # --- 1. 处理 demo_id_to_chemical.json ---
    if not os.path.exists(id_to_chem_json_path):
        print(f"❌ 错误: 条件化学品文件 '{id_to_chem_json_path}' 不存在。")
        return

    with open(id_to_chem_json_path, 'r') as f:
        id_to_smiles_map = json.load(f)

    print(f"处理来自 '{id_to_chem_json_path}' 的 {len(id_to_smiles_map)} 个条件化学品...")
    for chem_id, smiles in tqdm(id_to_smiles_map.items(), desc="Processing condition chemicals"):
        graph = mol_to_numpy_graphs(smiles)
        graphs_to_save[f'chem_x_{chem_id}'] = graph['x']
        graphs_to_save[f'chem_ei_{chem_id}'] = graph['edge_index']
        graphs_to_save[f'chem_ea_{chem_id}'] = graph['edge_attr']

    # --- 2. 处理新的主数据集CSV文件 ---
    if not os.path.exists(source_csv_path):
        print(f"❌ 错误: 源文件 '{source_csv_path}' 不存在。")
        return

    print(f"\n清洗并处理主数据集: '{source_csv_path}'")
    raw_df = pd.read_csv(source_csv_path)

    # 【新增】检查 'products_smiles' 列并重命名为 'product_smiles'
    if 'products_smiles' in raw_df.columns and 'product_smiles' not in raw_df.columns:
        print("检测到 'products_smiles' 列，重命名为 'product_smiles' 以兼容...")
        raw_df.rename(columns={'products_smiles': 'product_smiles'}, inplace=True)
    elif 'product_smiles' not in raw_df.columns:
        print(f"❌ 错误: 在 '{source_csv_path}' 中未找到 'product_smiles' 或 'products_smiles' 列。")
        return

    def check_row_data(row):
        # 检查 reactant1
        if not is_valid_smiles(row['reactant1_smiles']): return False

        # 检查 reactant2 (如果存在)
        # .get() 确保即使 'reactant2_smiles' 列不存在也不会报错
        r2_s = row.get('reactant2_smiles')
        if pd.notna(r2_s) and str(r2_s).strip() and not is_valid_smiles(str(r2_s)): return False

        # 检查 product
        p_s = str(row['product_smiles'])
        if pd.isna(p_s) or not any(is_valid_smiles(s) for s in p_s.split(';') if s.strip()): return False

        # 检查 condition_fingerprint (必须存在)
        if 'predicted_fingerprint_json' not in row or pd.isna(row['predicted_fingerprint_json']): return False

        return True

    valid_mask = raw_df.apply(check_row_data, axis=1)
    cleaned_df = raw_df[valid_mask].reset_index(drop=True)

    # 【新增】为推理添加一个虚拟的 'y_val' 列，以匹配 ReactionDataset
    if 'y_val' not in cleaned_df.columns:
        print("为推理兼容性添加虚拟 'y_val' 列...")
        cleaned_df['y_val'] = 0.0

    print(f"数据清洗完成。有效数据: {len(cleaned_df)}行。保存至: {cleaned_csv_path}")
    cleaned_df.to_csv(cleaned_csv_path, index=False)

    print(f"处理来自 '{cleaned_csv_path}' 的主反应分子...")
    for i in tqdm(range(len(cleaned_df)), desc="Processing main reaction molecules"):
        row = cleaned_df.iloc[i]

        r1_graph = mol_to_numpy_graphs(row['reactant1_smiles'])
        graphs_to_save[f'r1_x_{i}'] = r1_graph['x'];
        graphs_to_save[f'r1_ei_{i}'] = r1_graph['edge_index'];
        graphs_to_save[f'r1_ea_{i}'] = r1_graph['edge_attr']

        # .get() 确保 'reactant2_smiles' 列不存在时也能安全处理
        r2_smiles = str(row.get('reactant2_smiles', "")) if pd.notna(row.get('reactant2_smiles')) else ""
        r2_graph = mol_to_numpy_graphs(r2_smiles)
        graphs_to_save[f'r2_x_{i}'] = r2_graph['x'];
        graphs_to_save[f'r2_ei_{i}'] = r2_graph['edge_index'];
        graphs_to_save[f'r2_ea_{i}'] = r2_graph['edge_attr']

        p_graph = {}
        prods = [s.strip() for s in str(row['product_smiles']).split(';') if s.strip()]
        for p_smiles in prods:
            p_graph_candidate = mol_to_numpy_graphs(p_smiles)
            if p_graph_candidate['x'].shape[0] > 0: p_graph = p_graph_candidate; break
        if not p_graph: p_graph = mol_to_numpy_graphs(prods[0]) if prods else mol_to_numpy_graphs("")
        graphs_to_save[f'p_x_{i}'] = p_graph['x'];
        graphs_to_save[f'p_ei_{i}'] = p_graph['edge_index'];
        graphs_to_save[f'p_ea_{i}'] = p_graph['edge_attr']

    # --- 3. 将所有图数据一次性保存 ---
    print(f"\n保存所有图数据 ({len(graphs_to_save) // 3}个分子) 到: {graphs_npz_path}")
    np.savez_compressed(graphs_npz_path, **graphs_to_save)
    print("✅ 新数据预处理完成！")


if __name__ == '__main__':
    main()