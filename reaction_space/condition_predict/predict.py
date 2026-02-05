import os
import torch
import pandas as pd
import json
import math
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

# --- PyG Imports ---
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter

# --- RDKit Import ---
from rdkit import Chem

# --- 本地模块导入 ---
try:
    from data_utils import ReactionConditionTokenizer
    from model import AutoregressiveModel
    from ChemSReactMPNN import ChemSReactMPNN
except ImportError as e:
    print(f"Error: 无法导入本地模块。 {e}")
    print("请确保 'data_utils.py', 'model.py', 和 'ChemSReactMPNN.py' 都在此脚本的同一目录中。")
    exit()

# ======== 常量配置 ========

# --- Atom/Bond Feature Configuration ---
ONE_HOT_ATOMIC_NUM_SIZE = 35
NUM_HYBRIDIZATION_TYPES = 6
ATOM_FEATURE_SIZE = (
        ONE_HOT_ATOMIC_NUM_SIZE + 1 + 1 + 1 + NUM_HYBRIDIZATION_TYPES + 1 + 1 + 1
)
NUM_BOND_TYPES = 4
EDGE_FEATURE_SIZE = NUM_BOND_TYPES

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

# --- 模型和训练配置 ---
RESULTS_DIR = 'results_final'
MPNN_READOUT_FEATS = 512
MAX_TOKEN_SEQ_LEN = 640
EMBEDDING_DIM = 256
TRANSFORMER_LAYERS = 6
TRANSFORMER_HEADS = 8
MPNN_HIDDEN_FEATS = 64
MPNN_NUM_STEP_MESSAGE_PASSING = 3


# ======== 辅助函数 (保持不变) ========

def is_valid_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
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


# ======== 预测核心功能 ========

@torch.no_grad()
def predict_single_reaction(r1_smiles, r2_smiles, product_smiles_list, model, mpnn_model, tokenizer, device,
                            max_gen_len):
    default_graph = mol_to_atomic_graph('C')
    r1_graph = mol_to_atomic_graph(r1_smiles)
    if r1_graph.num_nodes == 0: r1_graph = default_graph
    r2_graph = mol_to_atomic_graph(r2_smiles)
    if r2_graph.num_nodes == 0: r2_graph = default_graph
    product_graphs = [mol_to_atomic_graph(s) for s in product_smiles_list if s]
    product_graphs = [g for g in product_graphs if g.num_nodes > 0]
    if not product_graphs:
        product_graphs = [default_graph]

    r1_batch = Batch.from_data_list([r1_graph]).to(device)
    r2_batch = Batch.from_data_list([r2_graph]).to(device)
    p_batch = Batch.from_data_list(product_graphs).to(device)
    p_graph_idx = torch.zeros(len(product_graphs), dtype=torch.long, device=device)

    aggregated_r1_feats, _ = mpnn_model(r1_batch)
    aggregated_r2_feats, _ = mpnn_model(r2_batch)
    p_graph_feats, _ = mpnn_model(p_batch)
    aggregated_p_feats = scatter(p_graph_feats, p_graph_idx, dim=0, reduce='sum')

    aggregated_r_feats = aggregated_r1_feats + aggregated_r2_feats
    graph_contexts = torch.cat([aggregated_r_feats, aggregated_p_feats], dim=1)

    memory = model.condition_projection(graph_contexts).unsqueeze(1)
    input_tokens = torch.full((1, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    generated_token_ids = [tokenizer.bos_id]

    for _ in range(max_gen_len):
        token_embeds = model.token_embedding(input_tokens) * math.sqrt(model.model_dim)
        token_embeds = model.pos_encoder(token_embeds)
        tgt_mask = model._generate_square_subsequent_mask(input_tokens.size(1), device)
        output = token_embeds
        for layer in model.transformer_decoder:
            output = layer(output, memory, tgt_mask=tgt_mask)
        output = model.output_norm(output)
        logits = model.to_logits(output)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        input_tokens = torch.cat([input_tokens, next_token_id], dim=1)
        next_token_item = next_token_id.item()
        generated_token_ids.append(next_token_item)

        if next_token_item == tokenizer.eos_id:
            break

    fingerprint_tensor = tokenizer.tokens_to_fingerprint(generated_token_ids)
    return fingerprint_tensor


# ======== 工作进程 (Worker Process) ========

def worker_process(rank, df_chunk, result_queue):
    """
    单个 GPU 的工作进程
    """
    try:
        # 1. 配置当前进程的设备
        device_id = rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{device_id}')

        # 2. 在进程内初始化 Tokenizer 和 模型
        tokenizer = ReactionConditionTokenizer()
        VOCAB_SIZE = len(tokenizer.vocab)

        mpnn_model = ChemSReactMPNN(
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

        # 加载权重
        MPNN_MODEL_PATH = os.path.join(RESULTS_DIR, 'mpnn_model_best.pth')
        MODEL_PATH = os.path.join(RESULTS_DIR, 'model_best.pth')

        mpnn_model.load_state_dict(torch.load(MPNN_MODEL_PATH, map_location=device))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

        mpnn_model.eval()
        model.eval()

        local_results = []

        # 3. 遍历分配给该进程的数据块
        iterator = tqdm(df_chunk.iterrows(), total=len(df_chunk),
                        desc=f"GPU {device_id} Prediction", position=rank)

        for idx, row in iterator:
            r1 = str(row['reactant1_smiles'])
            r2 = str(row['reactant2_smiles'])
            p_list = []
            if pd.notna(row['product_smiles']):
                p_list = [s for s in str(row['product_smiles']).split('.') if s]

            fingerprint_tensor = predict_single_reaction(
                r1, r2, p_list,
                model, mpnn_model, tokenizer,
                device, max_gen_len=MAX_TOKEN_SEQ_LEN
            )

            # 1. 转换为 list
            fp_list = fingerprint_tensor.cpu().tolist()

            # 2. 检查索引 1 到 13 的子数组 (包含1, 不包含14) 是否全为 0
            # 如果数组长度不足14 (极少见情况), 也跳过或按需处理
            if len(fp_list) < 14:
                continue

            sub_array = fp_list[1:14]
            # 如果该切片全为 0, 则认为该预测无效，跳过
            if all(x == 0 for x in sub_array):
                continue

            # 构建结果字典 (保留原始信息 + 预测结果)
            res_dict = row.to_dict()
            res_dict['predicted_fingerprint_json'] = json.dumps(fp_list)
            local_results.append(res_dict)

        # 4. 将结果放入队列
        result_queue.put(local_results)

    except Exception as e:
        print(f"Process {rank} encountered an error: {e}")
        result_queue.put([])  # 出错返回空列表


# ======== 主执行脚本 ========

def main():
    # --- 必须在 main 中设置 start method 为 spawn (CUDA 要求) ---
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    # --- 1. 配置 ---
    INPUT_CSV_PATH = '../data/candidate_reactions_from_pool.csv'
    OUTPUT_CSV_PATH = '../data/predicted_valid_reactions.csv'
    NUM_GPUS = 4

    # 检查 GPU 可用性
    available_gpus = torch.cuda.device_count()
    if available_gpus < NUM_GPUS:
        print(f"Warning: 请求使用 {NUM_GPUS} 张卡, 但系统只有 {available_gpus} 张卡。")
        NUM_GPUS = available_gpus

    if NUM_GPUS == 0:
        print("Error: 未检测到 GPU，无法进行 GPU 并行预测。")
        return

    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: 输入文件未找到: {INPUT_CSV_PATH}")
        return

    # --- 2. 加载数据并拆分 ---
    print(f"Loading data from {INPUT_CSV_PATH}...")
    df = pd.read_csv(INPUT_CSV_PATH)
    required_cols = ['reactant1_smiles', 'reactant2_smiles', 'product_smiles']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: 输入 CSV 缺少必要列: {required_cols}")
        return

    total_samples = len(df)
    print(f"Total samples: {total_samples}. Splitting across {NUM_GPUS} GPUs.")

    # 使用 numpy array_split 均匀切分 DataFrame
    df_chunks = np.array_split(df, NUM_GPUS)

    # --- 3. 启动多进程 ---
    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []

    for rank in range(NUM_GPUS):
        p = mp.Process(target=worker_process, args=(rank, df_chunks[rank], result_queue))
        p.start()
        processes.append(p)

    # --- 4. 收集结果 ---
    combined_results = []

    # 等待所有进程完成
    for p in processes:
        p.join()

    print("All processes finished. Collecting results...")

    # 从队列中提取所有结果
    while not result_queue.empty():
        combined_results.extend(result_queue.get())

    print(
        f"Collected {len(combined_results)} valid predictions (filtered out {total_samples - len(combined_results)} invalid samples).")

    # --- 5. 保存结果 ---
    if combined_results:
        df_output = pd.DataFrame(combined_results)

        # 确保列顺序整齐
        cols = list(df_output.columns)
        if 'predicted_fingerprint_json' in cols:
            cols.remove('predicted_fingerprint_json')
            target_idx = cols.index('product_smiles') + 1 if 'product_smiles' in cols else len(cols)
            cols.insert(target_idx, 'predicted_fingerprint_json')
            df_output = df_output[cols]

        df_output.to_csv(OUTPUT_CSV_PATH, index=False)
        print("\n" + "=" * 50)
        print("Prediction complete.")
        print(f"Valid results saved to: {OUTPUT_CSV_PATH}")
        print("=" * 50)
    else:
        print("Warning: No valid predictions found.")


if __name__ == '__main__':
    main()