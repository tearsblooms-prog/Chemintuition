import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from rdkit import Chem
from tqdm import tqdm
from functools import partial

# 导入你项目中的模块
from data_yield import ReactionDataset, ATOM_FEATURE_SIZE, EDGE_FEATURE_SIZE
from model_yield import YieldNet
# 复用 predict_yield.py 中的辅助函数
from predict_yield import custom_collate_fn

# ================= 配置 =================
# 路径配置
MODEL_PATH = '../results/suzuki_miyaura_DualChannel_Contrastive_seed46/best_model.pth'

# 1. 数据文件路径
KNOWN_CSV_PATH = '../data/suzuki_miyaura_smiles_with_fingerprint_cleaned.csv'
KNOWN_NPZ_PATH = '../data/suzuki_miyaura_graphs.npz'

CANDIDATE_CSV_PATH = 'predictions_output.csv'
CANDIDATE_NPZ_PATH = '../data/predicted_valid_reactions_graphs.npz'

# 2. 缓存配置 (新增)
CACHE_DIR = './cache_features'  # 缓存文件夹
KNOWN_FEATS_CACHE = os.path.join(CACHE_DIR, 'known_features.npy')
CANDIDATE_FEATS_CACHE = os.path.join(CACHE_DIR, 'candidate_features.npy')
FORCE_RECOMPUTE = False  # 如果设为 True，即使有缓存也会强制重新提取

# 3. 目标反应 (星星)
TARGET_REACTIONS = [
    {
        "r1": "O=C(OC)[C@@H](NC(C1=CC=CC=C1)=O)CSSC[C@@H](C(OC)=O)NC(C2=CC=CC=C2)=O",
        "r2": "O=P(C1=CC=CC=C1)C2=CC=CC=C2",
        "p": "O=C(OC)[C@H](CSP(C5=CC=CC=C5)(C6=CC=CC=C6)=O)NC(C7=CC=CC=C7)=O"
    },
    {
        "r1": "O=C(OC)[C@@H](NC(C)=O)CSSC[C@@H](C(OC)=O)NC(C)=O",
        "r2": "O=P(C1=CC=CC=C1)C2=CC=CC=C2",
        "p": "O=C(OC)[C@H](CSP(C3=CC=CC=C3)(C4=CC=CC=C4)=O)NC(C)=O"
    }
]

# 模型参数
MPNN_HIDDEN_FEATS = 128
MPNN_NUM_STEP_MESSAGE_PASSING = 4
MPNN_READOUT_FEATS = 512
PREDICT_HIDDEN_FEATS = 1024
PROB_DROPOUT = 0.19174391519919806
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================= 辅助函数 =================

def canonicalize_smiles(smi):
    """标准化 SMILES 以便精确匹配"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        pass
    return smi


def get_features(model, loader, device):
    """
    运行推理并提取特征。
    使用 Hook 获取全连接层之前的 Feature Vector。
    """
    features_list = []
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0].detach()

        return hook

    # 注册 Hook (根据你的模型结构，这里可能需要调整层名称)
    # 假设 predict 层的最后一个子模块是输出层，我们取它的输入
    handle = list(model.predict.children())[-1].register_forward_hook(get_activation('final_layer_input'))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Features"):
            g1, g2, gp, _, c_graphs, c_indices = batch

            g1 = g1.to(device)
            g2 = g2.to(device)
            gp = gp.to(device)
            if c_graphs is not None:
                c_graphs = c_graphs.to(device)
            c_indices = c_indices.to(device)

            _ = model(g1, g2, gp, c_graphs, c_indices)

            feats = activation['final_layer_input']
            features_list.append(feats.cpu().numpy())

    handle.remove()
    return np.concatenate(features_list, axis=0)


def load_or_extract_features(cache_path, csv_path, npz_path, model_loader_func):
    """
    通用函数：检查缓存，如果有则加载，无则调用模型提取并保存
    model_loader_func: 一个返回 (model, data_loader) 的函数，只有在需要计算时才调用
    """
    if os.path.exists(cache_path) and not FORCE_RECOMPUTE:
        print(f"Loading features from cache: {cache_path}")
        return np.load(cache_path), pd.read_csv(csv_path)

    print(f"Cache miss or forced recompute. extracting features for {csv_path}...")

    # 只有在这里才真正加载模型和数据加载器
    model, loader = model_loader_func()

    # 提取特征
    feats = get_features(model, loader, DEVICE)

    # 保存缓存
    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))
    np.save(cache_path, feats)
    print(f"Features saved to {cache_path}")

    return feats, pd.read_csv(csv_path)


# ================= 主逻辑 =================

def main():
    print(f"Using Device: {DEVICE}")

    # 确保缓存目录存在
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # 定义延迟加载模型的函数
    # 这样如果缓存都存在，我们根本不需要加载模型到显存里
    _model = None

    def get_model_instance():
        nonlocal _model
        if _model is None:
            print("Loading Neural Network Model...")
            m = YieldNet(node_in_feats=ATOM_FEATURE_SIZE, edge_in_feats=EDGE_FEATURE_SIZE,
                         mpnn_hidden_feats=MPNN_HIDDEN_FEATS,
                         mpnn_num_step_message_passing=MPNN_NUM_STEP_MESSAGE_PASSING,
                         mpnn_readout_feats=MPNN_READOUT_FEATS,
                         predict_hidden_feats=PREDICT_HIDDEN_FEATS,
                         prob_dropout=PROB_DROPOUT).to(DEVICE)
            m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            _model = m
        return _model

    # 准备数据加载器生成函数 (闭包)
    def prepare_known_loader():
        print("Preparing Known Data Loader...")
        model = get_model_instance()
        ds = ReactionDataset(KNOWN_CSV_PATH, KNOWN_NPZ_PATH)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=partial(custom_collate_fn, graph_store=ds.graph_store))
        return model, loader

    def prepare_candidate_loader():
        print("Preparing Candidate Data Loader...")
        model = get_model_instance()
        ds = ReactionDataset(CANDIDATE_CSV_PATH, CANDIDATE_NPZ_PATH)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=partial(custom_collate_fn, graph_store=ds.graph_store))
        return model, loader

    # 1. 获取特征 (自动判断是读取缓存还是重新计算)
    print("--- Step 1: Retrieving Features ---")
    feats_known, df_known = load_or_extract_features(
        KNOWN_FEATS_CACHE, KNOWN_CSV_PATH, KNOWN_NPZ_PATH, prepare_known_loader
    )
    print(f"Known features shape: {feats_known.shape}")

    feats_candidate, df_candidate = load_or_extract_features(
        CANDIDATE_FEATS_CACHE, CANDIDATE_CSV_PATH, CANDIDATE_NPZ_PATH, prepare_candidate_loader
    )
    print(f"Candidate features shape: {feats_candidate.shape}")

    # 2. t-SNE 降维
    print("\n--- Step 2: Running t-SNE ---")
    # 如果你也想缓存 t-SNE 的结果，可以在这里加类似的逻辑
    TSNE_CACHE = os.path.join(CACHE_DIR, 'tsne_results.npy')

    if os.path.exists(TSNE_CACHE) and not FORCE_RECOMPUTE:
        print(f"Loading t-SNE coordinates from cache: {TSNE_CACHE}")
        all_emb = np.load(TSNE_CACHE)
        emb_known = all_emb[:len(feats_known)]
        emb_candidate = all_emb[len(feats_known):]
    else:
        print("Running t-SNE calculation (this may take a while)...")
        all_feats = np.concatenate([feats_known, feats_candidate], axis=0)

        # 使用 PCA 初始化通常能加速并让结果更稳定
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', learning_rate='auto', random_state=42)
        all_emb = tsne.fit_transform(all_feats)

        np.save(TSNE_CACHE, all_emb)
        print("t-SNE result saved.")

        emb_known = all_emb[:len(feats_known)]
        emb_candidate = all_emb[len(feats_known):]

    # 3. 准备绘图元数据
    print("\n--- Step 3: Preparing Plot Metadata ---")

    # A. 处理 High Yield 标记
    if 'predicted_yield' not in df_candidate.columns:
        # 如果是第一次跑 predict_yield.py，列名可能是 'yield' 或者需要在 predict 脚本里确认
        # 这里做一个简单的兼容性检查，如果没有 predicted_yield，尝试用 yield
        pred_col = 'predicted_yield' if 'predicted_yield' in df_candidate.columns else 'yield'
        if pred_col not in df_candidate.columns:
            raise ValueError("Candidate CSV missing yield column.")
    else:
        pred_col = 'predicted_yield'

    high_yield_mask = df_candidate[pred_col] > 80.0
    indices_high_yield = df_candidate[high_yield_mask].index.to_numpy()
    indices_normal = df_candidate[~high_yield_mask].index.to_numpy()

    # B. 处理 Star Reactions (Target)
    print("Standardizing Target SMILES...")
    target_keys = []
    for tgt in TARGET_REACTIONS:
        k = (canonicalize_smiles(tgt['r1']),
             canonicalize_smiles(tgt['r2']),
             canonicalize_smiles(tgt['p']))
        target_keys.append(k)

    print("Matching candidates to targets...")
    r1_col = 'reactant1_smiles' if 'reactant1_smiles' in df_candidate.columns else 'reactant_1'
    r2_col = 'reactant2_smiles' if 'reactant2_smiles' in df_candidate.columns else 'reactant_2'
    p_col = 'product_smiles' if 'product_smiles' in df_candidate.columns else 'product'

    star_indices = []
    # 使用 set 查找加速（如果不需要知道具体匹配哪一个 target，只需要知道是否是 target）
    # 但为了兼容 list 查找逻辑，这里还是遍历

    # 这是一个耗时操作，如果 candidate 很大，建议也缓存 indices
    # 这里我们只构建一次 lookup keys
    candidate_keys = []
    for idx, row in tqdm(df_candidate.iterrows(), total=len(df_candidate), desc="Building SMILES keys"):
        k = (canonicalize_smiles(row[r1_col]),
             canonicalize_smiles(row[r2_col]),
             canonicalize_smiles(row[p_col]))
        candidate_keys.append(k)

    for i, t_key in enumerate(target_keys):
        try:
            idx = candidate_keys.index(t_key)
            star_indices.append(idx)
            print(f"  [Match] Target reaction #{i + 1} found at index {idx}")
        except ValueError:
            print(f"  [Miss] Target reaction #{i + 1} not found in candidates")

    # 4. 绘图
    print("\n--- Step 4: Plotting ---")
    plt.figure(figsize=(12, 10), dpi=300)

    # 层级 1: 普通候选 (Grey)
    plt.scatter(emb_candidate[indices_normal, 0], emb_candidate[indices_normal, 1],
                c='lightgrey', s=5, alpha=0.3, label='Virtual Reactions', edgecolors='none')

    # 层级 2: 已知数据 (Blue)
    plt.scatter(emb_known[:, 0], emb_known[:, 1],
                c='#1f77b4', s=10, alpha=0.6, label='Known Training Data', edgecolors='none')

    # 层级 3: 高产率候选 (Red)
    plt.scatter(emb_candidate[indices_high_yield, 0], emb_candidate[indices_high_yield, 1],
                c='#d62728', s=15, alpha=0.7, label='Top Candidates (>80%)', edgecolors='none')

    # 层级 4: Star Case (Orange Star)
    if star_indices:
        star_emb = emb_candidate[star_indices]
        plt.scatter(star_emb[:, 0], star_emb[:, 1],
                    c='#ff7f0e', s=300, marker='*', edgecolors='black', linewidth=1.5,
                    label='Case Validated', zorder=10)

    plt.title('Chemical Space Exploration (t-SNE Visualization)', fontsize=16)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right')  # 调整 Legend 位置以免遮挡
    plt.tight_layout()

    save_path = 'chemical_space_tsne.png'
    plt.savefig(save_path)
    print(f"Done! Plot saved to {save_path}")


if __name__ == "__main__":
    main()