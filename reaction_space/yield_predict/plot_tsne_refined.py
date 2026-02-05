import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 激活3D绘图
from sklearn.manifold import TSNE
from rdkit import Chem
from tqdm import tqdm

# ================= 配置 =================
CACHE_DIR = './cache_features'
KNOWN_FEATS_CACHE = os.path.join(CACHE_DIR, 'known_features.npy')
CANDIDATE_FEATS_CACHE = os.path.join(CACHE_DIR, 'candidate_features.npy')

# 两个不同的 t-SNE 缓存文件
TSNE_CACHE_2D = os.path.join(CACHE_DIR, 'tsne_results.npy')  # 原来的2D
TSNE_CACHE_3D = os.path.join(CACHE_DIR, 'tsne_results_3d.npy')  # 新增的3D

# 数据文件路径
KNOWN_CSV_PATH = '../data/suzuki_miyaura_smiles_with_fingerprint_cleaned.csv'
CANDIDATE_CSV_PATH = 'predictions_output.csv'

# 目标反应 (星星)
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


# ================= 辅助函数 =================

def canonicalize_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        pass
    return smi


def load_data():
    """只读取缓存的特征和CSV"""
    print("--- Loading Feature Data ---")
    if not os.path.exists(KNOWN_FEATS_CACHE) or not os.path.exists(CANDIDATE_FEATS_CACHE):
        raise FileNotFoundError("特征缓存文件不存在，请先确保 cache_features 文件夹内有 .npy 文件")

    feats_known = np.load(KNOWN_FEATS_CACHE)
    df_known = pd.read_csv(KNOWN_CSV_PATH)
    feats_candidate = np.load(CANDIDATE_FEATS_CACHE)
    df_candidate = pd.read_csv(CANDIDATE_CSV_PATH)
    return feats_known, df_known, feats_candidate, df_candidate


def get_tsne_embedding(feats_known, feats_candidate, n_components=2, cache_path=None):
    """通用 t-SNE 计算函数，支持 2D 和 3D"""
    if os.path.exists(cache_path):
        print(f"Loading cached t-SNE ({n_components}D) from {cache_path}...")
        all_emb = np.load(cache_path)
    else:
        print(f"Calculating t-SNE ({n_components}D)... This may take a while.")
        all_feats = np.concatenate([feats_known, feats_candidate], axis=0)
        # 初始化 PCA 加速，random_state 固定复现结果
        tsne = TSNE(n_components=n_components, perplexity=30, n_iter=1000,
                    init='pca', learning_rate='auto', random_state=42,n_jobs=-1)
        all_emb = tsne.fit_transform(all_feats)
        np.save(cache_path, all_emb)
        print(f"Saved t-SNE result to {cache_path}")

    emb_known = all_emb[:len(feats_known)]
    emb_candidate = all_emb[len(feats_known):]
    return emb_known, emb_candidate


# ================= 核心绘图逻辑 =================

def main():
    # 1. 准备数据
    feats_known, df_known, feats_candidate, df_candidate = load_data()

    # 2. 筛选索引 (High Yield & Targets)
    print("Processing metadata indices...")
    pred_col = 'predicted_yield' if 'predicted_yield' in df_candidate.columns else 'yield'
    high_yield_mask = df_candidate[pred_col] > 80.0
    indices_high_yield = df_candidate[high_yield_mask].index.to_numpy()
    indices_normal = df_candidate[~high_yield_mask].index.to_numpy()

    # 匹配星星
    target_keys = set()
    for tgt in TARGET_REACTIONS:
        k = (canonicalize_smiles(tgt['r1']), canonicalize_smiles(tgt['r2']), canonicalize_smiles(tgt['p']))
        target_keys.add(k)

    r1_col = 'reactant1_smiles' if 'reactant1_smiles' in df_candidate.columns else 'reactant_1'
    r2_col = 'reactant2_smiles' if 'reactant2_smiles' in df_candidate.columns else 'reactant_2'
    p_col = 'product_smiles' if 'product_smiles' in df_candidate.columns else 'product'

    star_indices = []
    # 建立临时的 key 列表加速查找
    for idx, row in tqdm(df_candidate.iterrows(), total=len(df_candidate), desc="Matching Targets"):
        k = (canonicalize_smiles(row[r1_col]), canonicalize_smiles(row[r2_col]), canonicalize_smiles(row[p_col]))
        if k in target_keys:
            star_indices.append(idx)

    # 全局字体设置 (加大加粗)
    plt.rcParams.update({
        'font.size': 28,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'font.family': 'sans-serif'
    })

    # ================= 绘制 2D 图 =================
    print("\n--- Generating 2D Plot ---")
    emb_known_2d, emb_candidate_2d = get_tsne_embedding(
        feats_known, feats_candidate, n_components=2, cache_path=TSNE_CACHE_2D
    )

    fig_2d, ax_2d = plt.subplots(figsize=(20, 14), dpi=600)
    ax_2d.set_axis_off()  # 去边框

    # 绘制散点
    # 1. Virtual (Grey) - rasterized=True 防止SVG过大
    ax_2d.scatter(emb_candidate_2d[indices_normal, 0], emb_candidate_2d[indices_normal, 1],
                  c='lightgrey', s=50, alpha=0.3, edgecolors='none', rasterized=True,
                  label='Virtual Reactions')

    # 2. Known (Blue)
    ax_2d.scatter(emb_known_2d[:, 0], emb_known_2d[:, 1],
                  c='#1f77b4', s=50, alpha=0.7, edgecolors='none',
                  label='Known Training Data')

    # 3. High Yield (Red)
    ax_2d.scatter(emb_candidate_2d[indices_high_yield, 0], emb_candidate_2d[indices_high_yield, 1],
                  c='#d62728', s=50, alpha=0.8, edgecolors='none',
                  label='Top Candidates (>80%)')

    # 4. Stars (Orange)
    if star_indices:
        star_emb = emb_candidate_2d[star_indices]
        ax_2d.scatter(star_emb[:, 0], star_emb[:, 1],
                      c='#ff7f0e', s=600, marker='*', edgecolors='black', linewidth=2.0, zorder=10,
                      label='Case Validated')

    # Legend 设置 (右上角，大字体，无框)
    leg = ax_2d.legend(loc='upper left', fontsize=28, frameon=False, markerscale=2.0,handletextpad=0.1)
    # 强制让 Legend 的字也加粗
    for text in leg.get_texts():
        text.set_weight('bold')

    # plt.title('Chemical Space Exploration', fontsize=24, pad=10)
    plt.tight_layout()
    plt.savefig('chemical_space_2d.svg', format='svg', bbox_inches='tight')
    plt.savefig('chemical_space_2d.png', dpi=600, bbox_inches='tight')
    print("2D Plot Saved.")
    plt.close(fig_2d)

    # ================= 绘制 3D 图 =================
    print("\n--- Generating 3D Plot ---")
    emb_known_3d, emb_candidate_3d = get_tsne_embedding(
        feats_known, feats_candidate, n_components=3, cache_path=TSNE_CACHE_3D
    )

    fig_3d = plt.figure(figsize=(16, 14), dpi=300)
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # 3D 去除背景和刻度，营造悬浮感
    ax_3d.set_axis_off()
    # 如果想保留网格但不要背景色，可以如下设置 (可选):
    # ax_3d.xaxis.pane.fill = False
    # ax_3d.yaxis.pane.fill = False
    # ax_3d.zaxis.pane.fill = False
    # ax_3d.grid(False)

    # 绘制 3D 散点
    ax_3d.scatter(emb_candidate_3d[indices_normal, 0], emb_candidate_3d[indices_normal, 1],
                  emb_candidate_3d[indices_normal, 2],
                  c='lightgrey', s=5, alpha=0.1, edgecolors='none', rasterized=True)  # 3D点可以更小一点，增加通透感

    ax_3d.scatter(emb_known_3d[:, 0], emb_known_3d[:, 1], emb_known_3d[:, 2],
                  c='#1f77b4', s=15, alpha=0.5, edgecolors='none')

    ax_3d.scatter(emb_candidate_3d[indices_high_yield, 0], emb_candidate_3d[indices_high_yield, 1],
                  emb_candidate_3d[indices_high_yield, 2],
                  c='#d62728', s=20, alpha=0.6, edgecolors='none')

    if star_indices:
        star_emb = emb_candidate_3d[star_indices]
        ax_3d.scatter(star_emb[:, 0], star_emb[:, 1], star_emb[:, 2],
                      c='#ff7f0e', s=400, marker='*', edgecolors='black', linewidth=1.5, zorder=10)

    # 3D 图为了视觉效果，通常还是需要一个简单的 Legend
    # 为了不挡住图，我们把 Legend 放在图外
    # 创建一些 Proxy Artists 用来显示 Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Virtual Reactions', markerfacecolor='lightgrey', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Known Training Data', markerfacecolor='#1f77b4', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Top Candidates', markerfacecolor='#d62728', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Case Validated', markerfacecolor='#ff7f0e', markersize=10,
               markeredgecolor='black'),
    ]

    leg_3d = ax_3d.legend(handles=legend_elements, loc='upper right', fontsize=24, frameon=False,handletextpad=0.1)
    for text in leg_3d.get_texts():
        text.set_weight('bold')

    plt.title('Chemical Space Exploration', fontsize=28, pad=0)
    plt.tight_layout()

    # 调整视角 (Elev=30度俯视, Azim=45度旋转) - 你可以修改这些值来找最佳角度
    ax_3d.view_init(elev=30, azim=-60)

    plt.savefig('chemical_space_3d.png', dpi=300, bbox_inches='tight')
    # 3D 导出 SVG 可能会有遮挡排序问题，但也可以保留一份
    plt.savefig('chemical_space_3d.svg', format='svg', bbox_inches='tight')

    print("3D Plot Saved. Done!")


if __name__ == "__main__":
    main()