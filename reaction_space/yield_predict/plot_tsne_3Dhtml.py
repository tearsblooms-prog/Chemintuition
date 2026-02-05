import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go  # 引入 Plotly
from sklearn.manifold import TSNE
from rdkit import Chem
from tqdm import tqdm

# ================= 配置 =================
CACHE_DIR = './cache_features'
KNOWN_FEATS_CACHE = os.path.join(CACHE_DIR, 'known_features.npy')
CANDIDATE_FEATS_CACHE = os.path.join(CACHE_DIR, 'candidate_features.npy')

# 两个不同的 t-SNE 缓存文件
TSNE_CACHE_2D = os.path.join(CACHE_DIR, 'tsne_results.npy')
TSNE_CACHE_3D = os.path.join(CACHE_DIR, 'tsne_results_3d.npy')

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
    """通用 t-SNE 计算函数"""
    if os.path.exists(cache_path):
        print(f"Loading cached t-SNE ({n_components}D) from {cache_path}...")
        all_emb = np.load(cache_path)
    else:
        print(f"Calculating t-SNE ({n_components}D)... This may take a while.")
        all_feats = np.concatenate([feats_known, feats_candidate], axis=0)
        # 初始化 PCA 加速
        tsne = TSNE(n_components=n_components, perplexity=30, n_iter=1000,
                    init='pca', learning_rate='auto', random_state=42, n_jobs=-1)
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
    print("Matching Targets...")
    for idx, row in tqdm(df_candidate.iterrows(), total=len(df_candidate), desc="Matching"):
        k = (canonicalize_smiles(row[r1_col]), canonicalize_smiles(row[r2_col]), canonicalize_smiles(row[p_col]))
        if k in target_keys:
            star_indices.append(idx)

    # ================= 绘制 3D 交互图 (Plotly) =================
    print("\n--- Generating Interactive 3D Plot (Plotly) ---")

    # 获取 3D 数据
    emb_known_3d, emb_candidate_3d = get_tsne_embedding(
        feats_known, feats_candidate, n_components=3, cache_path=TSNE_CACHE_3D
    )

    fig = go.Figure()

    # 1. Virtual Reactions (Grey) - 数量最多，放在最底层
    # 为了防止 HTML 文件过大导致浏览器卡顿，如果点数超过 50,000，建议适当降采样或减小 size
    fig.add_trace(go.Scatter3d(
        x=emb_candidate_3d[indices_normal, 0],
        y=emb_candidate_3d[indices_normal, 1],
        z=emb_candidate_3d[indices_normal, 2],
        mode='markers',
        marker=dict(
            size=2,  # 点的大小
            color='lightgrey',  # 颜色
            opacity=0.2  # 透明度，设低一点增加通透感
        ),
        name='Virtual Reactions',
        hoverinfo='skip'  # 为了性能，大量的背景点可以跳过悬停显示
    ))

    # 2. Known Training Data (Blue)
    fig.add_trace(go.Scatter3d(
        x=emb_known_3d[:, 0],
        y=emb_known_3d[:, 1],
        z=emb_known_3d[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='#1f77b4',
            opacity=0.6
        ),
        name='Known Training Data'
    ))

    # 3. Top Candidates (Red)
    fig.add_trace(go.Scatter3d(
        x=emb_candidate_3d[indices_high_yield, 0],
        y=emb_candidate_3d[indices_high_yield, 1],
        z=emb_candidate_3d[indices_high_yield, 2],
        mode='markers',
        marker=dict(
            size=4,
            color='#d62728',
            opacity=0.8
        ),
        name='Top Candidates (>80%)'
    ))

    # 4. Stars (Case Validated) - 突出显示
    if star_indices:
        star_emb = emb_candidate_3d[star_indices]
        # 获取相关信息用于 Hover
        star_info = df_candidate.iloc[star_indices]
        hover_texts = [f"Yield: {row[pred_col]:.2f}%" for _, row in star_info.iterrows()]

        fig.add_trace(go.Scatter3d(
            x=star_emb[:, 0],
            y=star_emb[:, 1],
            z=star_emb[:, 2],
            mode='markers',
            marker=dict(
                size=12,
                color='#ff7f0e',
                symbol='diamond',  # 3D 中 diamond 比较接近星星的效果
                line=dict(color='black', width=2),  # 黑色描边
                opacity=1.0
            ),
            text=hover_texts,  # 鼠标悬停显示具体产率
            name='Case Validated'
        ))

    # 设置布局样式
    fig.update_layout(
        title='Chemical Space Exploration',
        scene=dict(
            xaxis=dict(visible=False),  # 隐藏坐标轴，更像“星空”
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'  # 背景透明
        ),
        paper_bgcolor='white',
        legend=dict(
            itemsizing='constant',
            font=dict(size=14)
        ),
        margin=dict(l=0, r=0, b=0, t=40)  # 减少边距
    )

    output_file = 'chemical_space_3d_interactive.html'
    fig.write_html(output_file)
    print(f"Interactive 3D plot saved to {output_file}")
    print("You can open this file in any web browser.")


if __name__ == "__main__":
    main()