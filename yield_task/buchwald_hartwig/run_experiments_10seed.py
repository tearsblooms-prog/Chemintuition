# Run_YieldMPNN_Experiments.py
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from data import ReactionDataset, ATOM_FEATURE_SIZE, EDGE_FEATURE_SIZE
from model import YieldNet
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Data, Batch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from rdkit import RDLogger
from functools import partial
import time

# 禁用 RDKit 的一些警告信息，保持输出整洁
RDLogger.DisableLog('rdApp.warning')

# --- 原始配置参数 ---
# 注意: 为了加快示例运行速度，您可以适当减少 EPOCHS 的数量
BATCH_SIZE = 16
EPOCHS = 250
LEARNING_RATE = 0.00036402483412231295
WEIGHT_DECAY = 1.2899495348204742e-05
UNCERTAINTY_WEIGHT = 0.05522186434467659
CONTRASTIVE_LAMBDA = 0.18708920225157885
SPARSITY_ALPHA = 0.3627651040538812
MPNN_HIDDEN_FEATS = 128
MPNN_NUM_STEP_MESSAGE_PASSING = 4
MPNN_READOUT_FEATS = 512
PREDICT_HIDDEN_FEATS = 1024
PROB_DROPOUT = 0.19174391519919806

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- 原始脚本中的辅助函数 (无需修改) ---

def _load_condition_graph(chem_id: int, graph_store) -> Data:
    x = torch.from_numpy(graph_store[f'chem_x_{chem_id}'])
    edge_index = torch.from_numpy(graph_store[f'chem_ei_{chem_id}'])
    edge_attr = torch.from_numpy(graph_store[f'chem_ea_{chem_id}'])
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.shape[0])


def get_condition_seq_length(condition_vector, num_steps, feats_per_step):
    condition_reshaped = condition_vector.view(num_steps, feats_per_step)
    for i in range(num_steps - 1, -1, -1):
        if condition_reshaped[i, 0] == 1.0: return i + 1
    return 1


def custom_collate_fn(batch_list, graph_store):
    g1_list, g2_list, gp_list, condition_list, y_list = zip(*batch_list)
    condition_lengths = [get_condition_seq_length(c, 20, 32) for c in condition_list]
    sorted_indices = sorted(range(len(condition_lengths)), key=lambda k: condition_lengths[k], reverse=True)

    g1_sorted = [g1_list[i] for i in sorted_indices]
    g2_sorted = [g2_list[i] for i in sorted_indices]
    gp_sorted = [gp_list[i] for i in sorted_indices]
    cond_sorted = [condition_list[i] for i in sorted_indices]
    y_sorted = [y_list[i] for i in sorted_indices]

    g1_b = Batch.from_data_list(g1_sorted)
    g2_b = Batch.from_data_list(g2_sorted)
    gp_b = Batch.from_data_list(gp_sorted)
    y_b = torch.stack(y_sorted, dim=0)

    cond_b_reshaped = torch.stack(cond_sorted, dim=0).view(len(batch_list), 20, 32)
    RCS_ID_INDICES = list(range(1, 14))
    rcs_ids = cond_b_reshaped[:, :, RCS_ID_INDICES].long()

    unique_ids_list = torch.unique(rcs_ids[rcs_ids != 0]).cpu().tolist()
    valid_graphs, valid_id_to_graph_idx = [], {}
    for id_val in unique_ids_list:
        graph = _load_condition_graph(id_val, graph_store)
        if graph.num_nodes > 0:
            valid_id_to_graph_idx[id_val] = len(valid_graphs)
            valid_graphs.append(graph)

    cond_chem_graph_b = Batch.from_data_list(valid_graphs) if valid_graphs else None

    rcs_graph_indices_b = torch.full_like(rcs_ids, -1)
    for id_val, graph_idx in valid_id_to_graph_idx.items():
        rcs_graph_indices_b[rcs_ids == id_val] = graph_idx

    g1_b, g2_b, gp_b = g1_b.to(device), g2_b.to(device), gp_b.to(device)
    y_b = y_b.to(device)
    cond_chem_graph_b = cond_chem_graph_b.to(device) if cond_chem_graph_b else None
    rcs_graph_indices_b = rcs_graph_indices_b.to(device)

    return (g1_b, g2_b, gp_b, y_b, cond_chem_graph_b, rcs_graph_indices_b)


def enable_mc_dropout(model_to_set):
    for module in model_to_set.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def run_evaluation_with_uncertainty(model_to_eval, dataloader, current_device, mean_y, std_y, num_forward_passes=5):
    model_to_eval.eval()
    if num_forward_passes > 1: enable_mc_dropout(model_to_eval)
    preds_norm, logvars_norm, true_y = [], [], []
    with torch.no_grad():
        for (g1, g2, gp, y, c_graphs, c_indices) in dataloader:
            y = y.to(current_device)
            batch_preds, batch_logvars = [], []
            for _ in range(num_forward_passes):
                p_norm, l_norm, _, _, _ = model_to_eval(g1, g2, gp, c_graphs, c_indices)
                batch_preds.append(p_norm)
                batch_logvars.append(l_norm)
            preds_norm.append(torch.stack(batch_preds, dim=0))
            logvars_norm.append(torch.stack(batch_logvars, dim=0))
            true_y.append(y)
    if not true_y or len(true_y) == 0: return np.nan, np.nan, np.nan
    preds, y_true = torch.cat(preds_norm, dim=1), torch.cat(true_y, dim=0)
    preds_scaled = preds * std_y + mean_y
    mean_pred = torch.mean(preds_scaled, dim=0)
    y_true_np, y_pred_np = y_true.cpu().numpy(), mean_pred.cpu().numpy()

    # 防止验证集为空时出错
    if len(y_true_np) == 0:
        return np.nan, np.nan, np.nan

    return mean_absolute_error(y_true_np, y_pred_np), np.sqrt(mean_squared_error(y_true_np, y_pred_np)), r2_score(
        y_true_np, y_pred_np)


# --- 修改后的 run_experiment 函数 ---
# 增加了返回值，用于收集最佳性能指标
def run_experiment(csv_path, npz_path, val_ratio, random_seed, experiment_name):
    """
    运行单次训练和评估实验。

    返回:
        dict: 包含最佳 MAE, RMSE 和 R2 的字典。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n▶️ 开始实验: {experiment_name} on {device}")
    results_dir = os.path.join('results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    try:
        dataset = ReactionDataset(csv_path, npz_path)
    except FileNotFoundError:
        print(f"错误: 无法找到数据文件 {csv_path} 或 {npz_path}。请检查路径。")
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    # 确保训练集至少有一个样本
    if train_size == 0:
        print(f"警告: 划分比例 {val_ratio} 导致训练集大小为0。跳过此次实验。")
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}

    train_subset, val_subset = random_split(dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(random_seed))
    train_y_vals = dataset.data['y_val'].iloc[train_subset.indices].values
    mean, std = np.mean(train_y_vals), np.std(train_y_vals)
    mean_t, std_t = torch.tensor(mean, device=device), torch.tensor(std if std > 1e-6 else 1.0, device=device)
    collate_with_graphs = partial(custom_collate_fn, graph_store=dataset.graph_store)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_with_graphs,
                              drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_with_graphs)

    model = YieldNet(node_in_feats=ATOM_FEATURE_SIZE, edge_in_feats=EDGE_FEATURE_SIZE,
                     mpnn_hidden_feats=MPNN_HIDDEN_FEATS,
                     mpnn_num_step_message_passing=MPNN_NUM_STEP_MESSAGE_PASSING,
                     mpnn_readout_feats=MPNN_READOUT_FEATS,
                     predict_hidden_feats=PREDICT_HIDDEN_FEATS,
                     prob_dropout=PROB_DROPOUT).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss(reduction='none')
    cos_sim = nn.CosineSimilarity(dim=1)
    scheduler = MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)

    history = []
    best_val_mae = float('inf')
    best_val_rmse = float('inf')
    best_val_r2 = -float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_epoch_loss = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} (Seed {random_seed}, Val {val_ratio * 100}%)",
                          leave=False)
        for (g1, g2, gp, y, c_graphs, c_indices) in train_loop:
            optimizer.zero_grad()
            preds_norm, log_vars, reactant_feats, product_feats, all_gates = model(g1, g2, gp, c_graphs, c_indices)
            labels_norm = (y - mean_t) / std_t
            loss_terms = criterion(preds_norm, labels_norm)
            main_loss = ((1 - UNCERTAINTY_WEIGHT) * loss_terms + UNCERTAINTY_WEIGHT * (
                    loss_terms * torch.exp(-log_vars) + log_vars)).mean()
            positive_sim = cos_sim(reactant_feats, product_feats)
            shuffled_product_feats = product_feats[torch.randperm(product_feats.size(0))]
            negative_sim = cos_sim(reactant_feats, shuffled_product_feats)
            contrastive_loss = -torch.log(
                torch.exp(positive_sim) / (torch.exp(positive_sim) + torch.exp(negative_sim))).mean()
            if all_gates:
                all_gate_tensors = torch.cat(all_gates)
                sparsity_loss = torch.mean(all_gate_tensors)
            else:
                sparsity_loss = torch.tensor(0.0, device=device)
            total_loss = main_loss + CONTRASTIVE_LAMBDA * contrastive_loss + SPARSITY_ALPHA * sparsity_loss
            total_loss.backward()
            optimizer.step()
            total_epoch_loss += total_loss.item() * y.size(0)
            train_loop.set_postfix(loss=total_loss.item())

        avg_train_loss = total_epoch_loss / len(train_subset) if len(train_subset) > 0 else 0
        scheduler.step()
        val_mae, val_rmse, val_r2 = run_evaluation_with_uncertainty(model, val_loader, device, mean_t, std_t)

        if epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"\n--- Epoch {epoch}/{EPOCHS} Val (Seed {random_seed}, Val {val_ratio * 100}%) --- MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R2: {val_r2:.4f} ---")

        if not np.isnan(val_mae) and val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            best_val_r2 = val_r2
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))

        history.append(
            {'epoch': epoch, 'train_loss': avg_train_loss, 'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2})

    pd.DataFrame(history).to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    print(f"✅ 完成 '{experiment_name}': 最佳验证 MAE={best_val_mae:.4f}")

    return {'mae': best_val_mae, 'rmse': best_val_rmse, 'r2': best_val_r2}


# --- ★★★ 修改: 主执行模块 ★★★ ---
def main():
    """
    主函数，用于执行所有实验并生成结果报告。
    """
    # 定义要测试的验证集比例和总共10个随机种子
    SPLIT_RATIOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.975]

    # 扩展到10个种子：保留原来的 [46, 47, 124]，并添加7个新的
    # 你可以修改这7个新种子的值
    SEEDS = [46, 47, 124, 42, 88, 99, 101, 256, 512, 1024]
    print(f"ℹ️ 将为 {len(SEEDS)} 个种子运行实验: {SEEDS}")

    # 数据文件路径
    CLEANED_CSV_PATH = '../data/buchwald_hartwig_cleaned.csv'
    GRAPHS_NPZ_PATH = '../data/preprocessed_graphs.npz'

    # 检查核心数据文件是否存在
    if not os.path.exists(CLEANED_CSV_PATH) or not os.path.exists(GRAPHS_NPZ_PATH):
        print(f"错误: 核心数据文件未找到。请确保 '{CLEANED_CSV_PATH}' 和 '{GRAPHS_NPZ_PATH}' 存在。")
        return

    all_results = []
    start_time = time.time()

    for val_ratio in SPLIT_RATIOS:
        results_for_ratio = []
        train_ratio_percent = (1 - val_ratio) * 100
        val_ratio_percent = val_ratio * 100

        print("\n" + "=" * 60)
        print(f"🚀 开始新一轮划分比例: (训练/验证): {train_ratio_percent:.1f}/{val_ratio_percent:.1f}")
        print("=" * 60)

        for seed in SEEDS:

            experiment_name = f"Split_{train_ratio_percent:.1f}_{val_ratio_percent:.1f}_Seed_{seed}"
            results_dir = os.path.join('results', experiment_name)
            history_file_path = os.path.join(results_dir, 'training_history.csv')

            # ★★★ 新增逻辑: 检查结果是否已存在 ★★★
            if os.path.exists(history_file_path):
                print(f"✅ 实验 '{experiment_name}' 结果已存在，正在加载...")
                try:
                    history_df = pd.read_csv(history_file_path)
                    # 确保 val_mae 列是数值类型，并丢弃 NaN
                    history_df['val_mae'] = pd.to_numeric(history_df['val_mae'], errors='coerce')
                    history_df = history_df.dropna(subset=['val_mae'])

                    if history_df.empty:
                        print(f"   ...警告: 历史文件 '{history_file_path}' 为空或无有效MAE，将跳过。")
                        best_metrics = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}
                    else:
                        # 找到最佳MAE对应的行
                        best_epoch_idx = history_df['val_mae'].idxmin()
                        best_row = history_df.loc[best_epoch_idx]
                        best_metrics = {
                            'mae': best_row['val_mae'],
                            'rmse': best_row['val_rmse'],
                            'r2': best_row['val_r2']
                        }
                        print(f"   ...加载成功: 最佳 MAE={best_metrics['mae']:.4f}")
                except Exception as e:
                    print(f"   ...❌ 加载 '{history_file_path}' 失败: {e}。将跳过此种子。")
                    best_metrics = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}
            else:
                # 结果不存在，正常运行实验
                print(f"🚀 实验 '{experiment_name}' 结果不存在，开始新实验...")
                print("-" * 60)
                best_metrics = run_experiment(
                    csv_path=CLEANED_CSV_PATH,
                    npz_path=GRAPHS_NPZ_PATH,
                    val_ratio=val_ratio,
                    random_seed=seed,
                    experiment_name=experiment_name
                )

            # 收集结果 (无论是加载的还是新跑的)
            results_for_ratio.append(best_metrics)

        # 汇总当前划分比例下的所有种子结果
        maes = [r['mae'] for r in results_for_ratio if not np.isnan(r['mae'])]
        rmses = [r['rmse'] for r in results_for_ratio if not np.isnan(r['rmse'])]
        r2s = [r['r2'] for r in results_for_ratio if not np.isnan(r['r2'])]

        if maes:  # 仅当成功运行时才记录
            all_results.append({
                'val_ratio': val_ratio,
                'mae_mean': np.mean(maes), 'mae_std': np.std(maes),
                'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
                'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s),
                'num_seeds': len(maes)  # ★ 新增：记录实际汇总了多少个种子的结果
            })

    end_time = time.time()
    total_duration = (end_time - start_time) / 60
    print(f"\n\n所有实验已完成，总耗时: {total_duration:.2f} 分钟。")

    # --- 格式化并打印最终的结果表格 ---
    if not all_results:
        print("没有可供汇总的有效实验结果。")
        return

    results_df = pd.DataFrame(all_results)

    # 创建符合要求的最终表格
    final_table = pd.DataFrame()
    final_table['Model'] = ['YieldNet'] * len(results_df)

    # 格式化划分比例
    def format_split(r):
        train_p = 100 * (1 - r)
        val_p = 100 * r
        # 如果是整数，则不显示小数点
        return f"{int(train_p) if train_p == int(train_p) else f'{train_p:.1f}'}/{int(val_p) if val_p == int(val_p) else f'{val_p:.1f}'}"

    final_table['Split Ratio (Train/Val)'] = results_df['val_ratio'].apply(format_split)

    # 格式化性能指标
    final_table['MAE'] = results_df.apply(
        lambda row: f"{row['mae_mean']:.3f} ± {row['mae_std']:.3f}", axis=1
    )
    final_table['RMSE'] = results_df.apply(
        lambda row: f"{row['rmse_mean']:.3f} ± {row['rmse_std']:.3f}", axis=1
    )
    final_table['R²'] = results_df.apply(
        lambda row: f"{row['r2_mean']:.3f} ± {row['r2_std']:.3f}", axis=1
    )
    # ★ 新增：显示用于计算的种子数
    final_table['Seeds (N)'] = results_df['num_seeds'].astype(int)

    print("\n\n" + "🎉" * 15 + " 实验结果汇总 " + "🎉" * 15)
    # 使用 to_string() 保证对齐打印
    print(final_table.to_string(index=False))
    print("\n" + "=" * (55 + len("Seeds (N)") + 3))  # 调整分隔线宽度

    # 保存表格到本地文件
    try:
        final_table.to_csv('experiment_summary_formatted_10_seeds.csv', index=False)
        print("\n✅ 结果已成功保存到 'experiment_summary_formatted_10_seeds.csv'")
    except Exception as e:
        print(f"\n❌ 保存结果失败: {e}")


if __name__ == "__main__":
    # 运行主程序
    main()