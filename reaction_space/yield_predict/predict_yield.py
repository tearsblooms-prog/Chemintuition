# predict_multigpu.py
import os
import glob
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from torch_geometric.data import Data, Batch
from rdkit import RDLogger

# 从你的项目中导入必要的模块
from data_yield import ReactionDataset, ATOM_FEATURE_SIZE, EDGE_FEATURE_SIZE
from model_yield import YieldNet

RDLogger.DisableLog('rdApp.warning')

# ==============================================================================
# 预测配置
# ==============================================================================
MODEL_PATH = '../results/suzuki_miyaura_DualChannel_Contrastive_seed46/best_model.pth'
CLEANED_CSV_PATH = '../data/predicted_valid_reactions_cleaned.csv'
GRAPHS_NPZ_PATH = '../data/predicted_valid_reactions_graphs.npz'
OUTPUT_CSV_PATH = 'predictions_output.csv'

# 原始训练数据配置 (用于计算 Mean/Std)
ORIGINAL_TRAIN_CSV = '../data/suzuki_miyaura_smiles_with_fingerprint.csv'
TRAIN_VAL_RATIO = 0.3
TRAIN_RANDOM_SEED = 46

# 模型超参数
MPNN_HIDDEN_FEATS = 128
MPNN_NUM_STEP_MESSAGE_PASSING = 4
MPNN_READOUT_FEATS = 512
PREDICT_HIDDEN_FEATS = 1024
PROB_DROPOUT = 0.19174391519919806

# 推理时每个GPU使用的批量大小
BATCH_SIZE = 32


# ==============================================================================
# 辅助函数
# ==============================================================================

def get_train_stats(original_csv_path, val_ratio, random_seed):
    """
    加载原始训练CSV，执行与 Train_YieldMPNN.py 相同的分割，
    并计算训练集的 mean 和 std。
    """
    if not os.path.exists(original_csv_path):
        print(f"错误: 找不到原始训练数据文件 '{original_csv_path}'")
        return None, None

    try:
        df = pd.read_csv(original_csv_path)
        if 'y_val' not in df.columns:
            print(f"错误: 原始数据中缺少 'y_val' 列。")
            return None, None

        total_size = len(df)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size

        indices = list(range(total_size))
        train_subset_indices, _ = random_split(
            indices,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )

        train_y_vals = df['y_val'].iloc[train_subset_indices.indices].values
        mean = np.mean(train_y_vals)
        std = np.std(train_y_vals)

        if std < 1e-6: std = 1.0
        return mean, std

    except Exception as e:
        print(f"计算 Mean/Std 时出错: {e}")
        return None, None


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

    # 注意：这里我们不再在 collate 中 to(device)，而是在循环中处理
    # 这样可以兼容多进程 DataLoader
    return (g1_b, g2_b, gp_b, y_b, cond_chem_graph_b, rcs_graph_indices_b)


# ==============================================================================
# Worker: 单个 GPU 的预测逻辑
# ==============================================================================
def run_inference_worker(rank, world_size, train_mean, train_std, total_len):
    """
    这是每个 GPU 进程运行的函数
    """
    try:
        # 1. 设置设备
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        print(f"[Rank {rank}] 启动，使用设备: {device}")

        # 2. 转换 Mean/Std 为 Tensor
        mean_t = torch.tensor(train_mean, device=device, dtype=torch.float)
        std_t = torch.tensor(train_std, device=device, dtype=torch.float)

        # 3. 加载模型
        model = YieldNet(node_in_feats=ATOM_FEATURE_SIZE, edge_in_feats=EDGE_FEATURE_SIZE,
                         mpnn_hidden_feats=MPNN_HIDDEN_FEATS,
                         mpnn_num_step_message_passing=MPNN_NUM_STEP_MESSAGE_PASSING,
                         mpnn_readout_feats=MPNN_READOUT_FEATS,
                         predict_hidden_feats=PREDICT_HIDDEN_FEATS,
                         prob_dropout=PROB_DROPOUT).to(device)

        # 加载权重，指定 map_location 避免显存浪费
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # 4. 加载数据集
        # 每个进程都需要重新实例化 Dataset (读取 npz)，这会消耗一些内存，但对于多卡推理是安全的
        full_dataset = ReactionDataset(CLEANED_CSV_PATH, GRAPHS_NPZ_PATH)

        # 5. 计算当前 GPU 应该处理的数据索引范围 (Sequential Split)
        # 我们使用 array_split 保证顺序，方便后续拼接
        all_indices = np.arange(len(full_dataset))
        my_indices = np.array_split(all_indices, world_size)[rank]

        if len(my_indices) == 0:
            print(f"[Rank {rank}] 没有分配到数据，跳过。")
            return

        # 创建子集
        subset = Subset(full_dataset, my_indices)

        # Collate fn
        collate_with_graphs = partial(custom_collate_fn, graph_store=full_dataset.graph_store)

        loader = DataLoader(subset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,  # 必须为 False 保证顺序
                            num_workers=0,  # 在子进程中设为 0 比较安全
                            collate_fn=collate_with_graphs)

        print(f"[Rank {rank}] 开始预测 {len(my_indices)} 条数据...")

        local_preds = []

        with torch.no_grad():
            # 使用 tqdm 仅在 Rank 0 显示进度条，避免混乱
            iterator = loader
            if rank == 0:
                iterator = tqdm(loader, desc=f"Rank {rank} Processing")

            for (g1, g2, gp, _, c_graphs, c_indices) in iterator:
                # 移动数据到当前 GPU
                g1 = g1.to(device)
                g2 = g2.to(device)
                gp = gp.to(device)
                if c_graphs is not None:
                    c_graphs = c_graphs.to(device)
                c_indices = c_indices.to(device)

                preds_norm, _, _, _, _ = model(g1, g2, gp, c_graphs, c_indices)

                # 反归一化
                preds_scaled = preds_norm * std_t + mean_t
                local_preds.append(preds_scaled.cpu().numpy())

        # 6. 保存当前 GPU 的结果到临时文件
        if local_preds:
            final_local_preds = np.concatenate(local_preds)
        else:
            final_local_preds = np.array([])

        temp_filename = f"temp_preds_rank_{rank}.npy"
        np.save(temp_filename, final_local_preds)
        print(f"[Rank {rank}] 完成。已保存 {len(final_local_preds)} 条结果到 {temp_filename}")

    except Exception as e:
        print(f"[Rank {rank}] 发生错误: {e}")
        raise e


# ==============================================================================
# 主控制逻辑
# ==============================================================================
def main():
    # 1. 检测 GPU
    if not torch.cuda.is_available():
        print("错误: 未检测到 GPU，无法使用多卡预测。")
        return

    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 张 GPU。准备启动 {world_size} 个进程。")

    # 2. 检查文件
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 {MODEL_PATH}")
        return
    if not os.path.exists(CLEANED_CSV_PATH):
        print(f"错误: 数据文件不存在 {CLEANED_CSV_PATH}")
        return

    # 3. 计算 Mean/Std (主进程计算一次即可)
    print(f"正在计算训练集统计数据 (Mean/Std)...")
    TRAIN_MEAN, TRAIN_STD = get_train_stats(ORIGINAL_TRAIN_CSV, TRAIN_VAL_RATIO, TRAIN_RANDOM_SEED)

    if TRAIN_MEAN is None:
        print("无法计算统计数据，退出。")
        return
    print(f"TRAIN_MEAN = {TRAIN_MEAN:.6f}, TRAIN_STD = {TRAIN_STD:.6f}")

    # 4. 获取数据集总长度 (用于传递给子进程，虽然后面重新加载了，但这步确认文件没问题)
    temp_df = pd.read_csv(CLEANED_CSV_PATH)
    total_len = len(temp_df)
    del temp_df

    # 5. 启动多进程
    print("启动多进程预测...")
    try:
        mp.spawn(
            run_inference_worker,
            args=(world_size, TRAIN_MEAN, TRAIN_STD, total_len),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"多进程执行出错: {e}")
        return

    # 6. 合并结果
    print("正在合并所有 GPU 的预测结果...")
    all_predictions = []

    for rank in range(world_size):
        fname = f"temp_preds_rank_{rank}.npy"
        if os.path.exists(fname):
            part_pred = np.load(fname)
            all_predictions.append(part_pred)
            # 删除临时文件
            os.remove(fname)
        else:
            print(f"警告: 缺少 Rank {rank} 的结果文件 ({fname})")
            return

    final_predictions = np.concatenate(all_predictions)

    # 7. 保存最终 CSV
    results_df = pd.read_csv(CLEANED_CSV_PATH)

    # 安全性检查：长度是否匹配
    if len(final_predictions) != len(results_df):
        print(f"警告: 预测结果数量 ({len(final_predictions)}) 与 CSV 行数 ({len(results_df)}) 不一致!")
        print("将尝试截断或填充对齐。")
        if len(final_predictions) > len(results_df):
            final_predictions = final_predictions[:len(results_df)]
        else:
            # 可能是最后没整除丢弃了一点，通常截断 CSV
            results_df = results_df.iloc[:len(final_predictions)]

    results_df['predicted_yield'] = final_predictions
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("=" * 60)
    print(f"多卡预测完成！")
    print(f"总计预测: {len(final_predictions)}")
    print(f"结果已保存: {OUTPUT_CSV_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    # 设置启动方法，Linux下 spawn 通常更安全 (对于 CUDA context)
    # Windows下默认就是 spawn
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()