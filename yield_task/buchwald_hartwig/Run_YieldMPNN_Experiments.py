# Run_YieldMPNN_Experiments.py
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from data import ReactionDataset, ATOM_FEATURE_SIZE, EDGE_FEATURE_SIZE
from model import YieldMPNN
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Data, Batch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from rdkit import RDLogger
from functools import partial
import time

RDLogger.DisableLog('rdApp.warning')

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

def _load_condition_graph(chem_id: int, graph_store) -> Data:
    x = torch.from_numpy(graph_store[f'chem_x_{chem_id}'])
    edge_index = torch.from_numpy(graph_store[f'chem_ei_{chem_id}'])
    edge_attr = torch.from_numpy(graph_store[f'chem_ea_{chem_id}'])
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.shape[0])


def get_condition_seq_length(condition_vector, num_steps, feats_per_step):
    condition_reshaped = condition_vector.view(num_steps, feats_per_step)
    for i in range(num_steps - 1, -1, -1):
        if condition_reshaped[i, 0] == 1.0:
            return i + 1
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
    if num_forward_passes > 1:
        enable_mc_dropout(model_to_eval)
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
    if not true_y or len(true_y) == 0:
        return np.nan, np.nan, np.nan
    preds, y_true = torch.cat(preds_norm, dim=1), torch.cat(true_y, dim=0)
    preds_scaled = preds * std_y + mean_y
    mean_pred = torch.mean(preds_scaled, dim=0)
    y_true_np, y_pred_np = y_true.cpu().numpy(), mean_pred.cpu().numpy()

    if len(y_true_np) == 0:
        return np.nan, np.nan, np.nan

    return mean_absolute_error(y_true_np, y_pred_np), np.sqrt(mean_squared_error(y_true_np, y_pred_np)), r2_score(
        y_true_np, y_pred_np)


def run_experiment(csv_path, npz_path, val_ratio, random_seed, experiment_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStart experiment: {experiment_name} on {device}")
    results_dir = os.path.join('results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    try:
        dataset = ReactionDataset(csv_path, npz_path)
    except FileNotFoundError:
        print(f"Error: Cannot find data files {csv_path} or {npz_path}. Please check paths.")
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    if train_size == 0:
        print(f"Warning: Split ratio {val_ratio} results in 0 training samples. Skipping experiment.")
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

    model = YieldMPNN(node_in_feats=ATOM_FEATURE_SIZE, edge_in_feats=EDGE_FEATURE_SIZE,
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

        avg_train_loss = total_epoch_loss / len(train_subset) if len(train_subset) > 0 else 0
        scheduler.step()
        val_mae, val_rmse, val_r2 = run_evaluation_with_uncertainty(model, val_loader, device, mean_t, std_t)

        if not np.isnan(val_mae) and val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            best_val_r2 = val_r2
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))

        history.append(
            {'epoch': epoch, 'train_loss': avg_train_loss, 'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2})

    pd.DataFrame(history).to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    print(f"Finished '{experiment_name}': Best Validation MAE={best_val_mae:.4f}")

    return {'mae': best_val_mae, 'rmse': best_val_rmse, 'r2': best_val_r2}


def main():
    SPLIT_RATIOS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.975]
    SEEDS = [46, 47, 124]

    CLEANED_CSV_PATH = 'data/buchwald_hartwig_cleaned.csv'
    GRAPHS_NPZ_PATH = 'data/preprocessed_graphs.npz'

    if not os.path.exists(CLEANED_CSV_PATH) or not os.path.exists(GRAPHS_NPZ_PATH):
        print(f"Error: Core data files not found. Please ensure '{CLEANED_CSV_PATH}' and '{GRAPHS_NPZ_PATH}' exist.")
        return

    all_results = []
    start_time = time.time()

    for val_ratio in SPLIT_RATIOS:
        results_for_ratio = []
        train_ratio_percent = (1 - val_ratio) * 100
        val_ratio_percent = val_ratio * 100

        for seed in SEEDS:
            experiment_name = f"Split_{train_ratio_percent:.1f}_{val_ratio_percent:.1f}_Seed_{seed}"

            best_metrics = run_experiment(
                csv_path=CLEANED_CSV_PATH,
                npz_path=GRAPHS_NPZ_PATH,
                val_ratio=val_ratio,
                random_seed=seed,
                experiment_name=experiment_name
            )
            results_for_ratio.append(best_metrics)

        maes = [r['mae'] for r in results_for_ratio if not np.isnan(r['mae'])]
        rmses = [r['rmse'] for r in results_for_ratio if not np.isnan(r['rmse'])]
        r2s = [r['r2'] for r in results_for_ratio if not np.isnan(r['r2'])]

        if maes:
            all_results.append({
                'val_ratio': val_ratio,
                'mae_mean': np.mean(maes), 'mae_std': np.std(maes),
                'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
                'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s),
            })

    end_time = time.time()
    total_duration = (end_time - start_time) / 60
    print(f"All experiments finished. Total time: {total_duration:.2f} minutes.")

    if not all_results:
        print("No valid results to summarize.")
        return

    results_df = pd.DataFrame(all_results)

    final_table = pd.DataFrame()
    final_table['Model'] = ['YieldMPNN'] * len(results_df)

    def format_split(r):
        train_p = 100 * (1 - r)
        val_p = 100 * r
        return f"{int(train_p) if train_p == int(train_p) else f'{train_p:.1f}'}/{int(val_p) if val_p == int(val_p) else f'{val_p:.1f}'}"

    final_table['Split Ratio (Train/Val)'] = results_df['val_ratio'].apply(format_split)
    final_table['MAE'] = results_df.apply(
        lambda row: f"{row['mae_mean']:.3f} ± {row['mae_std']:.3f}", axis=1
    )
    final_table['RMSE'] = results_df.apply(
        lambda row: f"{row['rmse_mean']:.3f} ± {row['rmse_std']:.3f}", axis=1
    )
    final_table['R²'] = results_df.apply(
        lambda row: f"{row['r2_mean']:.3f} ± {row['r2_std']:.3f}", axis=1
    )

    print(final_table.to_string(index=False))

    try:
        final_table.to_csv('experiment_summary_formatted.csv', index=False)
        print("Results saved to 'experiment_summary_formatted.csv'")
    except Exception as e:
        print(f"Failed to save results: {e}")


if __name__ == "__main__":
    main()
