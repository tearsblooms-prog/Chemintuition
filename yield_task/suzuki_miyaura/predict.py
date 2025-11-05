# predict.py
import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial
from torch_geometric.data import Data, Batch
from rdkit import RDLogger

from data import ReactionDataset, ATOM_FEATURE_SIZE, EDGE_FEATURE_SIZE
from model import YieldMPNN

RDLogger.DisableLog('rdApp.warning')

MODEL_PATH = 'results/suzuki_miyaura_DualChannel_Contrastive_seed46/best_model.pth'

CLEANED_CSV_PATH = 'data/data/new_reaction_cleaned.csv'
GRAPHS_NPZ_PATH = 'data/data/new_data_graphs.npz'

OUTPUT_CSV_PATH = 'out/predictions_output.csv'

ORIGINAL_TRAIN_CSV = 'data/suzuki_miyaura_smiles_with_fingerprint.csv'
TRAIN_VAL_RATIO = 0.3
TRAIN_RANDOM_SEED = 46
MPNN_HIDDEN_FEATS = 128
MPNN_NUM_STEP_MESSAGE_PASSING = 4
MPNN_READOUT_FEATS = 512
PREDICT_HIDDEN_FEATS = 1024
PROB_DROPOUT = 0.19174391519919806
BATCH_SIZE = 32

def get_train_stats(original_csv_path, val_ratio, random_seed):

    if not os.path.exists(original_csv_path):
        print(f"Original training data file '{original_csv_path}' not found.")
        print("Cannot compute TRAIN_MEAN and TRAIN_STD.")
        return None, None

    try:
        df = pd.read_csv(original_csv_path)
        if 'y_val' not in df.columns:
            print(f"'y_val' column is missing in '{original_csv_path}'.")
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

        if std < 1e-6:
            std = 1.0

        return mean, std

    except Exception as e:
        print(f"Error computing Mean/Std: {e}")
        return None, None

def _load_condition_graph(chem_id: int, graph_store) -> Data:
    """Load a single conditional chemistry graph from npz storage"""
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g1_b, g2_b, gp_b = g1_b.to(device), g2_b.to(device), gp_b.to(device)
    y_b = y_b.to(device)
    cond_chem_graph_b = cond_chem_graph_b.to(device) if cond_chem_graph_b else None
    rcs_graph_indices_b = rcs_graph_indices_b.to(device)

    return (g1_b, g2_b, gp_b, y_b, cond_chem_graph_b, rcs_graph_indices_b)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting prediction on {device}...")
    TRAIN_MEAN, TRAIN_STD = get_train_stats(
        ORIGINAL_TRAIN_CSV,
        TRAIN_VAL_RATIO,
        TRAIN_RANDOM_SEED
    )

    if TRAIN_MEAN is None or TRAIN_STD is None:
        print("Training statistics unavailable. Prediction aborted.")
        return

    print(f"TRAIN_MEAN = {TRAIN_MEAN:.6f}, TRAIN_STD = {TRAIN_STD:.6f}")

    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found.")
        return

    if not os.path.exists(CLEANED_CSV_PATH) or not os.path.exists(GRAPHS_NPZ_PATH):
        print(f"Preprocessed files '{CLEANED_CSV_PATH}' or '{GRAPHS_NPZ_PATH}' not found.")
        print("Please run 'preprocess_demo_data.py' first.")
        return

    mean_t = torch.tensor(TRAIN_MEAN, device=device, dtype=torch.float)
    std_t = torch.tensor(TRAIN_STD, device=device, dtype=torch.float)

    print(f"Loading model: {MODEL_PATH}")
    model = YieldMPNN(node_in_feats=ATOM_FEATURE_SIZE, edge_in_feats=EDGE_FEATURE_SIZE,
                     mpnn_hidden_feats=MPNN_HIDDEN_FEATS,
                     mpnn_num_step_message_passing=MPNN_NUM_STEP_MESSAGE_PASSING,
                     mpnn_readout_feats=MPNN_READOUT_FEATS,
                     predict_hidden_feats=PREDICT_HIDDEN_FEATS,
                     prob_dropout=PROB_DROPOUT).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"Loading new data: {CLEANED_CSV_PATH} and {GRAPHS_NPZ_PATH}")
    dataset = ReactionDataset(CLEANED_CSV_PATH, GRAPHS_NPZ_PATH)
    collate_with_graphs = partial(custom_collate_fn, graph_store=dataset.graph_store)
    loader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        collate_fn=collate_with_graphs)

    print("Running prediction...")
    all_preds_scaled = []

    with torch.no_grad():
        for (g1, g2, gp, y_dummy, c_graphs, c_indices) in tqdm(loader, desc="Predicting"):
            preds_norm, log_vars, _, _, _ = model(g1, g2, gp, c_graphs, c_indices)

            preds_scaled = preds_norm * std_t + mean_t

            all_preds_scaled.append(preds_scaled.cpu().numpy())

    predictions = np.concatenate(all_preds_scaled)

    print("Prediction complete. Saving results...")
    results_df = pd.read_csv(CLEANED_CSV_PATH)

    results_df['predicted_yield'] = predictions

    results_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Prediction finished.")
    print(f"Total reactions predicted: {len(predictions)}")
    print(f"Results saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
