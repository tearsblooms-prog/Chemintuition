import torch
import pandas as pd
from torch_geometric.data import Data
import numpy as np

# The atom feature size is 47, calculated as follows:
# 35 (one-hot atomic_num) + 1 (degree) + 1 (formal_charge) + 1 (rad_electrons) +
# 6 (one-hot hybridization) + 1 (is_aromatic) + 1 (total_hs) + 1 (is_potential_center)
ATOM_FEATURE_SIZE = 47
EDGE_FEATURE_SIZE = 4


class ReactionDataset(torch.utils.data.Dataset):
    """
    Load reaction data from preprocessed CSV and NPZ files containing all graphs (main reaction + conditions).
    """

    def __init__(self, csv_file_path, graphs_npz_path):
        """
        Args:
            csv_file_path (str): Path to the cleaned CSV file.
            graphs_npz_path (str): Path to the NPZ file containing preprocessed graphs.
        """
        print(f"Loading cleaned data from: {csv_file_path}")
        self.data = pd.read_csv(csv_file_path)

        print(f"Loading preprocessed graphs from: {graphs_npz_path}...")
        self.graph_store = np.load(graphs_npz_path)
        print("Graph data loaded successfully.")

    def __len__(self):
        return len(self.data)

    def _load_graph(self, prefix: str, index: int) -> Data:
        """
        Load a graph from NPZ storage and convert it to a torch_geometric.data.Data object.
        """
        x = torch.from_numpy(self.graph_store[f'{prefix}_x_{index}'])
        edge_index = torch.from_numpy(self.graph_store[f'{prefix}_ei_{index}'])
        edge_attr = torch.from_numpy(self.graph_store[f'{prefix}_ea_{index}'])

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.shape[0])

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        g1 = self._load_graph('r1', idx)
        g2 = self._load_graph('r2', idx)
        gp = self._load_graph('p', idx)

        reaction_condition = torch.tensor(eval(row.get('condition_fingerprint', '[]')), dtype=torch.float)
        y = torch.tensor(float(row['y_val']), dtype=torch.float)

        return g1, g2, gp, reaction_condition, y
