import torch
import torch.nn as nn
from torch_geometric.nn import Set2Set, SAGEConv
from torch_geometric.data import Batch


class ChemSReactMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, hidden_feats=64,
                 num_step_message_passing=3, num_step_set2set=3, num_layer_set2set=1,
                 readout_feats=1024):
        super(ChemSReactMPNN, self).__init__()
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        self.gnn_layer = SAGEConv(hidden_feats, hidden_feats)
        self.activation = nn.ReLU()

        # --- Dual-Channel Attention Mechanism ---
        self.gate_nn_structure = nn.Sequential(nn.Linear(hidden_feats, 1), nn.Sigmoid())
        self.gate_nn_reactivity = nn.Sequential(nn.Linear(hidden_feats, 1), nn.Sigmoid())
        self.gru_s = nn.GRU(hidden_feats, hidden_feats)
        self.gru_r = nn.GRU(hidden_feats, hidden_feats)

        # Input to Set2Set is now projected_feats + structure_feats + reactivity_feats
        self.readout = Set2Set(in_channels=hidden_feats * 3,
                               processing_steps=num_step_set2set,
                               num_layers=num_layer_set2set)

        # Input to the final linear layer is doubled by Set2Set
        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 6, readout_feats), nn.PReLU()
        )

    def forward(self, g: Batch, return_node_features=False):
        if g is None or g.num_nodes == 0:
            device = next(self.parameters()).device
            # Default return for training/evaluation
            default_return = (torch.zeros((1, 512), device=device), [])
            if return_node_features:
                # Return empty node features and gates for visualization
                node_vis_data = (torch.zeros((0, self.readout.in_channels), device=device), [])
                return default_return[0], node_vis_data
            return default_return

        node_feats, edge_index, edge_feats, batch = g.x, g.edge_index, g.edge_attr, g.batch

        projected_node_feats = self.project_node_feats(node_feats)
        h_gru_s = projected_node_feats.unsqueeze(0)
        h_gru_r = projected_node_feats.unsqueeze(0)
        current_node_feats_s = projected_node_feats
        current_node_feats_r = projected_node_feats

        all_reactivity_gates = []

        for _ in range(self.num_step_message_passing):
            # Message calculation is shared across channels
            m = self.gnn_layer(current_node_feats_s, edge_index)  # Using structure feats as input
            activated_m = self.activation(m)

            # Structure Channel Update
            gate_s = self.gate_nn_structure(current_node_feats_s)
            gated_message_s = gate_s * activated_m
            _, h_gru_s = self.gru_s(gated_message_s.unsqueeze(0), h_gru_s)
            current_node_feats_s = h_gru_s.squeeze(0)

            # Reactivity Channel Update
            gate_r = self.gate_nn_reactivity(current_node_feats_r)
            all_reactivity_gates.append(gate_r)  # Store for auxiliary loss and visualization
            gated_message_r = gate_r * activated_m
            _, h_gru_r = self.gru_r(gated_message_r.unsqueeze(0), h_gru_r)
            current_node_feats_r = h_gru_r.squeeze(0)

        # Concatenate features from all channels for graph-level readout
        node_aggr_for_readout = torch.cat([projected_node_feats, current_node_feats_s, current_node_feats_r], dim=1)
        readout_output = self.readout(node_aggr_for_readout, batch)
        graph_feats = self.sparsify(readout_output)

        if return_node_features:
            # Return graph features and the detailed node-level data for visualization
            node_vis_data = (node_aggr_for_readout, all_reactivity_gates)
            return graph_feats, node_vis_data

        # Return graph features and reactivity gates for loss calculation
        return graph_feats, all_reactivity_gates


class YieldMPNN(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats,
                 mpnn_hidden_feats=64, mpnn_num_step_message_passing=3,
                 mpnn_num_step_set2set=3, mpnn_num_layer_set2set=1,
                 mpnn_readout_feats=1024,
                 predict_hidden_feats=512, prob_dropout=0.1):
        super(YieldMPNN, self).__init__()
        self.mpnn = ChemSReactMPNN(node_in_feats, edge_in_feats,
                         hidden_feats=mpnn_hidden_feats,
                         num_step_message_passing=mpnn_num_step_message_passing,
                         num_step_set2set=mpnn_num_step_set2set,
                         num_layer_set2set=mpnn_num_layer_set2set,
                         readout_feats=mpnn_readout_feats)

        self.predict = nn.Sequential(
            nn.Linear(mpnn_readout_feats * 2, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, 2)
        )

    def forward(self, r1_graph_batch, r2_graph_batch, p_graph_batch,
                condition_chem_graph_batch, rcs_graph_indices, return_node_features=False, **kwargs):

        if return_node_features:
            feat_r1, vis_r1 = self.mpnn(r1_graph_batch, return_node_features=True)
            feat_r2, vis_r2 = self.mpnn(r2_graph_batch, return_node_features=True)
            p_graph_feats, vis_p = self.mpnn(p_graph_batch, return_node_features=True)

            unified_reactant_feats = feat_r1 + feat_r2

            final_concat_feats = torch.cat([unified_reactant_feats, p_graph_feats], dim=1)
            out = self.predict(final_concat_feats)

            return out[:, 0], out[:, 1], (vis_r1, vis_r2, vis_p)

        # --- Training/Evaluation Mode ---
        all_gates = []
        feat_r1, gates_r1 = self.mpnn(r1_graph_batch)
        feat_r2, gates_r2 = self.mpnn(r2_graph_batch)
        p_graph_feats, gates_p = self.mpnn(p_graph_batch)
        all_gates.extend(gates_r1 + gates_r2 + gates_p)

        unified_reactant_feats = feat_r1 + feat_r2

        if condition_chem_graph_batch is not None and condition_chem_graph_batch.num_graphs > 0:
            unique_chem_feats, unique_chem_gates = self.mpnn(condition_chem_graph_batch)
            all_gates.extend(unique_chem_gates)

            padding_vec = torch.zeros(1, unique_chem_feats.size(1), device=unique_chem_feats.device)
            unique_chem_feats_with_padding = torch.cat([unique_chem_feats, padding_vec], dim=0)
            gather_indices = rcs_graph_indices.clone()
            gather_indices[gather_indices == -1] = unique_chem_feats.size(0)
            rcs_feats = unique_chem_feats_with_padding[gather_indices]

            summed_rcs_feats_per_step = torch.sum(rcs_feats, dim=2)
            aggregated_mol_cond_feats = torch.sum(summed_rcs_feats_per_step, dim=1)
            unified_reactant_feats = unified_reactant_feats + aggregated_mol_cond_feats

        final_concat_feats = torch.cat([unified_reactant_feats, p_graph_feats], dim=1)
        out = self.predict(final_concat_feats)

        # Return everything needed for the combined loss function
        return out[:, 0], out[:, 1], unified_reactant_feats, p_graph_feats, all_gates