
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import Set2Set, GINEConv
class DualChannelMPNN(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, hidden_feats=64,
                 num_step_message_passing=3, num_step_set2set=3, num_layer_set2set=1,
                 readout_feats=1024):
        super(DualChannelMPNN, self).__init__()
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing

        gine_nn = nn.Linear(hidden_feats, hidden_feats)
        self.gnn_layer = GINEConv(nn=gine_nn, edge_dim=edge_in_feats)
        self.activation = nn.ReLU()

        self.gate_nn_structure = nn.Sequential(nn.Linear(hidden_feats, 1), nn.Sigmoid())
        self.gate_nn_reactivity = nn.Sequential(nn.Linear(hidden_feats, 1), nn.Sigmoid())
        self.gru_s = nn.GRU(hidden_feats, hidden_feats)
        self.gru_r = nn.GRU(hidden_feats, hidden_feats)

        self.readout = Set2Set(in_channels=hidden_feats * 3,
                               processing_steps=num_step_set2set,
                               num_layers=num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 6, readout_feats), nn.PReLU()
        )

    def forward(self, g: Batch):
        if g is None or g.num_nodes == 0:
            device = next(self.parameters()).device
            return torch.zeros((1, self.sparsify[0].out_features), device=device), []

        node_feats, edge_index, edge_feats, batch = g.x, g.edge_index, g.edge_attr, g.batch

        projected_node_feats = self.project_node_feats(node_feats)
        h_gru_s = projected_node_feats.unsqueeze(0)
        h_gru_r = projected_node_feats.unsqueeze(0)
        current_node_feats_s = projected_node_feats
        current_node_feats_r = projected_node_feats

        all_reactivity_gates = []

        for _ in range(self.num_step_message_passing):
            m = self.gnn_layer(current_node_feats_s, edge_index, edge_feats)
            activated_m = self.activation(m)

            gate_s = self.gate_nn_structure(current_node_feats_s)
            gated_message_s = gate_s * activated_m
            _, h_gru_s = self.gru_s(gated_message_s.unsqueeze(0), h_gru_s)
            current_node_feats_s = h_gru_s.squeeze(0)

            gate_r = self.gate_nn_reactivity(current_node_feats_r)
            all_reactivity_gates.append(gate_r)
            gated_message_r = gate_r * activated_m
            _, h_gru_r = self.gru_r(gated_message_r.unsqueeze(0), h_gru_r)
            current_node_feats_r = h_gru_r.squeeze(0)

        node_aggr_for_readout = torch.cat([projected_node_feats, current_node_feats_s, current_node_feats_r], dim=1)
        readout_output = self.readout(node_aggr_for_readout, batch)
        graph_feats = self.sparsify(readout_output)

        return graph_feats, all_reactivity_gates
