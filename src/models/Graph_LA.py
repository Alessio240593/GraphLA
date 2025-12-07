import torch.nn.functional as NNF
from torch import nn
from torch_geometric.nn import GCNConv

from src.models.LeaderScoreNET import LeaderScoreGNN


class GraphLA(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.linear_self = nn.Linear(in_feats, hidden_feats)
        self.leader_score_net = LeaderScoreGNN(in_feats, hidden_feats // 2)

    def forward(self, x, edge_index, edge_weight=None):
        h_self = x

        h_neighbors = NNF.relu(self.conv1(x, edge_index, edge_weight=edge_weight))

        leader_score = self.leader_score_net(h_self, edge_index, edge_weight).unsqueeze(-1)

        h_self_proj = self.linear_self(h_self)

        h_new = (1 - leader_score) * h_neighbors + leader_score * h_self_proj

        h_final = NNF.relu(self.conv2(h_new, edge_index, edge_weight=edge_weight))

        return h_final, leader_score