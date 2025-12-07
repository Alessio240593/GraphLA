import torch
import torch.nn.functional as NNF
from torch import nn
from torch_geometric.nn import GCNConv


class LeaderScoreGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, 1)

    def forward(self, x, edge_index, edge_weight=None):
        h = NNF.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = self.conv2(h, edge_index, edge_weight=edge_weight)

        score = torch.sigmoid(h).squeeze(-1)

        deg_weighted = torch.zeros(x.size(0), device=x.device)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        deg_weighted = deg_weighted.scatter_add_(0, edge_index[0], edge_weight)
        deg_norm = deg_weighted / deg_weighted.max()

        score = score * (1 + deg_norm)

        return score
