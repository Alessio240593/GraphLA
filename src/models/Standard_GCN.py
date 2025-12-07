import torch.nn.functional as NNF
from torch import nn
from torch_geometric.nn import GCNConv


class StandardGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)

    def forward(self, x, edge_index, edge_weight=None):
        h = NNF.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = NNF.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        return h, None
