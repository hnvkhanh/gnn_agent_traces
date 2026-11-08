import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

from ..graph.build import EDGE_TYPES, F, NODE_TYPES


class HeteroGNN(nn.Module):
    def __init__(self, hidden=64, layers=2, classes=2, dropout=0.2):
        super().__init__()
        self.proj = nn.ModuleDict({t: nn.Linear(F, hidden) for t in NODE_TYPES})
        self.convs = nn.ModuleList([
            HeteroConv({et: SAGEConv((hidden, hidden), hidden) for et in EDGE_TYPES}, aggr="sum") for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, classes))

    def forward(self, data):
        x = {t: torch.relu(self.proj[t](data[t].x)) for t in NODE_TYPES}
        for conv in self.convs:
            out = conv(x, data.edge_index_dict)
            x = {t: (torch.relu(out[t]) + x[t]) if t in out else x[t] for t in NODE_TYPES}
        return self.head(x["action"])


class ActionMLP(nn.Module):
    """Baseline: same action-node features, no graph."""

    def __init__(self, hidden=64, classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(F, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, classes))

    def forward(self, data):
        return self.net(data["action"].x)
