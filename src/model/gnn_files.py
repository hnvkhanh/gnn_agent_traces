"""HeteroGNN over an explicit node / edge type list (src/model/gnn.py is left unchanged: src/cache.py hashes it).

    FileHeteroGNN()                                   # graphs from src/graph/files.py (file nodes)
    FileHeteroGNN(node_types=NODE_TYPES, edge_types=EDGE_TYPES)   # same architecture on plain build.py graphs
"""
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

from ..graph.build import F
from ..graph.files import FILE_EDGE_TYPES, FILE_NODE_TYPES


class FileHeteroGNN(nn.Module):
    """Same as HeteroGNN: per-type Linear(F, hidden) + ReLU, `layers` HeteroConv(SAGEConv per edge type, sum) with ReLU
    and residual, MLP head on action nodes.

    prompt=True (guard gnn+files+prompt): the frozen prompt embedding g["task"].prompt (src/prompt_embed.py, one row per
    graph) is projected to h_prompt; the task node starts as relu(proj_task(x)) + h_prompt and the head reads
    [h_action ; h_prompt of the action's graph]. prompt=False builds exactly the modules above (same init)."""

    def __init__(self, hidden=64, layers=2, classes=2, dropout=0.2, node_types=FILE_NODE_TYPES,
                 edge_types=FILE_EDGE_TYPES, prompt=False, prompt_dim=384):
        super().__init__()
        self.node_types, self.edge_types = list(node_types), [tuple(e) for e in edge_types]
        self.proj = nn.ModuleDict({t: nn.Linear(F, hidden) for t in self.node_types})
        self.convs = nn.ModuleList([
            HeteroConv({et: SAGEConv((hidden, hidden), hidden) for et in self.edge_types}, aggr="sum")
            for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden * (2 if prompt else 1), hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, classes))
        self.prompt = prompt
        if prompt:
            self.prompt_proj = nn.Linear(prompt_dim, hidden)

    def forward(self, data):
        x = {t: torch.relu(self.proj[t](data[t].x)) for t in self.node_types}
        if self.prompt:
            hp = torch.relu(self.prompt_proj(data["task"].prompt))   # one task node (and prompt row) per graph
            x["task"] = x["task"] + hp
        ei = {et: data[et].edge_index for et in self.edge_types}
        for conv in self.convs:
            out = conv(x, ei)
            x = {t: (torch.relu(out[t]) + x[t]) if t in out and out[t] is not None else x[t] for t in self.node_types}
        if not self.prompt:
            return self.head(x["action"])
        b = getattr(data["action"], "batch", None)
        if b is None:
            b = torch.zeros(x["action"].shape[0], dtype=torch.long, device=x["action"].device)
        return self.head(torch.cat([x["action"], hp[b]], 1))
