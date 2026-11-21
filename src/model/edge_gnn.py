"""GNN over the edge graph (src/graph/edges.py): artifacts are nodes, actions are edges, the scored action is a query.

    EdgeGNN()                 # edge-gnn
    EdgeGNN(prompt=True)      # edge-gnn+prompt: adds the frozen prompt embedding g["task"].prompt (src/prompt_embed.py)

Per-type Linear(F, hidden) + ReLU, `layers` HeteroConv(GINEConv per context edge type, with the 66-d edge features,
sum) with ReLU and residual. Readout per query (action k):
    MLP([h_agent ; h_task ; h_tool ; mean h_args ; proj(query.x) (; h_prompt)])
h_args averages the entity / file nodes of action k's arguments (zeros without any). With prompt=True the task node
starts as relu(proj_task(x)) + relu(proj_prompt(p)), and h_prompt = relu(proj_prompt(p)) of the query's task is
concatenated too.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, HeteroConv
from torch_geometric.utils import scatter

from ..graph.build import F
from ..graph.edges import EDGE_TYPES, NODE_TYPES

PROMPT_DIM = 384


def _gine(hidden):
    return GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)), edge_dim=F)


def _gather(h, data, et, n):
    """h[src] placed at each dst query (one edge per query)."""
    out = h.new_zeros((n, h.shape[1]))
    ei = data[et].edge_index
    if ei.numel():
        out[ei[1]] = h[ei[0]]
    return out


class EdgeGNN(nn.Module):
    def __init__(self, hidden=64, layers=2, classes=2, dropout=0.2, prompt=False, prompt_dim=PROMPT_DIM):
        super().__init__()
        self.ntypes = [t for t in NODE_TYPES if t != "query"]
        self.proj = nn.ModuleDict({t: nn.Linear(F, hidden) for t in NODE_TYPES})
        self.convs = nn.ModuleList([HeteroConv({et: _gine(hidden) for et in EDGE_TYPES}, aggr="sum")
                                    for _ in range(layers)])
        self.prompt = prompt
        if prompt:
            self.prompt_proj = nn.Linear(prompt_dim, hidden)
        k = 6 if prompt else 5
        self.head = nn.Sequential(nn.Linear(k * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, classes))

    def forward(self, data):
        x = {t: torch.relu(self.proj[t](data[t].x)) for t in self.ntypes}
        if self.prompt:
            hp = torch.relu(self.prompt_proj(data["task"].prompt))
            x["task"] = x["task"] + hp
        ei = {et: data[et].edge_index for et in EDGE_TYPES}
        ea = {et: data[et].edge_attr for et in EDGE_TYPES}
        for conv in self.convs:
            out = conv(x, ei, edge_attr_dict=ea)
            x = {t: (torch.relu(out[t]) + x[t]) if out.get(t) is not None else x[t] for t in self.ntypes}
        nq = data["query"].x.shape[0]
        hq = torch.relu(self.proj["query"](data["query"].x))
        parts = [_gather(x["agent"], data, ("agent", "q_agent", "query"), nq),
                 _gather(x["task"], data, ("task", "q_task", "query"), nq),
                 _gather(x["tool"], data, ("tool", "q_tool", "query"), nq)]
        src, dst = [], []
        for nt in ("entity", "file"):
            e = data[(nt, "q_arg", "query")].edge_index
            src.append(x[nt][e[0]])
            dst.append(e[1])
        src, dst = torch.cat(src), torch.cat(dst)
        parts.append(scatter(src, dst, 0, dim_size=nq, reduce="mean") if dst.numel() else hq.new_zeros(hq.shape))
        parts.append(hq)
        if self.prompt:
            parts.append(_gather(hp, data, ("task", "q_task", "query"), nq))
        return self.head(torch.cat(parts, 1))
