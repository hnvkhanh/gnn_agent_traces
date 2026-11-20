"""Transformer guard over the same graphs as FileHeteroGNN (src/graph/files.py): same nodes, same 66-d features, same
relations, same causality.

    GraphTransformer()                                   # graphs from src/graph/files.py
    GraphTransformer(node_types=NODE_TYPES, edge_types=EDGE_TYPES)   # plain build.py graphs

Tokens: every node of the graph (task, memory, tool, file, action, observation), projected per type (Linear + ReLU as in
the GNN) plus a learned type embedding. The step is already in the features, so there is no extra positional encoding.

Attention mask (the GNN's causality, docs/GRAPH.md): node time is -1 for context nodes (task / memory / tool),
2i for action i, 2i+1 for observation i, and for a file node the time the trace first names it (g["file"].time, -1 when
the user task names it); a query attends to a key only when time(key) <= time(query). Action i thus sees
the context, actions <= i and observations < i, never its own observation or anything later; the rule is transitive, so
it holds through stacked layers.

Edge bias (Graphormer-style): every edge type adds a learned per-layer, per-head scalar to the logit (query = dst,
key = src). That is how the transformer gets copies / mentions / targets / invoked_as, and why the graph ablations, which
drop edges, act on it the same way they act on the GNN.
"""
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from ..graph.build import F
from ..graph.files import FILE_EDGE_TYPES, FILE_NODE_TYPES

CONTEXT_TIME = -1


class _Block(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, h, attn_mask=mask, need_weights=False)[0])
        return x + self.drop(self.ff(self.ln2(x)))


class GraphTransformer(nn.Module):
    def __init__(self, hidden=64, layers=2, classes=2, heads=4, dropout=0.2, node_types=FILE_NODE_TYPES,
                 edge_types=FILE_EDGE_TYPES):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.node_types, self.edge_types = list(node_types), [tuple(e) for e in edge_types]
        self.heads = heads
        self.proj = nn.ModuleDict({t: nn.Linear(F, hidden) for t in self.node_types})
        self.type_emb = nn.Parameter(torch.zeros(len(self.node_types), hidden))
        self.edge_bias = nn.Parameter(torch.zeros(layers, len(self.edge_types), heads))
        self.blocks = nn.ModuleList([_Block(hidden, heads, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, classes))
        nn.init.normal_(self.type_emb, std=0.02)

    def forward(self, data):
        dev = self.type_emb.device
        B = int(getattr(data, "num_graphs", 1) or 1)
        xs, bs, ts, off, n0 = [], [], [], {}, 0
        for k, t in enumerate(self.node_types):
            x = data[t].x
            n = x.size(0)
            off[t] = n0
            n0 += n
            b = getattr(data[t], "batch", None)
            b = torch.zeros(n, dtype=torch.long, device=dev) if b is None else b
            xs.append(torch.relu(self.proj[t](x)) + self.type_emb[k])
            ft = getattr(data[t], "time", None)
            if ft is not None:   # file nodes: first time the trace names them (src/graph/files.py file_times)
                ts.append(ft.to(dev))
            elif t in ("action", "observation"):
                cnt = torch.bincount(b, minlength=B)
                local = torch.arange(n, device=dev) - (torch.cumsum(cnt, 0) - cnt)[b]
                ts.append(2 * local + (1 if t == "observation" else 0))
            else:
                ts.append(torch.full((n,), CONTEXT_TIME, dtype=torch.long, device=dev))
            bs.append(b)
        x, b, tm = torch.cat(xs), torch.cat(bs), torch.cat(ts)

        # dense packing: position of every node inside its graph's row
        order = torch.argsort(b, stable=True)
        X, valid = to_dense_batch(x[order], b[order], batch_size=B)
        N = X.size(1)
        cnt = torch.bincount(b, minlength=B)
        pos = torch.empty_like(b)
        pos[order] = torch.arange(b.numel(), device=dev) - (torch.cumsum(cnt, 0) - cnt)[b[order]]
        T = torch.full((B, N), 1 << 30, dtype=torch.long, device=dev)
        T[b, pos] = tm

        allowed = (T[:, None, :] <= T[:, :, None]) & valid[:, None, :]          # [B, query, key]
        allowed |= torch.eye(N, dtype=torch.bool, device=dev)[None]            # padding rows: never all -inf
        neg = torch.zeros(B, N, N, device=dev).masked_fill(~allowed, float("-inf"))

        # edge index in dense coordinates, one id per edge type
        eb, eq, ek, eid = [], [], [], []
        for j, et in enumerate(self.edge_types):
            if et not in data.edge_types:
                continue
            ei = data[et].edge_index
            if ei.numel() == 0:
                continue
            src, dst = off[et[0]] + ei[0], off[et[2]] + ei[1]
            eb.append(b[dst]); eq.append(pos[dst]); ek.append(pos[src])
            eid.append(torch.full((ei.size(1),), j, dtype=torch.long, device=dev))
        has_e = bool(eb)
        if has_e:
            eb, eq, ek, eid = (torch.cat(v) for v in (eb, eq, ek, eid))

        for l, blk in enumerate(self.blocks):
            bias = torch.zeros(B, N, N, self.heads, device=dev)
            if has_e:
                bias.index_put_((eb, eq, ek), self.edge_bias[l][eid], accumulate=True)
            mask = (bias.permute(0, 3, 1, 2) + neg[:, None]).reshape(B * self.heads, N, N)
            X = blk(X, mask)
        X = self.ln(X)

        a = off["action"] + torch.arange(data["action"].x.size(0), device=dev)   # batched action order (= y order)
        return self.head(X[b[a], pos[a]])
