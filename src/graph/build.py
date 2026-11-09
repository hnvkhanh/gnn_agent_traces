"""Trace -> causal heterogeneous graph. Action i only receives messages from nodes that exist before its decision."""
import json
import re
import zlib

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ..sandbox.policy import LABELS, _strings

D = 64
F = D + 2
NODE_TYPES = ["task", "memory", "tool", "action", "observation"]
EDGE_TYPES = [
    ("task", "proposes", "action"),
    ("memory", "recalled", "action"),
    ("tool", "invoked_as", "action"),
    ("action", "returns", "observation"),
    ("observation", "precedes", "action"),
    ("observation", "mentions", "action"),
    ("observation", "copies", "action"),
    ("action", "next", "action"),
]


def hash_vec(text: str) -> np.ndarray:
    v = np.zeros(D, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9_]+", text.lower()):
        h = zlib.crc32(tok.encode())
        v[h % D] += 1.0 if (h >> 16) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def feat(text: str, step: int = 0) -> np.ndarray:
    return np.concatenate([hash_vec(text), [step / 10.0, 1.0]]).astype(np.float32)


def _stack(rows):
    return torch.tensor(np.stack(rows)) if rows else torch.zeros((0, F))


def _ei(pairs):
    return torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.zeros((2, 0), dtype=torch.long)


ABLATIONS = ["tool_feat", "observation", "memory", "injected", "copies", "mentions"]


def trace_to_graph(tr: dict, ablate=frozenset()):
    ev = tr["events"]
    if not ev:
        return None
    inj = tr.get("injected", "")
    strip = (lambda t: t.replace(inj, "")) if ("injected" in ablate and inj) else (lambda t: t)
    tools = tr["tools"]
    tidx = {t["name"]: i for i, t in enumerate(tools)}
    g = HeteroData()
    g["task"].x = _stack([feat(strip(tr["user_task"]))])
    g["memory"].x = _stack([feat(strip(m)) for m in tr["memory"]])
    g["tool"].x = _stack([feat("" if "tool_feat" in ablate else t["name"] + " " + t["description"]) for t in tools])
    g["action"].x = _stack([feat(e["tool"] + " " + json.dumps(e["args"], sort_keys=True), e["step"]) for e in ev])
    g["observation"].x = _stack([feat(strip(e["observation"]), e["step"]) for e in ev])
    g["action"].y = torch.tensor([LABELS.index(e["label"]) for e in ev])

    n = len(ev)
    E = {et: [] for et in EDGE_TYPES}
    for i, e in enumerate(ev):
        E[("task", "proposes", "action")].append([0, i])
        for m in range(len(tr["memory"])):
            E[("memory", "recalled", "action")].append([m, i])
        if e["tool"] in tidx:
            E[("tool", "invoked_as", "action")].append([tidx[e["tool"]], i])
        E[("action", "returns", "observation")].append([i, i])
        for j in range(i):
            E[("observation", "precedes", "action")].append([j, i])
            E[("action", "next", "action")].append([j, i])
            oj = strip(ev[j]["observation"]).lower()
            if e["tool"].lower() in oj:
                E[("observation", "mentions", "action")].append([j, i])
            if any(len(v) >= 4 and v.lower() in oj for v in _strings(e["args"])):
                E[("observation", "copies", "action")].append([j, i])
    drop = set()
    if "observation" in ablate:
        drop |= {("action", "returns", "observation"), ("observation", "precedes", "action"),
                 ("observation", "mentions", "action"), ("observation", "copies", "action")}
    if "memory" in ablate:
        drop.add(("memory", "recalled", "action"))
    if "copies" in ablate:
        drop.add(("observation", "copies", "action"))
    if "mentions" in ablate:
        drop.add(("observation", "mentions", "action"))
    for et, pairs in E.items():
        g[et].edge_index = _ei([] if et in drop else pairs)
    return g


def attack_key(tr: dict):
    """Identity of the attack in a trace (None if benign): the thing that decides its reject labels. Train and test must
    not share it, or the model can memorise the attacker tool / attacker arguments instead of learning the signal.
    ASB, InjecAgent: the attacker tool. AgentDojo: the injection task (attacker tool + attacker args)."""
    a = tr["attack"]
    if a is None:
        return None
    if a.get("id"):
        return a["id"]
    if tr.get("benchmark") == "agentdojo":  # traces collected before ids were recorded
        from ..benchmarks.agentdojo import injection_id
        return injection_id(tr.get("injected", "")) or tr.get("injected", "")
    if tr.get("benchmark") == "agentpoison":
        from ..benchmarks.agentpoison import trigger_id
        return trigger_id(tr.get("user_task", "")) or tr.get("injected", "")
    return a.get("tool") or tr.get("injected", "")


def meta_of(tr: dict) -> dict:
    a = tr["attack"]
    return dict(benchmark=tr.get("benchmark", "asb"), model=tr.get("model", ""), agent=tr["agent"],
                attack_id=attack_key(tr),
                synthetic=[bool(e["synthetic"]) for e in tr["events"]], attack=tr["attack"] is not None,
                group=tr.get("group") or tr["agent"] + "|" + tr["task"],
                style=a["style"] if a else None, mode=a["mode"] if a else "none",
                category=(a.get("category") if a else None), tools=[e["tool"] for e in tr["events"]])
