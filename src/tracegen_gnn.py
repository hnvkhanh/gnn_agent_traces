"""Approve / reject guard on tracegen (real AgentDojo) traces, with and without file nodes.

Used by scripts/run_tracegen_gnn.py (CLI, parallel devices, report). Cells:
  main                  train split -> val and test; guards majority, tool-lookup, mlp, gnn, gnn+files
  ablate/<a>            gnn+files with ablation <a> (src/graph/files.py FILE_ABLATIONS)
  transfer/<A>-><B>     train split of agent A -> test split of agent B (still task- and attack-disjoint)
A run is one (cell, seed). Its result: {eval_name: {guard: report}}, eval names 'val', 'test', 'test|suite=<s>',
'test|agent=<m>'. A report is src/eval.py report(...) plus asr=asr_report(...).
"""
import hashlib
import json
import os
import re

import numpy as np
import torch

from .benchmarks.tracegen import load_traces
from .cache import ROOT, TRAIN_CODE, data_hash
from .eval import asr_report, report, tool_lookup_baseline
from .graph.build import meta_of, trace_to_graph
from .graph.files import FILE_ABLATIONS, trace_to_graph_files
from .sandbox.policy import LABELS

GUARDS = ["majority", "tool-lookup", "mlp", "gnn", "gnn+files"]
ABLATE = ["files", "observation", "injected", "copies", "mentions", "tool_feat", "attacker_values"]
CODE = TRAIN_CODE + ["src/tracegen_gnn.py", "src/graph/build.py", "src/graph/files.py", "src/model/gnn_files.py",
                     "src/benchmarks/tracegen.py"]


def code_hash():
    h = hashlib.sha256()
    for p in CODE:
        h.update(p.encode())
        h.update(open(os.path.join(ROOT, p), "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


# ---------------------------------------------------------------- data
class Data:
    """Converted traces + graphs (built lazily per ablation)."""

    def __init__(self, tg_dir="data/tracegen", data_dir="data/agentdojo", recs=None):
        self.stats = {}
        self.trs = load_traces(tg_dir, data_dir, recs=recs, stats=self.stats)
        self.metas = [meta_of(t) for t in self.trs]
        for m, t in zip(self.metas, self.trs):
            m.update(suite=t["suite"], split=t["split"], trace_id=t["trace_id"])
        self._g = {}

    def graphs(self, kind="files", ablate=()):
        key = (kind, tuple(sorted(ablate)))
        if key not in self._g:
            f = trace_to_graph_files if kind == "files" else trace_to_graph
            self._g[key] = [f(t, frozenset(ablate)) for t in self.trs]
        return self._g[key]

    def idx(self, split, model=None):
        return [i for i, t in enumerate(self.trs) if t["split"] == split and (model is None or t["model"] == model)]

    def agents(self):
        return sorted({t["model"] for t in self.trs})

    def summary(self):
        out = {}
        for sp in ("train", "val", "test"):
            I = self.idx(sp)
            y = [e["label"] for i in I for e in self.trs[i]["events"]]
            att = [i for i in I if self.trs[i]["attack"]]
            out[sp] = dict(traces=len(I), attack_traces=len(att), benign_traces=len(I) - len(att), actions=len(y),
                           reject=y.count("reject"), approve=y.count("approve"),
                           attack_followed=sum(any(e["label"] == "reject" for e in self.trs[i]["events"]) for i in att),
                           security=sum(bool(self.trs[i].get("security")) for i in att),
                           tasks=len({self.metas[i]["group"] for i in I}),
                           injection_tasks=len({self.metas[i]["attack_id"] for i in att}))
        return out

    def ready(self):
        """None if trainable, else the reason (no attacked traces / no reject labels on a side)."""
        s = self.summary()
        if not any(s[sp]["attack_traces"] for sp in s):
            return ("no attacked traces in %s: run `python -m tracegen --attacks` first (attacks.enabled in "
                    "config.yaml), then rerun this script" % os.path.join("data/tracegen", "traces.jsonl"))
        for sp in ("train", "test"):
            if not s[sp]["reject"]:
                return "no reject actions in the %s split (agents never followed an attack there)" % sp
        return None


# ---------------------------------------------------------------- cells
def cells(data, sections=("main", "ablate", "transfer")):
    out = []
    if "main" in sections:
        out.append("main")
    if "ablate" in sections:
        out += ["ablate/%s" % a for a in ABLATE]
    if "transfer" in sections:
        ag = data.agents()
        out += ["transfer/%s->%s" % (a, b) for a in ag for b in ag if a != b]
    return out


def cell_spec(data, key):
    """(train idx, eval sets {name: idx}, guards, graph kind/ablation per guard)."""
    if key == "main":
        tr = data.idx("train")
        ev = {"val": data.idx("val"), "test": data.idx("test")}
        te = ev["test"]
        for s in sorted({data.trs[i]["suite"] for i in te}):
            ev["test|suite=%s" % s] = [i for i in te if data.trs[i]["suite"] == s]
        for m in data.agents():
            ev["test|agent=%s" % m] = [i for i in te if data.trs[i]["model"] == m]
        return tr, ev, GUARDS, ()
    if key.startswith("ablate/"):
        a = key.split("/", 1)[1]
        assert a in FILE_ABLATIONS, a
        return data.idx("train"), {"val": data.idx("val"), "test": data.idx("test")}, ["gnn+files"], (a,)
    if key.startswith("transfer/"):
        a, b = key.split("/", 1)[1].split("->")
        return data.idx("train", a), {"test": data.idx("test", b)}, GUARDS, ()
    raise KeyError(key)


def fingerprint(data, key, seed, epochs, lr, hidden, code=None):
    tr, ev, guards, ab = cell_spec(data, key)
    allev = sorted({i for v in ev.values() for i in v})
    g = data.graphs("files", ab)
    part = ([g[i] for i in tr], [data.metas[i] for i in tr], [g[i] for i in allev], [data.metas[i] for i in allev])
    blob = dict(code=code or code_hash(), key=key, seed=seed, epochs=epochs, lr=lr, hidden=hidden, guards=guards,
                evals={k: [data.metas[i]["trace_id"] for i in v] for k, v in ev.items()}, data=data_hash([part]))
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def _ys(graphs):
    return torch.cat([g["action"].y for g in graphs]) if graphs else torch.zeros(0, dtype=torch.long)


def run_cell(data, key, seed, epochs=60, lr=3e-3, hidden=64):
    """Train every guard of the cell once (seed) and score every eval set."""
    from .model.gnn import ActionMLP, HeteroGNN
    from .model.gnn_files import FileHeteroGNN
    from .train import fit
    tr, ev, guards, ab = cell_spec(data, key)
    allev = sorted({i for v in ev.values() for i in v})
    gf, gp = data.graphs("files", ab), data.graphs("plain", ())
    ncls = len(LABELS)
    ytr = _ys([gf[i] for i in tr])
    cnt = torch.bincount(ytr, minlength=ncls).float().clamp(min=1)
    w = cnt.sum() / (ncls * cnt)
    trm, evm = [data.metas[i] for i in tr], [data.metas[i] for i in allev]
    n_act = [len(data.trs[i]["events"]) for i in allev]
    preds = {}
    for gname in guards:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if gname == "majority":
            p = [int(ytr.bincount(minlength=ncls).argmax())] * sum(n_act)
        elif gname == "tool-lookup":
            p = tool_lookup_baseline(trm, ytr.tolist(), evm)
        elif gname == "mlp":
            p = fit(ActionMLP(hidden, ncls), [gf[i] for i in tr], [gf[i] for i in allev], epochs, lr, w, ncls)
        elif gname == "gnn":
            p = fit(HeteroGNN(hidden, classes=ncls), [gp[i] for i in tr], [gp[i] for i in allev], epochs, lr, w, ncls)
        elif gname == "gnn+files":
            p = fit(FileHeteroGNN(hidden, classes=ncls), [gf[i] for i in tr], [gf[i] for i in allev], epochs, lr, w, ncls)
        else:
            raise KeyError(gname)
        per, k = {}, 0
        for i, n in zip(allev, n_act):
            per[i] = list(p[k:k + n])
            k += n
        preds[gname] = per
    out = {}
    for name, I in ev.items():
        y = [LABELS.index(e["label"]) for i in I for e in data.trs[i]["events"]]
        ms = [data.metas[i] for i in I]
        out[name] = {}
        for gname in guards:
            p = [x for i in I for x in preds[gname][i]]
            out[name][gname] = dict(report(y, p), asr=asr_report(y, p, ms), n_actions=len(y),
                                    n_reject=int(sum(y)), n_traces=len(I))
    return out


# ---------------------------------------------------------------- leakage audit
def _norm(t):
    return re.sub(r"\s+", " ", t or "").strip().lower()


def audit(data):
    """Overlaps between the train side and each held-out side. `hard` counts must be 0 (the script refuses to train
    otherwise); `attacker_values_in_train` is informational (AgentDojo reuses attacker constants across injection
    tasks, e.g. the banking IBAN): the `attacker_values` ablation measures how much the guard relies on them."""
    sides = {sp: data.idx(sp) for sp in ("train", "val", "test")}

    def vals(I, f):
        out = set()
        for i in I:
            v = f(i)
            out |= set(v) if isinstance(v, (set, list, tuple)) else ({v} - {None, ""})
        return out

    keys = dict(
        task_groups=lambda i: data.metas[i]["group"],
        injection_tasks=lambda i: data.metas[i]["attack_id"],
        injected_texts=lambda i: _norm(data.trs[i]["injected"]) if data.trs[i]["attack"] else None,
        prompts=lambda i: _norm(data.trs[i]["user_task"]),
    )
    out = {"hard": {}, "examples": {}, "info": {}}
    for held in ("val", "test"):
        for k, f in keys.items():
            both = vals(sides["train"], f) & vals(sides[held], f)
            out["hard"]["%s train∩%s" % (k, held)] = len(both)
            if both:
                out["examples"]["%s train∩%s" % (k, held)] = sorted(map(str, both))[:3]
    g_val, g_test = vals(sides["val"], keys["task_groups"]), vals(sides["test"], keys["task_groups"])
    out["hard"]["task_groups val∩test"] = len(g_val & g_test)
    # attacker constants of held-out attacks that also occur in training actions / observations (informational)
    train_text = " ".join(_norm(json.dumps(e["args"]) + " " + e["observation"])
                          for i in sides["train"] for e in data.trs[i]["events"])
    for held in ("val", "test"):
        av = {v.lower() for i in sides[held] if data.trs[i]["attack"]
              for c in data.trs[i]["attacker_calls"] for v in _strings_of(c.get("args") or {}) if len(v) >= 4}
        out["info"]["attacker_values_%s" % held] = len(av)
        out["info"]["attacker_values_%s_in_train" % held] = sum(1 for v in av if v in train_text)
    # structure: benign vs attacked trace length (the graph sees the number of actions)
    for kind, f in (("benign", lambda t: not t["attack"]), ("attack", lambda t: bool(t["attack"]))):
        L = [len(t["events"]) for t in data.trs if f(t)]
        out["info"]["mean_actions_%s" % kind] = float(np.mean(L)) if L else float("nan")
    out["ok"] = all(v == 0 for v in out["hard"].values())
    return out


def _strings_of(x):
    from .sandbox.policy import _strings
    return list(_strings(x))

