"""Approve / reject guard on tracegen (real AgentDojo) traces, with and without file nodes.

Used by scripts/run_tracegen_gnn.py (CLI, parallel devices, report). Cells:
  main                  train split -> val and test; guards majority, tool-lookup, mlp, gnn, gnn+files
  ablate/<a>            gnn+files with ablation <a> (src/graph/files.py FILE_ABLATIONS)
  transfer/<A>-><B>     train split of agent A -> test split of agent B (still task- and attack-disjoint)
  cv/<k>/<a>-<b>        cross-validation over k task folds x k injection-task folds (cv_folds). Cell (a, b) tests the
                        attack runs of task fold a with an injection task of fold b, plus the benign runs of task fold
                        a assigned to b; validation is the block (a+1, b+1); training is every run whose task is in
                        neither task fold and whose injection task is in neither injection fold. So each cell is task-
                        and attack-disjoint like the fixed split, and over the k*k cells every run is tested exactly
                        once. cv_reports pools the cells of a seed. The fixed train/val/test split is not used.
A run is one (cell, seed). Its result: {eval_name: {guard: report}}, eval names 'val', 'test', 'test|suite=<s>',
'test|agent=<m>'.

Model selection uses the cell's validation traces (the val split; for transfer, the val split of the training agent):
the checkpoint is the epoch with the best validation AUPRC, and the reject threshold is the one with the best validation
reject F1. Validation scores are therefore optimistic; test is the unbiased number. With no reject action in validation
the last epoch and threshold 0.5 are used (`selected` / `threshold_tuned` false in the report).

A report holds
  src/rank_metrics.py rank_report(...)   auprc, recall_at_fpr and their bootstrap intervals over traces (threshold free)
  src/eval.py report(...) + asr          at the validation threshold; macro_f1 is None when the set has no reject action
  threshold, threshold_tuned, best_epoch, val_auprc, selected, n_actions, n_reject, n_traces
"""
import hashlib
import json
import os
import random
import re
import zlib

import copy

import numpy as np
import torch
import torch.nn.functional as Fn

from .benchmarks.tracegen import load_traces
from .cache import ROOT, TRAIN_CODE, data_hash
from .eval import asr_report, report
from .graph.build import meta_of, trace_to_graph
from .graph.files import FILE_ABLATIONS, trace_to_graph_files
from .rank_metrics import average_precision, rank_report, threshold_for_f1
from .sandbox.policy import LABELS

GUARDS = ["majority", "tool-lookup", "mlp", "gnn", "gnn+files"]
ABLATE = ["files", "observation", "injected", "copies", "mentions", "tool_feat", "attacker_values"]
CODE = TRAIN_CODE + ["src/tracegen_gnn.py", "src/graph/build.py", "src/graph/files.py", "src/model/gnn_files.py",
                     "src/benchmarks/tracegen.py", "src/rank_metrics.py"]


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
        self.trs = load_traces(tg_dir, data_dir, recs=recs, stats=self.stats, keep_mismatch=True)
        self.metas = [meta_of(t) for t in self.trs]
        for m, t in zip(self.metas, self.trs):
            m.update(suite=t["suite"], split=t["split"], trace_id=t["trace_id"])
        self._g, self._folds = {}, {}

    def graphs(self, kind="files", ablate=()):
        key = (kind, tuple(sorted(ablate)))
        if key not in self._g:
            f = trace_to_graph_files if kind == "files" else trace_to_graph
            self._g[key] = [f(t, frozenset(ablate)) for t in self.trs]
        return self._g[key]

    def idx(self, split, model=None):
        """Traces of a fixed split. Attack runs whose injection task is in another split belong to no fixed split
        (only the cross-validation uses them)."""
        return [i for i, t in enumerate(self.trs) if t["split"] == split and t["inj_split"] in (None, split)
                and (model is None or t["model"] == model)]

    def folds(self, k, seed=0):
        if (k, seed) not in self._folds:
            self._folds[(k, seed)] = cv_folds(self.trs, k, seed)
        return self._folds[(k, seed)]

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
def cv_folds(trs, k, seed=0):
    """(task fold, injection fold) per trace. Per suite, the user tasks and the injection tasks are shuffled (seeded)
    and dealt round-robin into k folds, so every fold holds tasks and attacks of every suite. A benign run has no
    injection task: its injection fold is a hash of its trace id, which spreads the benign runs of a task fold over
    the k test cells of that fold."""
    def deal(keys_by_suite):
        out = {}
        for s in sorted(keys_by_suite):
            ks = sorted(keys_by_suite[s])
            random.Random("%s|%d" % (s, seed)).shuffle(ks)
            out.update({x: i % k for i, x in enumerate(ks)})
        return out
    tasks, injs = {}, {}
    for t in trs:
        tasks.setdefault(t["suite"], set()).add(t["group"])
        if t["attack"]:
            injs.setdefault(t["suite"], set()).add(t["attack"]["id"])
    tf, jf = deal(tasks), deal(injs)
    return ([tf[t["group"]] for t in trs],
            [jf[t["attack"]["id"]] if t["attack"] else zlib.crc32(t["trace_id"].encode()) % k for t in trs])


def cv_spec(data, k, a, b):
    """(train idx, validation idx, test idx) of cross-validation cell (a, b); see the module doc."""
    assert k >= 3, "cross-validation needs at least 3 folds (test, validation, train)"
    ta, ib = data.folds(k)
    a2, b2 = (a + 1) % k, (b + 1) % k
    att = [bool(t["attack"]) for t in data.trs]
    R = range(len(data.trs))
    return ([i for i in R if ta[i] not in (a, a2) and (not att[i] or ib[i] not in (b, b2))],
            [i for i in R if ta[i] == a2 and (not att[i] or ib[i] == b2)],
            [i for i in R if ta[i] == a and ib[i] == b])


def cells(data, sections=("main", "ablate", "transfer"), folds=5):
    out = []
    if "main" in sections:
        out.append("main")
    if "cv" in sections:   # a cell with nothing to test (no run of that task fold x injection fold) is skipped
        out += ["cv/%d/%d-%d" % (folds, a, b) for a in range(folds) for b in range(folds)
                if cv_spec(data, folds, a, b)[2]]
    if "ablate" in sections:
        out += ["ablate/%s" % a for a in ABLATE]
    if "transfer" in sections:
        ag = data.agents()
        out += ["transfer/%s->%s" % (a, b) for a in ag for b in ag if a != b]
    return out


def cell_spec(data, key):
    """(train idx, validation idx for model selection, eval sets {name: idx}, guards, ablation)."""
    if key == "main":
        tr = data.idx("train")
        ev = {"val": data.idx("val"), "test": data.idx("test")}
        te = ev["test"]
        for s in sorted({data.trs[i]["suite"] for i in te}):
            ev["test|suite=%s" % s] = [i for i in te if data.trs[i]["suite"] == s]
        for m in data.agents():
            ev["test|agent=%s" % m] = [i for i in te if data.trs[i]["model"] == m]
        return tr, ev["val"], ev, GUARDS, ()
    if key.startswith("ablate/"):
        a = key.split("/", 1)[1]
        assert a in FILE_ABLATIONS, a
        va = data.idx("val")
        return data.idx("train"), va, {"val": va, "test": data.idx("test")}, ["gnn+files"], (a,)
    if key.startswith("transfer/"):
        a, b = key.split("/", 1)[1].split("->")
        return data.idx("train", a), data.idx("val", a), {"test": data.idx("test", b)}, GUARDS, ()
    if key.startswith("cv/"):
        k, ab = key.split("/")[1:]
        tr, va, te = cv_spec(data, int(k), *map(int, ab.split("-")))
        return tr, va, {"test": te}, GUARDS, ()
    raise KeyError(key)


def fingerprint(data, key, seed, epochs, lr, hidden, code=None):
    tr, va, ev, guards, ab = cell_spec(data, key)
    allev = sorted({i for v in ev.values() for i in v})
    g = data.graphs("files", ab)
    parts = [([g[i] for i in tr], [data.metas[i] for i in tr], [g[i] for i in allev], [data.metas[i] for i in allev]),
             ([g[i] for i in va], [data.metas[i] for i in va], [], [])]
    blob = dict(code=code or code_hash(), key=key, seed=seed, epochs=epochs, lr=lr, hidden=hidden, guards=guards,
                evals={k: [data.metas[i]["trace_id"] for i in v] for k, v in ev.items()}, data=data_hash(parts))
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def _ys(graphs):
    return torch.cat([g["action"].y for g in graphs]) if graphs else torch.zeros(0, dtype=torch.long)


def _scores(model, graphs, device):
    """Reject probability per action, in graph order."""
    from torch_geometric.loader import DataLoader
    if not graphs:
        return np.zeros(0)
    model.eval()
    with torch.no_grad():
        return torch.cat([torch.softmax(model(b.to(device)), 1)[:, 1].cpu()
                          for b in DataLoader(graphs, batch_size=64)]).numpy()


def fit_select(model, train, val, evalg, epochs, lr, weights):
    """Train like src/train.py fit, keep the epoch with the best validation AUPRC (the last epoch when validation has
    no reject action). Returns (validation scores, eval scores, info)."""
    from torch_geometric.loader import DataLoader
    from .train import DEVICE
    model, weights = model.to(DEVICE), weights.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=32, shuffle=True)
    yv = _ys(val).numpy()
    best, state, best_epoch = None, None, epochs
    for ep in range(1, epochs + 1):
        model.train()
        for b in loader:
            b = b.to(DEVICE)
            opt.zero_grad()
            Fn.cross_entropy(model(b), b["action"].y, weight=weights).backward()
            opt.step()
        ap = average_precision(yv, _scores(model, val, DEVICE)) if yv.any() else None
        if ap is not None and (best is None or ap > best):
            best, best_epoch, state = ap, ep, copy.deepcopy(model.state_dict())
    if state is not None:
        model.load_state_dict(state)
    return _scores(model, val, DEVICE), _scores(model, evalg, DEVICE), dict(
        best_epoch=best_epoch, val_auprc=best, selected=state is not None)


def tool_rate_scores(train_meta, train_y, metas):
    """Tool lookup as a score: the training reject rate of the action's tool (unseen tool: the overall rate)."""
    n, r, k = {}, {}, 0
    for m in train_meta:
        for t in m["tools"]:
            n[t] = n.get(t, 0) + 1
            r[t] = r.get(t, 0) + train_y[k]
            k += 1
    glob = sum(r.values()) / max(1, sum(n.values()))
    return np.array([r[t] / n[t] if t in n else glob for m in metas for t in m["tools"]], dtype=float)


def run_cell(data, key, seed, epochs=60, lr=3e-3, hidden=64, n_boot=200):
    """Train every guard of the cell once (seed) and score every eval set."""
    from .model.gnn import ActionMLP, HeteroGNN
    from .model.gnn_files import FileHeteroGNN
    tr, va, ev, guards, ab = cell_spec(data, key)
    allev = sorted({i for v in ev.values() for i in v})
    gf, gp = data.graphs("files", ab), data.graphs("plain", ())
    ncls = len(LABELS)
    ytr = _ys([gf[i] for i in tr])
    yva = _ys([gf[i] for i in va]).numpy()
    cnt = torch.bincount(ytr, minlength=ncls).float().clamp(min=1)
    w = cnt.sum() / (ncls * cnt)
    trm, vam, evm = ([data.metas[i] for i in I] for I in (tr, va, allev))
    n_act = [len(data.trs[i]["events"]) for i in allev]
    scores, info = {}, {}
    for gname in guards:
        torch.manual_seed(seed)
        np.random.seed(seed)
        inf = dict(best_epoch=None, val_auprc=None, selected=False)
        if gname == "majority":   # constant score: predicts the majority training label, no threshold to tune
            sv, se = None, np.full(sum(n_act), float(ytr.bincount(minlength=ncls).argmax()))
        elif gname == "tool-lookup":
            sv, se = tool_rate_scores(trm, ytr.tolist(), vam), tool_rate_scores(trm, ytr.tolist(), evm)
        else:
            G = gp if gname == "gnn" else gf
            model = {"mlp": lambda: ActionMLP(hidden, ncls), "gnn": lambda: HeteroGNN(hidden, classes=ncls),
                     "gnn+files": lambda: FileHeteroGNN(hidden, classes=ncls)}[gname]()
            sv, se, inf = fit_select(model, [G[i] for i in tr], [G[i] for i in va], [G[i] for i in allev], epochs, lr, w)
        thr = threshold_for_f1(yva, sv) if sv is not None and yva.any() else None
        info[gname] = dict(inf, threshold=0.5 if thr is None else thr, threshold_tuned=thr is not None)
        per, k = {}, 0
        for i, n in zip(allev, n_act):
            per[i] = se[k:k + n]
            k += n
        scores[gname] = per
    out = {}
    for name, I in ev.items():
        out[name] = {}
        for gname in guards:
            s = np.concatenate([scores[gname][i] for i in I]) if I else np.zeros(0)
            p = (s >= info[gname]["threshold"]).astype(int).tolist()
            out[name][gname] = dict(score_set(data, I, s, p, n_boot, seed), **info[gname])
    if key.startswith("cv/"):   # per-action scores and decisions of the test cell, pooled later by cv_reports
        I = ev["test"]
        out["_pool"] = dict(
            trace_ids=[data.metas[i]["trace_id"] for i in I],
            scores={g: [round(float(x), 6) for i in I for x in scores[g][i]] for g in guards},
            preds={g: [int(x >= info[g]["threshold"]) for i in I for x in scores[g][i]] for g in guards})
    return out


def score_set(data, I, s, p, n_boot=200, seed=0):
    """Report of scores s / decisions p (per action, trace order) on the traces I."""
    y = [LABELS.index(e["label"]) for i in I for e in data.trs[i]["events"]]
    ms, lens = [data.metas[i] for i in I], [len(data.trs[i]["events"]) for i in I]
    r = dict(report(y, list(p)), asr=asr_report(y, list(p), ms), n_actions=len(y), n_reject=int(sum(y)), n_traces=len(I),
             n_followed=sum(any(e["label"] == "reject" for e in data.trs[i]["events"]) for i in I))
    if not sum(y):   # macro-F1 is capped at 0.5 without a reject action: not a score
        r["macro_f1"] = None
        r["recall"]["reject"] = None
    r.update(rank_report(y, s, lens, n_boot, seed))
    return r


def cv_reports(data, store, seeds, folds, n_boot=200):
    """Pooled cross-validation result: one {eval_name: {guard: report}} per seed that has every cell stored. Eval names
    as in the main cell ('test', 'test|suite=<s>', 'test|agent=<m>', and 'test|attack=<injection task>').

    Scores of different cells come from different models; pooling them into one AUPRC assumes they are comparable.
    The thresholded columns do not: each cell uses its own validation threshold."""
    keys = cells(data, ["cv"], folds)
    pos = {m["trace_id"]: i for i, m in enumerate(data.metas)}
    runs = []
    for sd in seeds:
        pools = [store.get(k, {}).get(str(sd), {}).get("run", {}).get("_pool") for k in keys]
        if not keys or any(p is None for p in pools):
            continue
        I, sc, pr = [], {g: {} for g in GUARDS}, {g: {} for g in GUARDS}
        for p in pools:
            k = 0
            for tid in p["trace_ids"]:
                i = pos[tid]
                n = len(data.trs[i]["events"])
                for g in GUARDS:
                    sc[g][i], pr[g][i] = p["scores"][g][k:k + n], p["preds"][g][k:k + n]
                k += n
                I.append(i)
        sets = {"test": I}
        for s in sorted({data.trs[i]["suite"] for i in I}):
            sets["test|suite=%s" % s] = [i for i in I if data.trs[i]["suite"] == s]
        for m in data.agents():
            sets["test|agent=%s" % m] = [i for i in I if data.trs[i]["model"] == m]
        for a in sorted({data.metas[i]["attack_id"] for i in I} - {None}):   # the attack's runs + all benign runs
            sets["test|attack=%s" % a] = [i for i in I if data.metas[i]["attack_id"] in (a, None)
                                          and data.trs[i]["suite"] == a.split("|")[0]]
        run = {}
        for name, J in sets.items():
            boot = n_boot if "|attack=" not in name else 0
            run[name] = {g: score_set(data, J, np.array([x for i in J for x in sc[g][i]], dtype=float),
                                      [x for i in J for x in pr[g][i]], boot, sd) for g in GUARDS}
        runs.append(run)
    return runs


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

