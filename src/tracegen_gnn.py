"""Approve / reject guard on tracegen (real AgentDojo) traces, with and without file nodes.

Used by scripts/run_tracegen_gnn.py (CLI, parallel devices, report). Cells:
  main                  train split -> val and test; guards majority, tool-lookup, mlp, gnn, gnn+files, transformer
                        (src/model/transformer.py: attention over the gnn+files graph, same nodes / features / causality),
                        edge-gnn (src/graph/edges.py + src/model/edge_gnn.py: artifacts are nodes, actions are edges),
                        gnn+files+prompt and edge-gnn+prompt (plus the frozen prompt embedding, src/prompt_embed.py),
                        mlp+content and gnn+files+content (plus frozen MiniLM embeddings of action / observation
                        text, src/node_embed.py, after Zavrak, arXiv:2605.11053)
  ablate/<a>            gnn+files with ablation <a> (src/graph/files.py FILE_ABLATIONS)
  ablate-content/<a>    the two content guards with ablation <a> (CONTENT_ABLATE); the text ablations also apply to
                        the embedded text
  ablate-edge/<a>       edge-gnn with ablation <a> (src/graph/edges.py EDGE_ABLATIONS)
  ablate-prompt/<a>     the two prompt guards with the prompt control <a>: prompt_zero (all-zero embedding) or
                        prompt_shuffle (the prompt of another user task of the same suite)
  cvprompt/<a>/<k>/<i>-<j>  the cross-validation cell cv/<k>/<i>-<j> for the prompt guards under control <a>
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


import numpy as np
import torch
import torch.nn.functional as Fn

from .benchmarks.tracegen import load_traces
from .cache import ROOT, TRAIN_CODE, data_hash
from .eval import asr_report, report
from .graph.build import meta_of, trace_to_graph
from .graph.edges import EDGE_ABLATIONS, trace_to_graph_edges
from .graph.files import FILE_ABLATIONS, trace_to_graph_files
from .rank_metrics import average_precision, paired_delta, rank_report, threshold_for_f1
from .sandbox.policy import LABELS

GUARDS = ["majority", "tool-lookup", "mlp", "gnn", "gnn+files", "transformer", "edge-gnn", "gnn+files+prompt",
          "edge-gnn+prompt", "mlp+content", "gnn+files+content"]
PROMPT_GUARDS = ["gnn+files+prompt", "edge-gnn+prompt"]
PROMPT_BASE = {"gnn+files+prompt": "gnn+files", "edge-gnn+prompt": "edge-gnn"}   # the same model without the prompt
CONTENT_GUARDS = ["mlp+content", "gnn+files+content"]
CONTENT_BASE = {"mlp+content": "mlp", "gnn+files+content": "gnn+files"}   # the same model without the content
CONTENT_ABLATE = ["observation", "injected", "attacker_values"]
ABLATE = ["files", "observation", "injected", "copies", "mentions", "tool_feat", "attacker_values"]
EDGE_ABLATE = list(EDGE_ABLATIONS)
PROMPT_ABLATE = ["prompt_zero", "prompt_shuffle"]
CODE = TRAIN_CODE + ["src/tracegen_gnn.py", "src/graph/build.py", "src/graph/files.py", "src/model/gnn_files.py",
                     "src/model/transformer.py", "src/benchmarks/tracegen.py", "src/rank_metrics.py",
                     "src/graph/edges.py", "src/model/edge_gnn.py", "src/prompt_embed.py", "src/node_embed.py"]
POOLED = ("main", "cv/", "cvprompt/", "ablate-prompt/")   # cells that store per-action test scores (_pool)


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
        self._g, self._folds, self._prompted, self._pvec = {}, {}, set(), None
        self._contented, self._cvec = set(), {}

    def graphs(self, kind="files", ablate=(), prompt=False, content=False):
        """Graphs of every trace. kind: plain (build.py), files (files.py) or edges (edges.py). An ablation name
        'edge:<a>' applies to the edges graphs only, 'prompt:<a>' is the prompt control; plain names go to plain /
        files. prompt=True attaches g['task'].prompt (one embedding row per task node). content=True (plain / files
        only) attaches g['action'].emb and g['observation'].emb (src/node_embed.py, under the same text ablations)."""
        key = (kind, tuple(sorted(ablate)))
        if key not in self._g:
            if kind == "edges":
                f, own = trace_to_graph_edges, frozenset(a[5:] for a in ablate if a.startswith("edge:"))
            else:
                f = trace_to_graph_files if kind == "files" else trace_to_graph
                own = frozenset(a for a in ablate if ":" not in a)
            self._g[key] = [f(t, own) for t in self.trs]
        gs = self._g[key]
        if prompt and key not in self._prompted:
            ctl = [a[7:] for a in ablate if a.startswith("prompt:")]
            for g, v in zip(gs, self.prompt_rows(ctl[0] if ctl else None)):
                g["task"].prompt = v.unsqueeze(0).expand(g["task"].x.shape[0], -1).contiguous()
            self._prompted.add(key)
        if content and key not in self._contented:
            assert kind != "edges", "content embeddings are for action / observation nodes (plain or files graphs)"
            for g, (va, vo) in zip(gs, self.content_rows([a for a in ablate if ":" not in a])):
                if g is not None:
                    g["action"].emb, g["observation"].emb = va, vo
            self._contented.add(key)
        return gs

    def content_rows(self, ablate=()):
        """Per trace (action embeddings [n, DIM], observation embeddings [n, DIM]) under the text ablations in `ablate`
        (src/node_embed.py; computed once per text, then read from its cache)."""
        from .node_embed import TEXT_ABLATIONS, node_texts, node_vectors
        key = tuple(sorted(set(ablate) & set(TEXT_ABLATIONS)))
        if key not in self._cvec:
            T = [node_texts(t, key) for t in self.trs]
            V = node_vectors([x for a, o in T for x in a + o])
            out, k = [], 0
            for a, _ in T:
                n = len(a)
                out.append((V[k:k + n], V[k + n:k + 2 * n]))
                k += 2 * n
            self._cvec[key] = out
        return self._cvec[key]

    def prompt_table(self):
        """{user prompt: frozen embedding} (src/prompt_embed.py; computed once, then read from its cache)."""
        if self._pvec is None:
            from .prompt_embed import prompt_vectors
            self._pvec = prompt_vectors([t["user_task"] for t in self.trs])
        return self._pvec

    def shuffle_donors(self):
        """Per trace, the index of a trace of the same suite with a different user task (None if there is none):
        deterministic, from a hash of the trace id."""
        by = {}
        for i, t in enumerate(self.trs):
            by.setdefault(t["suite"], []).append(i)
        out = []
        for i, t in enumerate(self.trs):
            c, g = by[t["suite"]], self.metas[i]["group"]
            start = zlib.crc32(t["trace_id"].encode()) % len(c)
            out.append(next((c[(start + d) % len(c)] for d in range(len(c))
                             if self.metas[c[(start + d) % len(c)]]["group"] != g), None))
        return out

    def prompt_rows(self, control=None):
        """Prompt embedding per trace under a control: None (its own prompt), prompt_zero, prompt_shuffle."""
        from .prompt_embed import DIM
        if control == "prompt_zero":
            return [torch.zeros(DIM) for _ in self.trs]
        tab = self.prompt_table()
        own = [tab[t["user_task"]] for t in self.trs]
        if control is None:
            return own
        assert control == "prompt_shuffle", control
        return [own[j] if j is not None else torch.zeros(DIM) for j in self.shuffle_donors()]

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
        out += ["ablate-edge/%s" % a for a in EDGE_ABLATE]
        out += ["ablate-prompt/%s" % a for a in PROMPT_ABLATE]
        out += ["ablate-content/%s" % a for a in CONTENT_ABLATE]
    if "cvprompt" in sections:
        out += ["cvprompt/%s/%d/%d-%d" % (p, folds, a, b) for p in PROMPT_ABLATE for a in range(folds)
                for b in range(folds) if cv_spec(data, folds, a, b)[2]]
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
    if key.startswith("ablate-edge/"):
        a = key.split("/", 1)[1]
        assert a in EDGE_ABLATE, a
        va = data.idx("val")
        return data.idx("train"), va, {"val": va, "test": data.idx("test")}, ["edge-gnn"], ("edge:" + a,)
    if key.startswith("ablate-prompt/"):
        a = key.split("/", 1)[1]
        assert a in PROMPT_ABLATE, a
        va = data.idx("val")
        return data.idx("train"), va, {"val": va, "test": data.idx("test")}, list(PROMPT_GUARDS), ("prompt:" + a,)
    if key.startswith("ablate-content/"):
        a = key.split("/", 1)[1]
        assert a in CONTENT_ABLATE, a
        va = data.idx("val")
        return data.idx("train"), va, {"val": va, "test": data.idx("test")}, list(CONTENT_GUARDS), (a,)
    if key.startswith("cvprompt/"):
        p, k, ab = key.split("/")[1:]
        assert p in PROMPT_ABLATE, p
        tr, va, te = cv_spec(data, int(k), *map(int, ab.split("-")))
        return tr, va, {"test": te}, list(PROMPT_GUARDS), ("prompt:" + p,)
    if key.startswith("transfer/"):
        a, b = key.split("/", 1)[1].split("->")
        return data.idx("train", a), data.idx("val", a), {"test": data.idx("test", b)}, GUARDS, ()
    if key.startswith("cv/"):
        k, ab = key.split("/")[1:]
        tr, va, te = cv_spec(data, int(k), *map(int, ab.split("-")))
        return tr, va, {"test": te}, GUARDS, ()
    raise KeyError(key)


def fingerprint(data, key, seed, epochs, lr, hidden, code=None, opts=None):
    """opts: the other run_cell options (clip, score, select, extra, guards); only non-default ones enter the hash."""
    tr, va, ev, guards, ab = cell_spec(data, key)
    opts = {k: list(v) if isinstance(v, tuple) else v for k, v in (opts or {}).items()
            if v is not None and v != REPO.get(k) and not (isinstance(v, (tuple, list, dict)) and not v)}
    if opts.get("guards"):
        guards = [g for g in guards if g in opts["guards"]]
    allev = sorted({i for v in ev.values() for i in v})
    g = data.graphs("files", tuple(a for a in ab if ":" not in a))   # edge / prompt controls are in `key`
    parts =[([g[i] for i in tr], [data.metas[i] for i in tr], [g[i] for i in allev], [data.metas[i] for i in allev]),
             ([g[i] for i in va], [data.metas[i] for i in va], [], [])]
    blob = dict(code=code or code_hash(), key=key, seed=seed, epochs=epochs, lr=lr, hidden=hidden, guards=guards,
                evals={k: [data.metas[i]["trace_id"] for i in v] for k, v in ev.items()}, data=data_hash(parts))
    if opts:
        blob["opts"] = opts
    if set(guards) & set(CONTENT_GUARDS):   # encoder / windowing decide the content vectors
        from .node_embed import config
        blob["content"] = config()
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def _y(g):
    """Action labels in action order: action nodes, or the queries of an edge graph."""
    return g["action"].y if "action" in g.node_types else g["query"].y


def _ys(graphs):
    return torch.cat([_y(g) for g in graphs]) if graphs else torch.zeros(0, dtype=torch.long)


def _scores(model, graphs, device, score="prob"):
    """Reject score per action, in graph order: the reject probability ('prob') or the reject-minus-approve logit
    margin ('logit': the same order, but no ties from probabilities that round to 1.0)."""
    from torch_geometric.loader import DataLoader
    if not graphs:
        return np.zeros(0)
    model.eval()
    out = []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=64):
            z = model(b.to(device)).double()
            out.append((torch.softmax(z, 1)[:, 1] if score == "prob" else z[:, 1] - z[:, 0]).cpu())
    return torch.cat(out).numpy()


# Training options (run_cell / fit_select). REPO reproduces the original runs; STABLE is the setting that fixed the
# seed sensitivity in results/tracegen_all_edge/ANALYSIS.md: lower learning rate, gradient clipping, and the mean score
# of the second half of training instead of the epoch picked on validation.
REPO = dict(lr=3e-3, clip=None, score="prob", select="best")
STABLE = dict(lr=1e-3, clip=1.0, score="logit", select="avg")


def _rule(rule, val_ap, epochs):
    """Epochs (1-based) whose scores a selection rule uses: 'best' the best validation AUPRC, 'best<N>' the same within
    the first N epochs, 'avg' every epoch of the second half (> epochs // 2), 'last' the last epoch. A best rule with no
    validation AUPRC (no reject action in validation) falls back to its last epoch."""
    if rule == "avg":
        return list(range(epochs // 2 + 1, epochs + 1))
    if rule == "last":
        return [epochs]
    if rule.startswith("best"):
        n = int(rule[4:] or epochs)
        aps = val_ap[:n]
        if any(a is None for a in aps):
            return [n]
        return [int(np.argmax(aps)) + 1]
    raise ValueError(rule)


def fit_select(model, train, val, evalg, epochs, lr, weights, clip=None, score="prob", rules=("best",), test_mask=None):
    """Train like src/train.py fit (Adam, batch 32, class-weighted cross entropy), optionally clipping the gradient norm
    at `clip`, and score validation and eval graphs after every epoch. Returns {rule: (validation scores, eval scores,
    info)} for every selection rule in `rules` (see _rule); an averaging rule's scores are the per-action mean over its
    epochs. One training run serves every rule. test_mask (bool per eval action): also log the test AUPRC of every
    epoch (diagnostic only, never used for selection)."""
    from torch_geometric.loader import DataLoader
    from .train import DEVICE
    model, weights = model.to(DEVICE), weights.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=32, shuffle=True)
    yv = _ys(val).numpy()
    yt = _ys(evalg).numpy()[test_mask] if test_mask is not None else None
    SV, SE, VA, LOSS, GN, TA = [], [], [], [], [], []
    for ep in range(1, epochs + 1):
        model.train()
        tot, gn, nb = 0.0, 0.0, 0
        for b in loader:
            b = b.to(DEVICE)
            opt.zero_grad()
            loss = Fn.cross_entropy(model(b), _y(b), weight=weights)
            loss.backward()
            if clip is not None:
                gn += float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
            opt.step()
            tot, nb = tot + float(loss), nb + 1
        SV.append(_scores(model, val, DEVICE, score))
        SE.append(_scores(model, evalg, DEVICE, score))
        VA.append(average_precision(yv, SV[-1]) if yv.any() else None)
        TA.append(round(float(average_precision(yt, SE[-1][test_mask])), 4) if yt is not None and yt.any() else None)
        LOSS.append(round(tot / max(1, nb), 5))
        GN.append(round(gn / max(1, nb), 4) if clip is not None else None)
    out = {}
    for r in rules:
        E = _rule(r, VA, epochs)
        sv, se = np.mean([SV[e - 1] for e in E], 0), np.mean([SE[e - 1] for e in E], 0)
        va = VA[E[0] - 1] if len(E) == 1 else (average_precision(yv, sv) if yv.any() else None)
        out[r] = (sv, se, dict(best_epoch=E[0] if len(E) == 1 else None, epochs_used=[E[0], E[-1]], val_auprc=va,
                               selected=r.startswith("best") and VA[0] is not None, rule=r))
    out[rules[0]][2].update(train_loss=LOSS, val_auprc_by_epoch=[None if a is None else round(a, 4) for a in VA],
                            test_auprc_by_epoch=TA, grad_norm=GN)
    return out


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


def run_cell(data, key, seed, epochs=60, lr=3e-3, hidden=64, n_boot=200, clip=None, score="prob", select="best",
             extra=(), guards=None, guard_opts=None):
    """Train every guard of the cell once (seed) and score every eval set.

    lr / clip / score / select: see fit_select and _rule (REPO and STABLE above). guard_opts {guard: {lr, clip, score,
    select}} overrides them per guard. extra: more selection rules, read off the same training run and reported as
    pseudo-guards '<guard>@<rule>' (e.g. 'edge-gnn@best10'). guards: train only these of the cell's guards (None: all)."""
    from .model.edge_gnn import EdgeGNN
    from .model.gnn import ActionMLP, HeteroGNN
    from .model.gnn_files import ContentMLP, FileHeteroGNN
    from .model.transformer import GraphTransformer
    tr, va, ev, cell_guards, ab = cell_spec(data, key)
    guards = [g for g in cell_guards if guards is None or g in guards]
    allev = sorted({i for v in ev.values() for i in v})
    fab = tuple(a for a in ab if ":" not in a)
    gf, gp = data.graphs("files", fab), data.graphs("plain", ())
    graphs_of = {"gnn": lambda: gp, "edge-gnn": lambda: data.graphs("edges", ab),
                 "edge-gnn+prompt": lambda: data.graphs("edges", ab, prompt=True),
                 "gnn+files+prompt": lambda: data.graphs("files", ab, prompt=True),
                 "mlp+content": lambda: data.graphs("files", fab, content=True),
                 "gnn+files+content": lambda: data.graphs("files", fab, content=True)}
    ncls = len(LABELS)
    ytr = _ys([gf[i] for i in tr])
    yva = _ys([gf[i] for i in va]).numpy()
    cnt = torch.bincount(ytr, minlength=ncls).float().clamp(min=1)
    w = cnt.sum() / (ncls * cnt)
    trm, vam, evm = ([data.metas[i] for i in I] for I in (tr, va, allev))
    n_act = [len(data.trs[i]["events"]) for i in allev]
    test = set(ev.get("test", ()))
    test_mask = np.concatenate([np.full(n, i in test) for i, n in zip(allev, n_act)]) if allev else None
    scores, info, names, used = {}, {}, [], {}
    for gname in guards:
        torch.manual_seed(seed)
        np.random.seed(seed)
        o = dict(dict(lr=lr, clip=clip, score=score, select=select), **(guard_opts or {}).get(gname, {}))
        rules = [o["select"]] + [r for r in extra if r != o["select"]]
        inf = dict(best_epoch=None, val_auprc=None, selected=False)
        if gname == "majority":   # constant score: predicts the majority training label, no threshold to tune
            res = {None: (None, np.full(sum(n_act), float(ytr.bincount(minlength=ncls).argmax())), inf)}
        elif gname == "tool-lookup":
            res = {None: (tool_rate_scores(trm, ytr.tolist(), vam), tool_rate_scores(trm, ytr.tolist(), evm), inf)}
        else:
            G = graphs_of.get(gname, lambda: gf)()
            model = {"mlp": lambda: ActionMLP(hidden, ncls), "gnn": lambda: HeteroGNN(hidden, classes=ncls),
                     "gnn+files": lambda: FileHeteroGNN(hidden, classes=ncls),
                     "transformer": lambda: GraphTransformer(hidden, classes=ncls),
                     "edge-gnn": lambda: EdgeGNN(hidden, classes=ncls),
                     "gnn+files+prompt": lambda: FileHeteroGNN(hidden, classes=ncls, prompt=True),
                     "edge-gnn+prompt": lambda: EdgeGNN(hidden, classes=ncls, prompt=True),
                     "mlp+content": lambda: ContentMLP(hidden, ncls),
                     "gnn+files+content": lambda: FileHeteroGNN(hidden, classes=ncls, content=True)}[gname]()
            res = fit_select(model, [G[i] for i in tr], [G[i] for i in va], [G[i] for i in allev], epochs, o["lr"], w,
                             o["clip"], o["score"], rules, test_mask)
            res = {(None if r == o["select"] else r): v for r, v in res.items()}
            used[gname] = o
        for r, (sv, se, inf) in res.items():
            name = gname if r is None else "%s@%s" % (gname, r)
            thr = threshold_for_f1(yva, sv) if sv is not None and yva.any() else None
            dflt = 0.5 if o["score"] == "prob" or gname in ("majority", "tool-lookup") else 0.0
            info[name] = dict(inf, threshold=dflt if thr is None else thr, threshold_tuned=thr is not None)
            per, k = {}, 0
            for i, n in zip(allev, n_act):
                per[i] = se[k:k + n]
                k += n
            scores[name] = per
            names.append(name)
    out = {}
    for name, I in ev.items():
        out[name] = {}
        for gname in names:
            s = np.concatenate([scores[gname][i] for i in I]) if I else np.zeros(0)
            p = (s >= info[gname]["threshold"]).astype(int).tolist()
            out[name][gname] = dict(score_set(data, I, s, p, n_boot, seed), **info[gname])
    if key.startswith(POOLED):   # per-action test scores and decisions: pooled by cv_reports, paired by prompt_effects
        I = ev["test"]
        out["_pool"] = dict(
            trace_ids=[data.metas[i]["trace_id"] for i in I],
            scores={g: [round(float(x), 6) for i in I for x in scores[g][i]] for g in names},
            preds={g: [int(x >= info[g]["threshold"]) for i in I for x in scores[g][i]] for g in names})
    out["_train"] = dict(epochs=epochs, extra=list(extra), guards=used)   # the settings each trained guard used
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


def pooled(data, store, keys, seed, guards):
    """(trace indices, {guard: {i: scores}}, {guard: {i: decisions}}) of the stored _pool of every cell in keys for one
    seed; None if a cell (or a guard of it) is missing."""
    pos = {m["trace_id"]: i for i, m in enumerate(data.metas)}
    pools = [store.get(k, {}).get(str(seed), {}).get("run", {}).get("_pool") for k in keys]
    if not keys or any(p is None or any(g not in p["scores"] for g in guards) for p in pools):
        return None
    I, sc, pr = [], {g: {} for g in guards}, {g: {} for g in guards}
    for p in pools:
        k = 0
        for tid in p["trace_ids"]:
            i = pos[tid]
            n = len(data.trs[i]["events"])
            for g in guards:
                sc[g][i], pr[g][i] = p["scores"][g][k:k + n], p["preds"][g][k:k + n]
            k += n
            I.append(i)
    return I, sc, pr


def cv_reports(data, store, seeds, folds, n_boot=200, keys=None, guards=None):
    """Pooled cross-validation result: one {eval_name: {guard: report}} per seed that has every cell stored. Eval names
    as in the main cell ('test', 'test|suite=<s>', 'test|agent=<m>', and 'test|attack=<injection task>').
    keys / guards default to the cv cells and every guard (cvprompt cells: pass their keys and PROMPT_GUARDS).

    Scores of different cells come from different models; pooling them into one AUPRC assumes they are comparable.
    The thresholded columns do not: each cell uses its own validation threshold."""
    keys = cells(data, ["cv"], folds) if keys is None else keys
    guards = guards or GUARDS
    runs = []
    for sd in seeds:
        got = pooled(data, store, keys, sd, guards)
        if got is None:
            continue
        I, sc, pr = got
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
                                      [x for i in J for x in pr[g][i]], boot, sd) for g in guards}
        runs.append(run)
    return runs


def prompt_comparisons(data, folds):
    """[(name, evaluation, (keys a, guard a), (keys b, guard b))]: each prompt guard against the same model without the
    prompt, against itself with an all-zero prompt and against itself with a shuffled prompt, on the pooled
    cross-validation ('cv') and on the fixed test split ('test')."""
    cv = cells(data, ["cv"], folds)
    cvp = {p: [k for k in cells(data, ["cvprompt"], folds) if k.startswith("cvprompt/%s/" % p)] for p in PROMPT_ABLATE}
    out = []
    for g in PROMPT_GUARDS:
        out.append(("%s vs %s" % (g, PROMPT_BASE[g]), "cv", (cv, g), (cv, PROMPT_BASE[g])))
        out.append(("%s vs %s" % (g, PROMPT_BASE[g]), "test", (["main"], g), (["main"], PROMPT_BASE[g])))
        for p in PROMPT_ABLATE:
            out.append(("%s vs %s" % (g, p), "cv", (cv, g), (cvp[p], g)))
            out.append(("%s vs %s" % (g, p), "test", (["main"], g), (["ablate-prompt/%s" % p], g)))
    return out


def content_comparisons(data, folds):
    """Like prompt_comparisons, for the content embedding (src/node_embed.py): each content guard against the same model
    without content, and gnn+files+content against mlp+content (does the graph still add anything once both see the
    content? the question Zavrak, arXiv:2605.11053, Sec. 6.2, answers 'barely' for whole sessions)."""
    cv = cells(data, ["cv"], folds)
    pairs = [(g, CONTENT_BASE[g]) for g in CONTENT_GUARDS] + [("gnn+files+content", "mlp+content")]
    return [("%s vs %s" % (a, b), ev, (k, a), (k, b)) for a, b in pairs for ev, k in (("cv", cv), ("test", ["main"]))]


def prompt_effects(data, store, seeds, folds, n_boot=200):
    """Does the prompt embedding help? See paired_effects."""
    return paired_effects(data, store, seeds, prompt_comparisons(data, folds), n_boot)


def content_effects(data, store, seeds, folds, n_boot=200):
    """Does the action / observation content embedding help? See paired_effects."""
    return paired_effects(data, store, seeds, content_comparisons(data, folds), n_boot)


def paired_effects(data, store, seeds, comparisons, n_boot=200):
    """Per comparison and seed: paired ΔAUPRC / Δ recall@FPR (src/rank_metrics.py paired_delta) on the traces both sides
    scored, plus ΔAUPRC per suite. Comparisons with a side missing are left out.
    Returns [{name, eval, seeds: [{seed, n_traces, n_reject, d_auprc, d_auprc_ci, ..., suites: {s: d}}]}]."""
    out = []
    for name, ev, (ka, ga), (kb, gb) in comparisons:
        per = []
        for sd in seeds:
            A, B = pooled(data, store, ka, sd, [ga]), pooled(data, store, kb, sd, [gb])
            if A is None or B is None:
                continue
            sa, sb = A[1][ga], B[1][gb]
            I = [i for i in A[0] if i in sb]
            y = [LABELS.index(e["label"]) for i in I for e in data.trs[i]["events"]]
            r = paired_delta(y, [x for i in I for x in sa[i]], [x for i in I for x in sb[i]],
                             [len(data.trs[i]["events"]) for i in I], n_boot, sd)
            r.update(seed=sd, n_traces=len(I), n_reject=int(sum(y)), suites={})
            for s in sorted({data.trs[i]["suite"] for i in I}):
                J = [i for i in I if data.trs[i]["suite"] == s]
                yj = [LABELS.index(e["label"]) for i in J for e in data.trs[i]["events"]]
                r["suites"][s] = paired_delta(yj, [x for i in J for x in sa[i]], [x for i in J for x in sb[i]],
                                              [len(data.trs[i]["events"]) for i in J], 0, sd)["d_auprc"]
            per.append(r)
        if per:
            out.append(dict(name=name, eval=ev, seeds=per))
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

