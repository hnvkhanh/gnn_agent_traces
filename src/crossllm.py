"""Cross-LLM transfer: does a guard trained on the traces of one LLM work on the traces of another?

Design (every choice below is a leakage or confound control; tests/test_crossllm.py checks each one):
  * Every LLM answered the same scenarios (same task groups and attacks), so ONE set of held-out task groups H is drawn per
    (scope, seed) and applied to every LLM. All test columns are the same scenarios, only the LLM behaviour differs.
  * Training traces never come from a group in H, and never share an attack (meta['attack_id']) with ANY LLM's test traces.
    The attack filter uses the test traces of all LLMs, so the training set depends only on the source LLM, never on the
    target: one training run is scored on every target (diagonal = in-LLM baseline on the same held-out scenarios).
  * Every source gets the same number of training traces (min over LLMs, uniform random subsample), because the qwen2.5:1.5b
    file has twice as many traces as the others and size alone would otherwise explain a better row.
  * Leave-one-LLM-out (LOLO): train on the other LLMs, each contributing an equal share of the SAME total size.
  * The LLM name is not a graph feature (checked in tests); the appended synthetic attacker steps are identical in every LLM,
    so each cell is also scored on real (LLM-chosen) actions only.
"""
import hashlib
import os
import random

import numpy as np
import torch

from .eval import asr_report, report, tool_lookup_baseline
from .model.gnn import ActionMLP, HeteroGNN
from .sandbox.policy import LABELS
from .train import fit, ys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ("majority", "tool-lookup", "mlp", "gnn")


def recipe_hash():
    return hashlib.sha256(open(os.path.join(ROOT, "src", "crossllm.py"), "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def held_groups(groups, seed, frac=0.2):
    gs = sorted(set(groups))
    random.Random(seed).shuffle(gs)
    return set(gs[: max(1, int(len(gs) * frac))])


def make_split(metas, seed, frac=0.2):
    """metas: {llm: [meta]} for one scope. Returns dict(held, test={llm: idx}, train={llm: idx}) with train independent of any target."""
    held = held_groups([m["group"] for ms in metas.values() for m in ms], seed, frac)
    test = {l: [i for i, m in enumerate(ms) if m["group"] in held] for l, ms in metas.items()}
    test_attacks = {metas[l][i]["attack_id"] for l in metas for i in test[l]} - {None}
    train = {l: [i for i, m in enumerate(ms) if m["group"] not in held and m["attack_id"] not in test_attacks]
             for l, ms in metas.items()}
    return dict(held=held, test=test, train=train, test_attacks=test_attacks)


def sample(idx, n, seed, tag):
    r = random.Random("%s|%s" % (seed, tag))
    idx = list(idx)
    return sorted(r.sample(idx, n)) if n < len(idx) else idx


def train_sets(split, seed):
    """{'single:<llm>': [(llm, idx)], 'lolo:<llm>': [(llm, idx)]}, all of the same total size n."""
    llms = sorted(split["train"])
    n = min(len(v) for v in split["train"].values())
    out = {}
    for l in llms:
        out["single:" + l] = [(l, sample(split["train"][l], n, seed, "single" + l))]
    for t in llms:
        others = [l for l in llms if l != t]
        share = n // len(others)
        out["lolo:" + t] = [(l, sample(split["train"][l], share, seed, "lolo%s%s" % (t, l))) for l in others]
    return out


def predict_many(train, train_meta, tests, epochs=60, lr=3e-3, hidden=64, seed=0, models=MODELS):
    """Train once on `train`, score every target in `tests` ({name: (graphs, metas)}). Same recipe as train.evaluate.
    Returns {target: {model or 'model|real': report incl. 'asr'}} plus label counts under key '_n'."""
    from .train import DEVICE  # noqa: F401  (device is picked by GNN_DEVICE at import)
    torch.manual_seed(seed)
    names = LABELS
    ncls = len(names)
    ytr = ys(train)
    cnt = torch.bincount(ytr, minlength=ncls).float().clamp(min=1)
    w = cnt.sum() / (ncls * cnt)
    all_g = [g for t in tests for g in tests[t][0]]
    all_m = [m for t in tests for m in tests[t][1]]
    full = {}
    if "majority" in models:
        full["majority"] = [int(ytr.bincount().argmax())] * len(ys(all_g))
    if "tool-lookup" in models:
        full["tool-lookup"] = tool_lookup_baseline(train_meta, ytr.tolist(), all_m)
    if "mlp" in models:
        full["mlp"] = fit(ActionMLP(hidden, ncls), train, all_g, epochs, lr, w, ncls)
    if "gnn" in models:
        full["gnn"] = fit(HeteroGNN(hidden, classes=ncls), train, all_g, epochs, lr, w, ncls)
    out, a0 = {}, 0
    for t, (g, m) in tests.items():
        y = ys(g).numpy()
        n = len(y)
        synth = np.array([s for mm in m for s in mm["synthetic"]], dtype=bool)
        res = {}
        for k, P in full.items():
            p = list(P[a0:a0 + n])
            asr = asr_report(y.tolist(), p, m)
            res[k] = dict(report(y.tolist(), p, names), asr=asr)
            keep = ~synth
            res[k + "|real"] = dict(report(y[keep].tolist(), [q for q, s in zip(p, keep) if s], names), asr=asr)
        res["_n"] = dict(actions=int(n), reject=int(y.sum()), traces=len(m), attack_traces=sum(1 for mm in m if mm["attack"]))
        out[t] = res
        a0 += n
    return out


def audit(split, sets, metas):
    """Overlap counts between each training set and the test side (all must be 0) and sizes."""
    te = [(l, i) for l in split["test"] for i in split["test"][l]]
    te_g = {metas[l][i]["group"] for l, i in te}
    te_a = {metas[l][i]["attack_id"] for l, i in te} - {None}
    te_t = {metas[l][i]["trace_id"] for l, i in te}
    rows = {}
    for k, parts in sets.items():
        tr = [(l, i) for l, idx in parts for i in idx]
        rows[k] = dict(n=len(tr), attacks=sum(1 for l, i in tr if metas[l][i]["attack"]),
                       groups=len({metas[l][i]["group"] for l, i in tr} & te_g),
                       attack_ids=len({metas[l][i]["attack_id"] for l, i in tr} & te_a),
                       trace_ids=len({metas[l][i]["trace_id"] for l, i in tr} & te_t))
    return rows
