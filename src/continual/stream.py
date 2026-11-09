"""The benchmark stream: traces, orders and leakage-checked train/test splits.

Each experience is one benchmark of qwen2.5:1.5b traces. Every split comes from train.split(metas, "task", None, seed),
the main suite's split, so it is task-disjoint and attack-disjoint, and the test sets match inbench/<bench> in
results/qwen2.5_1.5b. Benchmarks share no task groups and no attack ids. audit() checks this, and also the
normalised task text and attacker tool-name tokens, over the union of every train and every test set in the stream.
"""
import glob
import itertools
import json
import re

from ..graph.build import meta_of, trace_to_graph
from ..train import split

BENCHES = ["asb", "injecagent", "agentdojo"]  # AgentPoison is excluded on purpose
MODEL = "qwen2.5:1.5b"
ORDERS = [list(p) for p in itertools.permutations(BENCHES)]
CANONICAL = list(BENCHES)
TUNE_SALT = 1000  # validation split = train.split(train part of seed 0, seed TUNE_SALT)


class Experience:
    """One benchmark: graphs, metas, trace ids and user-task texts, all aligned by index."""

    def __init__(self, name, graphs, metas, ids, tasks):
        self.name, self.graphs, self.metas, self.ids, self.tasks = name, graphs, metas, ids, tasks

    def __len__(self):
        return len(self.graphs)


def load_experience(bench, model=MODEL, traces="traces/%s__*.jsonl"):
    """Same graphs and metas as train.load(traces % bench, models=[model]), plus trace ids and task texts."""
    graphs, metas, ids, tasks = [], [], [], []
    for p in sorted(glob.glob(traces % bench)):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            tr = json.loads(line)
            if tr.get("model") != model:
                continue
            g = trace_to_graph(tr)
            if g is None:
                continue
            m = meta_of(tr)
            if m["benchmark"] != bench:
                raise ValueError("%s holds a %s trace" % (p, m["benchmark"]))
            graphs.append(g)
            metas.append(m)
            ids.append(tr["trace_id"])
            tasks.append(tr["user_task"])
    if not graphs:
        raise ValueError("no %s traces for %s in %s" % (bench, model, traces % bench))
    return Experience(bench, graphs, metas, ids, tasks)


def load_stream(model=MODEL, benches=BENCHES, traces="traces/%s__*.jsonl"):
    return {b: load_experience(b, model, traces) for b in benches}


def splits(exps, seed):
    """{bench: (train_idx, test_idx)}: the main suite's held-out-task split of each benchmark."""
    return {b: split(e.metas, "task", None, seed) for b, e in exps.items()}


def tune_splits(exps, seed=0):
    """{bench: (tune_train_idx, val_idx)}, both inside the seed-`seed` training part. Validation never touches test."""
    out = {}
    for b, e in exps.items():
        tr, _ = split(e.metas, "task", None, seed)
        sub_tr, sub_val = split([e.metas[i] for i in tr], "task", None, seed + TUNE_SALT)
        out[b] = ([tr[i] for i in sub_tr], [tr[i] for i in sub_val])
    return out


_norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
_tokens = lambda s: set(re.findall(r"[a-z0-9_]+", (s or "").lower()))


def side(exps, sp, k):
    """All (bench, index) pairs on side k (0 = train, 1 = test) of the stream."""
    return [(b, i) for b, idx in sp.items() for i in idx[k]]


def overlaps(exps, a, b):
    """Overlap counts between two sides, each a list of (bench, index) pairs."""
    def get(pairs, f):
        return [f(exps[x], i) for x, i in pairs]
    out = {}
    out["traces"] = len(set(get(a, lambda e, i: (e.name, e.ids[i]))) & set(get(b, lambda e, i: (e.name, e.ids[i]))))
    out["groups"] = len(set(get(a, lambda e, i: e.metas[i]["group"])) & set(get(b, lambda e, i: e.metas[i]["group"])))
    att = lambda pairs: set(get(pairs, lambda e, i: e.metas[i]["attack_id"])) - {None}
    out["attacks"] = len(att(a) & att(b))
    out["task_text"] = len(set(get(a, lambda e, i: _norm(e.tasks[i]))) & set(get(b, lambda e, i: _norm(e.tasks[i]))))
    # ASB / InjecAgent: the attack id is the attacker tool, whose name is a hashed token in the action features
    tok = lambda pairs: set().union(*[_tokens(exps[x].metas[i]["attack_id"]) for x, i in pairs
                                      if exps[x].name in ("asb", "injecagent")] or [set()])
    out["attacker_tokens"] = len(tok(a) & tok(b))
    return out


def audit(exps, sp):
    """Leakage audit of one split of the whole stream: train side vs test side. Every count must be 0."""
    return overlaps(exps, side(exps, sp, 0), side(exps, sp, 1))


def check(exps, sp, what="split"):
    """Raise if the split leaks anything between the union of train sets and the union of test sets."""
    bad = {k: v for k, v in audit(exps, sp).items() if v}
    if bad:
        raise AssertionError("%s leaks between train and test: %s" % (what, bad))
    return sp


def sizes(exps, sp):
    """{bench: {train, train_attack, test, test_attack}} trace counts."""
    out = {}
    for b, (tr, te) in sp.items():
        m = exps[b].metas
        out[b] = dict(train=len(tr), train_attack=sum(m[i]["attack"] for i in tr),
                      test=len(te), test_attack=sum(m[i]["attack"] for i in te))
    return out
