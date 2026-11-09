"""Training engine and per-cell results of the continual-learning experiments.

A cell is one (mode, order, strategy, hparams, buffer, seed). It trains the HeteroGNN through the stream and stores
R[step][bench] = scores on every benchmark's test set (validation set in tune mode) after every step, plus row "init"
(the untrained model, for FWT). Each cell is a JSON file under <outdir>/cells/ with a fingerprint over the code,
the exact train/test graphs and the config. A stored cell whose fingerprint matches is reused, never retrained.

    python -m src.continual.run --strategy ewc --mode tune               # grid on the validation stream (seed 0)
    python -m src.continual.run --strategy ewc --seeds 0,1,2             # all 6 orders x seeds, hparams.json lambda
Normally scripts/run_continual.py runs these jobs (both GPUs) and writes the report.
"""
import argparse
import hashlib
import itertools
import json
import os
import time

import torch
import torch.nn.functional as Fn
from torch_geometric.loader import DataLoader

from .. import cache, train
from ..eval import asr_report, report
from ..model.gnn import HeteroGNN
from ..sandbox.policy import LABELS
from . import stream as S
from .strategies import STRATEGIES, TUNED, batches, make, resolved_hp

# the same settings as train.fit / train.evaluate
LR, WD, BATCH, HIDDEN, NCLS = 3e-3, 1e-4, 32, 64, len(LABELS)
EPOCHS, BUFFER = 60, 50
ROOT = cache.ROOT
CL_CODE = ["src/continual/stream.py", "src/continual/strategies.py", "src/continual/run.py"]


def device():
    return train.DEVICE  # read at call time, so GNN_DEVICE (and tests) decide


# ---------------------------------------------------------------- engine
def class_weights(graphs):
    cnt = torch.bincount(torch.cat([g["action"].y for g in graphs]), minlength=NCLS).float().clamp(min=1)
    return cnt.sum() / (NCLS * cnt)


def predict(model, graphs):
    dev = device()
    model.eval()
    with torch.no_grad():
        return torch.cat([model(b.to(dev))[:, :NCLS].argmax(1).cpu() for b in batches(graphs)]).tolist()


def score(model, exp, idx):
    """{'all': report + asr over every action, 'real': report over actions the LLM chose itself}."""
    graphs, metas = [exp.graphs[i] for i in idx], [exp.metas[i] for i in idx]
    y = torch.cat([g["action"].y for g in graphs]).tolist()
    p = predict(model, graphs)
    synth = [s for m in metas for s in m["synthetic"]]
    keep = [i for i, s in enumerate(synth) if not s]
    return dict(all=dict(report(y, p), asr=asr_report(y, p, metas)),
                real=report([y[i] for i in keep], [p[i] for i in keep]))


def train_experience(model, strat, graphs, epochs, lr=LR):
    """train.fit's loop (Adam, class-weighted CE, batch 32) with the strategy's loss and gradient hooks."""
    dev = device()
    w = class_weights(graphs + strat.replay_graphs())
    model, w = model.to(dev), w.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    loader = DataLoader(graphs, batch_size=BATCH, shuffle=True)
    for _ in range(epochs):
        model.train()
        for b in loader:
            b = b.to(dev)
            opt.zero_grad()
            out = model(b)
            loss = strat.loss(model, b, out, Fn.cross_entropy(out, b["action"].y, weight=w), w)
            loss.backward()
            strat.after_backward(model, w)
            opt.step()
    return model, w


def _no_rng(dev):
    """Strategy bookkeeping must not shift the training RNG stream (so strategies stay comparable)."""
    return torch.random.fork_rng(devices=[dev.index or 0] if dev.type == "cuda" else [])


def run_cell(exps, sp, order, strategy, hp, buffer, seed, epochs):
    """Train through `order` and score every test set after every step. Returns the cell record (no fingerprint)."""
    dev = device()
    torch.manual_seed(seed)

    def new_model():
        torch.manual_seed(seed)
        return HeteroGNN(HIDDEN, classes=NCLS)

    model = HeteroGNN(HIDDEN, classes=NCLS)
    strat = make(strategy, hp, buffer, seed, new_model)
    t0 = time.time()
    R = {"init": {b: score(model.to(dev), exps[b], sp[b][1]) for b in order}}
    steps = []
    for k, b in enumerate(order):
        cur = [exps[b].graphs[i] for i in sp[b][0]]
        past = [[exps[x].graphs[i] for i in sp[x][0]] for x in order[:k]]
        model, graphs = strat.start(k, model, cur, past)
        t = time.time()
        model, w = train_experience(model, strat, graphs, epochs)
        with _no_rng(dev):
            strat.end(k, model, cur, [(b, exps[b].ids[i]) for i in sp[b][0]], w)
        R[str(k)] = {x: score(model, exps[x], sp[x][1]) for x in order}
        steps.append(dict(bench=b, trained_graphs=len(graphs), seconds=round(time.time() - t, 1),
                          memory=sorted(strat.memory_ids())))
    return dict(R=R, steps=steps, seconds=round(time.time() - t0, 1), config=dict(order=list(order)))


# ---------------------------------------------------------------- fingerprints and planning
def code_digest():
    h = hashlib.sha256(cache.code_hash().encode())
    for p in CL_CODE:
        h.update(p.encode())
        h.update(open(os.path.join(ROOT, p), "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def side_digest(exp, idx):
    h = hashlib.sha256(b"%d" % len(idx))
    for i in idx:
        h.update(exp.ids[i].encode())
        h.update(cache.graph_digest(exp.graphs[i]))
    h.update(json.dumps([exp.metas[i] for i in idx], sort_keys=True, default=str).encode())
    return h.hexdigest()


class Planner:
    """Builds the cells of a job (or of the whole run) with their fingerprints. Splits are computed once per seed."""

    def __init__(self, exps, epochs=EPOCHS, device_type=None):
        self.exps, self.epochs = exps, epochs
        self.device_type = device_type or device().type
        self.code = code_digest()
        self._sp, self._dig = {}, {}

    def split(self, mode, seed):
        key = (mode, seed)
        if key not in self._sp:
            sp = S.tune_splits(self.exps, seed) if mode == "tune" else S.splits(self.exps, seed)
            self._sp[key] = S.check(self.exps, sp, "%s split, seed %d" % (mode, seed))
            if mode == "tune":  # validation must also stay away from the test sets
                test = S.splits(self.exps, seed)
                bad = {k: v for k, v in S.overlaps(self.exps, S.side(self.exps, sp, 1), S.side(self.exps, test, 1)).items() if v}
                bad.update({"train:" + k: v for k, v in
                            S.overlaps(self.exps, S.side(self.exps, sp, 0), S.side(self.exps, test, 1)).items() if v})
                if bad:
                    raise AssertionError("tune split touches the test sets: %s" % bad)
        return self._sp[key]

    def digest(self, mode, seed, bench):
        key = (mode, seed, bench)
        if key not in self._dig:
            tr, te = self.split(mode, seed)[bench]
            self._dig[key] = [side_digest(self.exps[bench], tr), side_digest(self.exps[bench], te)]
        return self._dig[key]

    def cell(self, outdir, mode, strategy, order, seed, hp, buffer):
        hp = resolved_hp(strategy, hp)
        buf = buffer if STRATEGIES[strategy].uses_buffer else None
        config = dict(mode=mode, order=list(order), strategy=strategy, hp=hp, buffer=buf, seed=seed,
                      epochs=self.epochs, lr=LR, wd=WD, batch=BATCH, hidden=HIDDEN, model=S.MODEL,
                      device=self.device_type)
        data = [self.digest(mode, seed, b) for b in order]
        fp = hashlib.sha256(json.dumps(dict(code=self.code, data=data, config=config), sort_keys=True).encode()).hexdigest()
        hps = "_".join("%s=%g" % kv for kv in sorted(hp.items())) or "default"
        name = "%s__%s%s__s%d.json" % ("-".join(order), ("%s__" % hps) if mode == "tune" else "",
                                       "m%d" % buf if buf is not None else "nomem", seed)
        path = os.path.join(outdir, "cells", mode, strategy, name)
        return dict(path=path, fp=fp, config=config, mode=mode, strategy=strategy, order=list(order), seed=seed,
                    hp=hp, buffer=buf)

    def job_cells(self, outdir, mode, strategy, seeds, orders, buffer, hparams):
        if mode == "tune":
            return [self.cell(outdir, "tune", strategy, S.CANONICAL, 0, hp, buffer) for hp in grid_points(strategy)]
        hp = chosen_hp(strategy, hparams)
        return [self.cell(outdir, "main", strategy, o, s, hp, buffer) for s in seeds for o in orders]


def grid_points(strategy):
    """Every hyperparameter combination of a tuned strategy's grid, in a fixed order."""
    grid = STRATEGIES[strategy].grid or {}
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def chosen_hp(strategy, hparams):
    if strategy not in TUNED:
        return {}
    if not hparams or strategy not in hparams:
        raise SystemExit("no tuned hyperparameters for %s: run the tune phase first (scripts/run_continual.py)" % strategy)
    return hparams[strategy]["hp"]


def load_cell(c):
    """The stored record of planned cell c if it is fresh (same fingerprint), else None."""
    if not os.path.exists(c["path"]):
        return None
    try:
        rec = json.load(open(c["path"], encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if rec.get("fingerprint") == c["fp"] else None


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def compute(planner, c):
    sp = planner.split(c["mode"], c["seed"])
    rec = run_cell(planner.exps, sp, c["order"], c["strategy"], c["hp"], c["buffer"] or 0, c["seed"], planner.epochs)
    rec.update(fingerprint=c["fp"], config=c["config"], sizes=S.sizes(planner.exps, sp),
               time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(c["path"], rec)
    return rec


def val_ap(rec):
    """Final average macro-F1 over the validation sets (the tuning criterion)."""
    last = rec["R"][str(len(rec["config"]["order"]) - 1)]
    return sum(v["all"]["macro_f1"] for v in last.values()) / len(last)


def choose(records):
    """records: tune cells of one strategy -> {'hp': best, 'val_ap': .., 'grid': [[hp, ap], ..]}. Ties: smaller values."""
    keys = sorted(STRATEGIES[records[0]["config"]["strategy"]].grid)
    rows = sorted((tuple(r["config"]["hp"][k] for k in keys), val_ap(r)) for r in records)
    pos, (best, ap) = max(enumerate(rows), key=lambda x: (x[1][1], -x[0]))
    return dict(hp=dict(zip(keys, best)), val_ap=round(ap, 4),
                grid=[[dict(zip(keys, v)), round(a, 4)] for v, a in rows])


# ---------------------------------------------------------------- CLI (one job)
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    ap.add_argument("--mode", default="main", choices=["main", "tune"])
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds (main mode)")
    ap.add_argument("--orders", default="all", choices=["all", "canonical"])
    ap.add_argument("--buffer", type=int, default=BUFFER, help="memory size in graphs (agem, replay, ergnn)")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="epochs per experience")
    ap.add_argument("--hparams", default=None, help="hparams.json from the tune phase (default: <outdir>/hparams.json)")
    ap.add_argument("--outdir", default="results/continual")
    a = ap.parse_args()
    hp_path = a.hparams or os.path.join(a.outdir, "hparams.json")
    hparams = json.load(open(hp_path, encoding="utf-8")) if os.path.exists(hp_path) else {}
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    orders = S.ORDERS if a.orders == "all" else [S.CANONICAL]
    t = time.time()
    exps = S.load_stream()
    print("loaded %s in %.0fs, device %s" % (", ".join("%s %d" % (b, len(e)) for b, e in exps.items()),
                                              time.time() - t, device()), flush=True)
    planner = Planner(exps, a.epochs)
    cells = planner.job_cells(a.outdir, a.mode, a.strategy, seeds, orders, a.buffer, hparams)
    n = dict(reused=0, computed=0)
    for c in cells:
        tag = "%s %s %s s%d %s" % (c["mode"], c["strategy"], ">".join(c["order"]), c["seed"],
                                   json.dumps(c["hp"], sort_keys=True) if c["hp"] else "")
        if load_cell(c) is not None:
            n["reused"] += 1
            print("[cache] reused   %s" % tag, flush=True)
            continue
        rec = compute(planner, c)
        n["computed"] += 1
        print("[cache] computed %s (%.0fs)" % (tag, rec["seconds"]), flush=True)
    print("[cache] %s/%s: %d reused, %d computed" % (a.mode, a.strategy, n["reused"], n["computed"]), flush=True)


if __name__ == "__main__":
    main()
