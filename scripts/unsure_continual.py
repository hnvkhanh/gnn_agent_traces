#!/usr/bin/env python3
"""Continue training the guard on its own unsure decisions (after temperature scaling), then test again.

Pipeline per (model, seed), on the usual task- and attack-disjoint split:
  1. train on a fit split, fit a temperature T on a task+attack-disjoint validation split, softmax(logits / T)
  2. "unsure" = test action whose confidence (max class probability after TS) is below --threshold. A trace with at least
     one unsure action is queried: its true labels are revealed and the model is trained on it (continual learning)
  3. repeat for --rounds rounds (T re-fitted on the same validation split every round, queries accumulate)
  4. score the same test traces again and compare with the model before any update

Because the queried test actions are trained on, scoring them again is contaminated. Three scenarios are therefore run
from the same trained base model:
  full      query from all test traces, score all test traces ("literal": includes queried) and the non-queried ones
  heldout   test task groups are split into folds A/B with disjoint groups; query from A, score B (and B -> A). The
            primary, leakage-free number. B's attack traces are also split into attack_shared / attack_novel (the attack
            id was / was not seen among the queried A traces); benign traces carry no attack id and are only in "all"
Learning technique: the repo's continual-learning strategies (src/continual/strategies.py, the CGLB set used by
scripts/run_continual.py), driven exactly as src/continual/run.py drives them. The fit split is experience 0 and each
query round is a new experience: strat.start -> train_experience (class-weighted CE, Adam, strategy loss and gradient
hooks) -> strat.end (fills the replay memory / computes the EWC, MAS, TWP importances). Strategies: finetune, replay,
ergnn, agem by default; ewc, mas, lwf, twp are used when results/continual/hparams.json holds their tuned
hyperparameters (run scripts/run_continual.py first), otherwise they are skipped with a note. GNN only, because the
strategies target HeteroGNN (ER-GNN and TWP hook model.head).
Variants are "<strategy>/<selection>". Selections: targeted (unsure traces), random (same number of traces as targeted,
per round), confident (the most confident traces, same number), oracle (every pool trace, one round). The random /
confident / oracle controls run with --control-strategy only.

Usage:
  python scripts/unsure_continual.py --bench agentdojo --models gnn mlp --seeds 0 1 2
  python scripts/unsure_continual.py --bench agentdojo --devices cuda:0,cuda:1      # seeds run in parallel on both GPUs
  python scripts/unsure_continual.py --bench agentdojo --report-only                # rebuild report from stored cells

Cells are stored in <outdir>/cells and reused when their fingerprint (code, trace file, arguments) still matches.
Writes REPORT.md, report.html and unsure_cl.json to <outdir> (default results/confidence/unsure_cl_<bench>_<split>).
"""
import argparse
import concurrent.futures
import glob
import hashlib
import html
import json
import math
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

DEFAULT_STRATEGIES = ["finetune", "replay", "ergnn", "agem"]
TUNED_NEEDED = ["ewc", "mas", "lwf", "twp"]
SELECTIONS = ["targeted", "random", "confident", "oracle"]
VIEWS = {"full": ["all", "nonqueried"], "heldout": ["all", "attack_shared", "attack_novel"]}
METRICS = ["accuracy", "macro_f1", "reject_recall", "ece", "cw_fn", "cw_fp", "unsure"]
CELL_VERSION = 1  # bump when run_cell / selection / metrics change what a cell contains (printing and report changes do not)
CODE_FILES = ["src/train.py", "src/model/gnn.py", "src/graph/build.py",
              "src/continual/strategies.py", "src/continual/run.py"]


# ---------------------------------------------------------------- pure helpers (no torch)
def softmax_p1(z, T=1.0):
    z = np.asarray(z, float).reshape(-1, 2) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(1)


def nll(logits, y, T=1.0):
    z = np.asarray(logits, float) / T
    z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-lp[np.arange(len(y)), np.asarray(y, int)].mean())


def fit_temperature(logits, y):
    """T > 0 minimising validation NLL (log-grid, then local refinement). 1.0 if the validation set is empty/single-class."""
    y = np.asarray(y, int)
    if len(y) < 2 or len(set(y.tolist())) < 2:
        return 1.0
    grid = np.logspace(-1.5, 2, 400)
    best = grid[int(np.argmin([nll(logits, y, t) for t in grid]))]
    fine = np.linspace(best / 1.05, best * 1.05, 101)
    return float(fine[int(np.argmin([nll(logits, y, t) for t in fine]))])


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if not len(conf):
        return None
    edges = np.linspace(0.5, 1.0, bins + 1)
    tot = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= edges[i + 1]))
        if m.any():
            tot += m.sum() / len(conf) * abs(conf[m].mean() - correct[m].mean())
    return float(tot)


def metrics(Z, y, T, t):
    """Scores for logits Z [n,2] and labels y at temperature T and confidence threshold t. Rates are over all n actions."""
    y = np.asarray(y, int)
    n = len(y)
    if n == 0:
        return None
    p1 = softmax_p1(Z, T)
    pred = (p1 >= 0.5).astype(int)
    conf = np.maximum(p1, 1 - p1)
    ok = pred == y
    sure = conf >= t
    f1s = []
    for c in (0, 1):
        tp, fp, fn = int(((pred == c) & (y == c)).sum()), int(((pred == c) & (y != c)).sum()), int(((pred != c) & (y == c)).sum())
        if tp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    pos = int((y == 1).sum())
    return {"n": n, "accuracy": float(ok.mean()), "macro_f1": float(np.mean(f1s)) if f1s else None,
            "reject_recall": float(((pred == 1) & (y == 1)).sum() / pos) if pos else None, "ece": ece(conf, ok),
            "cw_fn": float((sure & ~ok & (y == 1)).mean()), "cw_fp": float((sure & ~ok & (y == 0)).mean()),
            "unsure": float((~sure).mean())}


def trace_min_conf(Zs, T):
    """Per trace: lowest action confidence under temperature T. Zs is a list of [n_actions, 2] logit arrays."""
    out = []
    for z in Zs:
        p = softmax_p1(z, T)
        out.append(float(np.maximum(p, 1 - p).min()) if len(p) else 1.0)
    return out


def select_queries(variant, pool, min_conf, t, n_target, rng, budget=1.0):
    """Pick trace ids from `pool` (unqueried candidates). Depends on confidences and, for the controls, on a count only,
    never on labels. targeted: min action confidence < t, lowest first, capped at ceil(budget * #unsure).
    random / confident: n_target traces (random / highest min-confidence). oracle: everything."""
    pool = list(pool)
    if variant == "oracle":
        return pool
    if variant == "targeted":
        uns = sorted((i for i in pool if min_conf[i] < t), key=lambda i: (min_conf[i], i))
        return uns[: max(0, math.ceil(budget * len(uns)))] if budget < 1 else uns
    n = min(n_target, len(pool))
    if variant == "random":
        return rng.sample(sorted(pool), n)
    if variant == "confident":
        return sorted(pool, key=lambda i: (-min_conf[i], i))[:n]
    raise ValueError(variant)


def drop_overlap(train, held, metas):
    """Remove from `train` every trace sharing a task group or an attack id with `held` (indices into metas)."""
    groups = {metas[i]["group"] for i in held}
    attacks = {metas[i]["attack_id"] for i in held} - {None}
    return [i for i in train if metas[i]["group"] not in groups and metas[i]["attack_id"] not in attacks]


def val_split(tr_i, metas, seed):
    """Validation traces for T: ~20% of the training task groups; fit traces share neither a task nor an attack with them."""
    groups = sorted({metas[i]["group"] for i in tr_i})
    random.Random(1000 + seed).shuffle(groups)
    held = set(groups[: max(1, len(groups) // 5)])
    val = [i for i in tr_i if metas[i]["group"] in held]
    vs = set(val)
    return drop_overlap([i for i in tr_i if i not in vs], val, metas), val


def fold_split(te_i, metas, seed):
    """Two folds of the test traces with disjoint task groups."""
    groups = sorted({metas[i]["group"] for i in te_i})
    random.Random(2000 + seed).shuffle(groups)
    a = set(groups[: len(groups) // 2])
    return [i for i in te_i if metas[i]["group"] in a], [i for i in te_i if metas[i]["group"] not in a]


def views_for(scenario, evalset, queried, metas):
    """{view name: trace ids} for the traces scored after the update."""
    q = set(queried)
    if scenario == "full":
        return {"all": list(evalset), "nonqueried": [i for i in evalset if i not in q]}
    seen = {metas[i]["attack_id"] for i in queried} - {None}
    atk = [i for i in evalset if metas[i]["attack_id"] is not None]
    return {"all": list(evalset), "attack_shared": [i for i in atk if metas[i]["attack_id"] in seen],
            "attack_novel": [i for i in atk if metas[i]["attack_id"] not in seen]}


def variant_names(strategies, control):
    return ["%s/targeted" % s for s in strategies] + ["%s/%s" % (control, sel) for sel in SELECTIONS if sel != "targeted"]


def fingerprint(args_dict, trace_paths):
    h = hashlib.sha256(str(CELL_VERSION).encode())
    for f in CODE_FILES:
        h.update(open(os.path.join(ROOT, f), "rb").read())
    for p in sorted(trace_paths):
        h.update(hashlib.sha256(open(p, "rb").read()).digest())
    h.update(json.dumps(args_dict, sort_keys=True).encode())
    return h.hexdigest()[:16]


def fingerprint_args(a):
    keys = ["split", "holdout", "epochs", "hidden", "update_epochs", "update_lr", "rounds", "threshold", "budget", "buffer",
            "strategies", "control_strategy"]
    d = {k: getattr(a, k) for k in keys}
    d["hp"] = {s: hp_for(s, a.hparams) for s in a.strategies}
    return d


def hp_for(strategy, hparams_path):
    """Tuned hyperparameters from run_continual's hparams.json ({} for strategies without a grid); None if missing."""
    if strategy not in TUNED_NEEDED:
        return {}
    try:
        return json.load(open(hparams_path, encoding="utf-8"))[strategy]["hp"]
    except (OSError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------- training (torch imported lazily)
def load_graphs(pattern):
    from src.graph.build import meta_of, trace_to_graph
    graphs, metas = [], []
    for p in sorted(glob.glob(pattern)):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                tr = json.loads(line)
                g = trace_to_graph(tr)
                if g is not None:
                    graphs.append(g)
                    metas.append(dict(meta_of(tr), trace_id=tr["trace_id"]))
    return graphs, metas


def make_strategy(strategy, hp, buffer, seed, new_model):
    """src.continual.strategies.make, except ER-GNN's end() is replaced by the same code with its labels moved to the CPU.
    The repo version indexes CPU embeddings with labels left on the GPU by Batch.to(), so it raises on CUDA. It is
    patched here, not in src/, because editing strategies.py would invalidate the stored results/continual cells."""
    from src.continual.strategies import ERGNN, ActionEmbeddings, batches, make
    strat = make(strategy, hp, buffer, seed, new_model)
    if strategy != "ergnn":
        return strat
    import torch

    def end(self, k, model, cur, cur_ids, w):
        model.eval()
        dev = next(model.parameters()).device
        emb = ActionEmbeddings(model)
        hs, ys = [], []
        try:
            with torch.no_grad():
                for b in batches(cur, 1):
                    model(b.to(dev))
                    hs.append(emb.h.detach().cpu())
                    ys.append(b["action"].y.cpu())
        finally:
            emb.close()
        rank = {}
        for c in (0, 1):
            pts = [h[y == c] for h, y in zip(hs, ys)]
            allc = [p for p in pts if len(p)]
            if not allc:
                continue
            mu = torch.cat(allc).mean(0)
            rank[c] = [i for _, i in sorted((float((p.mean(0) - mu).norm()), i) for i, p in enumerate(pts) if len(p))]
        order, seen = [], set()
        for pos in range(max((len(r) for r in rank.values()), default=0)):
            for c in sorted(rank):
                if pos < len(rank[c]) and rank[c][pos] not in seen:
                    seen.add(rank[c][pos])
                    order.append(rank[c][pos])
        self.ranked.append([(cur_ids[i][0], cur_ids[i][1], cur[i]) for i in order])
        n = len(self.ranked)
        quota = [self.buffer // n + (1 if i < self.buffer % n else 0) for i in range(n)]
        self.mem = [x for r, q in zip(self.ranked, quota) for x in r[:q]]

    strat.__class__ = type("ERGNNCpuLabels", (ERGNN,), {"end": end})
    return strat


def run_cell(kind, seed, graphs, metas, a):
    """All scenarios x variants x rounds for one seed. Returns the cell dict."""
    import copy
    import torch
    from src.continual.run import _no_rng, class_weights, train_experience
    from src.continual.strategies import batches
    from src.model.gnn import HeteroGNN
    from src.train import DEVICE, split

    if kind != "gnn":
        raise ValueError("the continual strategies target the HeteroGNN guard; only --models gnn is supported")

    def G(idx):
        return [graphs[i] for i in idx]

    def ids(idx):
        return [(metas[i]["benchmark"], metas[i]["trace_id"]) for i in idx]

    def labels(i):
        return graphs[i]["action"].y.numpy()

    def trace_logits(model, idx):
        """{trace index: logits [n_actions, 2]} under the current model (batched; stored graphs are never modified)."""
        idx = list(idx)
        model.eval()
        with torch.no_grad():
            z = torch.cat([model(b.to(DEVICE))[:, :2].cpu() for b in batches(G(idx))]).numpy() if idx else np.zeros((0, 2))
        out, k = {}, 0
        for i in idx:
            n = int(graphs[i]["action"].y.shape[0])
            out[i] = z[k:k + n]
            k += n
        assert k == len(z)
        return out

    tr_i, te_i = split(metas, a.split, a.holdout, seed)
    fit_i, val_i = val_split(tr_i, metas, seed)
    if not fit_i or not val_i or not te_i:
        raise RuntimeError("empty split: %d fit / %d val / %d test traces" % (len(fit_i), len(val_i), len(te_i)))
    yv = np.concatenate([labels(i) for i in val_i])

    def new_model():
        torch.manual_seed(seed)
        return HeteroGNN(a.hidden, classes=2)

    # experience 0: the fit split, trained once; strategies only differ in what they store afterwards (memory, importances)
    base = new_model().to(DEVICE)
    from src.continual.strategies import Finetune
    base, _ = train_experience(base, Finetune({}, a.buffer, seed, new_model), G(fit_i), a.epochs, lr=3e-3)
    base_state = copy.deepcopy(base.state_dict())

    def temperature(model):
        z = trace_logits(model, val_i)
        return fit_temperature(np.concatenate([z[i] for i in val_i]), yv)

    T0 = temperature(base)
    Z0 = trace_logits(base, te_i)
    fa, fb = fold_split(te_i, metas, seed)
    scenarios = [("full", "full", te_i, te_i)]
    if fa and fb:
        scenarios += [("heldout", "A->B", fa, fb), ("heldout", "B->A", fb, fa)]

    def score(Z, idx, T):
        if not idx:
            return None
        return metrics(np.concatenate([Z[i] for i in idx]), np.concatenate([labels(i) for i in idx]), T, a.threshold)

    plan = [(s, "targeted") for s in a.strategies] + [(a.control_strategy, sel) for sel in SELECTIONS if sel != "targeted"]
    records = []
    for scen, direction, pool_all, evalset in scenarios:
        sizes = {}  # strategy -> queried-trace counts per round of its targeted run (sizes for the random/confident controls)
        for strategy, sel in plan:
            variant = "%s/%s" % (strategy, sel)
            model = new_model().to(DEVICE)
            model.load_state_dict(base_state)
            strat = make_strategy(strategy, hp_for(strategy, a.hparams), a.buffer, seed, new_model)
            w0 = class_weights(G(fit_i)).to(DEVICE)
            past = [G(fit_i)]  # experience 0
            model, _ = strat.start(0, model, G(fit_i), [])
            with _no_rng(DEVICE):
                strat.end(0, model, G(fit_i), ids(fit_i), w0)
            rng = random.Random(seed * 101 + plan.index((strategy, sel)))
            queried, T, Zc = [], T0, Z0
            for r in range(1, (1 if sel == "oracle" else a.rounds) + 1):
                qs = set(queried)
                cand = [i for i in pool_all if i not in qs]
                mc = dict(zip(cand, trace_min_conf([Zc[i] for i in cand], T)))
                n_t = sizes.get(a.control_strategy, [])
                n_t = n_t[r - 1] if r - 1 < len(n_t) else 0
                new = select_queries(sel, cand, mc, a.threshold, n_t, rng, a.budget)
                if sel == "targeted":
                    sizes.setdefault(strategy, []).append(len(new))
                if new:
                    queried += new
                    model, graphs_k = strat.start(r, model, G(new), past)
                    model, w = train_experience(model, strat, graphs_k, a.update_epochs, lr=a.update_lr)
                    with _no_rng(DEVICE):
                        strat.end(r, model, G(new), ids(new), w)
                    past.append(G(new))
                    T = temperature(model)
                    Zc = trace_logits(model, te_i)
                    if scen == "heldout" and set(strat.memory_ids()) & set(ids(evalset)):
                        raise RuntimeError("leakage: replay memory holds a scored held-out trace")
                qact = np.bincount(np.concatenate([labels(i) for i in queried]), minlength=2).tolist() if queried else [0, 0]
                for view, idx in views_for(scen, evalset, queried, metas).items():
                    records.append({"scenario": scen, "direction": direction, "variant": variant, "round": r, "view": view,
                                    "queried_traces": len(queried), "queried_actions": qact, "T": T, "T0": T0,
                                    "before": score(Z0, idx, T0), "after": score(Zc, idx, T)})
    return {"model": kind, "seed": seed, "T0": T0, "n_fit": len(fit_i), "n_val": len(val_i), "n_test": len(te_i),
            "folds": [len(fa), len(fb)], "records": records}


# ---------------------------------------------------------------- reporting
def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def pct(x):
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def fmt(m, x):
    return "n/a" if x is None else ("%.3f" % x if m == "ece" else pct(x))


def aggregate(cells):
    """{(model, scenario, variant, round, view): {metric: (before, after), queried_traces, T}} averaged over seeds/directions."""
    buckets = {}
    for c in cells:
        for r in c["records"]:
            buckets.setdefault((c["model"], r["scenario"], r["variant"], r["round"], r["view"]), []).append(r)
    out = {}
    for k, rs in buckets.items():
        d = {m: (mean([r["before"][m] for r in rs if r["before"]]), mean([r["after"][m] for r in rs if r["after"]])) for m in METRICS}
        d["n_actions"] = mean([r["after"]["n"] for r in rs if r["after"]]) or 0.0  # 0 when the view is empty in every cell
        d["queried_traces"] = mean([r["queried_traces"] for r in rs])
        d["T"] = mean([r["T"] for r in rs])
        d["T0"] = mean([r["T0"] for r in rs])
        d["cells"] = len(rs)
        out[k] = d
    return out


def table_md(head, body):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in body])


def table_html(head, body):
    h = "".join("<th>%s</th>" % html.escape(str(x)) for x in head)
    b = "".join("<tr>" + "".join("<td>%s</td>" % html.escape(str(x)) for x in r) + "</tr>" for r in body)
    return "<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (h, b)


PALETTE = ["#1f6fb5", "#e08a1e", "#2a9d5c", "#7a5bb5", "#c0392b", "#16a0a8", "#b58a1f", "#d1578a"]
COLORS = {}  # variant -> colour, filled by build()


def line_chart(series, title, pctfmt=True, w=340, h=190):
    """series: {name: [(x, y)]} -> inline SVG."""
    pts = [p for s in series.values() for p in s if p[1] is not None]
    if not pts:
        return ""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 0.01, y1 + 0.01
    L, R, T_, B = 44, 8, 22, 28
    sx = lambda x: L + (x - x0) / ((x1 - x0) or 1) * (w - L - R)  # noqa: E731
    sy = lambda y: T_ + (1 - (y - y0) / (y1 - y0)) * (h - T_ - B)  # noqa: E731
    f = (lambda v: "%.0f%%" % (100 * v)) if pctfmt else (lambda v: "%.3f" % v)
    g = ['<svg viewBox="0 0 %d %d" width="%d" role="img"><text x="%d" y="14" font-size="12" font-weight="600" fill="currentColor">%s</text>' % (w, h, w, L, html.escape(title))]
    for v in (y0, (y0 + y1) / 2, y1):
        g.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="#8884" /><text x="%d" y="%.1f" font-size="10" text-anchor="end" fill="currentColor">%s</text>' % (L, w - R, sy(v), sy(v), L - 4, sy(v) + 3, f(v)))
    for x in sorted(set(xs)):
        g.append('<text x="%.1f" y="%d" font-size="10" text-anchor="middle" fill="currentColor">%s</text>' % (sx(x), h - 12, "before" if x == 0 else "r%d" % x))
    for name, s in series.items():
        s = [p for p in s if p[1] is not None]
        if s:
            g.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (COLORS.get(name, "#444"), " ".join("%.1f,%.1f" % (sx(x), sy(y)) for x, y in s)))
            g += ['<circle cx="%.1f" cy="%.1f" r="2.5" fill="%s"/>' % (sx(x), sy(y), COLORS.get(name, "#444")) for x, y in s]
    return "".join(g) + "</svg>"


def build(cells, args):
    agg = aggregate(cells)
    models = sorted({k[0] for k in agg})
    seeds = sorted({c["seed"] for c in cells})
    rounds = max(k[3] for k in agg)
    variants = list(dict.fromkeys(k[2] for k in agg))
    targeted = [v for v in variants if v.endswith("/targeted")]
    for i, v in enumerate(variants):
        COLORS[v] = PALETTE[i % len(PALETTE)] if not v.endswith(("/random", "/confident", "/oracle")) else {"random": "#888888", "confident": "#b58a1f", "oracle": "#2a9d5c"}[v.split("/")[1]]
    last = lambda v: 1 if v.endswith("/oracle") else rounds  # noqa: E731
    md = ["# Continual learning on unsure decisions: %s, split=%s" % (args.bench or "custom", args.split), "",
          "Unsure = confidence after temperature scaling < **%.2f**; a trace is queried (labels revealed, model trained on it) if any of its "
          "actions is unsure. %d round(s), seeds %s, update %d epochs at lr %g, strategies %s, buffer %d. Values are means over seeds"
          " (and over A->B / B->A for held-out); `before -> after` compares with the same base model without any update on the same "
          "traces." % (args.threshold, rounds, seeds, args.update_epochs, args.update_lr, ", ".join(t.split("/")[0] for t in targeted), args.buffer), "",
          "**Read this first.** `full` / `all` scores traces that were trained on when queried, so it is optimistic. The leakage-free "
          "number is `heldout` (query fold A, score fold B with disjoint task groups). `nonqueried` shows what happened to the traces "
          "that were never queried.", ""]
    hh = ["<h1>%s</h1>" % html.escape(md[0][2:]), "<p>%s</p>" % html.escape(md[2]), "<p><b>Read this first.</b> %s</p>" % html.escape(md[4].replace("**Read this first.** ", "").replace("`", ""))]
    js = {"threshold": args.threshold, "seeds": seeds, "aggregate": {"|".join(map(str, k)): v for k, v in agg.items()}}
    head = ["variant", "queried traces", "accuracy", "macro-F1", "reject recall", "ECE", "missed attack (conf FN)", "false block (conf FP)", "unsure"]
    for model in models:
        md += ["## %s" % model, ""]
        hh.append("<h2>%s</h2>" % html.escape(model))
        for scen, views in VIEWS.items():
            for view in views:
                rows = []
                for v in variants:
                    d = agg.get((model, scen, v, last(v), view))
                    if d:
                        rows.append([v, "%.0f" % d["queried_traces"]] + ["%s -> %s" % (fmt(m, d[m][0]), fmt(m, d[m][1])) for m in METRICS])
                if not rows:
                    continue
                d0 = agg[(model, scen, targeted[0], rounds, view)]
                if not d0["n_actions"]:  # empty view (e.g. no attack traces of that kind in the scored fold)
                    continue
                cap = "%s / %s (~%.0f test actions, T %.2f -> %.2f at final round)" % (scen, view, d0["n_actions"], d0["T0"], d0["T"])
                md += ["### %s" % cap, "", table_md(head, rows), ""]
                hh += ["<h3>%s</h3>" % html.escape(cap), table_html(head, rows)]
        # round curves, primary held-out view (fall back to full/all)
        scen = "heldout" if any(k[1] == "heldout" for k in agg if k[0] == model) else "full"
        charts = []
        for m, label in (("macro_f1", "macro-F1"), ("reject_recall", "reject recall"), ("cw_fn", "confident missed attacks"), ("unsure", "unsure fraction")):
            series = {}
            for v in variants:
                if v.endswith("/oracle"):
                    continue
                s = [(r, agg[(model, scen, v, r, "all")][m][1]) for r in range(1, rounds + 1) if (model, scen, v, r, "all") in agg]
                if s:
                    s.insert(0, (0, agg[(model, scen, v, 1, "all")][m][0]))
                    series[v] = s
            charts.append(line_chart(series, "%s (%s/all)" % (label, scen)))
        legend = " ".join('<span style="color:%s">&#9632; %s</span>' % (COLORS[v], v) for v in variants if not v.endswith("/oracle"))
        hh += ["<h3>Per round, %s / all</h3><div class='charts'>%s</div><p>%s</p>" % (scen, "".join(charts), legend)]
        md += ["### Per round, %s / all (%s)" % (scen, targeted[0]), "", table_md(["round", "queried traces", "T", "macro-F1", "reject recall", "conf FN", "conf FP", "unsure"],
               [[r, "%.0f" % agg[(model, scen, targeted[0], r, "all")]["queried_traces"], "%.2f" % agg[(model, scen, targeted[0], r, "all")]["T"]] +
                ["%s -> %s" % (fmt(m, agg[(model, scen, targeted[0], r, "all")][m][0]), fmt(m, agg[(model, scen, targeted[0], r, "all")][m][1])) for m in ("macro_f1", "reject_recall", "cw_fn", "cw_fp", "unsure")]
                for r in range(1, rounds + 1) if (model, scen, targeted[0], r, "all") in agg]), ""]
    ts = [(c["model"], c["seed"], c["T0"], c["n_fit"], c["n_val"], c["n_test"], c["folds"]) for c in cells]
    md += ["## Cells", "", table_md(["model", "seed", "T (base)", "fit traces", "val traces", "test traces", "fold A/B"], [[*t[:6], "%d/%d" % tuple(t[6])] for t in ts]), ""]
    hh += ["<h2>Cells</h2>" + table_html(["model", "seed", "T (base)", "fit", "val", "test", "fold A/B"], [[t[0], t[1], "%.2f" % t[2], t[3], t[4], t[5], "%d/%d" % tuple(t[6])] for t in ts])]
    css = ("body{font:14px system-ui;margin:0;padding:1rem 1.5rem;max-width:1200px;background:#fff;color:#222}"
           "@media(prefers-color-scheme:dark){body{background:#161616;color:#ddd}}"
           "table{border-collapse:collapse;margin:.5rem 0;display:block;overflow-x:auto}td,th{border:1px solid #8886;padding:3px 8px;text-align:left}"
           ".charts{display:flex;flex-wrap:wrap;gap:.8rem}")
    page = "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Unsure continual learning</title><style>%s</style>%s" % (css, "".join(hh))
    return js, "\n".join(md) + "\n", page


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def print_summary(agg, rounds):
    """Terminal summary: every variant, before -> after the last round, for the main views of each scenario."""
    variants = list(dict.fromkeys(k[2] for k in agg))
    cols = [("accuracy", "acc"), ("macro_f1", "mF1"), ("reject_recall", "rej-rec"), ("ece", "ECE"), ("cw_fn", "confFN"), ("cw_fp", "confFP"), ("unsure", "unsure")]
    for model in sorted({k[0] for k in agg}):
        for scen, view, note in (("heldout", "all", "leakage-free: query one fold, score the other"),
                                 ("heldout", "attack_novel", "held-out attacks not seen among queried traces"),
                                 ("full", "all", "optimistic: includes queried traces"),
                                 ("full", "nonqueried", "traces never queried")):
            rows = [(v, agg[(model, scen, v, 1 if v.endswith("/oracle") else rounds, view)]) for v in variants
                    if (model, scen, v, 1 if v.endswith("/oracle") else rounds, view) in agg]
            if not rows:
                continue
            print("\n== %s | %s / %s (%s) | before -> after, mean over seeds ==" % (model, scen, view, note))
            print("%-20s %7s " % ("variant", "queried") + " ".join("%-15s" % c[1] for c in cols))
            for v, d in rows:
                print("%-20s %7.0f " % (v, d["queried_traces"]) + " ".join("%-15s" % ("%s>%s" % (fmt(m, d[m][0]), fmt(m, d[m][1]))) for m, _ in cols))
    print()


# ---------------------------------------------------------------- driver
def devices_from(spec):
    if spec != "auto":
        return [d.strip() for d in spec.split(",") if d.strip()]
    if os.environ.get("GNN_DEVICE"):
        return [os.environ["GNN_DEVICE"]]
    try:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n = 0
    return ["cuda:%d" % i for i in range(n)] or ["cpu"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", help="injecagent|agentdojo (sets --traces to traces/<bench>__<model>.jsonl)")
    ap.add_argument("--llm", default="qwen2.5:1.5b", help="trace LLM, used with --bench")
    ap.add_argument("--traces", help="trace glob, overrides --bench")
    ap.add_argument("--split", default="task", choices=["task", "style", "agent", "mode", "category", "benchmark"])
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--models", nargs="+", default=["gnn"], choices=["gnn"], help="GNN only: the continual strategies target HeteroGNN")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.90, help="unsure if confidence after TS is below this")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--update-epochs", type=int, default=10)
    ap.add_argument("--update-lr", type=float, default=1e-3)
    ap.add_argument("--strategies", nargs="+", default=None, help="continual strategies (default: %s plus the tuned ones found in --hparams)" % " ".join(DEFAULT_STRATEGIES))
    ap.add_argument("--control-strategy", default="replay", help="strategy used for the random / confident / oracle controls")
    ap.add_argument("--buffer", type=int, default=50, help="memory size in graphs (agem, replay, ergnn), as run_continual.py")
    ap.add_argument("--hparams", default=os.path.join(ROOT, "results", "continual", "hparams.json"), help="tuned hyperparameters from run_continual.py")
    ap.add_argument("--budget", type=float, default=1.0, help="fraction of unsure traces queried per round, lowest confidence first")
    ap.add_argument("--devices", default="auto", help="auto | cpu | cuda:0,cuda:1 ; one worker process per device")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore stored cells")
    ap.add_argument("--cell", help=argparse.SUPPRESS)  # internal: kind:seed, runs one cell in this process
    a = ap.parse_args()
    if not a.traces:
        if not a.bench:
            ap.error("give --bench or --traces")
        a.traces = os.path.join(ROOT, "traces", "%s__%s.jsonl" % (a.bench, a.llm.replace(":", "_")))
    if a.strategies is None:
        a.strategies = DEFAULT_STRATEGIES + [t for t in TUNED_NEEDED if hp_for(t, a.hparams) is not None]
        skipped = [t for t in TUNED_NEEDED if t not in a.strategies]
        if skipped and not a.cell:
            print("note: %s skipped, no tuned hyperparameters in %s (run scripts/run_continual.py first)" % (", ".join(skipped), a.hparams))
    for t in a.strategies:
        if hp_for(t, a.hparams) is None:
            ap.error("strategy %s needs tuned hyperparameters in %s (run scripts/run_continual.py first)" % (t, a.hparams))
    if a.control_strategy not in a.strategies:
        a.strategies = a.strategies + [a.control_strategy]
    a.outdir = a.outdir or os.path.join(ROOT, "results", "confidence", "unsure_cl_%s_%s" % (a.bench or "custom", a.split))
    cells_dir = os.path.join(a.outdir, "cells")
    os.makedirs(cells_dir, exist_ok=True)
    paths = sorted(glob.glob(a.traces))
    if not paths:
        sys.exit("no traces matched %s" % a.traces)
    fp = fingerprint(fingerprint_args(a), paths)
    cell_path = lambda k, s: os.path.join(cells_dir, "%s_seed%d.json" % (k, s))  # noqa: E731

    def stored(k, s):
        try:
            c = json.load(open(cell_path(k, s), encoding="utf-8"))
            return c if c.get("fingerprint") == fp else None
        except (OSError, ValueError):
            return None

    if a.cell:
        kind, seed = a.cell.split(":")
        graphs, metas = load_graphs(a.traces)
        c = run_cell(kind, int(seed), graphs, metas, a)
        c["fingerprint"] = fp
        write(cell_path(kind, int(seed)), json.dumps(c))
        return

    todo = [(k, s) for k in a.models for s in a.seeds if a.force or not stored(k, s)]
    if a.report_only:
        todo = []
    reused = len(a.models) * len(a.seeds) - len(todo)
    if todo:
        devs = devices_from(a.devices)
        print("running %d cell(s) on %s (%d reused)" % (len(todo), devs, reused), flush=True)
        passthrough = [x for k in ("split", "holdout", "epochs", "hidden", "threshold", "rounds", "update_epochs", "update_lr", "buffer", "budget", "hparams", "control_strategy")
                       for x in (["--" + k.replace("_", "-"), str(getattr(a, k))] if getattr(a, k) is not None else [])]

        def work(j):
            i, (k, s) = j
            env = dict(os.environ, GNN_DEVICE=devs[i % len(devs)])
            cmd = [sys.executable, os.path.abspath(__file__), "--traces", a.traces, "--outdir", a.outdir, "--cell", "%s:%d" % (k, s), "--strategies", *a.strategies] + passthrough
            print("  start %s seed %d on %s" % (k, s, env["GNN_DEVICE"]), flush=True)
            r = subprocess.run(cmd, env=env)
            return k, s, r.returncode

        with concurrent.futures.ThreadPoolExecutor(len(devs)) as ex:
            for k, s, rc in ex.map(work, list(enumerate(todo))):
                print("  done %s seed %d%s" % (k, s, "" if rc == 0 else "  FAILED (exit %d)" % rc), flush=True)
    cells = [c for k in a.models for s in a.seeds for c in [stored(k, s)] if c]
    if not cells:
        sys.exit("no stored cells for the current code/arguments; run without --report-only")
    js, md, page = build(cells, a)
    write(os.path.join(a.outdir, "unsure_cl.json"), json.dumps(js, indent=1, sort_keys=True) + "\n")
    write(os.path.join(a.outdir, "REPORT.md"), md)
    write(os.path.join(a.outdir, "report.html"), page)
    print_summary(aggregate(cells), a.rounds)
    print("wrote", a.outdir)


if __name__ == "__main__":
    main()
