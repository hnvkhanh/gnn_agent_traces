#!/usr/bin/env python3
"""Is the GNN confidently wrong or confidently correct?

Stored results only hold hard predictions, so this retrains the model with the same recipe as src/train.py (class-weighted
cross-entropy, Adam lr 3e-3, wd 1e-4, batch 32) on the same task- and attack-disjoint split, keeps the softmax, and sorts
every test action into four groups at a confidence threshold t (confidence = max class probability):

  confident-correct   conf >= t, prediction right      unsure-correct   conf < t, right
  confident-wrong     conf >= t, prediction wrong      unsure-wrong     conf < t, wrong

Confident-wrong is split by true class: confident FN = attack approved with high confidence (the dangerous failure),
confident FP = benign call blocked with high confidence. Class 0 = approve, class 1 = reject.

Writes predictions.csv, confidence.json, REPORT.md and report.html to --outdir.

Usage:
  python scripts/confidence_analysis.py --bench asb
  python scripts/confidence_analysis.py --bench injecagent --seeds 0 1 2 --threshold 0.95 --models gnn mlp
  python scripts/confidence_analysis.py --traces "traces/*__qwen2.5:7b.jsonl" --split style --holdout paraphrase
  python scripts/confidence_analysis.py --bench asb --eval train     # same, scored on the training traces
  python scripts/confidence_analysis.py --bench all --eval both      # asb, injecagent, agentdojo; train+test, GPUs in parallel
  python scripts/confidence_analysis.py --bench asb --report-only      # rebuild report from predictions.csv
"""
import argparse
import csv
import glob
import html
import json
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

FIELDS = ["model", "seed", "trace_id", "step", "tool", "true", "pred", "p_reject", "conf", "temperature", "synthetic",
          "benchmark", "category", "style", "mode", "attack_id"]
NAMES = ["approve", "reject"]


# ---------- pure metric helpers (no torch; unit-testable) ----------
def quadrant(true, pred, conf, t):
    """One of cc, cw_fn, cw_fp, uc, uw. cw_fn: attack (reject) predicted approve; cw_fp: benign predicted reject."""
    if pred == true:
        return "cc" if conf >= t else "uc"
    if conf < t:
        return "uw"
    return "cw_fn" if true == 1 else "cw_fp"


def quadrant_counts(rows, t):
    c = dict.fromkeys(["cc", "cw_fn", "cw_fp", "uc", "uw"], 0)
    for r in rows:
        c[quadrant(r["true"], r["pred"], r["conf"], t)] += 1
    c["n"] = len(rows)
    return c


def ece(conf, correct, bins=10):
    """Expected calibration error over equal-width confidence bins, plus per-bin (lo, hi, n, mean conf, accuracy)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.5, 1.0, bins + 1)  # two classes: max prob is always >= 0.5
    out, total = [], 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= edges[i + 1]))
        n = int(m.sum())
        if n:
            total += n / len(conf) * abs(conf[m].mean() - correct[m].mean())
            out.append((float(edges[i]), float(edges[i + 1]), n, float(conf[m].mean()), float(correct[m].mean())))
        else:
            out.append((float(edges[i]), float(edges[i + 1]), 0, None, None))
    return (float(total) if len(conf) else None), out


def nll(logits, y, T=1.0):
    """Mean negative log-likelihood of labels y under softmax(logits / T)."""
    z = np.asarray(logits, float) / T
    z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-lp[np.arange(len(y)), np.asarray(y, int)].mean())


def fit_temperature(logits, y):
    """Temperature T > 0 minimising validation NLL (grid over log T, then local refinement). Returns 1.0 when the
    validation set is empty or single-class (NLL then just pushes T to an extreme, which says nothing about calibration)."""
    y = np.asarray(y, int)
    if len(y) < 2 or len(set(y.tolist())) < 2:
        return 1.0
    grid = np.logspace(-1.5, 2, 400)
    best = grid[int(np.argmin([nll(logits, y, t) for t in grid]))]
    fine = np.linspace(best / 1.05, best * 1.05, 101)
    return float(fine[int(np.argmin([nll(logits, y, t) for t in fine]))])


def selective(conf, correct, points=10):
    """Accuracy when only the most confident k% of predictions are kept: [(coverage, accuracy)]."""
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    c = np.asarray(correct, float)[order]
    return [(k / points, float(c[: max(1, int(round(len(c) * k / points)))].mean())) for k in range(1, points + 1)] if len(c) else []


def breakdown(rows, key, t):
    """Per value of `key`: n, confident-wrong count and rate among that value's rows."""
    g = defaultdict(list)
    for r in rows:
        g[str(r[key])].append(r)
    out = []
    for v, rs in g.items():
        q = quadrant_counts(rs, t)
        cw = q["cw_fn"] + q["cw_fp"]
        out.append({"value": v, "n": len(rs), "confident_wrong": cw, "rate": cw / len(rs), "cw_fn": q["cw_fn"], "cw_fp": q["cw_fp"]})
    return sorted(out, key=lambda d: (-d["confident_wrong"], d["value"]))


def prf(rows, cls):
    """Precision, recall, F1 and support for class `cls` (0 approve, 1 reject), from hard predictions."""
    tp = sum(r["pred"] == cls and r["true"] == cls for r in rows)
    fp = sum(r["pred"] == cls and r["true"] != cls for r in rows)
    fn = sum(r["pred"] != cls and r["true"] == cls for r in rows)
    pr = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    f = 2 * pr * rc / (pr + rc) if pr and rc else (0.0 if pr is not None and rc is not None else None)
    return {"precision": pr, "recall": rc, "f1": f, "support": tp + fn}


def classification(rows):
    per = {NAMES[c]: prf(rows, c) for c in (0, 1)}
    f1s = [v["f1"] for v in per.values() if v["f1"] is not None]
    return {"per_class": per, "macro_f1": sum(f1s) / len(f1s) if f1s else None,
            "micro_f1": sum(r["pred"] == r["true"] for r in rows) / len(rows) if rows else None}


def summarize(rows, t):
    q = quadrant_counts(rows, t)
    n = max(q["n"], 1)
    conf_n = q["cc"] + q["cw_fn"] + q["cw_fp"]
    e, bins = ece([r["conf"] for r in rows], [r["pred"] == r["true"] for r in rows])
    return {"counts": q, "rates": {k: q[k] / n for k in ("cc", "cw_fn", "cw_fp", "uc", "uw")},
            "accuracy": (q["cc"] + q["uc"]) / n, "classification": classification(rows),
            "confident_precision": q["cc"] / conf_n if conf_n else None,  # accuracy among confident predictions
            "confident_fraction": conf_n / n, "ece": e, "bins": bins,
            "selective": selective([r["conf"] for r in rows], [r["pred"] == r["true"] for r in rows])}


# ---------- training ----------
def load_with_traces(pattern):
    """Like train.load but also keeps the raw traces, aligned by index, for example context."""
    from src.graph.build import meta_of, trace_to_graph
    graphs, metas, trs = [], [], []
    for p in sorted(glob.glob(pattern)):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                tr = json.loads(line)
                g = trace_to_graph(tr)
                if g is not None:
                    graphs.append(g)
                    metas.append(meta_of(tr))
                    trs.append(tr)
    return graphs, metas, trs


def train_logits(kind, train, evals, epochs, seed, hidden=64, lr=3e-3):
    """Mirror of train.fit, but returns raw logits [n_actions, 2] for each graph list in `evals` instead of argmax."""
    import torch
    import torch.nn.functional as Fn
    from torch_geometric.loader import DataLoader
    from src.model.gnn import ActionMLP, HeteroGNN
    from src.train import DEVICE, ys
    torch.manual_seed(seed)
    cnt = torch.bincount(ys(train), minlength=2).float().clamp(min=1)
    w = (cnt.sum() / (2 * cnt)).to(DEVICE)
    model = (HeteroGNN(hidden, classes=2) if kind == "gnn" else ActionMLP(hidden, 2)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=32, shuffle=True)
    for _ in range(epochs):
        model.train()
        for b in loader:
            b = b.to(DEVICE)
            opt.zero_grad()
            Fn.cross_entropy(model(b), b["action"].y, weight=w).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return [torch.cat([model(b.to(DEVICE))[:, :2].cpu() for b in DataLoader(g, batch_size=64)]).numpy() if g else np.zeros((0, 2)) for g in evals]


def val_split(tr_i, metas, seed):
    """Validation set for temperature scaling, carved from the training indices: ~20% of task groups, and every remaining
    training trace that shares a task group or an attack with it is dropped, so the validation actions are task- and
    attack-disjoint from what the model fits on (same rule as the test split). Test traces are never touched."""
    from src.train import disjoint
    groups = sorted({metas[i]["group"] for i in tr_i})
    random.Random(1000 + seed).shuffle(groups)
    held = set(groups[: max(1, len(groups) // 5)])
    val = [i for i in tr_i if metas[i]["group"] in held]
    vs = set(val)
    return disjoint([i for i in tr_i if i not in vs], val, metas), val


def softmax_p1(z, T=1.0):
    z = np.asarray(z, float) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(1)


def rows_for(kind, seed, graphs, metas, args):
    """Rows for `kind` (raw softmax) and `kind+TS` (temperature fitted on the held-out validation set). Argmax is
    invariant to T, so predictions are identical and only confidences differ."""
    from src.train import split, ys
    tr_i, te_i = split(metas, args.split, args.holdout, seed)
    fit_i, val_i = val_split(tr_i, metas, seed)
    if not fit_i or not val_i:
        raise RuntimeError("validation split left %d fit / %d val traces; need more tasks" % (len(fit_i), len(val_i)))
    if args.eval == "train":
        te_i = fit_i  # score the very traces the model was fit on (shows how overconfident it is on seen data)
    train, val, test = [graphs[i] for i in fit_i], [graphs[i] for i in val_i], [graphs[i] for i in te_i]
    Zt, Zv = train_logits(kind, train, [test, val], args.epochs, seed, args.hidden)
    yv, y = ys(val).numpy(), ys(test).numpy()
    T = fit_temperature(Zv, yv)
    print("  %s seed %d: T=%.3f  val NLL %.4f -> %.4f  (fit %d, val %d, test %d traces)" % (
        kind, seed, T, nll(Zv, yv), nll(Zv, yv, T), len(fit_i), len(val_i), len(te_i)), flush=True)
    rows, k = [], 0
    for i in te_i:
        m = metas[i]
        for step, tool in enumerate(m["tools"]):
            for name, temp in ((kind, 1.0), (kind + "+TS", T)):
                p1 = float(softmax_p1(Zt[k:k + 1], temp)[0])
                rows.append({"model": name, "seed": seed, "trace_id": "", "step": step, "tool": tool, "true": int(y[k]),
                             "pred": int(p1 >= 0.5), "p_reject": p1, "conf": max(p1, 1 - p1), "temperature": temp,
                             "synthetic": int(bool(m["synthetic"][step])), "benchmark": m["benchmark"],
                             "category": m.get("category"), "style": m.get("style"), "mode": m.get("mode"),
                             "attack_id": m.get("attack_id"), "_trace_idx": i})
            k += 1
    if k != len(Zt):
        raise RuntimeError("action alignment broke: %d rows vs %d predictions" % (k, len(Zt)))
    return rows


# ---------- reporting ----------
def example_context(r, trs):
    tr = trs[r["_trace_idx"]] if trs and r["_trace_idx"] >= 0 else None
    ev = tr["events"][r["step"]] if tr and r["step"] < len(tr["events"]) else {}
    return {"trace_id": tr["trace_id"] if tr else "", "args": json.dumps(ev.get("args"), ensure_ascii=False)[:200],
            "observation": str(ev.get("observation", ""))[:200]}


def pct(x):
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def md_table(head, body):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in body])


def html_table(head, body):
    h = "".join("<th>%s</th>" % html.escape(str(x)) for x in head)
    b = "".join("<tr>" + "".join("<td>%s</td>" % html.escape(str(x)) for x in r) + "</tr>" for r in body)
    return "<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (h, b)


def bar(frac, color):
    return '<div class="bar"><span style="width:%.1f%%;background:%s"></span></div>' % (100 * max(0, min(1, frac)), color)


def build(rows, t, trs, title, topn):
    out = {"threshold": t, "models": {}}
    md, hh = ["# %s" % title, "", "Confident = max softmax >= **%.2f**. Rates are over all evaluated actions (pooled over seeds)." % t, ""], []
    for kind in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == kind]
        S = summarize(rs, t)
        per_seed = {s: summarize([r for r in rs if r["seed"] == s], t) for s in sorted({r["seed"] for r in rs})}
        S["per_seed"] = {str(s): {"rates": v["rates"], "accuracy": v["accuracy"], "ece": v["ece"], "classification": v["classification"]} for s, v in per_seed.items()}
        cw = [r for r in rs if quadrant(r["true"], r["pred"], r["conf"], t) in ("cw_fn", "cw_fp")]
        S["breakdown"] = {k: breakdown(rs, k, t)[:15] for k in ("tool", "category", "style", "mode", "synthetic")}
        out["models"][kind] = S
        c, r_ = S["counts"], S["rates"]
        quad = [["confident-correct", c["cc"], pct(r_["cc"])], ["confident-wrong: missed attack (FN)", c["cw_fn"], pct(r_["cw_fn"])],
                ["confident-wrong: false block (FP)", c["cw_fp"], pct(r_["cw_fp"])],
                ["unsure-correct", c["uc"], pct(r_["uc"])], ["unsure-wrong", c["uw"], pct(r_["uw"])]]
        summ = ["accuracy %s" % pct(S["accuracy"]), "confident fraction %s" % pct(S["confident_fraction"]),
                "accuracy among confident %s" % pct(S["confident_precision"]), "ECE %s" % ("n/a" if S["ece"] is None else "%.3f" % S["ece"])]
        cl = S["classification"]
        clr = [[k, pct(v["precision"]), pct(v["recall"]), pct(v["f1"]), v["support"]] for k, v in cl["per_class"].items()]
        clh = ["class", "precision", "recall", "F1", "support"]
        summ.append("macro-F1 %s" % pct(cl["macro_f1"]))
        sel = [["%d%%" % round(100 * a), pct(b)] for a, b in S["selective"]]
        rel = [["%.2f-%.2f" % (lo, hi), n, "" if mc is None else "%.3f" % mc, "" if ac is None else "%.3f" % ac] for lo, hi, n, mc, ac in S["bins"]]
        ex = sorted(cw, key=lambda r: -r["conf"])[:topn]
        exr = [[r["conf"].__round__(3), NAMES[r["true"]], NAMES[r["pred"]], r["tool"], r["synthetic"], *[example_context(r, trs)[k] for k in ("trace_id", "args", "observation")]] for r in ex]
        exh = ["conf", "true", "pred", "tool", "synthetic", "trace", "args", "observation"]
        md += ["## %s" % kind, "", "; ".join(summ), "", md_table(clh, clr), "", md_table(["group", "n", "share"], quad), "", "### Calibration (reliability bins)", "",
               md_table(["conf bin", "n", "mean conf", "accuracy"], rel), "", "### Selective prediction (keep most confident k%)", "",
               md_table(["coverage", "accuracy"], sel), ""]
        hh += ["<h2>%s</h2><p>%s</p>" % (html.escape(kind), html.escape("; ".join(summ)))]
        hh.append("<div class='quad'>" + "".join(
            "<div><b>%s</b><br>%d (%s)%s</div>" % (html.escape(n), k, pct(f), bar(f, col)) for (n, k, f), col in zip(
                [(q[0], q[1], r_[key]) for q, key in zip(quad, ("cc", "cw_fn", "cw_fp", "uc", "uw"))],
                ["#2a9d5c", "#d1342f", "#e08a1e", "#7fb99a", "#b9a27f"])) + "</div>")
        hh.append("<h3>Precision / recall / F1</h3>" + html_table(clh, clr))
        hh.append("<h3>Reliability</h3>" + html_table(["conf bin", "n", "mean conf", "accuracy"], rel))
        hh.append("<h3>Selective prediction</h3>" + html_table(["coverage", "accuracy"], sel))
        for k, v in S["breakdown"].items():
            body = [[d["value"], d["n"], d["confident_wrong"], pct(d["rate"]), d["cw_fn"], d["cw_fp"]] for d in v]
            md += ["### Confident-wrong by %s" % k, "", md_table([k, "n", "conf-wrong", "rate", "FN", "FP"], body), ""]
            hh.append("<h3>Confident-wrong by %s</h3>" % html.escape(k) + html_table([k, "n", "conf-wrong", "rate", "FN", "FP"], body))
        md += ["### Most confident wrong predictions", "", md_table(exh, [[str(x).replace("|", "/").replace("\n", " ") for x in r] for r in exr]), ""]
        hh.append("<h3>Most confident wrong predictions</h3>" + html_table(exh, exr))
    cmp_head = ["model", "T (mean over seeds)", "ECE", "accuracy", "confident fraction", "confident-wrong FN", "confident-wrong FP"]
    cmp_rows = []
    for kind, S in out["models"].items():
        ts = [r["temperature"] for r in rows if r["model"] == kind and "temperature" in r]
        cmp_rows.append([kind, "%.2f" % (sum(ts) / len(ts)) if ts else "n/a", "n/a" if S["ece"] is None else "%.3f" % S["ece"],
                         pct(S["accuracy"]), pct(S["confident_fraction"]), pct(S["rates"]["cw_fn"]), pct(S["rates"]["cw_fp"])])
    out["comparison"] = cmp_rows
    md[4:4] = ["## Raw vs temperature-scaled", "", "`+TS` = same trained model, logits divided by T fitted on a task- and attack-disjoint validation set "
               "carved from training data (test untouched). Predictions are identical; only confidence changes.", "", md_table(cmp_head, cmp_rows), ""]
    hh.insert(0, "<h2>Raw vs temperature-scaled</h2>" + html_table(cmp_head, cmp_rows))
    css = ("body{font:14px system-ui;margin:0;padding:1rem 1.5rem;max-width:1100px;background:#fff;color:#222}"
           "table{border-collapse:collapse;margin:.5rem 0;display:block;overflow-x:auto}td,th{border:1px solid #ccc;padding:3px 8px;text-align:left}"
           ".quad{display:flex;flex-wrap:wrap;gap:.6rem}.quad>div{border:1px solid #ccc;padding:.5rem;min-width:170px}"
           ".bar{background:#eee;height:8px;margin-top:4px}.bar span{display:block;height:8px}")
    page = "<!doctype html><meta charset=utf-8><title>%s</title><style>%s</style><h1>%s</h1><p>Confident = max softmax &ge; %.2f.</p>%s" % (
        html.escape(title), css, html.escape(title), t, "".join(hh))
    return out, "\n".join(md) + "\n", page


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


BENCHES = ["asb", "injecagent", "agentdojo"]


def out_dir(a, bench, ev):
    return a.outdir or os.path.join(ROOT, "results", "confidence", ("%s_%s" % (bench or "custom", a.split)) + ("_train" if ev == "train" else ""))


def dispatch(a, argv):
    """--bench all / several benches and/or --eval both: one subprocess per (bench, eval), spread over the GPUs, two at a time.
    Jobs whose predictions.csv already exists are skipped (pass --force to retrain)."""
    import subprocess
    import time
    benches = BENCHES if a.bench == "all" else a.bench.split(",")
    evals = ["test", "train"] if a.eval == "both" else [a.eval]
    try:
        import torch
        ngpu = max(1, torch.cuda.device_count())
    except Exception:
        ngpu = 1
    jobs = []
    for b in benches:
        for ev in evals:
            if not a.force and os.path.exists(os.path.join(out_dir(a, b, ev), "predictions.csv")):
                print("skip %s/%s: results exist (use --force)" % (b, ev))
                continue
            cmd = [sys.executable, os.path.abspath(__file__)] + [x for x in argv if x != "--force"]
            cmd = [x for i, x in enumerate(cmd) if not (cmd[i - 1] in ("--bench", "--eval") or x in ("--bench", "--eval"))]
            jobs.append(cmd + ["--bench", b, "--eval", ev])
    running, k = [], 0
    while jobs or running:
        while jobs and len(running) < min(2, ngpu):
            env = dict(os.environ)
            if ngpu > 1:
                env["GNN_DEVICE"] = "cuda:%d" % (k % ngpu)
            j = jobs.pop(0)
            print("launch:", " ".join(j[2:]), "on", env.get("GNN_DEVICE", "default"), flush=True)
            running.append(subprocess.Popen(j, env=env))
            k += 1
        for p in running[:]:
            if p.poll() is not None:
                running.remove(p)
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", help="asb|injecagent|agentdojo|agentpoison (sets --traces to traces/<bench>__<model>.jsonl)")
    ap.add_argument("--model", default="qwen2.5:1.5b", help="trace LLM, used with --bench")
    ap.add_argument("--traces", help="trace glob, overrides --bench")
    ap.add_argument("--split", default="task", choices=["task", "style", "agent", "mode", "category", "benchmark"])
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--models", nargs="+", default=["gnn"], choices=["gnn", "mlp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--eval", default="test", choices=["test", "train", "both"], help="which actions to score: held-out test (default), the training traces, or both")
    ap.add_argument("--force", action="store_true", help="with several jobs: retrain even if predictions.csv exists")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--examples", type=int, default=15)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--report-only", action="store_true", help="rebuild reports from an existing predictions.csv")
    a = ap.parse_args()
    if not a.traces and a.bench and (a.bench == "all" or "," in a.bench or a.eval == "both"):
        return dispatch(a, sys.argv[1:])
    if a.eval == "both":
        ap.error("--eval both needs --bench")
    if not a.traces:
        if not a.bench:
            ap.error("give --bench or --traces")
        a.traces = os.path.join(ROOT, "traces", "%s__%s.jsonl" % (a.bench, a.model.replace(":", "_")))
    tag = a.bench or "custom"
    outdir = out_dir(a, a.bench, a.eval)
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "predictions.csv")
    graphs, metas, trs = load_with_traces(a.traces)
    if not graphs:
        sys.exit("no traces matched %s" % a.traces)
    if a.report_only:
        by_id = {t["trace_id"]: i for i, t in enumerate(trs)}
        rows = []
        for r in csv.DictReader(open(csv_path, encoding="utf-8")):
            r.update({k: int(r[k]) for k in ("seed", "step", "true", "pred", "synthetic")}, p_reject=float(r["p_reject"]), conf=float(r["conf"]), temperature=float(r.get("temperature") or 1.0))
            rows.append(r)
        for r in rows:
            r["_trace_idx"] = by_id.get(r["trace_id"], -1)
    else:
        rows = []
        for kind in a.models:
            for s in a.seeds:
                print("training %s seed %d ..." % (kind, s), flush=True)
                rs = rows_for(kind, s, graphs, metas, a)
                for r in rs:
                    r["trace_id"] = trs[r["_trace_idx"]]["trace_id"]
                rows += rs
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    title = "Confidence analysis: %s, split=%s, eval=%s" % (tag, a.split, a.eval)
    js, md, page = build(rows, a.threshold, trs, title, a.examples)
    write(os.path.join(outdir, "confidence.json"), json.dumps(js, indent=1, sort_keys=True) + "\n")
    write(os.path.join(outdir, "REPORT.md"), md)
    write(os.path.join(outdir, "report.html"), page)
    for kind, S in js["models"].items():
        r = S["rates"]
        cl = S["classification"]
        print("\n== %s (threshold %.2f, %d actions) ==" % (kind, js["threshold"], S["counts"]["n"]))
        print("accuracy %s | macro-F1 %s | ECE %s | confident %s, accuracy among confident %s" % (
            pct(S["accuracy"]), pct(cl["macro_f1"]), "n/a" if S["ece"] is None else "%.3f" % S["ece"],
            pct(S["confident_fraction"]), pct(S["confident_precision"])))
        print("%-9s %9s %8s %7s %8s" % ("class", "precision", "recall", "F1", "support"))
        for k, v in cl["per_class"].items():
            print("%-9s %9s %8s %7s %8d" % (k, pct(v["precision"]), pct(v["recall"]), pct(v["f1"]), v["support"]))
        print("confident-correct %s | confident-wrong: missed attack (FN) %s, false block (FP) %s | unsure-correct %s, unsure-wrong %s" % (
            pct(r["cc"]), pct(r["cw_fn"]), pct(r["cw_fp"]), pct(r["uc"]), pct(r["uw"])))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
