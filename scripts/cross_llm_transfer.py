#!/usr/bin/env python3
"""Does a guard trained on the traces of one LLM generalise to the traces of another LLM?

For every source LLM a guard (GNN, plus MLP / tool-lookup / majority baselines) is trained on that LLM's traces and scored on
the held-out scenarios of EVERY LLM. The diagonal (train LLM = test LLM) is the in-LLM baseline; the off-diagonal cells are
cross-LLM transfer; "leave-one-LLM-out" trains on all the other LLMs. Scopes: all benchmarks pooled, and each benchmark alone.
Leakage and confound controls are described in src/crossllm.py and listed in the report ("Leakage controls").

  python scripts/cross_llm_transfer.py                          # 3 seeds, 60 epochs, every GPU in parallel (one worker per GPU)
  python scripts/cross_llm_transfer.py --seeds 5 --epochs 80
  python scripts/cross_llm_transfer.py --devices cuda:0,cuda:1  # explicit devices; --devices cpu for CPU
  python scripts/cross_llm_transfer.py --llms qwen2.5:7b,llama3.1:8b,gemma2:9b --benches injecagent,agentdojo
  python scripts/cross_llm_transfer.py --report-only            # rebuild the report from stored cells (no training)
  python scripts/cross_llm_transfer.py --seeds 1 --epochs 2 --benches agentdojo --outdir /tmp/xllm_smoke   # quick smoke run

Result cells are fingerprinted (src/cache.py): cells whose training code, seeds, epochs and train/test graphs are unchanged are
reused, so re-running only trains what is new or stale (--no-reuse recomputes everything). The script never calls an LLM.

Outputs in --outdir (default results/crossllm):  REPORT.md, report.html, summary.json, experiments_crossllm.json, fingerprints_crossllm.json
"""
import argparse
import glob
import html
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

DEFAULT_BENCHES = ["injecagent", "agentdojo"]
SHOW = ("gnn", "mlp", "tool-lookup", "majority")


# ---------------------------------------------------------------- data
def load_traces(benches, traces_dir):
    """-> data[llm][bench] = (graphs, metas), behaviour[(llm, bench)] = stats. Only LLMs with traces in EVERY bench are kept."""
    from src.graph.build import meta_of, trace_to_graph
    data, beh = defaultdict(dict), {}
    for b in benches:
        for p in sorted(glob.glob(os.path.join(traces_dir, "%s__*.jsonl" % b))):
            for line in open(p, encoding="utf-8"):
                if not line.strip():
                    continue
                tr = json.loads(line)
                g = trace_to_graph(tr)
                if g is None:
                    continue
                m = meta_of(tr)
                m["trace_id"] = tr["trace_id"]
                m["bench"] = b
                m["group"] = b + "|" + m["group"]  # task groups of different benchmarks never collide
                if m["attack_id"] is not None:
                    m["attack_id"] = b + "|" + str(m["attack_id"])
                llm = tr["model"]
                gs, ms = data[llm].setdefault(b, ([], []))
                gs.append(g)
                ms.append(m)
                s = beh.setdefault((llm, b), dict(traces=0, steps=0, attack=0, followed=0, reject=0, synthetic=0, invalid=0))
                s["traces"] += 1
                s["steps"] += len(tr["events"])
                s["invalid"] += tr.get("n_invalid", 0)
                s["reject"] += sum(e["label"] == "reject" for e in tr["events"])
                s["synthetic"] += sum(bool(e["synthetic"]) for e in tr["events"])
                if tr["attack"]:
                    s["attack"] += 1
                    s["followed"] += any(e["label"] == "reject" and not e["synthetic"] for e in tr["events"])
    return {l: d for l, d in data.items() if all(b in d for b in benches)}, beh


def scope_data(data, llms, scope, benches):
    bs = benches if scope == "pooled" else [scope]
    out = {}
    for l in llms:
        gs, ms = [], []
        for b in bs:
            gs += data[l][b][0]
            ms += data[l][b][1]
        out[l] = (gs, ms)
    return out


def scopes_for(benches):
    return ["pooled"] + list(benches) if len(benches) > 1 else list(benches)


# ---------------------------------------------------------------- cells
def plan(sd_by_scope, llms, seeds):
    """-> list of (key, scope, kind, name, per-seed parts, per-seed targets, per-seed audit)."""
    from src import crossllm as X
    cells = []
    for scope, sd in sd_by_scope.items():
        metas = {l: sd[l][1] for l in llms}
        splits = [X.make_split(metas, s) for s in seeds]
        sets = [X.train_sets(sp, s) for sp, s in zip(splits, seeds)]
        audits = [X.audit(sp, st, metas) for sp, st in zip(splits, sets)]
        for k in sets[0]:
            kind, name = k.split(":", 1)
            targets = llms if kind == "single" else [name]
            parts, tg = [], []
            for sp, st in zip(splits, sets):
                tr_g = [sd[l][0][i] for l, idx in st[k] for i in idx]
                tr_m = [sd[l][1][i] for l, idx in st[k] for i in idx]
                te_g = [sd[t][0][i] for t in targets for i in sp["test"][t]]
                te_m = [sd[t][1][i] for t in targets for i in sp["test"][t]]
                parts.append((tr_g, tr_m, te_g, te_m))
                tg.append({t: ([sd[t][0][i] for i in sp["test"][t]], [sd[t][1][i] for i in sp["test"][t]]) for t in targets})
            cells.append(dict(key="xllm/%s/%s" % (scope, k), scope=scope, kind=kind, name=name, parts=parts, targets=tg,
                              audit=[a[k] for a in audits]))
    return cells


def compute_cell(c, seeds, epochs):
    from src import crossllm as X
    return [dict(targets=X.predict_many(p[0], p[1], tg, epochs=epochs, seed=s),
                 train=dict(n=len(p[0]), attacks=sum(m["attack"] for m in p[1])))
            for s, p, tg in zip(seeds, c["parts"], c["targets"])]


def run_cells(cells, args, cache):
    from src import crossllm as X
    seeds = list(range(args.seeds))
    R = {}
    for c in cells:
        R[c["key"]] = cache.get(c["key"], c["parts"], lambda c=c: compute_cell(c, seeds, args.epochs),
                                recipe=X.recipe_hash(), models=list(X.MODELS))
    return R


# ---------------------------------------------------------------- aggregation
def val(run, tgt, model, metric):
    r = run["targets"][tgt][model]
    if metric == "reject_recall":
        return r["recall"]["reject"]
    if metric in r["asr"]:
        return r["asr"][metric]
    return r[metric]


def per_seed(runs, tgt, model, metric):
    return np.array([val(r, tgt, model, metric) for r in runs], dtype=float)


def ms(v):
    v = np.asarray(v, dtype=float)
    return (float(np.nanmean(v)), float(np.nanstd(v))) if v.size and not np.all(np.isnan(v)) else (float("nan"), 0.0)


def fmt(v, d=2):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else "%.*f" % (d, v)


def pm(t, d=2):
    return "%s±%s" % (fmt(t[0], d), fmt(t[1], d))


def transfer_summary(R, scope, llms, model, metric="macro_f1"):
    """Per source LLM: in-LLM (diagonal), cross-LLM mean over the other targets, paired gap, LOLO score of the source as target."""
    rows = []
    for s in llms:
        runs = R["xllm/%s/single:%s" % (scope, s)]
        diag = per_seed(runs, s, model, metric)
        off = np.array([[val(r, t, model, metric) for t in llms if t != s] for r in runs], dtype=float)
        offm = np.nanmean(off, 1)
        lruns = R["xllm/%s/lolo:%s" % (scope, s)]
        rows.append(dict(llm=s, diag=ms(diag), cross=ms(offm), gap=ms(diag - offm), worst=ms(np.nanmin(off, 1)),
                         lolo=ms(per_seed(lruns, s, model, metric))))
    return rows


def verdict(gap):
    if np.isnan(gap):
        return "n/a"
    return "transfers" if gap <= 0.03 else "partial" if gap <= 0.10 else "weak"


# ---------------------------------------------------------------- report
def matrix_md(R, scope, llms, model, metric):
    rows = []
    for s in llms:
        runs = R["xllm/%s/single:%s" % (scope, s)]
        rows.append([s] + [pm(ms(per_seed(runs, t, model, metric))) + (" ◆" if t == s else "") for t in llms])
    return md_table(["train ↓ / test →"] + llms, rows)


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def label_share(R, scope, llms):
    """Reject share of each target's test actions (what the majority/F1 floor depends on), mean over seeds."""
    runs = R["xllm/%s/single:%s" % (scope, llms[0])]
    return {t: float(np.mean([r["targets"][t]["_n"]["reject"] / max(r["targets"][t]["_n"]["actions"], 1) for r in runs])) for t in llms}


def build_markdown(R, cells, llms, benches, args, beh):
    scopes = scopes_for(benches)
    L = ["# Cross-LLM transfer of the trace guard", "",
         "Question: a guard trained on the traces of one LLM, does it still work on the traces of another LLM?",
         "", "- LLMs: %s" % ", ".join(llms), "- benchmarks: %s" % ", ".join(benches),
         "- seeds: %d (each seed draws different held-out task groups), epochs: %d" % (args.seeds, args.epochs),
         "- every number is mean±std over seeds; ◆ marks the diagonal (train LLM = test LLM, the in-LLM baseline)",
         "- metric: macro-F1 over approve/reject actions of the held-out scenarios", ""]
    pooled = scopes[0]
    L += ["## Headline (%s)" % pooled, "",
          "In-LLM = trained and tested on the same LLM; cross-LLM = mean over the other LLMs; gap = in-LLM − cross-LLM (paired per seed). "
          "Verdict is a rule of thumb: gap ≤ 0.03 transfers, ≤ 0.10 partial, otherwise weak.", ""]
    for model in ("gnn", "tool-lookup"):
        rows = [[r["llm"], pm(r["diag"]), pm(r["cross"]), pm(r["gap"]), pm(r["worst"]), pm(r["lolo"]), verdict(r["gap"][0])]
                for r in transfer_summary(R, pooled, llms, model)]
        L += ["**%s**" % model, "", md_table(["train LLM", "in-LLM", "cross-LLM mean", "gap", "worst target", "leave-one-LLM-out → this LLM", "verdict"], rows), ""]
    L += ["## Macro-F1 matrices (GNN)", ""]
    for sc in scopes:
        L += ["### %s" % sc, "", matrix_md(R, sc, llms, "gnn", "macro_f1"), ""]
    L += ["## Same matrices, real LLM actions only (appended synthetic attacker steps removed)", "",
          "The synthetic attacker step is appended identically to every LLM's trace, so it can make transfer look easier than it is.", "",
          matrix_md(R, pooled, llms, "gnn|real", "macro_f1"), ""]
    L += ["## Is the GNN better than simpler guards across LLMs? (%s, mean over all train/test pairs)" % pooled, ""]
    rows = []
    for model in SHOW:
        d, o = [], []
        for s in llms:
            runs = R["xllm/%s/single:%s" % (pooled, s)]
            for t in llms:
                (d if t == s else o).append(per_seed(runs, t, model, "macro_f1"))
        rows.append([model, pm(ms(np.mean(d, 0))), pm(ms(np.mean(o, 0)))])
    L += [md_table(["model", "in-LLM macro-F1", "cross-LLM macro-F1"], rows), ""]
    L += ["## Attack success rate and reject recall (GNN, %s)" % pooled, "",
          "ASR (no guard → guard): share of attack traces where an attacker action goes through; lower with the guard is better.", ""]
    rows = []
    for s in llms:
        runs = R["xllm/%s/single:%s" % (pooled, s)]
        r_ = lambda t_set, m: pm(ms(np.mean([per_seed(runs, t, "gnn", m) for t in t_set], 0)))
        cross = [t for t in llms if t != s]
        rows.append([s, r_([s], "asr_undefended") + " → " + r_([s], "asr_defended"), r_(cross, "asr_undefended") + " → " + r_(cross, "asr_defended"),
                     r_([s], "reject_recall"), r_(cross, "reject_recall"), r_(cross, "benign_block")])
    L += [md_table(["train LLM", "ASR in-LLM", "ASR cross-LLM", "reject recall in-LLM", "reject recall cross-LLM", "benign traces blocked (cross)"], rows), ""]
    L += ["## Per-benchmark in-LLM vs cross-LLM (GNN macro-F1)", ""]
    rows = []
    for sc in scopes[1:]:
        for r in transfer_summary(R, sc, llms, "gnn"):
            rows.append([sc, r["llm"], pm(r["diag"]), pm(r["cross"]), pm(r["gap"]), pm(r["lolo"]), verdict(r["gap"][0])])
    L += [md_table(["benchmark", "train LLM", "in-LLM", "cross-LLM", "gap", "leave-one-out", "verdict"], rows) if rows else "(single scope)", ""]
    L += ["## What differs between the LLMs", "",
          "Share of rejects in the test actions, and how each LLM behaved on the traces it produced (%s). "
          "If an LLM's behaviour differs a lot (steps, how often it follows the attack), a transfer gap is expected." % pooled, ""]
    sh = label_share(R, pooled, llms)
    rows = []
    for l in llms:
        t = defaultdict(float)
        for b in benches:
            for k, v in beh[(l, b)].items():
                t[k] += v
        rows.append([l, int(t["traces"]), "%.2f" % (t["steps"] / t["traces"]), "%.2f" % (t["followed"] / max(t["attack"], 1)),
                     "%.3f" % (t["invalid"] / max(t["invalid"] + t["steps"], 1)), "%.2f" % (t["reject"] / t["steps"]), "%.2f" % sh[l]])
    L += [md_table(["LLM", "traces", "avg steps", "follows attack (no guard)", "invalid-reply rate", "reject share (all)", "reject share (test)"], rows), ""]
    L += ["## Leakage controls", ""]
    rows = []
    for sc in scopes:
        for c in [c for c in cells if c["scope"] == sc]:
            a = c["audit"]
            rows.append([sc, c["kind"] + ":" + c["name"], "%d" % np.mean([x["n"] for x in a]), min(x["attacks"] for x in a),
                         max(x["groups"] for x in a), max(x["attack_ids"] for x in a), max(x["trace_ids"] for x in a)])
    L += ["Over all seeds. Overlap columns must be 0; min attack traces in train must be > 0.", "",
          md_table(["scope", "training set", "train traces (mean)", "min attack traces in train", "task groups shared with test",
                    "attack ids shared with test", "trace ids shared with test"], rows), "",
          "- Same held-out task groups for every LLM: the test columns are identical scenarios.",
          "- Training excludes every attack present in ANY LLM's test traces, so the training set does not depend on the target.",
          "- Every source gets the same number of training traces; leave-one-out uses the same total, split equally over the other LLMs.",
          "- The LLM name is not a graph feature (tests/test_crossllm.py).", ""]
    L += ["## Reading the results", "",
          "- Small gap and cross-LLM macro-F1 well above tool-lookup and majority: the guard learned the attack signal rather than one LLM's style.",
          "- Large gap on a row: that LLM's traces look different (see behaviour table); a column with low scores for every row: that LLM is harder to guard.",
          "- Leave-one-LLM-out near in-LLM: pooling several LLMs is enough to cover an unseen one.",
          "- Only %d seeds and one held-out split per seed: differences smaller than the ± are noise." % args.seeds, ""]
    return "\n".join(L)


CSS = """
:root{--bg:#fbfbfa;--fg:#1f2328;--mute:#656d76;--line:#d8dee4;--card:#fff;--accent:#2a6fdb;--gap:#c2570c}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14161a;--fg:#e6e8eb;--mute:#9aa3ad;--line:#2d333b;--card:#1b1f24;--accent:#5b9cf5;--gap:#e8893f}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:16px}
h1{font-size:22px}h2{font-size:17px;margin-top:32px}h3{font-size:14px;margin:18px 0 6px}
p,li{color:var(--mute)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow-x:auto}
table{border-collapse:separate;border-spacing:2px;font-variant-numeric:tabular-nums;font-size:13px}
th{font-weight:500;color:var(--mute);text-align:left;padding:4px 8px}
td{padding:6px 10px;border-radius:4px;text-align:center;min-width:74px}
td.l{text-align:left;color:var(--fg)}
td.diag{outline:2px solid var(--fg);outline-offset:-2px}
.bars div.row{display:grid;grid-template-columns:130px 1fr 150px;gap:8px;align-items:center;margin:4px 0}
.bar{height:10px;border-radius:0 4px 4px 0;margin:2px 0}
.b1{background:var(--accent)}.b2{background:var(--gap)}
.lg span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px}
.v{font-size:12px;color:var(--fg)}
.ok{color:#1a7f37}.mid{color:#9a6700}.bad{color:#cf222e}
"""


def heat(v):
    return "color-mix(in srgb,var(--accent) %d%%,var(--card))" % int(max(0, min(1, v if v == v else 0)) * 55)


def html_matrix(R, scope, llms, model="gnn"):
    h = ["<table><tr><th>train ↓ / test →</th>%s</tr>" % "".join("<th>%s</th>" % html.escape(t) for t in llms)]
    for s in llms:
        runs = R["xllm/%s/single:%s" % (scope, s)]
        h.append("<tr><td class=l>%s</td>" % html.escape(s))
        for t in llms:
            m = ms(per_seed(runs, t, model, "macro_f1"))
            h.append("<td class='%s' style='background:%s' title='%s → %s: %s'>%s</td>" % ("diag" if s == t else "", heat(m[0]), s, t, pm(m, 3), fmt(m[0])))
        h.append("</tr>")
    return "".join(h) + "</table>"


def build_html(R, cells, llms, benches, args, beh, md):
    pooled = scopes_for(benches)[0]
    S = transfer_summary(R, pooled, llms, "gnn")
    B = transfer_summary(R, pooled, llms, "tool-lookup")
    p = ["<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Cross-LLM transfer</title><style>%s</style><main>" % CSS,
         "<h1>Does a guard trained on one LLM work on another?</h1>",
         "<p>%d LLMs · %s · %d seeds · %d epochs. Macro-F1 on held-out task groups; the diagonal (outlined) is in-LLM.</p>" % (len(llms), ", ".join(benches), args.seeds, args.epochs),
         "<h2>In-LLM vs cross-LLM (GNN, %s)</h2><div class='card bars'>" % pooled,
         "<div class=lg><span class=b1></span>in-LLM<span class=b2></span>cross-LLM mean<span style='background:var(--mute)'></span>tool-lookup cross-LLM</div>"]
    for r, b in zip(S, B):
        cls = {"transfers": "ok", "partial": "mid", "weak": "bad"}.get(verdict(r["gap"][0]), "")
        p.append("<div class=row><div>%s</div><div><div class='bar b1' style='width:%d%%'></div><div class='bar b2' style='width:%d%%'></div>"
                 "<div class=bar style='background:var(--mute);width:%d%%'></div></div>"
                 "<div class=v>%s → %s · gap %s · <b class=%s>%s</b></div></div>"
                 % (html.escape(r["llm"]), 100 * (r["diag"][0] if r["diag"][0] == r["diag"][0] else 0), 100 * (r["cross"][0] if r["cross"][0] == r["cross"][0] else 0),
                    100 * (b["cross"][0] if b["cross"][0] == b["cross"][0] else 0), fmt(r["diag"][0]), fmt(r["cross"][0]), fmt(r["gap"][0]), cls, verdict(r["gap"][0])))
    p.append("</div><h2>GNN macro-F1, train → test</h2>")
    for sc in scopes_for(benches):
        p.append("<h3>%s</h3><div class=card>%s</div>" % (html.escape(sc), html_matrix(R, sc, llms)))
    p.append("<h3>%s, real LLM actions only</h3><div class=card>%s</div>" % (pooled, html_matrix(R, pooled, llms, "gnn|real")))
    p.append("<h2>Full report</h2><div class=card><pre style='white-space:pre-wrap;font-size:12px;color:var(--fg)'>%s</pre></div></main>" % html.escape(md))
    return "".join(p)


def write_outputs(R, cells, llms, benches, args, beh):
    md = build_markdown(R, cells, llms, benches, args, beh)
    os.makedirs(args.outdir, exist_ok=True)
    w = lambda n, s: open(os.path.join(args.outdir, n), "w", encoding="utf-8", newline="\n").write(s)
    w("REPORT.md", md + "\n")
    w("report.html", build_html(R, cells, llms, benches, args, beh, md))
    pooled = scopes_for(benches)[0]
    summ = {m: {r["llm"]: {k: v for k, v in r.items() if k != "llm"} for r in transfer_summary(R, pooled, llms, m)} for m in ("gnn", "tool-lookup")}
    w("summary.json", json.dumps(summ, indent=1))
    print(md)
    print("\nwrote %s/{REPORT.md,report.html,summary.json}" % os.path.relpath(args.outdir, ROOT))


# ---------------------------------------------------------------- driver
def devices(spec):
    if spec != "auto":
        return [d for d in spec.split(",") if d]
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return ["cuda:%d" % i for i in range(n)] or ["cpu"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--benches", default=",".join(DEFAULT_BENCHES))
    ap.add_argument("--llms", default="auto", help="comma list, or auto = every LLM with traces in all benchmarks")
    ap.add_argument("--traces", default=os.path.join(ROOT, "traces"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "crossllm"))
    ap.add_argument("--devices", default="auto", help="auto | cpu | cuda:0,cuda:1 (one worker process per device)")
    ap.add_argument("--no-reuse", dest="reuse", action="store_false")
    ap.add_argument("--report-only", action="store_true", help="only rebuild the report; fail if a cell is missing")
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--nworkers", type=int, default=1, help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.outdir = os.path.abspath(a.outdir)
    benches = [b for b in a.benches.split(",") if b]
    devs = devices(a.devices)

    # driver with several devices: one worker process per device, each computes a shard of the cells
    if a.worker is None and len(devs) > 1 and not a.report_only:
        procs = []
        for i, d in enumerate(devs):
            cmd = [sys.executable, os.path.abspath(__file__)] + [x for x in sys.argv[1:]] + ["--worker", str(i), "--nworkers", str(len(devs))]
            print("worker %d on %s" % (i, d), flush=True)
            procs.append(subprocess.Popen(cmd, env=dict(os.environ, GNN_DEVICE=d)))
        bad = [p.returncode for p in procs if p.wait() != 0]
        if bad:
            print("a worker failed (%s); the missing cells are computed here" % bad, flush=True)
    elif a.worker is None and len(devs) == 1 and not os.environ.get("GNN_DEVICE") and devs[0] != "cpu":
        os.environ["GNN_DEVICE"] = devs[0]
    elif a.worker is None and devs == ["cpu"]:
        os.environ["GNN_DEVICE"] = "cpu"

    from src.cache import ResultCache
    data, beh = load_traces(benches, a.traces)
    llms = sorted(data) if a.llms == "auto" else a.llms.split(",")
    missing = [l for l in llms if l not in data]
    if missing or len(llms) < 2:
        sys.exit("need >= 2 LLMs with traces in every benchmark %s; have %s, missing %s" % (benches, sorted(data), missing))
    print("LLMs:", ", ".join(llms), "| benches:", ", ".join(benches), flush=True)
    seeds = list(range(a.seeds))
    sd_by_scope = {sc: scope_data(data, llms, sc, benches) for sc in scopes_for(benches)}
    cells = plan(sd_by_scope, llms, seeds)

    if a.worker is not None:  # compute my shard only, write shard files, merged by the driver
        mine = [c for k, c in enumerate(cells) if k % a.nworkers == a.worker]
        cache = ResultCache(a.outdir, "crossllm_w%d" % a.worker, seeds, a.epochs, "cross-llm", reuse=a.reuse)
        R = run_cells(mine, a, cache)
        cache.save()
        with open(os.path.join(a.outdir, "experiments_crossllm_w%d.json" % a.worker), "w", encoding="utf-8", newline="\n") as f:
            json.dump(R, f)
        return

    cache = ResultCache(a.outdir, "crossllm", seeds, a.epochs, "cross-llm", reuse=a.reuse or len(devs) > 1)
    if a.report_only:
        stale = [c["key"] for c in cells if not cache.cands.get(c["key"])]
        if stale:
            sys.exit("--report-only: no stored result for %d cells, e.g. %s" % (len(stale), stale[:2]))
    R = run_cells(cells, a, cache)
    cache.save()
    with open(os.path.join(a.outdir, "experiments_crossllm.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(R, f)
    for p in glob.glob(os.path.join(a.outdir, "*_crossllm_w*.json")):
        os.remove(p)
    write_outputs(R, cells, llms, benches, a, beh)


if __name__ == "__main__":
    main()
