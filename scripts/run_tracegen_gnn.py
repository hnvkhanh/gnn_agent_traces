"""Train and evaluate the approve/reject guard on tracegen traces, with file nodes, and write the report.

    python scripts/run_tracegen_gnn.py                                # all sections, every CUDA GPU in parallel
    python scripts/run_tracegen_gnn.py --devices cuda:0,cuda:1 --seeds 3 --epochs 60
    python scripts/run_tracegen_gnn.py --seeds 1 --epochs 3 --devices cpu --outdir /tmp/x    # smoke
    python scripts/run_tracegen_gnn.py --report-only                  # re-render the report from stored results

Needs attacked traces: `python -m tracegen --attacks` (config.yaml `attacks`). Without them the script stops and says so.

Guards: majority, tool-lookup, mlp (no graph), gnn (docs/GRAPH.md graph), gnn+files (plus file nodes,
src/graph/files.py). Sections: main (val + test, per suite, per agent), ablate (gnn+files), transfer (agent -> agent).
Splits: tracegen's task-level split; injection tasks have their own split, so held-out sides have unseen tasks AND
unseen attacks (src/benchmarks/tracegen.py). The leakage audit runs first and the script refuses to train if a hard
overlap is not 0.

Each (cell, seed) is fingerprinted (training/graph code, options, every train/eval graph and meta). Stored runs with the
same fingerprint are reused, so only new or stale runs train. --no-reuse recomputes everything.

Outputs (under --outdir, default results/tracegen/):
    experiments.json   {cell: {seed: {fp, run, time, seconds}}}
    REPORT.md          tables
    report.html        self-contained page with charts and the same tables
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import multiprocessing as mp
import os
import platform
import queue
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

GUARD_NAME = {"majority": "Majority", "tool-lookup": "Tool lookup", "mlp": "MLP (no graph)", "gnn": "GNN",
              "gnn+files": "GNN + files"}
ABL_NAME = {"files": "file edges", "observation": "observations", "injected": "injected text", "copies": "copies edges",
            "mentions": "mentions edges", "tool_feat": "tool features", "attacker_values": "attacker constants masked"}


# ---------------------------------------------------------------- workers
_W = {}


def _init(device, tg_dir, data_dir, threads):
    os.environ["GNN_DEVICE"] = device
    import torch
    torch.set_num_threads(max(1, threads))
    from src.tracegen_gnn import Data
    _W["data"] = Data(tg_dir, data_dir)


def _job(key, seed, epochs, lr, hidden):
    from src.tracegen_gnn import run_cell
    t0 = time.time()
    return run_cell(_W["data"], key, seed, epochs, lr, hidden), time.time() - t0


def devices_of(spec):
    if spec != "auto":
        return [d.strip() for d in spec.split(",") if d.strip()]
    import torch
    n = torch.cuda.device_count()
    return ["cuda:%d" % i for i in range(n)] or ["cpu"]


def run_parallel(jobs, devices, a):
    """jobs: [(key, seed)] -> {(key, seed): (run, seconds)}. One worker process per device, pulling from one queue."""
    q, out, lock = queue.Queue(), {}, threading.Lock()
    for j in jobs:
        q.put(j)
    threads = max(1, (os.cpu_count() or 2) // len(devices))
    ctx = mp.get_context("spawn")
    errors = []

    def drive(dev):
        with cf.ProcessPoolExecutor(1, mp_context=ctx, initializer=_init,
                                    initargs=(dev, a.tracegen, a.agentdojo, threads)) as ex:
            while True:
                try:
                    key, seed = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    res = ex.submit(_job, key, seed, a.epochs, a.lr, a.hidden).result()
                except Exception as e:  # noqa: BLE001
                    errors.append((key, seed, repr(e)))
                    print("[%s] FAILED %s seed %d: %r" % (dev, key, seed, e), flush=True)
                    continue
                with lock:
                    out[(key, seed)] = res
                    print("[%s] done %-34s seed %d  %.0fs  (%d/%d)" % (dev, key, seed, res[1], len(out), len(jobs)),
                          flush=True)

    ts = [threading.Thread(target=drive, args=(d,)) for d in devices]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errors:
        print("%d runs failed: %s" % (len(errors), errors[:3]), flush=True)
    return out


# ---------------------------------------------------------------- report helpers
def nanmean(v):
    v = [x for x in v if x is not None and x == x]
    return sum(v) / len(v) if v else float("nan")


def nanstd(v):
    m = nanmean(v)
    v = [x for x in v if x is not None and x == x]
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 if v else float("nan")


def runs_of(store, key, seeds):
    c = store.get(key, {})
    return [c[str(s)]["run"] for s in seeds if str(s) in c]


def metric(runs, ev, guard, f):
    return [f(r[ev][guard]) for r in runs if ev in r and guard in r[ev]]


M = dict(f1=lambda r: r["macro_f1"], rej=lambda r: r["recall"]["reject"], appr=lambda r: r["recall"]["approve"],
         asr_u=lambda r: r["asr"]["asr_undefended"], asr_d=lambda r: r["asr"]["asr_defended"],
         bblock=lambda r: r["asr"]["benign_block"], fpr=lambda r: r["asr"]["approve_fpr"])


def pm(v, d=2):
    m, s = nanmean(v), nanstd(v)
    if m != m:
        return "–"
    return ("%.*f" % (d, m)) + (" ± %.*f" % (d, s) if len(v) > 1 else "")


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def guard_rows(runs, ev, guards):
    rows = []
    for g in guards:
        if not metric(runs, ev, g, M["f1"]):
            continue
        rows.append([GUARD_NAME[g], pm(metric(runs, ev, g, M["f1"])), pm(metric(runs, ev, g, M["rej"])),
                     pm(metric(runs, ev, g, M["appr"])),
                     "%s → %s" % (pm(metric(runs, ev, g, M["asr_u"])), pm(metric(runs, ev, g, M["asr_d"]))),
                     pm(metric(runs, ev, g, M["bblock"])), pm(metric(runs, ev, g, M["fpr"]))])
    return rows


GHEAD = ["guard", "macro-F1", "reject recall", "approve recall", "ASR no guard → guard", "benign traces blocked",
         "approve FPR"]


def build_report(a, store, summary, leak, stats, seeds, devices, status):
    from src.tracegen_gnn import ABLATE, GUARDS
    main = runs_of(store, "main", seeds)
    evs = sorted({k for r in main for k in r})
    suites = [e.split("=", 1)[1] for e in evs if e.startswith("test|suite=")]
    agents = [e.split("=", 1)[1] for e in evs if e.startswith("test|agent=")]
    transfer = sorted(k for k in store if k.startswith("transfer/"))
    L = ["# GNN guard on tracegen traces (AgentDojo, with file nodes)", "",
         "Generated %s by `scripts/run_tracegen_gnn.py` · seeds %s · epochs %d · devices %s · git %s." % (
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), list(seeds), a.epochs, ",".join(devices), _git()), "",
         "Per-action approve/reject. A held-out split has unseen user tasks (and all their paraphrases) and unseen "
         "injection tasks. Scores are mean ± std over seeds.", "",
         "## Data", "",
         md_table(["split", "traces", "benign", "attacked", "followed attack", "security true", "actions", "reject",
                   "tasks", "injection tasks"],
                  [[sp, s["traces"], s["benign_traces"], s["attack_traces"], s["attack_followed"], s["security"],
                    s["actions"], s["reject"], s["tasks"], s["injection_tasks"]] for sp, s in summary.items()]), "",
         "Records read: %d; dropped with no tool call: %d; attack runs dropped because the injection task is in another "
         "split: %d." % (stats.get("records", 0), stats.get("no_tool_calls", 0), stats.get("dropped_split_mismatch", 0)),
         ""]
    for ev, title in (("test", "Test"), ("val", "Validation")):
        L += ["## %s" % title, "", md_table(GHEAD, guard_rows(main, ev, GUARDS)), ""]
    if suites:
        L += ["## Test macro-F1 per suite", "",
              md_table(["guard"] + suites, [[GUARD_NAME[g]] + [pm(metric(main, "test|suite=%s" % s, g, M["f1"]))
                                                               for s in suites] for g in GUARDS]), ""]
    if agents:
        L += ["## Test macro-F1 per agent", "",
              md_table(["guard"] + agents, [[GUARD_NAME[g]] + [pm(metric(main, "test|agent=%s" % m, g, M["f1"]))
                                                               for m in agents] for g in GUARDS]), ""]
    base = metric(main, "test", "gnn+files", M["f1"])
    rows = []
    for ab in ABLATE:
        r = runs_of(store, "ablate/%s" % ab, seeds)
        if r:
            v = metric(r, "test", "gnn+files", M["f1"])
            rows.append([ABL_NAME[ab], pm(v), "%+.3f" % (nanmean(v) - nanmean(base)), pm(metric(r, "test", "gnn+files", M["rej"]))])
    if rows:
        L += ["## Ablations (GNN + files, test)", "", "Full model: macro-F1 %s." % pm(base), "",
              md_table(["removed", "macro-F1", "Δ vs full", "reject recall"], rows), ""]
    if transfer:
        L += ["## Cross-agent transfer (test macro-F1)", "",
              md_table(["train → test"] + [GUARD_NAME[g] for g in GUARDS],
                       [[k.split("/", 1)[1]] + [pm(metric(runs_of(store, k, seeds), "test", g, M["f1"])) for g in GUARDS]
                        for k in transfer]), ""]
    L += ["## Leakage audit", "", "Hard overlaps (must be 0): %s." % ("all 0" if leak["ok"] else "**NOT 0**"), "",
          md_table(["overlap", "count"], [[k, v] for k, v in leak["hard"].items()]), "",
          md_table(["check", "value"], [[k, ("%.2f" % v) if isinstance(v, float) else v] for k, v in leak["info"].items()]),
          "", "`attacker_values_*_in_train` counts attacker argument strings of held-out attacks that also occur in "
          "training actions or observations (AgentDojo reuses constants such as the banking attacker IBAN across "
          "injection tasks). The `attacker constants masked` ablation shows how much the guard depends on them.", "",
          "## Provenance", "",
          "Runs reused: %d, trained: %d, failed: %d." % (status.count("reused"), status.count("computed"),
                                                        status.count("failed")),
          "Platform: %s, Python %s." % (platform.platform(), platform.python_version()), ""]
    md = "\n".join(L)
    return md, render_html(a, store, summary, leak, seeds, main, suites, agents, transfer, base, md)


def _git():
    import subprocess
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def bars(items, title):
    """items: [(label, values, cls)] -> horizontal bars with a ± std whisker, 0..1 axis."""
    from report_page import TIP, esc
    W, Lm, rowh, T = 760, 200, 30, 8
    H = T + rowh * len(items) + 26
    x = lambda v: Lm + (W - Lm - 50) * max(0.0, min(1.0, v))
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="%s">' % (W, H, esc(title))]
    for t in (0, .25, .5, .75, 1):
        out.append('<line class="grid" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>' % (x(t), x(t), T, H - 24))
        out.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%.2f</text>' % (x(t), H - 8, t))
    for i, (lab, vals, cls) in enumerate(items):
        m, s = nanmean(vals), nanstd(vals)
        y = T + i * rowh + 6
        out.append('<text class="cat" x="%d" y="%.1f" text-anchor="end" dy="0.32em">%s</text>' % (Lm - 10, y + 9, esc(lab)))
        if m != m:
            continue
        out.append('<rect class="%s" x="%.1f" y="%.1f" width="%.1f" height="18" rx="3" %s/>'
                   % (cls, x(0), y, max(1.0, x(m) - x(0)), TIP % esc("%s: %.3f ± %.3f" % (lab, m, s))))
        if len(vals) > 1:
            out.append('<line class="whisker" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>' % (x(m - s), x(m + s), y + 9, y + 9))
        out.append('<text class="val" x="%.1f" y="%.1f" dy="0.32em">%.2f</text>' % (x(m) + 8, y + 9, m))
    out.append('<line class="axis" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/></svg>' % (x(0), x(0), T, H - 24))
    return "".join(out)


def render_html(a, store, summary, leak, seeds, main, suites, agents, transfer, base, md):
    from report_page import CSS, JS, div_color, esc, heatmap, seq_color, table
    from src.tracegen_gnn import ABLATE, GUARDS
    cls = {"majority": "nog", "tool-lookup": "s4", "mlp": "s3", "gnn": "s2", "gnn+files": "s1"}
    te = lambda g: metric(main, "test", g, M["f1"])
    tot = {k: sum(s[k] for s in summary.values()) for k in ("traces", "attack_traces", "reject", "actions")}
    best = max(GUARDS, key=lambda g: nanmean(te(g)) if te(g) else -1)
    tiles = [("traces", tot["traces"], "%d attacked" % tot["attack_traces"]),
             ("reject actions", tot["reject"], "of %d actions" % tot["actions"]),
             ("GNN + files, test macro-F1", pm(te("gnn+files")), "GNN without files %s" % pm(te("gnn"))),
             ("leakage audit", "clean" if leak["ok"] else "overlap", "hard overlaps all 0" if leak["ok"] else "see table")]
    S = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
         'initial-scale=1"><title>Tracegen GNN guard</title><style>%s</style></head><body><main>' % CSS,
         '<header><div class="eyebrow">tracegen · AgentDojo · %s</div><h1>GNN guard on real agent traces, with file nodes'
         '</h1><p class="lede">Per-action approve/reject on AgentDojo runs from tracegen. Held-out splits have unseen '
         'user tasks (with all their paraphrases) and unseen injection tasks. Best guard on test: <b>%s</b>.</p></header>'
         % (esc(", ".join(agents) or datetime.date.today().isoformat()), esc(GUARD_NAME[best])),
         '<div class="tiles">%s</div>' % "".join('<div class="tile"><div class="k">%s</div><div class="v">%s</div>'
                                                 '<div class="s">%s</div></div>' % (esc(k), esc(v), esc(s))
                                                 for k, v, s in tiles)]
    for ev, title in (("test", "Test"), ("val", "Validation")):
        items = [(GUARD_NAME[g], metric(main, ev, g, M["f1"]), cls[g]) for g in GUARDS]
        S.append('<section><h2>%s macro-F1</h2><div class="panel">%s%s</div></section>'
                 % (title, bars(items, "%s macro-F1 by guard" % title),
                    table(GHEAD, [[esc(c) for c in r] for r in guard_rows(main, ev, GUARDS)])))
    for kind, cols in (("suite", suites), ("agent", agents)):
        if cols:
            def cell(r, c, kind=kind):
                v = metric(main, "test|%s=%s" % (kind, c), [g for g in GUARDS if GUARD_NAME[g] == r][0], M["f1"])
                if not v:
                    return None
                m = nanmean(v)
                fill, ink = seq_color(m)
                return fill, ink, "%.2f" % m, "%s · %s: macro-F1 %s" % (r, c, pm(v))
            S.append('<section><h2>Test macro-F1 per %s</h2><div class="panel">%s</div></section>'
                     % (kind, heatmap([GUARD_NAME[g] for g in GUARDS], cols, cell, "macro-F1 per %s" % kind)))
    rows = [(ab, runs_of(store, "ablate/%s" % ab, seeds)) for ab in ABLATE]
    rows = [(ab, r) for ab, r in rows if r]
    if rows:
        bm = nanmean(base)

        def acell(r, c):
            ab = [k for k, v in ABL_NAME.items() if v == r][0]
            v = metric(dict(rows)[ab], "test", "gnn+files", M["f1"])
            d = nanmean(v) - bm
            fill, ink = div_color(d, 0.2)
            return fill, ink, "%+.3f" % d, "without %s: macro-F1 %s (full %.3f)" % (r, pm(v), bm)
        S.append('<section><h2>Ablations</h2><p>Change in test macro-F1 of GNN + files when one input is removed '
                 '(full model %s). Red means the input helps.</p><div class="panel">%s</div></section>'
                 % (esc(pm(base)), heatmap([ABL_NAME[ab] for ab, _ in rows], ["Δ macro-F1"], acell, "ablations")))
    if transfer:
        def tcell(r, c):
            g = [k for k, v in GUARD_NAME.items() if v == c][0]
            v = metric(runs_of(store, "transfer/" + r, seeds), "test", g, M["f1"])
            if not v:
                return None
            fill, ink = seq_color(nanmean(v))
            return fill, ink, "%.2f" % nanmean(v), "%s, %s: %s" % (r, c, pm(v))
        S.append('<section><h2>Cross-agent transfer</h2><div class="panel">%s</div></section>'
                 % heatmap([k.split("/", 1)[1] for k in transfer], [GUARD_NAME[g] for g in GUARDS], tcell, "transfer"))
    S.append('<section><h2>Data and leakage</h2><div class="panel">%s%s%s</div></section>' % (
        table(["split", "traces", "attacked", "followed attack", "actions", "reject", "tasks", "injection tasks"],
              [[sp, s["traces"], s["attack_traces"], s["attack_followed"], s["actions"], s["reject"], s["tasks"],
                s["injection_tasks"]] for sp, s in summary.items()]),
        table(["overlap (must be 0)", "count"],
              [[esc(k), '<span class="pill %s">%d</span>' % ("ok" if v == 0 else "bad", v)] for k, v in leak["hard"].items()]),
        table(["check", "value"], [[esc(k), esc(("%.2f" % v) if isinstance(v, float) else v)]
                                   for k, v in leak["info"].items()])))
    S.append('<details><summary>Markdown report</summary><pre style="white-space:pre-wrap">%s</pre></details>' % esc(md))
    S.append('<footer>scripts/run_tracegen_gnn.py · %s</footer></main><div id="tip" hidden></div><script>%s</script>'
             '</body></html>' % (esc(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")), JS))
    return "".join(S)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracegen", default=os.path.join(ROOT, "data", "tracegen"), help="tracegen output dir")
    ap.add_argument("--agentdojo", default=os.path.join(ROOT, "data", "agentdojo"), help="static AgentDojo export")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "tracegen"))
    ap.add_argument("--sections", default="main,ablate,transfer")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--devices", default="auto", help="auto | cpu | cuda:0,cuda:1 | ...")
    ap.add_argument("--no-reuse", dest="reuse", action="store_false")
    ap.add_argument("--report-only", action="store_true", help="only re-render the report from stored results")
    a = ap.parse_args()

    from src.tracegen_gnn import Data, audit, cells, code_hash, fingerprint
    data = Data(a.tracegen, a.agentdojo)
    summary, leak = data.summary(), audit(data)
    print("traces:", json.dumps({k: (v["traces"], v["attack_traces"], v["reject"]) for k, v in summary.items()}),
          "(traces, attacked, reject actions)")
    why = data.ready()
    if why:
        print("STOP:", why)
        sys.exit(2)
    if not leak["ok"]:
        print("STOP: leakage audit failed:", json.dumps(leak["hard"]), json.dumps(leak["examples"]))
        sys.exit(3)

    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, "experiments.json")
    store = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    seeds = list(range(a.seeds))
    keys = cells(data, a.sections.split(","))
    code = code_hash()
    fps = {(k, s): fingerprint(data, k, s, a.epochs, a.lr, a.hidden, code) for k in keys for s in seeds}
    status, jobs = [], []
    for (k, s), fp in fps.items():
        old = store.get(k, {}).get(str(s))
        if a.reuse and old and old.get("fp") == fp:
            status.append("reused")
        elif not a.report_only:
            jobs.append((k, s))
    devices = devices_of(a.devices)
    print("%d runs: %d reused, %d to train on %s" % (len(fps), status.count("reused"), len(jobs), ",".join(devices)),
          flush=True)
    if jobs:
        res = run_parallel(jobs, devices, a)
        for (k, s) in jobs:
            if (k, s) not in res:
                status.append("failed")
                continue
            run, sec = res[(k, s)]
            store.setdefault(k, {})[str(s)] = dict(fp=fps[(k, s)], run=run, seconds=round(sec, 1),
                                                    time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            status.append("computed")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(store, f, indent=1, sort_keys=True)
    # only report runs that match the current inputs
    cur = {k: {str(s): store[k][str(s)] for s in seeds if str(s) in store.get(k, {})
               and store[k][str(s)].get("fp") == fps[(k, s)]} for k in keys}
    md, page = build_report(a, cur, summary, leak, data.stats, seeds, devices, status)
    for name, text in (("REPORT.md", md), ("report.html", page)):
        with open(os.path.join(a.outdir, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print("wrote", os.path.join(a.outdir, "REPORT.md"), "and report.html")


if __name__ == "__main__":
    main()
