"""Train and evaluate the approve/reject guard on tracegen traces, with file nodes, and write the report.

    python scripts/run_tracegen_gnn.py                                # all sections, every CUDA GPU in parallel
    python scripts/run_tracegen_gnn.py --devices cuda:0,cuda:1 --seeds 3 --epochs 60
    python scripts/run_tracegen_gnn.py --seeds 1 --epochs 3 --devices cpu --outdir /tmp/x    # smoke
    python scripts/run_tracegen_gnn.py --sections cv --folds 5        # only the cross-validation
    python scripts/run_tracegen_gnn.py --report-only                  # re-render the report from stored results

Needs attacked traces: `python -m tracegen --attacks` (config.yaml `attacks`). Without them the script stops and says so.

Guards: majority, tool-lookup, mlp (no graph), gnn (docs/GRAPH.md graph), gnn+files (plus file nodes,
src/graph/files.py), transformer (attention over the gnn+files graph: same nodes, features, edges as attention bias,
causal mask; src/model/transformer.py). Sections: main (val + test, per suite, per agent), cv, ablate (gnn+files), transfer (agent -> agent).
cv: cross-validation over --folds task folds x --folds injection-task folds (src/tracegen_gnn.py). Every run is tested
once, by a guard that saw neither its user task nor its injection task, and the cells are pooled per seed. This is the
headline number: the fixed test split holds only a few followed attacks.
Splits: tracegen's task-level split; injection tasks have their own split, so held-out sides have unseen tasks AND
unseen attacks (src/benchmarks/tracegen.py). The leakage audit runs first and the script refuses to train if a hard
overlap is not 0.

Model selection: per guard, the epoch with the best validation AUPRC and the reject threshold with the best validation
reject F1 (src/tracegen_gnn.py). Scores: AUPRC and reject recall at 1% approve FPR with bootstrap intervals over traces
(src/rank_metrics.py), then the thresholded numbers. A set with no reject action shows "–", not a macro-F1 of 0.50.

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
              "gnn+files": "GNN + files", "transformer": "Transformer"}
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


def _hms(sec):
    sec = int(max(0, sec))
    return "%d:%02d:%02d" % (sec // 3600, sec % 3600 // 60, sec % 60)


def run_parallel(jobs, devices, a, weight=None, on_done=None, progress_path=None):
    """jobs: [(key, seed)] -> {(key, seed): (run, seconds)}. One worker process per device, pulling from one queue.

    Progress: a line per started / finished run with elapsed time and ETA, and (progress_path) a JSON file rewritten
    after every event. The ETA is elapsed wall time x remaining work / finished work, where a run's work is
    weight[(key, seed)] (default 1), e.g. its number of training traces. on_done(key, seed, res) runs under the lock
    after each finished run (used to save results incrementally)."""
    q, out, lock = queue.Queue(), {}, threading.Lock()
    for j in jobs:
        q.put(j)
    threads = max(1, (os.cpu_count() or 2) // len(devices))
    ctx = mp.get_context("spawn")
    errors = []
    w = {j: (weight or {}).get(j, 1) for j in jobs}
    total_w, t0 = float(sum(w.values())) or 1.0, time.time()
    running = {}

    def report(event=None):
        done_w = sum(w[j] for j in out) + sum(w[(k, s)] for k, s, _ in errors)
        el = time.time() - t0
        eta = el * (total_w - done_w) / done_w if done_w else None
        info = dict(started=datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
                    updated=datetime.datetime.now().isoformat(timespec="seconds"), event=event,
                    done=len(out), failed=len(errors), total=len(jobs), work_done_pct=round(100 * done_w / total_w, 1),
                    elapsed=_hms(el), eta=_hms(eta) if eta is not None else None,
                    expected_finish=(datetime.datetime.now() + datetime.timedelta(seconds=eta)).isoformat(
                        timespec="minutes") if eta is not None else None,
                    running={d: "%s seed %d" % r for d, r in running.items()})
        if progress_path:
            tmp = progress_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=1)
            os.replace(tmp, progress_path)
        return info

    def drive(dev):
        with cf.ProcessPoolExecutor(1, mp_context=ctx, initializer=_init,
                                    initargs=(dev, a.tracegen, a.agentdojo, threads)) as ex:
            while True:
                try:
                    key, seed = q.get_nowait()
                except queue.Empty:
                    with lock:
                        running.pop(dev, None)
                    return
                with lock:
                    running[dev] = (key, seed)
                    report("start %s seed %d on %s" % (key, seed, dev))
                    print("[%s] start %-33s seed %d  (elapsed %s)" % (dev, key, seed, _hms(time.time() - t0)),
                          flush=True)
                try:
                    res = ex.submit(_job, key, seed, a.epochs, a.lr, a.hidden).result()
                except Exception as e:  # noqa: BLE001
                    with lock:
                        errors.append((key, seed, repr(e)))
                        running.pop(dev, None)
                        report("failed %s seed %d" % (key, seed))
                    print("[%s] FAILED %s seed %d: %r" % (dev, key, seed, e), flush=True)
                    continue
                with lock:
                    out[(key, seed)] = res
                    running.pop(dev, None)
                    if on_done:
                        on_done(key, seed, res)
                    p = report("done %s seed %d" % (key, seed))
                    print("[%s] done  %-33s seed %d  %5.0fs  (%d/%d, %.0f%% of work)  elapsed %s  ETA %s  (~%s)"
                          % (dev, key, seed, res[1], len(out), len(jobs), p["work_done_pct"], p["elapsed"], p["eta"],
                             p["expected_finish"][11:]), flush=True)

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


def _ci(key, k):
    return lambda r: r[key][k] if r.get(key) else None


M = dict(f1=lambda r: r["macro_f1"], rej=lambda r: r["recall"]["reject"], appr=lambda r: r["recall"]["approve"],
         asr_u=lambda r: r["asr"]["asr_undefended"], asr_d=lambda r: r["asr"]["asr_defended"],
         bblock=lambda r: r["asr"]["benign_block"], fpr=lambda r: r["asr"]["approve_fpr"],
         ap=lambda r: r["auprc"], ap_lo=_ci("auprc_ci", 0), ap_hi=_ci("auprc_ci", 1),
         rf=lambda r: r["recall_at_fpr"], rf_lo=_ci("recall_at_fpr_ci", 0), rf_hi=_ci("recall_at_fpr_ci", 1),
         thr=lambda r: r["threshold"], epoch=lambda r: r["best_epoch"], val_ap=lambda r: r["val_auprc"])


def ci(runs, ev, guard, key):
    """Mean over seeds of the per-seed bootstrap interval, as '[lo, hi]'."""
    lo, hi = nanmean(metric(runs, ev, guard, M[key + "_lo"])), nanmean(metric(runs, ev, guard, M[key + "_hi"]))
    return "–" if lo != lo else "[%.2f, %.2f]" % (lo, hi)


def counts(runs, ev):
    """(reject actions, actions, traces) of an eval set, from any stored report."""
    for r in runs:
        for g in r.get(ev, {}).values():
            return g["n_reject"], g["n_actions"], g["n_traces"]
    return 0, 0, 0


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
        if not metric(runs, ev, g, M["fpr"]):
            continue
        rows.append([GUARD_NAME[g], pm(metric(runs, ev, g, M["ap"])), ci(runs, ev, g, "ap"),
                     pm(metric(runs, ev, g, M["rf"])), ci(runs, ev, g, "rf"),
                     pm(metric(runs, ev, g, M["rej"])), pm(metric(runs, ev, g, M["fpr"])),
                     pm(metric(runs, ev, g, M["f1"])),
                     "%s → %s" % (pm(metric(runs, ev, g, M["asr_u"])), pm(metric(runs, ev, g, M["asr_d"]))),
                     pm(metric(runs, ev, g, M["bblock"]))])
    return rows


GHEAD = ["guard", "AUPRC", "95% CI", "reject recall @ 1% FPR", "95% CI", "reject recall", "approve FPR", "macro-F1",
         "ASR no guard → guard", "benign traces blocked"]
GNOTE = ("AUPRC and reject recall at 1% approve FPR use the scores (no threshold); the interval is a bootstrap over "
         "traces, averaged over seeds. The last five columns use the reject threshold tuned on validation.")


def set_line(runs, ev):
    n, a, t = counts(runs, ev)
    return "%d reject actions of %d (%d traces). Chance AUPRC is the reject share, %.3f." % (n, a, t, n / a if a else 0)


def slice_cols(runs, kind, names, fmt="%s (%d reject)"):
    return [fmt % (c, counts(runs, "test|%s=%s" % (kind, c))[0]) for c in names]


def cv_section(cv, folds, n_cells, n_seeds):
    """Markdown lines of the cross-validation section ([] without a complete seed)."""
    from src.tracegen_gnn import GUARDS
    if not cv:
        return []
    n, acts, traces = counts(cv, "test")
    fol = next(iter(cv[0]["test"].values()))["n_followed"]
    L = ["## Cross-validation (%d task folds × %d injection-task folds)" % (folds, folds), "",
         "%d cells, pooled; %d of %d seeds complete. Every run is tested once, by a guard that saw neither its user task "
         "nor its injection task. %d reject actions of %d, in %d runs that followed an attack (%d runs in total). "
         "Chance AUPRC is the reject share, %.3f." % (n_cells, len(cv), n_seeds, n, acts, fol, traces,
                                                      n / acts if acts else 0), "",
         md_table(GHEAD, guard_rows(cv, "test", GUARDS)), "", GNOTE,
         "Scores of different cells come from different models and are pooled as they are; each cell uses its own "
         "validation threshold.", ""]
    evs = sorted({k for r in cv for k in r})
    for kind in ("suite", "agent", "attack"):
        names = [e.split("=", 1)[1] for e in evs if e.startswith("test|%s=" % kind)]
        if not names:
            continue
        if kind == "attack":   # one row per injection task: its runs plus the benign runs of its suite
            g = "gnn+files"
            rows = [[c.replace("|", " / "), next(iter(cv[0]["test|attack=%s" % c].values()))["n_followed"],
                     counts(cv, "test|attack=%s" % c)[0], pm(metric(cv, "test|attack=%s" % c, g, M["ap"])),
                     pm(metric(cv, "test|attack=%s" % c, g, M["rej"])),
                     "%s → %s" % (pm(metric(cv, "test|attack=%s" % c, g, M["asr_u"])),
                                  pm(metric(cv, "test|attack=%s" % c, g, M["asr_d"])))] for c in names]
            L += ["### Per injection task (GNN + files)", "",
                  "Each row scores the runs attacked with that injection task together with the benign runs of its "
                  "suite.", "",
                  md_table(["injection task", "runs that followed", "reject actions", "AUPRC", "reject recall",
                            "ASR no guard → guard"], rows), ""]
        else:
            L += ["### AUPRC per %s" % kind, "",
                  md_table(["guard"] + slice_cols(cv, kind, names),
                           [[GUARD_NAME[g]] + [pm(metric(cv, "test|%s=%s" % (kind, c), g, M["ap"])) for c in names]
                            for g in GUARDS]), ""]
    return L


def build_report(a, store, summary, leak, stats, seeds, devices, status, cv=(), n_cv=0):
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
         "injection tasks. Scores are mean ± std over seeds. Each guard uses the epoch with the best validation AUPRC "
         "and the reject threshold with the best validation reject F1, so validation scores are optimistic and test "
         "is the unbiased number. A set with no reject action has no AUPRC, recall or macro-F1 (shown as –).", "",
         "## Data", "",
         md_table(["split", "traces", "benign", "attacked", "followed attack", "security true", "actions", "reject",
                   "tasks", "injection tasks"],
                  [[sp, s["traces"], s["benign_traces"], s["attack_traces"], s["attack_followed"], s["security"],
                    s["actions"], s["reject"], s["tasks"], s["injection_tasks"]] for sp, s in summary.items()]), "",
         "Records read: %d; dropped with no tool call: %d; attack runs outside the fixed split because the injection task "
         "is in another split (used by the cross-validation only): %d." % (stats.get("records", 0), stats.get("no_tool_calls", 0), stats.get("dropped_split_mismatch", 0)),
         "", "Calls dropped as repeats of an earlier call with the same arguments and result: %d. Runs cut at the "
         "length cap: %d (%d calls, %d of them reject)." % (stats.get("repeats_dropped", 0), stats.get("traces_cut", 0),
                                                           stats.get("calls_cut", 0), stats.get("rejects_cut", 0)),
         "", "A call is `reject` only if it matches an attacker call, comes after a tool result that carried the "
         "injected text, and is not a call that benign runs of the same user task make.", ""]
    L += cv_section(cv, a.folds, n_cv, len(seeds))
    for ev, title in (("test", "Fixed split: test"), ("val", "Fixed split: validation (used for model selection)")):
        L += ["## %s" % title, "", set_line(main, ev), "", md_table(GHEAD, guard_rows(main, ev, GUARDS)), "", GNOTE, ""]
    sel = [[GUARD_NAME[g], pm(metric(main, "test", g, M["epoch"]), 0), pm(metric(main, "test", g, M["thr"])),
            pm(metric(main, "test", g, M["val_ap"]))] for g in GUARDS if metric(main, "test", g, M["thr"])]
    if sel:
        L += ["## Model selection (fixed split)", "", md_table(["guard", "best epoch", "reject threshold", "validation AUPRC"], sel), ""]
    for kind, names in (("suite", suites), ("agent", agents)):
        if names:
            L += ["## Fixed split: test AUPRC per %s" % kind, "",
                  md_table(["guard"] + slice_cols(main, kind, names),
                           [[GUARD_NAME[g]] + [pm(metric(main, "test|%s=%s" % (kind, c), g, M["ap"])) for c in names]
                            for g in GUARDS]), ""]
    base = metric(main, "test", "gnn+files", M["ap"])
    rows = []
    for ab in ABLATE:
        r = runs_of(store, "ablate/%s" % ab, seeds)
        if r:
            v = metric(r, "test", "gnn+files", M["ap"])
            rows.append([ABL_NAME[ab], pm(v), "%+.3f" % (nanmean(v) - nanmean(base)), ci(r, "test", "gnn+files", "ap"),
                         pm(metric(r, "test", "gnn+files", M["rf"]))])
    if rows:
        L += ["## Ablations (GNN + files, test)", "",
              "Full model: AUPRC %s, 95%% CI %s. A change smaller than the interval is not a finding." % (
                  pm(base), ci(main, "test", "gnn+files", "ap")), "",
              md_table(["removed", "AUPRC", "Δ vs full", "95% CI", "reject recall @ 1% FPR"], rows), ""]
    if transfer:
        L += ["## Cross-agent transfer (test AUPRC)", "",
              md_table(["train → test", "test reject actions"] + [GUARD_NAME[g] for g in GUARDS],
                       [[k.split("/", 1)[1], counts(runs_of(store, k, seeds), "test")[0]]
                        + [pm(metric(runs_of(store, k, seeds), "test", g, M["ap"])) for g in GUARDS]
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
    return md, render_html(a, store, summary, leak, seeds, main, suites, agents, transfer, base, md, cv)


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


def render_html(a, store, summary, leak, seeds, main, suites, agents, transfer, base, md, cv=()):
    from report_page import CSS, JS, div_color, esc, heatmap, seq_color, table
    from src.tracegen_gnn import ABLATE, GUARDS
    cls = {"majority": "nog", "tool-lookup": "s4", "mlp": "s3", "gnn": "s2", "gnn+files": "s1", "transformer": "s5"}
    head = cv or main   # the headline numbers: pooled cross-validation when it is complete
    te = lambda g: metric(head, "test", g, M["ap"])
    tot = {k: sum(s[k] for s in summary.values()) for k in ("traces", "attack_traces", "reject", "actions")}
    best = max(GUARDS, key=lambda g: nanmean(te(g)) if nanmean(te(g)) == nanmean(te(g)) else -1)
    tiles = [("traces", tot["traces"], "%d attacked" % tot["attack_traces"]),
             ("reject actions", tot["reject"], "of %d actions" % tot["actions"]),
             ("GNN + files, %s AUPRC" % ("cross-validation" if cv else "test"), pm(te("gnn+files")),
              "95%% CI %s · %d reject actions tested" % (ci(head, "test", "gnn+files", "ap"), counts(head, "test")[0])),
             ("leakage audit", "clean" if leak["ok"] else "overlap", "hard overlaps all 0" if leak["ok"] else "see table")]
    S = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
         'initial-scale=1"><title>Tracegen GNN guard</title><style>%s</style></head><body><main>' % CSS,
         '<header><div class="eyebrow">tracegen · AgentDojo · %s</div><h1>GNN guard on real agent traces, with file nodes'
         '</h1><p class="lede">Per-action approve/reject on AgentDojo runs from tracegen. Held-out splits have unseen '
         'user tasks (with all their paraphrases) and unseen injection tasks. Highest AUPRC: <b>%s</b>.</p></header>'
         % (esc(", ".join(agents) or datetime.date.today().isoformat()), esc(GUARD_NAME[best])),
         '<div class="tiles">%s</div>' % "".join('<div class="tile"><div class="k">%s</div><div class="v">%s</div>'
                                                 '<div class="s">%s</div></div>' % (esc(k), esc(v), esc(s))
                                                 for k, v, s in tiles)]
    if cv:
        items = [(GUARD_NAME[g], metric(cv, "test", g, M["ap"]), cls[g]) for g in GUARDS]
        S.append('<section><h2>Cross-validation AUPRC (%d × %d folds, pooled)</h2><p>Every run is tested once, by a '
                 'guard that saw neither its user task nor its injection task. %s</p><div class="panel">%s%s</div>'
                 '<p>%s</p></section>'
                 % (a.folds, a.folds, esc(set_line(cv, "test")), bars(items, "Cross-validation AUPRC by guard"),
                    table(GHEAD, [[esc(c) for c in r] for r in guard_rows(cv, "test", GUARDS)]), esc(GNOTE)))
    for ev, title in (("test", "Fixed split: test"), ("val", "Fixed split: validation (used for model selection)")):
        items = [(GUARD_NAME[g], metric(main, ev, g, M["ap"]), cls[g]) for g in GUARDS]
        S.append('<section><h2>%s AUPRC</h2><p>%s</p><div class="panel">%s%s</div><p>%s</p></section>'
                 % (title, esc(set_line(main, ev)), bars(items, "%s AUPRC by guard" % title),
                    table(GHEAD, [[esc(c) for c in r] for r in guard_rows(main, ev, GUARDS)]), esc(GNOTE)))
    for kind, cols in (("suite", suites), ("agent", agents)):
        if cols:
            shown = dict(zip(slice_cols(main, kind, cols, "%s (%d)"), cols))

            def cell(r, c, kind=kind, shown=shown):
                v = metric(main, "test|%s=%s" % (kind, shown[c]), [g for g in GUARDS if GUARD_NAME[g] == r][0], M["ap"])
                m = nanmean(v)
                if m != m:   # no reject action in this slice
                    return None
                fill, ink = seq_color(m)
                return fill, ink, "%.2f" % m, "%s · %s: AUPRC %s" % (r, c, pm(v))
            S.append('<section><h2>Test AUPRC per %s</h2><p>In brackets: reject actions in the slice. A slice with none has no score.</p>'
                     '<div class="panel">%s</div></section>'
                     % (kind, heatmap([GUARD_NAME[g] for g in GUARDS], list(shown), cell, "AUPRC per %s" % kind)))
    rows = [(ab, runs_of(store, "ablate/%s" % ab, seeds)) for ab in ABLATE]
    rows = [(ab, r) for ab, r in rows if r]
    if rows:
        bm = nanmean(base)

        def acell(r, c):
            ab = [k for k, v in ABL_NAME.items() if v == r][0]
            v = metric(dict(rows)[ab], "test", "gnn+files", M["ap"])
            d = nanmean(v) - bm
            if d != d:
                return None
            fill, ink = div_color(d, 0.2)
            return fill, ink, "%+.3f" % d, "without %s: AUPRC %s (full %.3f)" % (r, pm(v), bm)
        S.append('<section><h2>Ablations</h2><p>Change in test AUPRC of GNN + files when one input is removed '
                 '(full model %s, 95%% CI %s). Red means the input helps. A change smaller than the interval is not a '
                 'finding.</p><div class="panel">%s</div></section>'
                 % (esc(pm(base)), esc(ci(main, "test", "gnn+files", "ap")),
                    heatmap([ABL_NAME[ab] for ab, _ in rows], ["Δ AUPRC"], acell, "ablations")))
    if transfer:
        def tcell(r, c):
            g = [k for k, v in GUARD_NAME.items() if v == c][0]
            v = metric(runs_of(store, "transfer/" + r, seeds), "test", g, M["ap"])
            if nanmean(v) != nanmean(v):
                return None
            fill, ink = seq_color(nanmean(v))
            return fill, ink, "%.2f" % nanmean(v), "%s, %s: AUPRC %s" % (r, c, pm(v))
        S.append('<section><h2>Cross-agent transfer (test AUPRC)</h2><p>Blank: the target agent has no reject action '
                 'in test.</p><div class="panel">%s</div></section>'
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
    ap.add_argument("--sections", default="main,cv,ablate,transfer")
    ap.add_argument("--folds", type=int, default=5, help="cv: task folds and injection-task folds (>= 3)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--devices", default="auto", help="auto | cpu | cuda:0,cuda:1 | ...")
    ap.add_argument("--no-reuse", dest="reuse", action="store_false")
    ap.add_argument("--report-only", action="store_true", help="only re-render the report from stored results")
    a = ap.parse_args()

    from src.tracegen_gnn import Data, audit, cells, code_hash, cv_reports, fingerprint
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
    keys = cells(data, a.sections.split(","), a.folds)
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
        def save(k, s, r):  # after every finished run, so an interrupted run resumes from what is stored
            store.setdefault(k, {})[str(s)] = dict(fp=fps[(k, s)], run=r[0], seconds=round(r[1], 1),
                                                    time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
                json.dump(store, f, indent=1, sort_keys=True)
            os.replace(path + ".tmp", path)

        from src.tracegen_gnn import cell_spec
        ntrain = {k: len(cell_spec(data, k)[0]) for k in {k for k, _ in jobs}}
        prog = os.path.join(a.outdir, "progress.json")
        print("progress file:", prog, flush=True)
        res = run_parallel(jobs, devices, a, weight={(k, s): ntrain[k] for k, s in jobs}, on_done=save,
                           progress_path=prog)
        status += ["computed" if j in res else "failed" for j in jobs]
    # only report runs that match the current inputs
    cur = {k: {str(s): store[k][str(s)] for s in seeds if str(s) in store.get(k, {})
               and store[k][str(s)].get("fp") == fps[(k, s)]} for k in keys}
    n_cv = sum(k.startswith("cv/") for k in keys)
    cv = cv_reports(data, cur, seeds, a.folds) if n_cv else []
    if n_cv and len(cv) < len(seeds):
        print("cross-validation: %d of %d seeds have every cell; the others are left out of the report" % (len(cv), len(seeds)))
    md, page = build_report(a, cur, summary, leak, data.stats, seeds, devices, status, cv, n_cv)
    for name, text in (("REPORT.md", md), ("report.html", page)):
        with open(os.path.join(a.outdir, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print("wrote", os.path.join(a.outdir, "REPORT.md"), "and report.html")


if __name__ == "__main__":
    main()
