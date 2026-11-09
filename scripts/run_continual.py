"""Continual graph learning experiments: train the GNN guard on one benchmark after another, then write the report.

    python scripts/run_continual.py                        # full run on every CUDA GPU (2 jobs per GPU), then report
    python scripts/run_continual.py --seeds 1 --epochs 5   # quick pass
    python scripts/run_continual.py --report-only          # rebuild the report from stored cells, no training
    python scripts/run_continual.py --devices cpu --jobs 4 # CPU only

Stream: asb, injecagent, agentdojo, qwen2.5:1.5b traces only (AgentPoison excluded), all 6 orders. Strategies: finetune,
joint, ewc, mas, lwf, agem, twp, ergnn, replay (src/continual/strategies.py).

Phases:
  1. tune     ewc / mas / lwf / twp hyperparameters on a validation stream carved out of the seed-0 training traces
              (never the test sets), plus every cell that needs no tuning, plus the memory-size sweep
  2. main     the tuned strategies with the chosen hyperparameters (<outdir>/hparams.json)
  3. report   <outdir>/REPORT.md, <outdir>/report.html (charts), <outdir>/continual.json; headline printed here

Every cell is stored under <outdir>/cells/ with a fingerprint over the code, the exact train/test graphs and the config.
Cells whose fingerprint still matches are reused, so re-running resumes an interrupted run and trains only stale or
new cells. A GPU run refuses to start while run_experiments.py or trace collection holds the GPUs (--force overrides).
"""
import argparse
import concurrent.futures
import datetime
import glob
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from run_experiments import git_info, pick_devices, pid_alive  # noqa: E402  (same scripts/ directory)

from src.continual import report as REP  # noqa: E402
from src.continual import stream as S  # noqa: E402
from src.continual.run import BUFFER, EPOCHS, Planner, choose, load_cell, write_json  # noqa: E402
from src.continual.strategies import STRATEGIES, TUNED  # noqa: E402

MAIN_LOCK = os.path.join(ROOT, "results", ".run_experiments.lock")
BUSY = ("collect_traces.py", "src.collect", "src.experiments", "run_experiments.py")
_print_lock, _log_lock = threading.Lock(), threading.Lock()


# ---------------------------------------------------------------- GPU etiquette
def busy_processes():
    """Other processes that use the GPUs for this project (POSIX /proc only; elsewhere only the lock file is checked)."""
    found = []
    if os.path.isdir("/proc"):
        for d in glob.glob("/proc/[0-9]*"):
            pid = int(os.path.basename(d))
            if pid == os.getpid():
                continue
            try:
                argv = open(os.path.join(d, "cmdline"), "rb").read().decode(errors="replace").split("\0")
            except OSError:
                continue
            cmd = " ".join(argv)
            if argv and "python" in os.path.basename(argv[0]) and any(k in cmd for k in BUSY):
                found.append("%d %s" % (pid, cmd.strip()[:120]))
    if os.path.exists(MAIN_LOCK):
        try:
            pid = int(open(MAIN_LOCK, encoding="utf-8").read().split()[0])
        except (OSError, ValueError, IndexError):
            pid = -1
        if pid > 0 and pid_alive(pid):
            found.append("%d (holds %s)" % (pid, os.path.relpath(MAIN_LOCK, ROOT)))
    return found


def take_lock(path, force):
    if os.path.exists(path) and not force:
        try:
            pid = int(open(path, encoding="utf-8").read().split()[0])
        except (OSError, ValueError, IndexError):
            pid = -1
        if pid > 0 and pid != os.getpid() and pid_alive(pid):
            sys.exit("another run_continual.py (pid %d) is running; wait for it or pass --force" % pid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("%d %s\n" % (os.getpid(), datetime.datetime.now().isoformat(timespec="seconds")))


# ---------------------------------------------------------------- jobs
def run_job(job, device, threads, outdir):
    """Run `python -m src.continual.run ...` on `device`; echo prefixed output; append it to <outdir>/run.log."""
    cmd = [sys.executable, "-m", "src.continual.run"] + job["args"] + ["--outdir", outdir]
    tag = "[%s@%s] " % (job["name"], device)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", GNN_DEVICE=device)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(k, str(threads))
    buf = ["\n=== %s %s (device %s)\n" % (datetime.datetime.now().isoformat(timespec="seconds"), " ".join(cmd), device)]
    with _print_lock:
        print("=== %sstart (%d cells)" % (tag, job["n"]), flush=True)
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        buf.append(line)
        if "[cache]" in line or "Error" in line or "Traceback" in line:
            with _print_lock:
                sys.stdout.write(tag + line)
                sys.stdout.flush()
    p.wait()
    log = os.path.join(outdir, "run.log")
    with _log_lock, open(log, "a", encoding="utf-8", newline="\n") as f:
        f.write("".join(buf))
    if p.returncode != 0:
        return "job %s failed (exit %d), see %s" % (job["name"], p.returncode, log)
    return None


def run_jobs(jobs, devices, n_jobs, outdir):
    """At most n_jobs at once, devices handed out round-robin; no new jobs after the first failure."""
    if not jobs:
        return []
    free = queue.Queue()
    for i in range(n_jobs):
        free.put(devices[i % len(devices)])
    threads = max(1, (os.cpu_count() or 1) // n_jobs)
    failed = threading.Event()

    def work(job):
        if failed.is_set():
            return None
        dev = free.get()
        try:
            if failed.is_set():
                return None
            err = run_job(job, dev, threads, outdir)
            if err:
                failed.set()
            return err
        finally:
            free.put(dev)

    jobs = sorted(jobs, key=lambda j: -j["cost"])  # longest first
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as ex:
        return [e for e in ex.map(work, jobs) if e]


def make_job(name, cells, args):
    pending = [c for c in cells if load_cell(c) is None]
    weight = 2.0 if cells and cells[0]["strategy"] == "joint" else 1.0
    return dict(name=name, args=args, n=len(pending), cost=len(pending) * weight, cells=cells, pending=pending)


# ---------------------------------------------------------------- audit
def leakage_audit(planner, seeds, fresh):
    exps = planner.exps
    sizes, overlaps = {}, {}
    sets = {}
    for s in seeds:
        sp = planner.split("main", s)
        sizes[s], overlaps[s] = S.sizes(exps, sp), S.audit(exps, sp)
        sets[s] = [{(b, exps[b].ids[i]) for b, idx in sp.items() for i in idx[k]} for k in (0, 1)]
    tsp = planner.split("tune", 0)
    test0 = planner.split("main", 0)
    tune = dict(val_vs_tune_train=S.audit(exps, tsp),
                val_vs_test=S.overlaps(exps, S.side(exps, tsp, 1), S.side(exps, test0, 1)))
    tune_txt = "val vs tune-train: %s; val vs test: %s" % (
        ", ".join("%s %d" % kv for kv in tune["val_vs_tune_train"].items()),
        ", ".join("%s %d" % kv for kv in tune["val_vs_test"].items()))
    checked = not_train = in_test = joint_checked = joint_bad = 0
    for rec in fresh:
        cfg = rec["config"]
        if cfg["mode"] != "main":
            continue
        train_ids, test_ids = sets.get(cfg["seed"]) or [set(), set()]
        for k, st in enumerate(rec["steps"]):
            for b, t in st["memory"]:
                checked += 1
                not_train += (b, t) not in train_ids
                in_test += (b, t) in test_ids
            if cfg["strategy"] == "joint":
                joint_checked += 1
                expect = sum(len(planner.split("main", cfg["seed"])[b][0]) for b in cfg["order"][: k + 1])
                joint_bad += st["trained_graphs"] != expect
    caveats = []
    few = min(z["agentdojo"]["train_attack"] for z in sizes.values()) if sizes else None
    if few is not None and few < 20:
        caveats.append("AgentDojo has as few as %d training attack traces in some seeds (inherited from the main "
                       "suite's split), so its cells are noisier." % few)
    tz = S.sizes(exps, tsp)
    caveats.append("Validation sets are small (%s traces), so the tuned values are a coarse choice."
                   % ", ".join("%s %d" % (b, z["test"]) for b, z in tz.items()))
    return dict(sizes=sizes, overlaps=overlaps, tune=tune_txt, buffer_checked=checked, buffer_not_train=not_train,
                buffer_in_test=in_test, joint_checked=joint_checked, joint_bad=joint_bad, caveats=caveats)


# ---------------------------------------------------------------- main
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategies", default=",".join(STRATEGIES), help="comma-separated subset of " + ",".join(STRATEGIES))
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (0..n-1)")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs per experience")
    ap.add_argument("--buffer", type=int, default=BUFFER, help="memory size in graphs for agem / replay / ergnn")
    ap.add_argument("--sweep", default="20,100,200", help="extra memory sizes for replay / ergnn, or 'none'")
    ap.add_argument("--devices", default="auto", help="cuda:0,cuda:1 / cpu / mps; 'auto' = every CUDA GPU, else cpu")
    ap.add_argument("--jobs", default="auto", help="jobs at once; 'auto' = 2 per CUDA GPU, else 1")
    ap.add_argument("--outdir", default="results/continual")
    ap.add_argument("--report-only", action="store_true", help="no training: report the stored fresh cells")
    ap.add_argument("--force", action="store_true", help="start even if other GPU jobs or another run are active")
    a = ap.parse_args()
    t_start = time.time()
    strategies = [s for s in a.strategies.split(",") if s]
    bad = set(strategies) - set(STRATEGIES)
    if bad:
        ap.error("unknown strategies: %s" % ", ".join(sorted(bad)))
    sweep = [] if a.sweep.strip().lower() in ("", "none") else sorted({int(x) for x in a.sweep.split(",")} - {a.buffer})
    try:
        devices, _ = pick_devices(a.devices, "auto")
    except ValueError as e:
        ap.error(str(e))
    import torch
    types = {torch.device(d).type for d in devices}
    if len(types) > 1:
        ap.error("do not mix device types in one run (dropout RNG differs per device): %s" % ", ".join(devices))
    dev_type = types.pop()
    n_jobs = (2 * len(devices) if dev_type == "cuda" else len(devices)) if a.jobs == "auto" else int(a.jobs)
    if n_jobs < 1:
        ap.error("--jobs must be >= 1")
    outdir = os.path.abspath(a.outdir)
    seeds = list(range(a.seeds))

    lock = os.path.join(outdir, ".lock")
    if not a.report_only:
        busy = busy_processes() if dev_type != "cpu" else []  # a CPU run does not compete for the GPUs
        if busy and not a.force:
            sys.exit("GPU jobs of this project are running; wait for them or pass --force:\n  " + "\n  ".join(busy))
        take_lock(lock, a.force)
    try:
        t = time.time()
        exps = S.load_stream()
        print("traces (%s): %s, loaded in %.0fs" % (S.MODEL, ", ".join("%s %d" % (b, len(e)) for b, e in exps.items()),
                                                   time.time() - t), flush=True)
        print("devices: %s, parallel jobs: %d" % (", ".join(devices), n_jobs), flush=True)
        planner = Planner(exps, a.epochs, dev_type)
        common = ["--epochs", str(a.epochs)]
        hp_path = os.path.join(outdir, "hparams.json")
        tuned = [s for s in strategies if s in TUNED]

        # phase 1: tuning, untuned strategies, sweep
        tune_jobs = {s: make_job("tune/" + s, planner.job_cells(outdir, "tune", s, [0], None, a.buffer, None),
                                 ["--strategy", s, "--mode", "tune"] + common) for s in tuned}
        phase1 = list(tune_jobs.values())
        for s in strategies:
            if s in TUNED:
                continue
            for seed in seeds:
                phase1.append(make_job("%s/s%d" % (s, seed),
                                       planner.job_cells(outdir, "main", s, [seed], S.ORDERS, a.buffer, {}),
                                       ["--strategy", s, "--seeds", str(seed), "--buffer", str(a.buffer)] + common))
        sweep_jobs = []
        for s in [x for x in ("replay", "ergnn") if x in strategies]:
            for m in sweep:
                for seed in seeds:
                    sweep_jobs.append(make_job("%s/m%d/s%d" % (s, m, seed),
                                               planner.job_cells(outdir, "main", s, [seed], [S.CANONICAL], m, {}),
                                               ["--strategy", s, "--seeds", str(seed), "--orders", "canonical",
                                                "--buffer", str(m)] + common))
        phase1 += sweep_jobs
        planned = [c for j in phase1 for c in j["cells"]]
        reused = sum(len(j["cells"]) - j["n"] for j in phase1)
        todo = [j for j in phase1 if j["n"]]
        if not a.report_only:
            print("phase 1: %d cells planned, %d reused, %d to compute in %d jobs"
                  % (len(planned), reused, sum(j["n"] for j in todo), len(todo)), flush=True)
            errors = run_jobs(todo, devices, n_jobs, outdir)
            if errors:
                sys.exit("\n".join(errors))

        # choose hyperparameters from fresh tuning cells
        hparams = {}
        for s, j in tune_jobs.items():
            recs = [load_cell(c) for c in j["cells"]]
            if recs and all(recs):
                hparams[s] = choose(recs)
                print("tuned %-4s %s (val AP %.3f)" % (s, json.dumps(hparams[s]["hp"]), hparams[s]["val_ap"]), flush=True)
        if hparams:
            old = json.load(open(hp_path, encoding="utf-8")) if os.path.exists(hp_path) else {}
            write_json(hp_path, dict(old, **hparams))

        # phase 2: tuned strategies with the chosen hyperparameters
        phase2, missing = [], {}
        for s in tuned:
            if s not in hparams:
                missing[s] = len(seeds) * len(S.ORDERS)
                continue
            for seed in seeds:
                phase2.append(make_job("%s/s%d" % (s, seed),
                                       planner.job_cells(outdir, "main", s, [seed], S.ORDERS, a.buffer, hparams),
                                       ["--strategy", s, "--seeds", str(seed), "--hparams", hp_path] + common))
        planned += [c for j in phase2 for c in j["cells"]]
        reused += sum(len(j["cells"]) - j["n"] for j in phase2)
        todo = [j for j in phase2 if j["n"]]
        if not a.report_only:
            print("phase 2: %d cells planned, %d reused, %d to compute in %d jobs"
                  % (sum(len(j["cells"]) for j in phase2), sum(len(j["cells"]) - j["n"] for j in phase2),
                     sum(j["n"] for j in todo), len(todo)), flush=True)
            errors = run_jobs(todo, devices, n_jobs, outdir)
            if errors:
                sys.exit("\n".join(errors))

        # phase 3: report on fresh cells only
        main_recs, sweep_recs, tune_recs, fresh = {}, {}, {}, []
        for c in planned:
            rec = load_cell(c)
            if rec is None:
                missing[c["strategy"]] = missing.get(c["strategy"], 0) + 1
                continue
            fresh.append(rec)
            if c["mode"] == "tune":
                tune_recs.setdefault(c["strategy"], []).append(rec)
            elif c["buffer"] is not None and c["buffer"] != a.buffer:
                sweep_recs.setdefault(c["strategy"], []).append(rec)
            else:
                main_recs.setdefault(c["strategy"], []).append(rec)
        computed = len(fresh) - reused if not a.report_only else 0
        prov = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "git commit": git_info(),
                "python / torch": "%s / %s" % (platform.python_version(), torch.__version__),
                "devices / parallel jobs": "%s / %d" % (", ".join(devices), n_jobs),
                "seeds / epochs per experience": "%d / %d" % (len(seeds), a.epochs),
                "memory M (agem, replay, ergnn)": "%d graphs (sweep: %s)" % (a.buffer, ", ".join(map(str, sweep)) or "none"),
                "orders": "all %d permutations of %s" % (len(S.ORDERS), ", ".join(S.BENCHES)),
                "cells": "%d planned: %d reused, %d computed, %d missing" % (len(planned), reused, max(computed, 0),
                                                                             sum(missing.values())),
                "command": "`python %s`" % " ".join(["scripts/run_continual.py"] + sys.argv[1:]),
                "model": S.MODEL, "benches": list(S.BENCHES), "canonical": list(S.CANONICAL), "buffer": a.buffer,
                "missing": missing}
        audit = leakage_audit(planner, seeds, fresh)
        d = REP.build(main_recs, sweep_recs, tune_recs, hparams, audit, prov, os.path.join(ROOT, "results"))
        paths = REP.write(d, outdir)
        print("\n" + REP.terminal(d))
        for k in ("REPORT.md", "report.html", "continual.json"):
            rel = os.path.relpath(paths[k], os.getcwd())
            print("wrote", paths[k] if rel.startswith("..") else rel)
        print("cells: %s; total %.0fs" % (prov["cells"], time.time() - t_start))
        if missing:
            print("incomplete: %s" % ", ".join("%s %d" % kv for kv in missing.items()))
    finally:
        if not a.report_only and os.path.exists(lock):
            try:
                if open(lock, encoding="utf-8").read().split()[0] == str(os.getpid()):
                    os.remove(lock)
            except (OSError, IndexError):
                pass


if __name__ == "__main__":
    main()
