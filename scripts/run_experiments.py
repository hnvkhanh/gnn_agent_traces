"""Run every experiment on the existing traces and write a combined report.

    python scripts/run_experiments.py                       # all sections, every LLM in traces/
    python scripts/run_experiments.py --seeds 1 --epochs 5  # quick pass
    python scripts/run_experiments.py --skip-existing       # resume an interrupted run

Result cells are reused when their inputs are unchanged (src/cache.py: a fingerprint over the training code, seeds,
epochs and every train/test graph), so only new or stale cells are trained; --no-reuse recomputes everything.
--adopt-legacy accepts cells written before fingerprints existed if their results file is newer than their inputs.
Each model runs one job per section (inbench, transfer, ablate) so sections spread over the devices.

Steps run in parallel, one per device. `--devices auto` (default) uses every CUDA GPU, else the CPU
(also on macOS, where MPS is opt-in):
    python scripts/run_experiments.py --devices cuda:0,cuda:1   # two steps at once, one per GPU
    python scripts/run_experiments.py --devices cuda:1          # serial on one GPU
    python scripts/run_experiments.py --devices cpu --jobs 2    # two CPU steps, threads split between them
    python scripts/run_experiments.py --devices mps             # Apple GPU (PyG support on MPS is partial)

This script never calls an LLM. Generate traces first with scripts/collect_traces.py.

Outputs (under --outdir, default results/):
    <model_slug>/experiments_*.json, report_*.md, run.log   benchmark sections, one dir per trace-generating LLM
    llm/llm_experiments.json, llm_report.md, run.log        cross-LLM section
    REPORT.md                                               everything above plus run metadata
The README block between the results markers is updated with the headline table (unless --no-readme).
"""
import argparse
import concurrent.futures
import datetime
import glob
import json
import os
import platform
import re
import queue
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.experiments import BENCHES, stat, table  # noqa: E402

BENCH_SECTIONS = ["inbench", "transfer", "ablate"]
ALL_SECTIONS = BENCH_SECTIONS + ["llm"]
README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"


def slug(model):
    return model.replace(":", "_")


def trace_counts():
    """{model: {bench: n_traces}} from traces/<bench>__*.jsonl."""
    counts = {}
    for b in BENCHES:
        for p in glob.glob(os.path.join(ROOT, "traces", "%s__*.jsonl" % b)):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    m = json.loads(line).get("model")
                    counts.setdefault(m, {}).setdefault(b, 0)
                    counts[m][b] += 1
    return counts


def model_key(m):
    """Sort by family, then parameter count (qwen2.5:1.5b < qwen2.5:7b < qwen2.5:14b)."""
    fam, _, size = m.partition(":")
    num = re.match(r"[\d.]+", size)
    return fam, float(num.group()) if num else 0.0, size


def pick_devices(spec, jobs):
    """Resolve --devices/--jobs to (devices, n_jobs). Raises ValueError on an unusable device."""
    import torch

    if spec == "auto":
        if os.environ.get("GNN_DEVICE"):
            devices = [os.environ["GNN_DEVICE"]]
        elif torch.cuda.is_available():
            devices = ["cuda:%d" % i for i in range(torch.cuda.device_count())]
        else:
            devices = ["cpu"]  # also on macOS: MPS is opt-in via --devices mps
    else:
        devices = [d.strip() for d in spec.split(",") if d.strip()]
        if not devices:
            raise ValueError("--devices is empty")
    for d in devices:
        try:
            dev = torch.device(d)
        except RuntimeError as e:
            raise ValueError("bad device %r: %s" % (d, e))
        if dev.type == "cuda":
            n = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if (dev.index or 0) >= n:
                raise ValueError("device %s not available (%d CUDA GPU(s) found)" % (d, n))
        elif dev.type == "mps":
            if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                raise ValueError("device mps not available on this machine")
        elif dev.type != "cpu":
            raise ValueError("unsupported device %r (use cuda[:i], mps or cpu)" % d)
    n_jobs = len(devices) if jobs == "auto" else int(jobs)
    if n_jobs < 1:
        raise ValueError("--jobs must be >= 1")
    return devices, n_jobs


_print_lock = threading.Lock()
_log_lock = threading.Lock()


def run_step(name, args, outdir, expected, device, threads=None):
    """Run `python -m src.experiments ...` on `device`, echo prefixed output and append it to outdir/run.log.

    Returns an error message, or None on success. Safe to call from several threads at once."""
    os.makedirs(outdir, exist_ok=True)
    cmd = [sys.executable, "-m", "src.experiments"] + args + ["--outdir", outdir]
    tag = "[%s@%s] " % (name, device)
    with _print_lock:
        print("\n=== %s: %s" % (tag.strip(), " ".join(cmd)), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", GNN_DEVICE=device)
    if threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(k, str(threads))
    buf = ["\n=== %s %s (device %s)\n" % (datetime.datetime.now().isoformat(timespec="seconds"), " ".join(cmd), device)]
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        buf.append(line)
        with _print_lock:
            sys.stdout.write(tag + line)
            sys.stdout.flush()
    p.wait()
    log_path = os.path.join(outdir, "run.log")
    with _log_lock, open(log_path, "a", encoding="utf-8", newline="\n") as log:
        log.write("".join(buf))  # one write per step so parallel steps sharing a dir don't interleave
    if p.returncode != 0:
        return "step '%s' failed (exit %d), see %s" % (name, p.returncode, log_path)
    if not os.path.exists(expected):
        return "step '%s' finished but %s is missing" % (name, expected)
    with _print_lock:
        print("=== %sdone" % tag, flush=True)
    return None


LOCK = os.path.join(ROOT, "results", ".run_experiments.lock")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False


def take_lock(force):
    """One run at a time: parallel runs would overwrite each other's reports and fingerprints."""
    if os.path.exists(LOCK) and not force:
        try:
            pid = int(open(LOCK, encoding="utf-8").read().split()[0])
        except Exception:
            pid = -1
        if pid > 0 and pid != os.getpid() and pid_alive(pid):
            sys.exit("another run_experiments.py (pid %d) is running; wait for it or pass --force" % pid)
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    with open(LOCK, "w", encoding="utf-8") as f:
        f.write("%d %s\n" % (os.getpid(), datetime.datetime.now().isoformat(timespec="seconds")))


def retire_legacy(model_dir):
    """Move experiments_all.json / report_all.md aside once every per-section output exists (so REPORT.md does not
    show sections twice). Nothing is deleted."""
    if not all(os.path.exists(os.path.join(model_dir, "report_%s.md" % s)) for s in BENCH_SECTIONS):
        return
    for name in ("experiments_all.json", "report_all.md", "fingerprints_all.json"):
        src = os.path.join(model_dir, name)
        if os.path.exists(src):
            os.makedirs(os.path.join(model_dir, "legacy"), exist_ok=True)
            os.replace(src, os.path.join(model_dir, "legacy", name))
            print("moved %s to legacy/" % os.path.relpath(src, ROOT))


def run_all(jobs, devices, n_jobs):
    """Run (name, args, outdir, expected) jobs, at most n_jobs at once, devices handed out round-robin.

    After the first failure no new jobs start; running ones finish. Returns the list of error messages."""
    free = queue.Queue()
    for i in range(n_jobs):
        free.put(devices[i % len(devices)])
    per_dev = {d: sum(1 for i in range(n_jobs) if devices[i % len(devices)] == d) for d in devices}
    cpus = os.cpu_count() or 1
    failed = threading.Event()
    errors = []

    def work(job):
        if failed.is_set():
            return None
        dev = free.get()
        try:
            if failed.is_set():
                return None
            # several jobs on the CPU: split the cores so they don't oversubscribe
            threads = max(1, cpus // per_dev[dev]) if dev == "cpu" and per_dev[dev] > 1 else None
            err = run_step(*job, device=dev, threads=threads)
            if err:
                failed.set()
            return err
        finally:
            free.put(dev)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as ex:
        for err in ex.map(work, jobs):
            if err:
                errors.append(err)
    return errors


def git_info():
    def git(*a):
        try:
            return subprocess.run(["git"] + list(a), cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    head = git("rev-parse", "--short", "HEAD") or "unknown"
    return head + (" (uncommitted changes)" if git("status", "--porcelain", "--untracked-files=no") else "")


def demote(md):
    """Push markdown headings down one level so per-model reports nest under '## <model>'."""
    return re.sub(r"^(#+) ", r"#\1 ", md, flags=re.M)


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def inbench_runs(model_dir):
    for name in ("experiments_all.json", "experiments_inbench.json"):
        R = load_json(os.path.join(model_dir, name))
        if R and any(k.startswith("inbench/") for k in R):
            return R
    return None


def headline(outdir, models):
    """GNN macro-F1 on held-out tasks (tool-lookup baseline in brackets), model x benchmark."""
    rows = []
    for m in models:
        R = inbench_runs(os.path.join(outdir, slug(m)))
        if not R:
            continue
        row = [m]
        for b in BENCHES:
            runs = R.get("inbench/" + b)
            if not runs:
                row.append("-")
                continue
            gnn = stat([r["gnn"]["macro_f1"] for r in runs], 0)
            tl = sum(r["tool-lookup"]["macro_f1"] for r in runs) / len(runs)
            row.append("%s (%.2f)" % (gnn, tl))
        rows.append(row)
    if not rows:
        return ""
    return ("GNN macro-F1 on held-out tasks, mean±std over seeds, with the tool-lookup baseline in brackets. "
            "Each row trains and tests on traces from one LLM.\n\n" + table(rows, ["trace LLM"] + BENCHES))


def write_report(a, outdir, models, counts, devices, n_jobs):
    import torch

    md = ["# Experiment report", "",
          "| | |", "|---|---|",
          "| date | %s |" % datetime.datetime.now().isoformat(timespec="seconds"),
          "| git commit | %s |" % git_info(),
          "| python / torch | %s / %s |" % (platform.python_version(), torch.__version__),
          "| devices / parallel jobs | %s / %d |" % (", ".join(devices), n_jobs),
          "| seeds / epochs | %d / %d |" % (a.seeds, a.epochs),
          "| sections | %s |" % ", ".join(a.sections),
          "| command | `python %s` |" % " ".join(["scripts/run_experiments.py"] + sys.argv[1:]),
          "",
          "Cells are `macro-F1 / micro-F1 / reject recall` (mean±std over seeds) unless a table says otherwise. "
          "See the README for the definitions of ASR, `benign_block` and `approve_fpr`.",
          "",
          "## Traces",
          "",
          table([[m] + [counts.get(m, {}).get(b, 0) for b in BENCHES] + [sum(counts.get(m, {}).values())] for m in models],
                ["LLM"] + BENCHES + ["total"])]
    h = headline(outdir, models)
    if h:
        md += ["", "## Headline: in-benchmark", "", h]
    for m in models:
        d = os.path.join(outdir, slug(m))
        order = {s: i for i, s in enumerate(["all"] + BENCH_SECTIONS)}
        reports = sorted(glob.glob(os.path.join(d, "report_*.md")),
                         key=lambda p: order.get(os.path.basename(p)[len("report_"):-len(".md")], 99))
        parts = [open(p, encoding="utf-8").read().strip() for p in reports]
        if parts:
            md += ["", "## Benchmark sections, traces from %s" % m, "", demote("\n\n".join(parts))]
    llm = os.path.join(outdir, "llm", "llm_report.md")
    if os.path.exists(llm):
        md += ["", "## Cross-LLM (%s)" % ", ".join(models), "", demote(open(llm, encoding="utf-8").read().strip())]
    path = os.path.join(outdir, "REPORT.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print("\nwrote", path)
    return h


def update_readme(outdir, h, a):
    path = os.path.join(ROOT, "README.md")
    text = open(path, encoding="utf-8").read()
    if README_START not in text or README_END not in text:
        print("README has no results markers, not updated")
        return
    rel = os.path.relpath(os.path.join(outdir, "REPORT.md"), ROOT).replace(os.sep, "/")
    body = ("Generated by `scripts/run_experiments.py` on %s (commit %s, %d seeds, %d epochs). Full tables: [%s](%s).\n\n%s"
            % (datetime.date.today().isoformat(), git_info(), a.seeds, a.epochs, rel, rel, h or "_no in-benchmark results_"))
    pre, rest = text.split(README_START, 1)
    _, post = rest.split(README_END, 1)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(pre + README_START + "\n" + body + "\n" + README_END + post)
    print("updated README results block")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--models", default="auto", help="comma-separated trace LLMs, or 'auto' = every LLM found in traces/")
    ap.add_argument("--sections", default=",".join(ALL_SECTIONS), help="comma-separated subset of " + ",".join(ALL_SECTIONS))
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--skip-existing", action="store_true", help="skip steps whose report already exists")
    ap.add_argument("--no-readme", action="store_true", help="do not update the README results block")
    ap.add_argument("--devices", default="auto",
                    help="comma-separated torch devices to train on (cuda:0,cuda:1 / mps / cpu); "
                         "'auto' = $GNN_DEVICE if set, else every CUDA GPU, else cpu")
    ap.add_argument("--jobs", default="auto", help="steps to run at once; 'auto' = one per device")
    ap.add_argument("--no-reuse", action="store_true", help="recompute every result cell")
    ap.add_argument("--adopt-legacy", action="store_true",
                    help="accept results written before fingerprints existed when newer than their inputs")
    ap.add_argument("--force", action="store_true", help="start even if another run holds the lock")
    a = ap.parse_args()
    a.sections = [s for s in a.sections.split(",") if s]
    bad = set(a.sections) - set(ALL_SECTIONS)
    if bad:
        ap.error("unknown sections: %s" % ", ".join(sorted(bad)))
    try:
        devices, n_jobs = pick_devices(a.devices, a.jobs)
    except ValueError as e:
        ap.error(str(e))
    outdir = os.path.abspath(a.outdir)

    counts = trace_counts()
    models = sorted(counts, key=model_key) if a.models == "auto" else [m for m in a.models.split(",") if m]
    missing = [m for m in models if m not in counts]
    if missing:
        sys.exit("no traces for: %s (run scripts/collect_traces.py first)" % ", ".join(missing))
    if not models:
        sys.exit("no traces found in traces/ (run scripts/collect_traces.py first)")
    print("models:", ", ".join(models))
    print("devices: %s, parallel jobs: %d" % (", ".join(devices), n_jobs))

    take_lock(a.force)
    # heaviest sections first, so with several devices they start at once on different GPUs
    bench = [s for s in ["ablate", "transfer", "inbench"] if s in a.sections]
    onlys = bench
    common = ["--seeds", str(a.seeds), "--epochs", str(a.epochs)]
    common += ["--no-reuse"] if a.no_reuse else []
    common += ["--adopt-legacy"] if a.adopt_legacy else []
    jobs = []
    if "llm" in a.sections:  # usually the longest step, so start it first
        d = os.path.join(outdir, "llm")
        expected = os.path.join(d, "llm_report.md")
        if a.skip_existing and os.path.exists(expected):
            print("skip llm (exists)")
        else:
            jobs.append(("llm", ["--only", "llm", "--llms", ",".join(models)] + common, d, expected))
    bench_jobs = []
    for m in sorted(models, key=lambda m: -len(counts.get(m, {}))):  # models with the most benchmarks first
        d = os.path.join(outdir, slug(m))
        for only in onlys:
            expected = os.path.join(d, "report_%s.md" % only)
            if a.skip_existing and os.path.exists(expected):
                print("skip %s/%s (exists)" % (m, only))
                continue
            bench_jobs.append((onlys.index(only), ("%s/%s" % (m, only), ["--only", only, "--model", m] + common, d, expected)))
    jobs += [j for _, j in bench_jobs]  # e.g. 1.5B ablate -> cuda:0, 1.5B transfer -> cuda:1, ... then cached models

    try:
        errors = run_all(jobs, devices, n_jobs)
        if errors:
            sys.exit("\n".join(errors))
        for m in models:
            retire_legacy(os.path.join(outdir, slug(m)))

        h = write_report(a, outdir, models, counts, devices, n_jobs)
        if not a.no_readme:
            update_readme(outdir, h, a)
    finally:
        if os.path.exists(LOCK) and open(LOCK, encoding="utf-8").read().split()[:1] == [str(os.getpid())]:
            os.remove(LOCK)


if __name__ == "__main__":
    main()
