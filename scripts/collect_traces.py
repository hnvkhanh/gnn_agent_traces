"""Generate traces with Ollama for every LLM x benchmark (the only step that calls an LLM).

    python scripts/collect_traces.py                                  # default models, 200 scenarios per benchmark
    python scripts/collect_traces.py --models qwen2.5:7b --benches asb --n 50
    python scripts/collect_traces.py --dry-run                        # print the commands only

Each pair runs `python -m src.collect` and writes traces/<bench>__<model_slug>.jsonl, logging to
traces/<bench>__<model_slug>.log. Collection is resumable: trace IDs already in a file are skipped,
so re-running after an interruption continues where it stopped. Progress goes to traces/collect.status.

Several GPUs in parallel (--gpus):
    python scripts/collect_traces.py --models qwen2.5:1.5b --benches agentpoison --n 400 --gpus 0,1 --workers-per-gpu 4
starts one private `ollama serve` per GPU (CUDA_VISIBLE_DEVICES=<gpu>, port --port-base + k, OLLAMA_NUM_PARALLEL =
workers, model store --ollama-models, default ~/.ollama/models, pulled once if missing) and runs gpus x workers shards of
`src.collect` at once (scenario i goes to shard i % K, shard j to GPU j % gpus). Shards write traces/.shards/ and are
merged into the final file (deduplicated, sorted by scenario index) when all of them succeed; the servers are stopped
at the end. An Ollama already running (e.g. the system service on :11434) is never touched.
Run scripts/run_experiments.py afterwards.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:14b", "llama3.1:8b", "gemma2:9b"]
BENCHES = ["asb", "injecagent", "agentdojo", "agentpoison"]
SHARD_DIR = os.path.join(ROOT, "traces", ".shards")


def slug(m):
    return m.replace(":", "_")


def scenario_index(trace_id):
    """trace ids are '<bench>|<model>|<seed>|<i>'"""
    parts = trace_id.split("|")
    return int(parts[-2]), int(parts[-1])


def merge(final, shard_files):
    """Union of `final` and the shard files, first occurrence of a trace_id wins, sorted by (seed, scenario index).
    Written atomically; returns the number of traces in the merged file."""
    rows, seen = [], set()
    for p in [final] + list(shard_files):
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            if line.strip():
                tid = json.loads(line)["trace_id"]
                if tid not in seen:
                    seen.add(tid)
                    rows.append((scenario_index(tid), line.rstrip("\n")))
    rows.sort(key=lambda r: r[0])
    tmp = final + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for _, line in rows:
            f.write(line + "\n")
    os.replace(tmp, final)
    return len(rows)


def http_json(url, body=None, timeout=10):
    import requests
    r = requests.post(url, json=body, timeout=timeout) if body is not None else requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def start_servers(gpus, port_base, workers, models_dir, log_dir):
    procs = []
    for k, gpu in enumerate(gpus):
        host = "http://127.0.0.1:%d" % (port_base + k)
        try:
            http_json(host + "/api/tags", timeout=2)
            raise SystemExit("something already listens on %s; pick another --port-base" % host)
        except SystemExit:
            raise
        except Exception:
            pass
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OLLAMA_HOST="127.0.0.1:%d" % (port_base + k),
                   OLLAMA_NUM_PARALLEL=str(workers), OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_MODELS=models_dir,
                   OLLAMA_KEEP_ALIVE="30m")
        log = open(os.path.join(log_dir, "ollama_gpu%s.log" % gpu), "a", encoding="utf-8")
        procs.append((subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT), host, gpu))
    for p, host, gpu in procs:
        for _ in range(120):
            if p.poll() is not None:
                raise SystemExit("ollama serve for GPU %s exited (code %s), see traces/.shards/ollama_gpu%s.log"
                                 % (gpu, p.returncode, gpu))
            try:
                http_json(host + "/api/tags", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("ollama serve for GPU %s did not come up" % gpu)
        print("ollama for GPU %s at %s" % (gpu, host), flush=True)
    return procs


def stop_servers(procs):
    for p, _, _ in procs:
        p.terminate()
    for p, _, _ in procs:
        try:
            p.wait(15)
        except subprocess.TimeoutExpired:
            p.kill()


def ensure_model(host, model):
    names = [m["name"] for m in http_json(host + "/api/tags").get("models", [])]
    if model not in names:
        print("pulling %s into the shared model store ..." % model, flush=True)
        http_json(host + "/api/pull", {"name": model, "stream": False}, timeout=3600)


def collect_parallel(m, b, a, hosts, env):
    """Run K = len(hosts) * workers shards of src.collect at once, then merge them into the final trace file."""
    final = os.path.join(ROOT, "traces", "%s__%s.jsonl" % (b, slug(m)))
    log = os.path.join(ROOT, "traces", "%s__%s.log" % (b, slug(m)))
    K = len(hosts) * a.workers_per_gpu
    shards, procs = [], []
    for j in range(K):
        out = os.path.join(SHARD_DIR, "%s__%s.%d.jsonl" % (b, slug(m), j))
        shards.append(out)
        cmd = [sys.executable, "-m", "src.collect", "--bench", b, "--model", m, "--n", str(a.n), "--seed", str(a.seed),
               "--host", hosts[j % len(hosts)], "--shard", str(j), "--num-shards", str(K), "--out", out,
               "--skip-file", final]
        f = open(out[:-len(".jsonl")] + ".log", "a", encoding="utf-8", newline="\n")
        f.write("\n=== %s %s\n" % (datetime.datetime.now().isoformat(timespec="seconds"), " ".join(cmd)))
        f.flush()
        procs.append((subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT), f))
    rcs = []
    for p, f in procs:
        rcs.append(p.wait())
        f.close()
    with open(log, "a", encoding="utf-8", newline="\n") as lf:
        for out in shards:
            sl = out[:-len(".jsonl")] + ".log"
            if os.path.exists(sl):
                lf.write(open(sl, encoding="utf-8").read())
    if any(rcs):
        print("    %d shard(s) failed, kept in traces/.shards for resume; see %s" % (sum(1 for r in rcs if r), log),
              flush=True)
        return False
    n = merge(final, shards)
    for out in shards:
        for p in (out, out[:-len(".jsonl")] + ".log"):
            if os.path.exists(p):
                os.remove(p)
    print("    merged %d traces into %s" % (n, os.path.relpath(final, ROOT)), flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated Ollama model names")
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--n", type=int, default=200, help="scenarios per benchmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default="", help="comma-separated GPU indices: one private Ollama server per GPU")
    ap.add_argument("--workers-per-gpu", type=int, default=4, help="parallel shards (and OLLAMA_NUM_PARALLEL) per GPU")
    ap.add_argument("--port-base", type=int, default=11435)
    ap.add_argument("--ollama-models", default=os.path.join(os.path.expanduser("~"), ".ollama", "models"),
                    help="model store shared by the per-GPU servers")
    a = ap.parse_args()

    status = os.path.join(ROOT, "traces", "collect.status")
    os.makedirs(os.path.dirname(status), exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    models = [x for x in a.models.split(",") if x]
    benches = [x for x in a.benches.split(",") if x]
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    failed, servers, hosts = [], [], []
    try:
        if gpus and not a.dry_run:
            os.makedirs(SHARD_DIR, exist_ok=True)
            servers = start_servers(gpus, a.port_base, a.workers_per_gpu, a.ollama_models, SHARD_DIR)
            hosts = [h for _, h, _ in servers]
        for m in models:
            if hosts:
                ensure_model(hosts[0], m)
            for b in benches:
                log = os.path.join(ROOT, "traces", "%s__%s.log" % (b, slug(m)))
                print("=== %s / %s -> %s" % (m, b, os.path.relpath(log, ROOT)), flush=True)
                if gpus:
                    if a.dry_run:
                        print("    %d shards over GPUs %s" % (len(gpus) * a.workers_per_gpu, ",".join(gpus)))
                        continue
                    if not collect_parallel(m, b, a, hosts, env):
                        failed.append("%s/%s" % (m, b))
                    continue
                cmd = [sys.executable, "-m", "src.collect", "--bench", b, "--model", m, "--n", str(a.n), "--seed", str(a.seed)]
                if a.dry_run:
                    print("   ", " ".join(cmd))
                    continue
                with open(log, "a", encoding="utf-8", newline="\n") as f:
                    f.write("\n=== %s %s\n" % (datetime.datetime.now().isoformat(timespec="seconds"), " ".join(cmd)))
                    f.flush()
                    rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    print("    failed (exit %d), see the log" % rc, flush=True)
                    failed.append("%s/%s" % (m, b))
            if not a.dry_run:
                with open(status, "a", encoding="utf-8", newline="\n") as f:
                    f.write("finished %s\n" % m)
    finally:
        stop_servers(servers)
    if failed:
        sys.exit("failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
