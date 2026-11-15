"""`all`: parallel mode streams generate (GPU 0) → verify → trace (GPU 1); split mode runs the stages in order."""
import collections
import logging
import queue
import threading

from tracegen import config as C
from tracegen import gpu, io, ollama
from tracegen.stages import build as B
from tracegen.stages import generate as G
from tracegen.stages import trace as T
from tracegen.stages.verify import Verifier, verify

log = logging.getLogger("tracegen")


def server_specs(cfg):
    g = cfg["gpu"]
    if g["mode"] == "split":
        return [(g["split_url"], list(g["split_devices"]))]
    return [(g["generator_url"], [g["generator_device"]]), (g["agent_url"], [g["agent_device"]])]


def servers(cfg):
    g = cfg["gpu"]
    if not g["manage_servers"]:
        for url, dev in server_specs(cfg):
            if not ollama.up(url):
                raise SystemExit("no Ollama at %s. Start it with: CUDA_VISIBLE_DEVICES=%s OLLAMA_HOST=%s ollama serve"
                                 % (url, ",".join(map(str, dev)), url.split("://")[-1]))
        return _Null()
    return ollama.Servers(server_specs(cfg), ollama.models_dir(g["ollama_models_dir"]),
                          C.out_path(cfg, "logs/.keep").parent, parallel=cfg["trace"]["concurrency"],
                          context_length=g.get("context_length", 16384))


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _local_agents(cfg):
    return [a for a in C.agent_specs(cfg) if not a.get("base_url")]


def run_all(cfg, suites, limit=None, judge=None, dedup=None):
    gpus = gpu.probe()
    rows = G.extract(cfg, suites, limit)
    if cfg["gpu"]["mode"] == "split":
        run_split(cfg, gpus, suites, rows, limit, judge)
    else:
        run_parallel(cfg, gpus, suites, rows, limit, judge)
    return B.build(cfg, suites, do_dedup=dedup, limit=limit)


def run_split(cfg, gpus, suites, rows, limit, judge):
    url, gm = cfg["gpu"]["split_url"], C.generator_model(cfg)
    devs = list(cfg["gpu"]["split_devices"])
    if any((r["suite"], r["task_id"]) not in io.keys(C.out_path(cfg, "generate_log.jsonl"),
                                                     lambda x: (x["suite"], x["task_id"])) for r in rows):
        if gm not in ollama.loaded(url):
            gpu.check_fit(gpus, devs, [("generator " + gm, gpu.need_bytes(ollama.model_info(url, gm)["size"]))])
        G.generate(cfg, rows)
    ollama.unload(url, gm)
    verify(cfg, suites, judge=judge, limit=limit)     # NLI on cuda:<split_devices[0]>, freed afterwards
    if judge if judge is not None else cfg["verify"]["judge"]:
        ollama.unload(C.judge_url(cfg), cfg["verify"]["judge_model"])
    T.trace(cfg, suites, limit, urls=[url])


def run_parallel(cfg, gpus, suites, rows, limit, judge):
    g, v = cfg["gpu"], cfg["verify"]
    gen_url, ag_url, gm = g["generator_url"], g["agent_url"], C.generator_model(cfg)
    judge = v["judge"] if judge is None else judge
    gen_done = io.keys(C.out_path(cfg, "generate_log.jsonl"), lambda x: (x["suite"], x["task_id"]))
    todo_gen = [r for r in rows if (r["suite"], r["task_id"]) not in gen_done]
    local = _local_agents(cfg)
    first = local[:1]   # the streaming phase uses one agent on GPU 1; the rest run in the batch phase

    # fit checks before anything is loaded
    if todo_gen and gm not in ollama.loaded(gen_url):
        gpu.check_fit(gpus, [g["generator_device"]], [("generator " + gm, gpu.need_bytes(ollama.model_info(gen_url, gm)["size"]))])
    extra = [("NLI " + v["nli_model"], gpu.NLI_BYTES)]
    if judge and C.judge_url(cfg) == ag_url and (not first or v["judge_model"] != first[0]["name"]):
        log.warning("judge %s differs from the streaming agent; Ollama swaps them on GPU %s (slower)",
                    v["judge_model"], g["agent_device"])
    T.check_agent_fit(cfg, {ag_url: [a["name"] for a in first]}, gpus, extra={ag_url: extra})

    done = io.keys(C.out_path(cfg, "traces.jsonl"), io.trace_key)
    runner = T.TraceRunner(cfg, {ag_url: [a["name"] for a in first]}, desc="trace (streamed)") if first else None

    def submit(prows):
        if runner is None:
            return
        for j in T.plan_jobs(cfg, prows, first, done, attacks=False):
            done.add(io.trace_key(j))
            runner.submit(j)

    # originals first, then variants verified in an earlier (interrupted) run
    submit([T.orig_row(r) for r in rows] if cfg["trace"]["include_originals"] else [])
    keep = {(r["suite"], r["task_id"]) for r in rows}
    submit([T.variant_row(x) for x in io.read_jsonl(C.out_path(cfg, "verified_variants.jsonl"))
            if (x["suite"], x["task_id"]) in keep])

    ver = Verifier(cfg, judge=judge)
    q = queue.Queue(maxsize=64)
    pending = collections.defaultdict(list)   # candidates from an earlier run that were never verified
    for c in io.read_jsonl(C.out_path(cfg, "candidates.jsonl")):
        if (c["suite"], c["task_id"]) in keep and c["variant_id"] not in ver.done:
            pending[(c["suite"], c["task_id"])].append(c)
    gen_err = []

    def producer():
        try:
            for batch in pending.values():
                q.put(batch)
            if todo_gen:
                G.generate(cfg, todo_gen, on_candidates=lambda cs: cs and q.put(cs))
        except BaseException as e:  # noqa: BLE001
            gen_err.append(e)
        finally:
            q.put(None)
            ollama.unload(gen_url, gm)   # GPU 0 is free for agents from here on

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    try:
        while True:
            batch = q.get()
            if batch is None:
                break
            kept = ver.run(batch)
            submit([T.variant_row(x) for x in kept])
            log.info("verify %s/%s: %d/%d kept", batch[0]["suite"], batch[0]["task_id"], len(kept), len(batch))
    finally:
        ver.close()
        th.join()
        if runner is not None:
            log.info("waiting for streamed trace runs to finish")
            runner.close()
    if gen_err:
        raise gen_err[0]
    if judge and C.judge_url(cfg) == ag_url and (not first or v["judge_model"] != first[0]["name"]):
        ollama.unload(ag_url, v["judge_model"])
    # batch phase: remaining agents, extra seeds, attacks, anything missed; GPU 0 joins when allowed
    urls = [ag_url] + ([gen_url] if g["reuse_generator_gpu_for_agents"] else [])
    T.trace(cfg, suites, limit, urls=urls)
