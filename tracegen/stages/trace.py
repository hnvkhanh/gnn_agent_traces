"""Stage 6 (benign) and 6b (attacked) trace generation in AgentDojo, original prompts and verified variants."""
import collections
import logging
import queue
import threading
import time

from tracegen import config as C
from tracegen import gpu, io, ollama, progress
from tracegen.dojo import load_suite, run_one

log = logging.getLogger("tracegen")


def prompt_rows(cfg, suites, limit=None):
    """Originals (is_original) followed by verified variants, for the selected suites/limit."""
    from tracegen.stages.generate import originals
    origs = originals(cfg, suites, limit)
    keep = {(r["suite"], r["task_id"]) for r in origs}
    rows = []
    if cfg["trace"]["include_originals"]:
        rows += [orig_row(r) for r in origs]
    rows += [variant_row(v) for v in io.read_jsonl(C.out_path(cfg, "verified_variants.jsonl"))
             if (v["suite"], v["task_id"]) in keep]
    return rows


def orig_row(r):
    return {"suite": r["suite"], "task_id": r["task_id"], "variant_id": "%s/%s/orig" % (r["suite"], r["task_id"]),
            "is_original": True, "prompt": r["prompt"]}


def variant_row(v):
    return {"suite": v["suite"], "task_id": v["task_id"], "variant_id": v["variant_id"], "is_original": False,
            "prompt": v["paraphrase"]}


def seeds(cfg):
    t = cfg["trace"]
    k = max(1, int(t["samples_per_prompt"]))
    temp = t["temperature"] if k == 1 else t["sample_temperature"]
    return [(t["base_seed"] + i, temp) for i in range(k)]


def plan_jobs(cfg, rows, agents, done, attacks=None):
    """Every (row × agent × seed) benign job, plus attack jobs when enabled, minus keys already in `done`."""
    a_cfg = cfg["attacks"]
    attacks = a_cfg["enabled"] if attacks is None else attacks
    inj_ids = {}
    jobs = []
    task_split = inj_split = None
    if attacks and a_cfg.get("match_splits", True):
        # Only pair a user task with injection tasks of the same split (tracegen.stages.build assigns the task splits
        # the same way), so train / val / test stay attack-disjoint without discarding runs later.
        from tracegen.stages.build import assign_injection_splits, assign_splits
        b = cfg["build"]
        task_split = assign_splits([(r["suite"], r["task_id"]) for r in rows], b["split_ratios"], b["split_seed"])
        suites = sorted({r["suite"] for r in rows})
        inj_split = assign_injection_splits(
            {s: a_cfg["injection_tasks"] or sorted(load_suite(cfg["benchmark_version"], s).injection_tasks)
             for s in suites}, b["split_ratios"], b["split_seed"])
    for agent in agents:
        for r in rows:
            for seed, temp in seeds(cfg):
                base = dict(r, agent=agent, agent_model=agent["name"], seed=seed, temperature=temp)
                variants = [dict(base, mode="benign", injection_task_id=None, attack_name=None)]
                if attacks and (a_cfg.get("rows", "all") == "all" or r["is_original"]):
                    if r["suite"] not in inj_ids:
                        ids = a_cfg["injection_tasks"] or sorted(
                            load_suite(cfg["benchmark_version"], r["suite"]).injection_tasks)
                        inj_ids[r["suite"]] = ids
                    ids = inj_ids[r["suite"]]
                    if task_split is not None:
                        ids = [i for i in ids if inj_split.get((r["suite"], i)) == task_split[(r["suite"], r["task_id"])]]
                    variants += [dict(base, mode="attack", injection_task_id=i, attack_name=an)
                                 for an in a_cfg["names"] for i in ids]
                jobs += [j for j in variants if io.trace_key(j) not in done]
    return jobs


class TraceRunner:
    """Worker threads per endpoint. An endpoint drains its agents one at a time (per-agent queues), so a GPU
    keeps one model loaded instead of thrashing between models. Several endpoints may serve the same agent."""

    def __init__(self, cfg, routes, desc="trace"):
        """routes: {endpoint_url_or_'api': [agent_name, ...]} in priority order. desc labels the progress bar."""
        self.cfg, self.routes = cfg, routes
        self.out = C.out_path(cfg, "traces.jsonl")
        self.queues = {a: queue.Queue() for agents in routes.values() for a in agents}
        self.closed, self.threads, self.tags, self.n_done = False, [], {}, 0
        self.lock = threading.Lock()
        self.stats = collections.Counter()
        self.bar = progress.bar(desc)
        self.suites = {}
        for url, agents in routes.items():
            for _ in range(cfg["trace"]["concurrency"]):
                th = threading.Thread(target=self._work, args=(url, agents), daemon=True)
                th.start()
                self.threads.append(th)

    def submit(self, job):
        if job["agent_model"] not in self.queues:
            raise KeyError("agent %s has no route" % job["agent_model"])
        with self.lock:
            self.bar.total += 1
            self.bar.refresh()
        self.queues[job["agent_model"]].put(job)

    def close(self):
        self.closed = True
        for th in self.threads:
            th.join()
        self.bar.close()
        return self.n_done

    def _suite(self, name):
        with self.lock:
            if name not in self.suites:
                self.suites[name] = load_suite(self.cfg["benchmark_version"], name)
            return self.suites[name]

    def _tag(self, url, agent):
        key = (url, agent["name"])
        if key not in self.tags:
            self.tags[key] = agent["name"] if agent.get("base_url") else ollama.model_info(url, agent["name"])["tag"]
        return self.tags[key]

    def _work(self, url, agents):
        while True:
            job = None
            for a in agents:
                try:
                    job = self.queues[a].get_nowait()
                    break
                except queue.Empty:
                    continue
            if job is None:
                if self.closed:
                    return
                time.sleep(0.2)
                continue
            try:
                self._run(url, job)
            except Exception:  # noqa: BLE001  (run_one never raises; this guards bookkeeping bugs)
                log.exception("trace job %s %s failed outside run_one", job["variant_id"], job["agent_model"])
            finally:
                with self.lock:
                    self.bar.update(1)

    def _run(self, url, job):
        t = self.cfg["trace"]
        suite = self._suite(job["suite"])
        agent = job["agent"]
        inj = suite.get_injection_task_by_id(job["injection_task_id"]) if job["mode"] == "attack" else None
        res = run_one(suite, suite.user_tasks[job["task_id"]], job["prompt"], agent, url, seed=job["seed"],
                      temperature=job["temperature"], injection_task=inj, attack_name=job["attack_name"],
                      run_timeout=t["run_timeout_s"], request_timeout=t["request_timeout_s"], max_iters=t["max_iters"])
        try:
            tag = self._tag(url, agent)
        except BaseException as e:  # noqa: BLE001
            tag = "unknown (%r)" % e
        rec = {"trace_id": res.pop("trace_id"), "mode": job["mode"], "suite": job["suite"], "task_id": job["task_id"],
               "variant_id": job["variant_id"], "is_original": job["is_original"], "prompt": job["prompt"],
               "agent_model": agent["name"], "agent_model_tag": tag, "seed": job["seed"],
               "temperature": job["temperature"], "endpoint": url}
        if job["mode"] == "attack":
            rec.update(injection_task_id=job["injection_task_id"], attack_name=job["attack_name"])
        rec.update(res)
        io.append_jsonl(self.out, rec)
        with self.lock:
            self.n_done += 1
            n = self.n_done
            s = self.stats
            s["runs"] += 1
            s["utility"] += bool(rec["utility"])
            s["error"] += bool(rec["error"])
            if job["mode"] == "attack":
                s["attack"] += 1
                s["security"] += bool(rec["security"])
            self.bar.set_postfix_str(progress.run_stats(s), refresh=False)
        log.info("trace #%d %s %s %s seed=%d utility=%s%s steps=%d %.1fs%s", n, job["mode"], job["variant_id"],
                 agent["name"], job["seed"], rec["utility"],
                 " security=%s" % rec["security"] if job["mode"] == "attack" else "", rec["n_steps"],
                 rec["latency_s"], " ERROR " + rec["error"][:160] if rec["error"] else "", extra={"per_run": True})


def routes_for(cfg, urls):
    """Assign local agents to Ollama endpoints round-robin (one agent and two endpoints = both serve it).
    API agents get their own 'api' route."""
    agents = C.agent_specs(cfg)
    local = [a["name"] for a in agents if not a.get("base_url")]
    routes = {u: [] for u in urls}
    if local:
        if len(local) >= len(urls):
            for i, a in enumerate(local):
                routes[urls[i % len(urls)]].append(a)
        else:
            for i, u in enumerate(urls):
                routes[u] = local[i % len(local):] + local[:i % len(local)]
    api = [a["name"] for a in agents if a.get("base_url")]
    if api:
        routes["api"] = api
    return {u: a for u, a in routes.items() if a}


def check_agent_fit(cfg, routes, gpus, extra=None):
    """Fit check per endpoint: the largest agent routed there (Ollama keeps one loaded) plus `extra` (e.g. NLI)."""
    g = cfg["gpu"]
    dev_of = {g["agent_url"]: [g["agent_device"]], g["generator_url"]: [g["generator_device"]],
              g["split_url"]: list(g["split_devices"])}
    for url, agents in routes.items():
        if url == "api":
            continue
        resident = ollama.loaded(url)
        sizes = [(a, ollama.model_info(url, a)["size"]) for a in agents if a not in resident]
        items = [("agent " + max(sizes, key=lambda x: x[1])[0], gpu.need_bytes(max(s for _, s in sizes)))] if sizes else []
        items += (extra or {}).get(url, [])
        if items and url in dev_of:
            gpu.check_fit(gpus, dev_of[url], items)


def trace(cfg, suites, limit=None, urls=None):
    rows = prompt_rows(cfg, suites, limit)
    done = io.keys(C.out_path(cfg, "traces.jsonl"), io.trace_key)
    jobs = plan_jobs(cfg, rows, C.agent_specs(cfg), done)
    log.info("trace: %d prompts, %d jobs to run (%d traces already recorded)", len(rows), len(jobs), len(done))
    if not jobs:
        return 0
    routes = routes_for(cfg, urls or [C.agent_url(cfg)])
    check_agent_fit(cfg, routes, gpu.probe())
    runner = TraceRunner(cfg, routes)
    for j in sorted(jobs, key=lambda j: (not j["is_original"], j["mode"] != "benign")):
        runner.submit(j)
    return runner.close()
