"""python -m tracegen.agentlab {setup,extract,benign,generate,verify,trace,build,all} [options]

Agent traces on AgentLAB's tool-chaining track (https://github.com/TanqiuJiang/AgentLAB): local Ollama agents are
walked through the dataset's multi-turn attacks in the SHADE-Arena environments, next to benign tasks in the same
environments, on the original wording and on verified paraphrases.

    python -m tracegen.agentlab all              # everything; run the same command again to resume
    python -m tracegen.agentlab all --smoke      # 2 tasks per kind, 2 suites, 1 agent, into data/agentlab_smoke

Every stage is resumable: finished tasks, variants and runs are skipped (see tracegen/agentlab/stages.py).
"""
import argparse
import logging

from tracegen import config as C
from tracegen import progress
from tracegen.agentlab import config as LC

STAGES = ["setup", "extract", "benign", "generate", "verify", "trace", "build", "all"]


def trace_urls(cfg):
    g = cfg["gpu"]
    if g["mode"] == "split":
        return [g["split_url"]]
    # with adaptive turns the planner keeps the generator endpoint busy, so agents stay on their own GPU
    share = g["reuse_generator_gpu_for_agents"] and not cfg["attacks"]["adaptive_turns"]
    return [g["agent_url"]] + ([g["generator_url"]] if share else [])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tracegen.agentlab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config", default="config_agentlab.yaml")
    ap.add_argument("--suites", help="comma-separated: travel,banking,workspace,spam (default: config suites)")
    ap.add_argument("--limit", type=int, help="first K tasks per suite and kind (attack / main / generated)")
    ap.add_argument("--out", help="output directory (default: paths.out_dir)")
    ap.add_argument("--gpu-mode", choices=["parallel", "split"])
    ap.add_argument("--agents", help="comma-separated Ollama agent models (overrides trace.agents)")
    ap.add_argument("--variants", type=int, help="paraphrases per task (overrides generate.n)")
    ap.add_argument("--adaptive-turns", type=int, help="planner-written extra attack turns (overrides attacks.adaptive_turns)")
    ap.add_argument("--smoke", action="store_true",
                    help="small end-to-end run: --limit 2 --suites travel,banking --variants 2, the first agent only, "
                         "into data/agentlab_smoke (explicit options still win)")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging; also print every finished run")
    a = ap.parse_args(argv)

    over = {}
    if a.smoke:
        over = {"paths": {"out_dir": "data/agentlab_smoke"}, "generate": {"n": 2}}
        a.limit = 2 if a.limit is None else a.limit
        a.suites = a.suites or "travel,banking"
    if a.out:
        over.setdefault("paths", {})["out_dir"] = a.out
    if a.gpu_mode:
        over["gpu"] = {"mode": a.gpu_mode}
    if a.variants is not None:
        over.setdefault("generate", {})["n"] = a.variants
    if a.adaptive_turns is not None:
        over["attacks"] = {"adaptive_turns": a.adaptive_turns}
    cfg = LC.load(a.config, over)
    if a.agents:
        cfg["trace"]["agents"] = a.agents.split(",")
    elif a.smoke:
        cfg["trace"]["agents"] = cfg["trace"]["agents"][:1]
    suites = a.suites.split(",") if a.suites else cfg["suites"]
    bad = [s for s in suites if s not in LC.SUITES]
    if bad:
        ap.error("unknown suites %s (choose from %s)" % (bad, LC.SUITES))

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger("tracegen")
    root.setLevel(logging.DEBUG if a.verbose else logging.INFO)
    for h in (progress.ConsoleHandler(verbose=a.verbose),
              logging.FileHandler(C.out_path(cfg, "tracegen.log"), encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    for noisy in ("httpx", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root.info("tracegen.agentlab %s: suites=%s limit=%s mode=%s out=%s", a.stage, suites, a.limit, cfg["gpu"]["mode"],
              cfg["paths"]["out_dir"])

    from tracegen.agentlab import lab
    from tracegen.agentlab import stages as S
    lab.setup(LC.repo_dir(cfg), cfg["repo"]["url"], cfg["repo"]["commit"])
    if a.stage == "setup":
        return
    if a.stage == "extract":
        S.extract(cfg, suites, a.limit)
        return
    if a.stage == "build":
        S.build(cfg, S.originals(cfg, suites, a.limit))
        return

    from tracegen import ollama
    from tracegen import pipeline as P
    todo = STAGES[1:-1] if a.stage == "all" else [a.stage]
    with P.servers(cfg):
        if "extract" in todo:
            S.extract(cfg, suites, a.limit)
        if "benign" in todo:
            S.synthesize_benign(cfg, suites, a.limit)
        rows = S.originals(cfg, suites, a.limit)
        if "generate" in todo:
            S.generate(cfg, rows)
        if a.stage == "all" and not cfg["attacks"]["adaptive_turns"]:
            ollama.unload(C.generator_url(cfg), C.generator_model(cfg))   # frees the GPU for NLI and for agents
        if "verify" in todo:
            S.verify(cfg, rows)
        if "trace" in todo:
            S.trace(cfg, rows, urls=trace_urls(cfg))
        if "build" in todo:
            S.build(cfg, rows)


if __name__ == "__main__":
    main()
