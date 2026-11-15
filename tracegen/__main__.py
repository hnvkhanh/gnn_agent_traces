"""python -m tracegen {extract,generate,verify,trace,build,all} [options]"""
import argparse
import logging

from tracegen import config as C
from tracegen import gpu, progress

STAGES = ["extract", "generate", "verify", "trace", "build", "all"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tracegen", description=__doc__)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--suites", help="comma-separated (default: config suites)")
    ap.add_argument("--limit", type=int, help="first K user tasks per suite (smoke tests)")
    ap.add_argument("--out", help="output directory (default: paths.out_dir)")
    ap.add_argument("--gpu-mode", choices=["parallel", "split"])
    ap.add_argument("--judge", action=argparse.BooleanOptionalAction, default=None, help="Stage 5 LLM judge")
    ap.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--attacks", action=argparse.BooleanOptionalAction, default=None, help="Stage 6b attacked traces")
    ap.add_argument("--agents", help="comma-separated Ollama agent models (overrides trace.agents)")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging; also print every finished run")
    a = ap.parse_args(argv)

    over = {}
    if a.out:
        over["paths"] = {"out_dir": a.out}
    if a.gpu_mode:
        over["gpu"] = {"mode": a.gpu_mode}
    if a.attacks is not None:
        over["attacks"] = {"enabled": a.attacks}
    if a.agents:
        over["trace"] = {"agents": a.agents.split(",")}
    cfg = C.load(a.config, over)
    suites = a.suites.split(",") if a.suites else cfg["suites"]

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger("tracegen")
    root.setLevel(logging.DEBUG if a.verbose else logging.INFO)
    for h in (progress.ConsoleHandler(verbose=a.verbose),
              logging.FileHandler(C.out_path(cfg, "tracegen.log"), encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    for noisy in ("httpx", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root.info("tracegen %s: suites=%s limit=%s mode=%s out=%s", a.stage, suites, a.limit, cfg["gpu"]["mode"],
              cfg["paths"]["out_dir"])

    from tracegen.stages import build as B
    from tracegen.stages import generate as G
    if a.stage == "extract":
        G.extract(cfg, suites, a.limit)
        return
    if a.stage == "build":
        B.build(cfg, suites, do_dedup=a.dedup, limit=a.limit)
        return

    from tracegen import pipeline as P
    with P.servers(cfg):
        if a.stage == "all":
            P.run_all(cfg, suites, a.limit, judge=a.judge, dedup=a.dedup)
        elif a.stage == "generate":
            gpus = gpu.probe()
            from tracegen import ollama
            url, gm = C.generator_url(cfg), C.generator_model(cfg)
            devs = cfg["gpu"]["split_devices"] if cfg["gpu"]["mode"] == "split" else [cfg["gpu"]["generator_device"]]
            if gm not in ollama.loaded(url):
                gpu.check_fit(gpus, devs, [("generator " + gm, gpu.need_bytes(ollama.model_info(url, gm)["size"]))])
            G.generate(cfg, G.originals(cfg, suites, a.limit))
        elif a.stage == "verify":
            from tracegen.stages.verify import verify
            verify(cfg, suites, judge=a.judge, limit=a.limit)
        elif a.stage == "trace":
            from tracegen.stages import trace as T
            g = cfg["gpu"]
            urls = [g["split_url"]] if g["mode"] == "split" else \
                [g["agent_url"]] + ([g["generator_url"]] if g["reuse_generator_gpu_for_agents"] else [])
            T.trace(cfg, suites, a.limit, urls=urls)


if __name__ == "__main__":
    main()
