"""Config for `python -m tracegen.agentlab`: config_agentlab.yaml deep-merged over DEFAULTS.

The gpu / generate / verify / trace / build sections have the same keys as tracegen's config.yaml, so the helpers in
tracegen.config (out_path, generator_url, agent_specs, ...) and tracegen.pipeline.servers work on this config too.
"""
from pathlib import Path

import yaml

from tracegen import config as C

SUITES = ["travel", "banking", "workspace", "spam"]

DEFAULTS = C.merge(C.DEFAULTS, {
    "suites": SUITES,
    "paths": {"out_dir": "data/agentlab"},
    "repo": {
        "dir": "external/AgentLAB",
        "url": "https://github.com/TanqiuJiang/AgentLAB",
        "commit": "36f58e60c36bbd6d5b8e61d50d7db7d9ea7258d7",   # null = the default branch head
        "dataset": "data/filtered_top_200_attacks.json",
    },
    "attacks": {
        "enabled": True,
        "rows": "all",              # all = originals + verified variants | originals
        # Extra user turns written by a local planner model after the scripted turns, while the tool chain is not
        # complete (AgentLAB's adaptive planner). 0 = replay the dataset's turns only.
        "adaptive_turns": 0,
        "planner_model": None,      # null = the generator model
        "planner_temperature": 0.3,
    },
    "benign": {
        "enabled": True,
        "shade_main_tasks": True,   # the benign main tasks of SHADE-Arena's task pairs
        "per_suite": 40,            # benign tasks to synthesize per environment (fewer survive deduplication)
        "batch": 8,                 # tasks asked for per generator call
        "max_turns": 3,
        "temperature": 0.9,
        "num_predict": 2048,
    },
    "generate": {"n": 3},
    "trace": {"max_agent_rounds": 10, "max_result_chars": 6000, "run_timeout_s": 600},
})
for _k in ("benchmark_version",):
    DEFAULTS.pop(_k, None)


def load(path="config_agentlab.yaml", overrides=None):
    p = Path(path)
    user = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return C.merge(C.merge(DEFAULTS, user or {}), overrides or {})


def repo_dir(cfg):
    return Path(cfg["repo"]["dir"]).resolve()


def planner_model(cfg):
    return cfg["attacks"]["planner_model"] or C.generator_model(cfg)
