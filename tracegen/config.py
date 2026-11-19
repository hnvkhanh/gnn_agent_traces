"""Config loading: config.yaml deep-merged over DEFAULTS. The result is a plain nested dict."""
import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "suites": ["banking", "slack", "travel", "workspace"],
    "benchmark_version": "v1.2.2",
    "paths": {"out_dir": "data/tracegen"},
    "gpu": {
        "mode": "parallel", "generator_device": 0, "nli_device": 1, "agent_device": 1,
        "generator_url": "http://127.0.0.1:11436", "agent_url": "http://127.0.0.1:11435",
        "split_url": "http://127.0.0.1:11437", "split_devices": [0, 1],
        "manage_servers": True, "context_length": 16384, "ollama_models_dir": "auto", "reuse_generator_gpu_for_agents": True,
    },
    "generate": {"model": "qwen2.5:7b", "split_model": "qwen2.5:14b", "n": 5, "temperature": 0.9,
                 "base_seed": 0, "max_attempts": 3, "num_predict": 1024},
    "verify": {"entities": True, "nli_model": "microsoft/deberta-large-mnli", "nli_tau": 0.8, "nli_batch_size": 16,
               "judge": False, "judge_model": "gemma2:9b", "judge_system": None, "judge_url": None},
    "trace": {"agents": ["llama3.1:8b", "qwen2.5:7b"], "samples_per_prompt": 1, "temperature": 0.0,
              "sample_temperature": 0.7, "base_seed": 0, "run_timeout_s": 300, "request_timeout_s": 180,
              "concurrency": 2, "max_iters": 15, "include_originals": True},
    "attacks": {"enabled": False, "names": ["important_instructions"], "injection_tasks": None,
                "match_splits": True, "rows": "all",
                "search": {"enabled": False, "steps": 3, "candidates": 4, "islands": 2, "max_pairs": None,
                           "proposer_model": None, "max_format_retries": 5}},
    "build": {"dedup": False, "split_ratios": [0.7, 0.15, 0.15], "split_seed": 0},
}


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path="config.yaml", overrides=None):
    p = Path(path)
    user = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return merge(merge(DEFAULTS, user), overrides or {})


def out_path(cfg, name):
    p = Path(cfg["paths"]["out_dir"]) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def generator_model(cfg):
    return cfg["generate"]["split_model"] if cfg["gpu"]["mode"] == "split" else cfg["generate"]["model"]


def generator_url(cfg):
    return cfg["gpu"]["split_url"] if cfg["gpu"]["mode"] == "split" else cfg["gpu"]["generator_url"]


def agent_url(cfg):
    return cfg["gpu"]["split_url"] if cfg["gpu"]["mode"] == "split" else cfg["gpu"]["agent_url"]


def judge_url(cfg):
    return cfg["verify"]["judge_url"] or agent_url(cfg)


def nli_device(cfg):
    g = cfg["gpu"]
    return g["split_devices"][0] if g["mode"] == "split" else g["nli_device"]


def agent_specs(cfg):
    """Normalize trace.agents entries to {name, base_url|None, api_key_env|None}. base_url None = local Ollama."""
    out = []
    for a in cfg["trace"]["agents"]:
        a = {"name": a} if isinstance(a, str) else dict(a)
        a.setdefault("base_url", None)
        a.setdefault("api_key_env", None)
        out.append(a)
    return out
