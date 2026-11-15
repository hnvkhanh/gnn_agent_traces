"""Minimal Ollama HTTP client: chat with seed/temperature, model metadata, unload, and private per-GPU servers."""
import logging
import os
import subprocess
import time
from pathlib import Path

import requests

log = logging.getLogger("tracegen")
SYSTEM_STORE = "/usr/share/ollama/.ollama/models"


def chat(url, model, messages, temperature=0.0, seed=0, num_predict=1024, timeout=300, fmt=None, keep_alive=None):
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict}}
    if fmt is not None:
        body["format"] = fmt
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    last = None
    for attempt in range(4):  # transient 500s under load: back off and retry
        try:
            r = requests.post(url.rstrip("/") + "/api/chat", json=body, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            return j["message"]["content"], {"tokens_in": j.get("prompt_eval_count"), "tokens_out": j.get("eval_count")}
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(8, 2 ** attempt))
    raise RuntimeError("ollama chat failed on %s (%s): %r" % (url, model, last))


def up(url, timeout=2):
    try:
        requests.get(url.rstrip("/") + "/api/version", timeout=timeout).raise_for_status()
        return True
    except Exception:  # noqa: BLE001
        return False


def tags(url):
    r = requests.get(url.rstrip("/") + "/api/tags", timeout=10)
    r.raise_for_status()
    return {m["name"]: m for m in r.json().get("models", [])}


def model_info(url, model):
    """{'tag': <name>@<digest12>, 'size': bytes} or raises with a pull hint."""
    t = tags(url)
    m = t.get(model) or t.get(model + ":latest")
    if m is None:
        raise SystemExit("model %r is not available on %s. Run: ollama pull %s" % (model, url, model))
    return {"tag": "%s@%s" % (m["name"], m.get("digest", "")[:12]), "size": int(m.get("size", 0))}


def loaded(url):
    try:
        r = requests.get(url.rstrip("/") + "/api/ps", timeout=5)
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:  # noqa: BLE001
        return set()


def unload(url, model):
    try:
        requests.post(url.rstrip("/") + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
        log.info("unloaded %s from %s", model, url)
    except Exception as e:  # noqa: BLE001
        log.warning("could not unload %s from %s: %r", model, url, e)


def models_dir(setting):
    if setting and setting != "auto":
        return setting
    if os.access(SYSTEM_STORE + "/manifests", os.R_OK):
        return SYSTEM_STORE
    return str(Path.home() / ".ollama" / "models")


class Servers:
    """Start a private `ollama serve` per (url, devices) unless something already answers on that url.
    Servers that this object started are stopped on exit. Servers it found already running are left alone."""

    def __init__(self, specs, models, log_dir, parallel=2, context_length=16384):
        self.specs, self.models, self.log_dir, self.parallel, self.procs = specs, models, Path(log_dir), parallel, []
        self.ctx = context_length

    def __enter__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for url, devices in self.specs:
            if up(url):
                log.info("using already-running Ollama at %s (GPU pinning not verified)", url)
                continue
            hostport = url.split("://", 1)[-1].rstrip("/")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, devices)), OLLAMA_HOST=hostport,
                       OLLAMA_MODELS=self.models, OLLAMA_NUM_PARALLEL=str(self.parallel), OLLAMA_NOPRUNE="1",
                       OLLAMA_KEEP_ALIVE="30m", OLLAMA_SCHED_SPREAD="1" if len(devices) > 1 else "0",
                       # AgentDojo tool outputs overflow the default context; one model per instance avoids
                       # spilling to CPU when the runner switches agents
                       OLLAMA_CONTEXT_LENGTH=str(self.ctx), OLLAMA_MAX_LOADED_MODELS="1",
                       OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE="q8_0")
            f = open(self.log_dir / ("ollama_%s.log" % hostport.replace(":", "_")), "a", encoding="utf-8")
            p = subprocess.Popen(["ollama", "serve"], env=env, stdout=f, stderr=subprocess.STDOUT)
            self.procs.append((p, url, devices))
        for p, url, devices in self.procs:
            for _ in range(120):
                if p.poll() is not None:
                    raise SystemExit("ollama serve for GPUs %s exited with code %s; see %s"
                                     % (devices, p.returncode, self.log_dir))
                if up(url):
                    break
                time.sleep(0.5)
            else:
                raise SystemExit("ollama serve for GPUs %s did not come up at %s" % (devices, url))
            log.info("started Ollama on GPUs %s at %s (models: %s)", devices, url, self.models)
        return self

    def __exit__(self, *exc):
        for p, _, _ in self.procs:
            p.terminate()
        for p, _, _ in self.procs:
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                p.kill()
