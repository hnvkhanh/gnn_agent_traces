"""Run the LLM inside the mock sandbox over benchmark scenarios and write traces as JSONL.

Sharding (used by scripts/collect_traces.py --gpus): --shard j --num-shards K runs only scenarios with i % K == j, so K
processes (e.g. one group per GPU-pinned Ollama server, --host) together produce exactly the serial set of traces.
"""
import argparse
import json
import os

from .llm.ollama_client import HOST, OllamaChat, list_models
from .benchmarks import agentdojo, agentpoison, asb, injecagent
from .sandbox.env import run_scenario

BENCH = {"asb": (asb.scenarios, "data/asb"), "injecagent": (injecagent.scenarios, "data/injecagent"),
         "agentdojo": (agentdojo.scenarios, "data/agentdojo"), "agentpoison": (agentpoison.scenarios, "data/agentpoison")}


def trace_ids(path):
    if not path or not os.path.exists(path):
        return set()
    return {json.loads(l)["trace_id"] for l in open(path, encoding="utf-8") if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=list(BENCH), default="asb")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default=None, help="default: first locally installed ollama model")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benign-frac", type=float, default=0.25)
    ap.add_argument("--no-replay", action="store_true", help="do not append synthetic attacker-goal steps")
    ap.add_argument("--host", default=HOST, help="Ollama server URL")
    ap.add_argument("--shard", type=int, default=0, help="run only scenarios with index %% num-shards == shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--skip-file", default=None, help="also skip trace ids already in this file (the merged output)")
    a = ap.parse_args()

    if not 0 <= a.shard < a.num_shards:
        ap.error("need 0 <= --shard < --num-shards")
    model = a.model or list_models(a.host)[0]
    out = a.out or "traces/%s__%s.jsonl" % (a.bench, model.replace(":", "_"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = trace_ids(out) | trace_ids(a.skip_file)
    gen, data_dir = BENCH[a.bench]
    llm = OllamaChat(model, host=a.host)
    scs = gen(data_dir, a.n, a.seed, a.benign_frac)
    with open(out, "a", encoding="utf-8", newline="\n") as f:
        for i, sc in enumerate(scs):
            if i % a.num_shards != a.shard:
                continue
            tid = "%s|%s|%d|%d" % (a.bench, model, a.seed, i)
            if tid in done:
                continue
            tr = run_scenario(sc, llm, replay=not a.no_replay, trace_id=tid, model=model)
            if not tr["events"]:  # no action was taken: nothing to label, do not store
                print(i, "skipped (no actions)", flush=True)
                continue
            f.write(json.dumps(tr) + "\n")
            f.flush()
            print(i, sc.mode, sc.style, [e["label"] for e in tr["events"]], flush=True)


if __name__ == "__main__":
    main()
