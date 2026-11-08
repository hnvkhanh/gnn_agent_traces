"""Run the LLM inside the mock sandbox over ASB tasks x attacks and write traces as JSONL."""
import argparse
import json
import os

from .llm.ollama_client import OllamaChat, list_models
from .benchmarks import agentdojo, asb, injecagent
from .sandbox.env import run_scenario

BENCH = {"asb": (asb.scenarios, "data/asb"), "injecagent": (injecagent.scenarios, "data/injecagent"),
         "agentdojo": (agentdojo.scenarios, "data/agentdojo")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=list(BENCH), default="asb")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default=None, help="default: first locally installed ollama model")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benign-frac", type=float, default=0.25)
    ap.add_argument("--no-replay", action="store_true", help="do not append synthetic attacker-goal steps")
    a = ap.parse_args()

    model = a.model or list_models()[0]
    out = a.out or "traces/%s__%s.jsonl" % (a.bench, model.replace(":", "_"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["trace_id"] for l in open(out) if l.strip()}
    gen, data_dir = BENCH[a.bench]
    llm = OllamaChat(model)
    scs = gen(data_dir, a.n, a.seed, a.benign_frac)
    with open(out, "a") as f:
        for i, sc in enumerate(scs):
            tid = "%s|%s|%d|%d" % (a.bench, model, a.seed, i)
            if tid in done:
                continue
            tr = run_scenario(sc, llm, replay=not a.no_replay, trace_id=tid, model=model)
            f.write(json.dumps(tr) + "\n")
            f.flush()
            print(i, sc.mode, sc.style, [e["label"] for e in tr["events"]], flush=True)


if __name__ == "__main__":
    main()
