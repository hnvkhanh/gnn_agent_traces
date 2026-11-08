"""Multi-benchmark experiments: in-benchmark, cross-benchmark transfer, ASB shifts, ablations.

    python -m src.experiments --seeds 3 --epochs 60
Writes results/experiments.json and results/report.md.
"""
import argparse
import glob
import json
import os
import random

import numpy as np

from .graph.build import ABLATIONS
from .train import evaluate, load, split

BENCHES = ["asb", "injecagent", "agentdojo"]
SHIFTS = [("style", "combined_attack"), ("mode", "memory"), ("category", "aggressive"), ("category", "Stealthy"),
          ("agent", "system_admin_agent")]


def stat(runs, key):
    v = np.array(runs)
    return "%.2f±%.2f" % (v.mean(), v.std())


def cell(results, model):
    """results: list (per seed) of {model: report}. -> 'macroF1 | rejR'"""
    f1 = [r[model]["macro_f1"] for r in results]
    rj = [r[model]["recall"]["reject"] for r in results]
    return "%s / %s" % (stat(f1, 0), stat(rj, 0))


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--traces", default="traces/%s__*.jsonl")
    ap.add_argument("--only", default="all", help="all|inbench|transfer|shifts|ablate|llm")
    ap.add_argument("--model", default="qwen2.5:1.5b", help="trace-generating LLM for the benchmark sections")
    ap.add_argument("--llms", default="qwen2.5:1.5b,qwen2.5:7b,llama3.1:8b", help="LLMs for --only llm")
    ap.add_argument("--three-class", action="store_true", help="also score the rule-derived escalate class (default: approve/reject only)")
    a = ap.parse_args()
    seeds = range(a.seeds)
    ev = lambda *args, **kw: evaluate(*args, epochs=a.epochs, binary=not a.three_class, **kw)[0]

    if a.only == "llm":
        run_llm(a, seeds)
        return
    data = {b: load(a.traces % b, models=[a.model]) for b in BENCHES}
    for b, (g, m) in data.items():
        lab = np.bincount(np.concatenate([x["action"].y.numpy() for x in g]), minlength=3).tolist()
        print("%s: %d traces, labels approve/reject/escalate = %s" % (b, len(g), lab))
    R, md = {}, []

    if a.only in ("all", "inbench"):
        rows = []
        for b in BENCHES:
            g, m = data[b]
            runs = []
            for s in seeds:
                tr, te = split(m, "task", None, s)
                runs.append(ev([g[i] for i in tr], [m[i] for i in tr], [g[i] for i in te], [m[i] for i in te], seed=s))
            R["inbench/" + b] = runs
            rows.append([b] + [cell(runs, k) for k in ("majority", "tool-lookup", "mlp", "gnn")])
        md.append("## In-benchmark (held-out tasks), macro-F1 (approve/reject) / reject recall\n" + table(rows, ["benchmark", "majority", "tool-lookup", "mlp", "gnn"]))

    if a.only in ("all", "transfer"):
        rows = []
        for tgt in BENCHES:
            for src in [[b] for b in BENCHES if b != tgt] + [[b for b in BENCHES if b != tgt]]:
                tg = sum([data[b][0] for b in src], [])
                tm = sum([data[b][1] for b in src], [])
                runs = [ev(tg, tm, data[tgt][0], data[tgt][1], seed=s) for s in seeds]
                R["transfer/%s->%s" % ("+".join(src), tgt)] = runs
                rows.append(["+".join(src), tgt] + [cell(runs, k) for k in ("tool-lookup", "mlp", "gnn")])
        md.append("## Cross-benchmark transfer (train -> test), macro-F1 (approve/reject) / reject recall\n" + table(rows, ["train", "test", "tool-lookup", "mlp", "gnn"]))

    if a.only in ("all", "shifts"):
        g, m = data["asb"]
        rows = []
        for key, val in SHIFTS:
            tr, te = split(m, key, val, 0)
            if not te or not tr:
                continue
            runs = [ev([g[i] for i in tr], [m[i] for i in tr], [g[i] for i in te], [m[i] for i in te], seed=s) for s in seeds]
            R["shift/%s=%s" % (key, val)] = runs
            rows.append(["%s~%s" % (key, val), len(te)] + [cell(runs, k) for k in ("tool-lookup", "mlp", "gnn")])
        md.append("## ASB held-out shifts, macro-F1 (approve/reject) / reject recall\n" + table(rows, ["held out", "test traces", "tool-lookup", "mlp", "gnn"]))

    if a.only in ("all", "ablate"):
        rows = []
        for ab in ["none"] + ABLATIONS:
            row = [ab]
            for b in BENCHES:
                gs, m = load(a.traces % b, frozenset() if ab == "none" else frozenset([ab]), models=[a.model])
                runs = []
                for s in seeds:
                    tr, te = split(m, "task", None, s)
                    runs.append(evaluate([gs[i] for i in tr], [m[i] for i in tr], [gs[i] for i in te], [m[i] for i in te],
                                         epochs=a.epochs, seed=s, models=("gnn",), binary=not a.three_class)[0])
                R["ablate/%s/%s" % (ab, b)] = runs
                row.append(cell(runs, "gnn"))
            rows.append(row)
        md.append("## GNN ablations (held-out tasks), macro-F1 (approve/reject) / reject recall\n" + table(rows, ["removed"] + BENCHES))

    os.makedirs("results", exist_ok=True)
    json.dump(R, open("results/experiments_%s%s.json" % (a.only, "_3class" if a.three_class else ""), "w"), indent=1)
    open("results/report_%s%s.md" % (a.only, "_3class" if a.three_class else ""), "w").write("\n\n".join(md) + "\n")
    print("\n\n".join(md))


def behaviour_table(llms):
    """Per LLM x benchmark: how much real (non-synthetic) signal each LLM produced."""
    rows = []
    for llm in llms:
        for b in BENCHES:
            n = att = followed = steps = invalid = ev = rej = syn = appr = 0
            for f in glob.glob("traces/%s__*.jsonl" % b):
                for line in open(f):
                    tr = json.loads(line)
                    if tr.get("model") != llm:
                        continue
                    n += 1
                    evs = tr["events"]
                    steps += len(evs)
                    invalid += tr.get("n_invalid", 0)
                    ev += len(evs)
                    rej += sum(e["label"] == "reject" for e in evs)
                    syn += sum(e["synthetic"] for e in evs)
                    appr += sum(e["label"] == "approve" for e in evs)
                    if tr["attack"]:
                        att += 1
                        followed += any(e["label"] == "reject" and not e["synthetic"] for e in evs)
            if n:
                rows.append([llm, b, n, "%.2f" % (steps / n), "%.2f" % (followed / max(att, 1)),
                             "%d/%d" % (syn, rej), appr, "%.3f" % (invalid / max(invalid + ev, 1))])
    return table(rows, ["LLM", "benchmark", "traces", "avg steps", "attack-follow rate", "synthetic/reject", "approve", "invalid-reply rate"])


def run_llm(a, seeds):
    llms = [m for m in a.llms.split(",")]
    per = {m: load("traces/*__*.jsonl", models=[m]) for m in llms}
    per = {m: v for m, v in per.items() if v[0]}
    llms = list(per)
    print({m: len(v[0]) for m, v in per.items()})
    R, md = {}, ["## LLM behaviour (same scenarios for every LLM)\n" + behaviour_table(llms)]
    all_groups = sorted({x["group"] for m in llms for x in per[m][1]})
    ev = lambda tr, te, s, **kw: evaluate(tr[0], tr[1], te[0], te[1], epochs=a.epochs, seed=s, also_real=True, **kw)[0]

    def pick(m, groups, inside, bench=None):
        g, md_ = per[m]
        idx = [i for i, x in enumerate(md_) if (x["group"] in groups) == inside and (bench is None or x["benchmark"] == bench)]
        return [g[i] for i in idx], [md_[i] for i in idx]

    def pool(parts):
        return sum([p[0] for p in parts], []), sum([p[1] for p in parts], [])

    def held_groups(s):
        gs = list(all_groups)
        random.Random(s).shuffle(gs)
        return set(gs[: max(1, len(gs) // 5)])

    def fmt(runs, k):
        f1 = [r[k]["macro_f1"] for r in runs]
        rj = [r[k]["recall"]["reject"] for r in runs]
        return "%s / %s" % (stat(f1, 0), stat(rj, 0))

    # (2)+(3) cross-LLM transfer: task-disjoint (honest) and real-only; same-task = leaky upper bound
    rows = []
    configs = [([a_], b_) for a_ in llms for b_ in llms] + [([x for x in llms if x != b_], b_) for b_ in llms]
    for src, tgt in configs:
        if tgt in src and len(src) > 1:
            continue
        honest, leaky = [], []
        for s in seeds:
            h = held_groups(s)
            te = pick(tgt, h, True)
            honest.append(ev(pool([pick(m, h, False) for m in src]), te, s))
            # same-task variant: ALL of the source LLM's traces, including the test tasks (leaky upper bound)
            leaky.append(ev(pool([pick(m, set(), False) for m in src]), te, s))
        R["xllm/%s->%s/honest" % ("+".join(src), tgt)] = honest
        R["xllm/%s->%s/sametask" % ("+".join(src), tgt)] = leaky
        rows.append(["+".join(src), tgt] + [fmt(honest, k) for k in ("tool-lookup", "mlp", "gnn", "gnn|real")] + [fmt(leaky, "gnn")])
    md.append("## Cross-LLM transfer, pooled over benchmarks (macro-F1 / reject recall; task-disjoint)\n"
              + table(rows, ["train LLM(s)", "test LLM", "tool-lookup", "mlp", "gnn", "gnn real-only", "gnn same-task (leaky)"]))

    # (4) per-benchmark breakdown of the honest GNN transfer
    rows = []
    for src, tgt in [([x], y) for x in llms for y in llms]:
        for b in BENCHES:
            runs = []
            for s in seeds:
                h = held_groups(s)
                te = pick(tgt, h, True, b)
                if not te[0]:
                    break
                runs.append(evaluate(*pool([pick(m, h, False) for m in src]), *te, epochs=a.epochs, seed=s,
                                     models=("tool-lookup", "gnn"), also_real=True)[0])
            if runs:
                R["xllm_bench/%s->%s/%s" % (src[0], tgt, b)] = runs
                rows.append([src[0], tgt, b] + [fmt(runs, k) for k in ("tool-lookup", "gnn", "gnn|real")])
    md.append("## Cross-LLM transfer per benchmark (macro-F1 / reject recall)\n" + table(rows, ["train", "test", "benchmark", "tool-lookup", "gnn", "gnn real-only"]))

    # (5) does bigger-LLM data help? always test on each LLM's held-out tasks
    rows = []
    mixes = [llms[:k] for k in range(1, len(llms) + 1)]
    for tgt in llms:
        for mix in mixes:
            runs = []
            for s in seeds:
                h = held_groups(s)
                runs.append(evaluate(*pool([pick(m, h, False) for m in mix]), *pick(tgt, h, True), epochs=a.epochs, seed=s,
                                     models=("gnn",), also_real=True)[0])
            R["mix/%s/%s" % ("+".join(mix), tgt)] = runs
            rows.append(["+".join(mix), tgt, fmt(runs, "gnn"), fmt(runs, "gnn|real")])
    md.append("## Training-data mix (test = held-out tasks of the target LLM)\n" + table(rows, ["train mix", "test LLM", "gnn", "gnn real-only"]))

    os.makedirs("results", exist_ok=True)
    json.dump(R, open("results/llm_experiments.json", "w"), indent=1)
    open("results/llm_report.md", "w").write("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
