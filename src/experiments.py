"""Multi-benchmark experiments: in-benchmark, cross-benchmark transfer, ASB shifts, ablations.

    python -m src.experiments --seeds 3 --epochs 60
Writes <outdir>/experiments_<only>.json and <outdir>/report_<only>.md (outdir defaults to results/).
"""
import argparse
import glob
import json
import os
import random

import numpy as np

from .cache import ResultCache, common_benches
from .graph.build import ABLATIONS
from .train import disjoint, evaluate, load, split

BENCHES = ["asb", "injecagent", "agentdojo", "agentpoison"]
LEGACY_BENCHES = ["asb", "injecagent", "agentdojo"]  # benchmarks of the first results (pooled transfer rows kept)
SHIFTS = [("style", "combined_attack"), ("mode", "memory"), ("category", "aggressive"), ("category", "Stealthy"),
          ("agent", "system_admin_agent")]


def stat(runs, key):
    v = np.array(runs)
    return "%.2f±%.2f" % (v.mean(), v.std())


def cell(results, model):
    """results: list (per seed) of {model: report}. -> 'macroF1 / microF1 / rejR'"""
    f1 = [r[model]["macro_f1"] for r in results]
    mi = [r[model]["micro_f1"] for r in results]
    rj = [r[model]["recall"]["reject"] for r in results]
    return "%s / %s / %s" % (stat(f1, 0), stat(mi, 0), stat(rj, 0))


def asr_cell(results, model):
    """'ASR without guard -> with guard (worst-case with guard)', mean over seeds. NaN-safe."""
    m = lambda key: np.nanmean([r[model]["asr"][key] for r in results])
    return "%.2f→%.2f (%.2f)" % (m("asr_undefended"), m("asr_defended"), m("asr_forced"))


def cost_cell(results, model):
    """'benign traces blocked / approve actions rejected', mean over seeds."""
    m = lambda key: np.nanmean([r[model]["asr"][key] for r in results])
    return "%.2f / %.2f" % (m("benign_block"), m("approve_fpr"))


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
    ap.add_argument("--outdir", default="results", help="directory for the JSON and markdown outputs")
    ap.add_argument("--no-reuse", dest="reuse", action="store_false",
                    help="recompute every cell (default: reuse cells whose inputs and training code are unchanged)")
    ap.add_argument("--adopt-legacy", action="store_true",
                    help="accept stored cells without a fingerprint if the results file is newer than their inputs")
    a = ap.parse_args()
    seeds = range(a.seeds)

    if a.only == "llm":
        run_llm(a, seeds)
        return
    data = {b: load(a.traces % b, models=[a.model]) for b in BENCHES}
    benches = [b for b in BENCHES if data[b][0]]
    for b in BENCHES:
        g, m = data[b]
        if not g:
            print("%s: no traces for %s, skipped" % (b, a.model))
            continue
        lab = np.bincount(np.concatenate([x["action"].y.numpy() for x in g]), minlength=2).tolist()
        print("%s: %d traces, labels approve/reject = %s" % (b, len(g), lab))
    cache = ResultCache(a.outdir, a.only, seeds, a.epochs, a.model, reuse=a.reuse, adopt_legacy=a.adopt_legacy)
    R, md = {}, []
    DEFAULT = ("majority", "tool-lookup", "mlp", "gnn")

    def cell_runs(key, parts, models=DEFAULT):
        """parts: per seed (train graphs, train metas, test graphs, test metas). Reused when the inputs are unchanged."""
        R[key] = cache.get(key, parts, lambda: [evaluate(*p, epochs=a.epochs, seed=s, models=models)[0]
                                                 for s, p in zip(seeds, parts)], models=list(models))
        return R[key]

    def sub(g, m, idx):
        return [g[i] for i in idx], [m[i] for i in idx]

    if a.only in ("all", "inbench"):
        rows = []
        for b in benches:
            g, m = data[b]
            parts = [sub(g, m, tr) + sub(g, m, te) for tr, te in (split(m, "task", None, s) for s in seeds)]
            runs = cell_runs("inbench/" + b, parts)
            rows.append([b] + [cell(runs, k) for k in DEFAULT])
        md.append("## In-benchmark (held-out tasks), macro-F1 / micro-F1 / reject recall\n" + table(rows, ["benchmark", "majority", "tool-lookup", "mlp", "gnn"]))
        arows = [[b] + [asr_cell(R["inbench/" + b], k) for k in DEFAULT] + [cost_cell(R["inbench/" + b], "gnn")]
                 for b in benches]
        md.append("## Attack success rate, in-benchmark (no guard → guard (worst case: attack always attempted))\n"
                  + table(arows, ["benchmark", "majority", "tool-lookup", "mlp", "gnn", "gnn: benign blocked / approve rejected"]))

    if a.only in ("all", "transfer"):
        rows = []
        for tgt in benches:
            others = [b for b in benches if b != tgt]
            srcs = [[b] for b in others] + [others]
            old = [b for b in others if b in LEGACY_BENCHES]
            if len(old) > 1 and old != others:  # also keep the pooled row over the original benchmarks only
                srcs.append(old)
            for src in srcs:
                tg = sum([data[b][0] for b in src], [])
                tm = sum([data[b][1] for b in src], [])
                parts = [(tg, tm, data[tgt][0], data[tgt][1])] * len(seeds)
                runs = cell_runs("transfer/%s->%s" % ("+".join(src), tgt), parts)
                rows.append(["+".join(src), tgt] + [cell(runs, k) for k in ("tool-lookup", "mlp", "gnn")] + [asr_cell(runs, "gnn")])
        md.append("## Cross-benchmark transfer (train -> test), macro-F1 / micro-F1 / reject recall\n" + table(rows, ["train", "test", "tool-lookup", "mlp", "gnn", "gnn ASR"]))
        if "agentpoison" in benches:
            # AgentPoison has ONE target action per agent (DeleteDB / Finish("I don't know")), shared by every trigger, so
            # held-out triggers still share the target action with training. Training on one agent and testing on the
            # other also holds out the target action (and every task and trigger).
            g, m = data["agentpoison"]
            idx = lambda a_: [i for i, x in enumerate(m) if x["group"].startswith("agentpoison|%s|" % a_)]
            rows = []
            for s_, t_ in (("ehr", "qa"), ("qa", "ehr")):
                runs = cell_runs("transfer/agentpoison:%s->agentpoison:%s" % (s_, t_),
                                 [sub(g, m, idx(s_)) + sub(g, m, idx(t_))] * len(seeds))
                rows.append([s_, t_] + [cell(runs, k) for k in ("tool-lookup", "mlp", "gnn")] + [asr_cell(runs, "gnn")])
            md.append("## AgentPoison, held-out target action (train on one agent, test on the other)\n"
                      + table(rows, ["train", "test", "tool-lookup", "mlp", "gnn", "gnn ASR"]))

    if a.only in ("all", "shifts") and "asb" in benches:
        g, m = data["asb"]
        rows = []
        for key, val in SHIFTS:
            tr, te = split(m, key, val, 0)
            if not te or not tr:
                continue
            runs = cell_runs("shift/%s=%s" % (key, val), [sub(g, m, tr) + sub(g, m, te)] * len(seeds))
            rows.append(["%s~%s" % (key, val), len(te)] + [cell(runs, k) for k in ("tool-lookup", "mlp", "gnn")])
        md.append("## ASB held-out shifts, macro-F1 / micro-F1 / reject recall\n" + table(rows, ["held out", "test traces", "tool-lookup", "mlp", "gnn"]))

    if a.only in ("all", "ablate"):
        rows = []
        for ab in ["none"] + ABLATIONS:
            row = [ab]
            for b in benches:
                gs, m = load(a.traces % b, frozenset() if ab == "none" else frozenset([ab]), models=[a.model])
                parts = [sub(gs, m, tr) + sub(gs, m, te) for tr, te in (split(m, "task", None, s) for s in seeds)]
                row.append(cell(cell_runs("ablate/%s/%s" % (ab, b), parts, ("gnn",)), "gnn"))
            rows.append(row)
        md.append("## GNN ablations (held-out tasks), macro-F1 / micro-F1 / reject recall\n" + table(rows, ["removed"] + benches))

    cache.save()
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "experiments_%s.json" % a.only), "w", encoding="utf-8", newline="\n") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(a.outdir, "report_%s.md" % a.only), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\n".join(md) + "\n")
    print("\n\n".join(md))


def behaviour_table(llms, benches=BENCHES):
    """Per LLM x benchmark: how much real (non-synthetic) signal each LLM produced."""
    rows = []
    for llm in llms:
        for b in benches:
            n = att = followed = steps = invalid = ev = rej = syn = appr = 0
            for f in glob.glob("traces/%s__*.jsonl" % b):
                for line in open(f, encoding="utf-8"):
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
    return table(rows, ["LLM", "benchmark", "traces", "avg steps", "ASR, no guard (LLM attacks itself)", "synthetic/reject", "approve", "invalid-reply rate"])


def run_llm(a, seeds):
    llms = [m for m in a.llms.split(",")]
    # only benchmarks with traces from EVERY compared LLM, so all LLMs are compared on the same scenarios
    benches = common_benches(BENCHES, llms)
    print("cross-LLM benchmarks:", ", ".join(benches))
    per = {}
    for m in llms:  # same load order as before agentpoison existed: sorted glob over all trace files
        g, md_ = load("traces/*__*.jsonl", models=[m])
        keep = [i for i, x in enumerate(md_) if x["benchmark"] in benches]
        per[m] = ([g[i] for i in keep], [md_[i] for i in keep])
    per = {m: v for m, v in per.items() if v[0]}
    llms = list(per)
    print({m: len(v[0]) for m, v in per.items()})
    R, md = {}, ["## LLM behaviour (same scenarios for every LLM)\n" + behaviour_table(llms, benches)]
    all_groups = sorted({x["group"] for m in llms for x in per[m][1]})
    ev = lambda tr, te, s, **kw: evaluate(tr[0], tr[1], te[0], te[1], epochs=a.epochs, seed=s, also_real=True, **kw)[0]

    def pick(m, groups, inside, bench=None):
        g, md_ = per[m]
        idx = [i for i, x in enumerate(md_) if (x["group"] in groups) == inside and (bench is None or x["benchmark"] == bench)]
        return [g[i] for i in idx], [md_[i] for i in idx]

    def pool(parts):
        return sum([p[0] for p in parts], []), sum([p[1] for p in parts], [])

    def train_pool(models, groups, te):
        """Training traces of `models` outside `groups`, minus any trace sharing a task group or attack with test set te."""
        g, m = pool([pick(x, groups, False) for x in models])
        keep = disjoint(range(len(m)), range(len(te[1])), m, te[1])
        return [g[i] for i in keep], [m[i] for i in keep]

    def held_groups(s):
        gs = list(all_groups)
        random.Random(s).shuffle(gs)
        return set(gs[: max(1, len(gs) // 5)])

    def fmt(runs, k):
        f1 = [r[k]["macro_f1"] for r in runs]
        mi = [r[k]["micro_f1"] for r in runs]
        rj = [r[k]["recall"]["reject"] for r in runs]
        return "%s / %s / %s" % (stat(f1, 0), stat(mi, 0), stat(rj, 0))

    # (2)+(3) cross-LLM transfer: task- and attack-disjoint (honest) and real-only; same-task = leaky upper bound
    rows = []
    configs = [([a_], b_) for a_ in llms for b_ in llms] + [([x for x in llms if x != b_], b_) for b_ in llms]
    for src, tgt in configs:
        if tgt in src and len(src) > 1:
            continue
        honest, leaky = [], []
        for s in seeds:
            h = held_groups(s)
            te = pick(tgt, h, True)
            honest.append(ev(train_pool(src, h, te), te, s))
            # same-task variant: ALL of the source LLM's traces, including the test tasks (leaky upper bound)
            leaky.append(ev(pool([pick(m, set(), False) for m in src]), te, s))
        R["xllm/%s->%s/honest" % ("+".join(src), tgt)] = honest
        R["xllm/%s->%s/sametask" % ("+".join(src), tgt)] = leaky
        rows.append(["+".join(src), tgt] + [fmt(honest, k) for k in ("tool-lookup", "mlp", "gnn", "gnn|real")] + [fmt(leaky, "gnn"), asr_cell(honest, "gnn")])
    md.append("## Cross-LLM transfer, pooled over benchmarks (macro-F1 / micro-F1 / reject recall; task-disjoint)\n"
              + table(rows, ["train LLM(s)", "test LLM", "tool-lookup", "mlp", "gnn", "gnn real-only", "gnn same-task (leaky)", "gnn ASR (no guard→guard)"]))

    # (4) per-benchmark breakdown of the honest GNN transfer
    rows = []
    for src, tgt in [([x], y) for x in llms for y in llms]:
        for b in benches:
            runs = []
            for s in seeds:
                h = held_groups(s)
                te = pick(tgt, h, True, b)
                if not te[0]:
                    break
                runs.append(evaluate(*train_pool(src, h, te), *te, epochs=a.epochs, seed=s,
                                     models=("tool-lookup", "gnn"), also_real=True)[0])
            if runs:
                R["xllm_bench/%s->%s/%s" % (src[0], tgt, b)] = runs
                rows.append([src[0], tgt, b] + [fmt(runs, k) for k in ("tool-lookup", "gnn", "gnn|real")])
    md.append("## Cross-LLM transfer per benchmark (macro-F1 / micro-F1 / reject recall)\n" + table(rows, ["train", "test", "benchmark", "tool-lookup", "gnn", "gnn real-only"]))

    # (5) does bigger-LLM data help? always test on each LLM's held-out tasks
    rows = []
    mixes = [llms[:k] for k in range(1, len(llms) + 1)]
    for tgt in llms:
        for mix in mixes:
            runs = []
            for s in seeds:
                h = held_groups(s)
                te = pick(tgt, h, True)
                runs.append(evaluate(*train_pool(mix, h, te), *te, epochs=a.epochs, seed=s,
                                     models=("gnn",), also_real=True)[0])
            R["mix/%s/%s" % ("+".join(mix), tgt)] = runs
            rows.append(["+".join(mix), tgt, fmt(runs, "gnn"), fmt(runs, "gnn|real")])
    md.append("## Training-data mix (test = held-out tasks of the target LLM)\n" + table(rows, ["train mix", "test LLM", "gnn", "gnn real-only"]))

    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "llm_experiments.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(a.outdir, "llm_report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
