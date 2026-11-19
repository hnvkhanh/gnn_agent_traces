"""Stage 7: action signatures, optional dedup, task-level splits, and report.md / report.html."""
import collections
import hashlib
import html
import json
import logging
import random
import statistics

from tracegen import config as C
from tracegen import io

log = logging.getLogger("tracegen")


# ---------------------------------------------------------------- signatures / dedup
def norm_value(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        f = float(v)
        return int(f) if f.is_integer() else round(f, 6)
    if isinstance(v, str):
        s = v.strip()
        try:   # "98.70" and 98.7 are the same amount
            return norm_value(float(s)) if s and s.replace(".", "", 1).lstrip("-").isdigit() else s
        except ValueError:
            return s
    if isinstance(v, dict):
        return {str(k): norm_value(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [norm_value(x) for x in v]
    return str(v)


def signature(tool_calls):
    """Canonical JSON of the ordered (tool name, normalized args) sequence."""
    return json.dumps([[c["name"], norm_value(c.get("args") or {})] for c in tool_calls], sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)


def sig_hash(sig):
    return hashlib.sha1(sig.encode()).hexdigest()[:16]


def dedup_key(t):
    k = (t["suite"], t["task_id"], t["mode"], t["signature"], t["utility"])
    return k + (t.get("security"),) if t["mode"] == "attack" else k


def dedup(traces):
    seen, out = set(), []
    for t in traces:
        k = dedup_key(t)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


# ---------------------------------------------------------------- splits
def assign_splits(task_keys, ratios, seed):
    """{(suite, task_id): split}. Shuffled per suite with a seeded RNG, so every suite is spread over the splits
    and all traces of one task land in one split."""
    by_suite = collections.defaultdict(list)
    for s, t in sorted(set(task_keys)):
        by_suite[s].append(t)
    total = float(sum(ratios))
    a, b = ratios[0] / total, (ratios[0] + ratios[1]) / total
    out = {}
    for s, tasks in sorted(by_suite.items()):
        tasks = sorted(tasks)
        random.Random("%s/%s" % (seed, s)).shuffle(tasks)
        n = len(tasks)
        i, j = int(round(n * a)), int(round(n * b))
        for k, t in enumerate(tasks):
            out[(s, t)] = "train" if k < i else "val" if k < j else "test"
    return out


def assign_injection_splits(ids_by_suite, ratios, seed):
    """{(suite, injection_task_id): split}. Same seeded per-suite shuffle as assign_splits (with an 'inj' salt), but
    every split gets at least one injection task when a suite has >= 3, so val and test always contain attacks the
    training side never saw."""
    total = float(sum(ratios))
    out = {}
    for s, ids in sorted(ids_by_suite.items()):
        ids = sorted(ids)
        random.Random("%s/inj/%s" % (seed, s)).shuffle(ids)
        n = len(ids)
        i, j = int(round(n * ratios[0] / total)), int(round(n * (ratios[0] + ratios[1]) / total))
        if n >= 3:
            j = min(max(j, 2), n - 1)
            i = min(max(i, 1), j - 1)
        for k, t in enumerate(ids):
            out[(s, t)] = "train" if k < i else "val" if k < j else "test"
    return out


def first_divergence(trace):
    got = [c["name"] for c in trace["tool_calls"]]
    exp = [c["function"] for c in trace["ground_truth_calls"]]
    for i in range(max(len(got), len(exp))):
        g = got[i] if i < len(got) else "<stop>"
        e = exp[i] if i < len(exp) else "<stop>"
        if g != e:
            return i, e, g
    return None


# ---------------------------------------------------------------- build
def build(cfg, suites, do_dedup=None, limit=None):
    do_dedup = cfg["build"]["dedup"] if do_dedup is None else do_dedup
    from tracegen.stages.generate import originals
    keep = {(r["suite"], r["task_id"]) for r in originals(cfg, suites, limit)}
    traces = [t for t in io.read_jsonl(C.out_path(cfg, "traces.jsonl")) if (t["suite"], t["task_id"]) in keep]
    for t in traces:
        t["signature"] = signature(t["tool_calls"])
        t["signature_hash"] = sig_hash(t["signature"])
    data = dedup(traces) if do_dedup else traces
    splits = assign_splits(keep, cfg["build"]["split_ratios"], cfg["build"]["split_seed"])
    by_split = collections.defaultdict(list)
    for t in data:
        t["split"] = splits[(t["suite"], t["task_id"])]
        by_split[t["split"]].append(t)
    for name in ("train", "val", "test"):
        io.write_jsonl(C.out_path(cfg, "splits/%s.jsonl" % name), by_split.get(name, []))
    leak = leakage(by_split)
    if leak["tasks_in_multiple_splits"]:
        raise AssertionError("task ids in more than one split: %s" % leak["tasks_in_multiple_splits"][:5])
    stats = report_stats(cfg, keep, traces, data, by_split, leak, do_dedup)
    (C.out_path(cfg, "report.md")).write_text(render_md(stats), encoding="utf-8")
    (C.out_path(cfg, "report.html")).write_text(render_html(stats), encoding="utf-8")
    log.info("build: %d traces (%d after dedup) -> splits %s; report at %s", len(traces), len(data),
             {k: len(v) for k, v in by_split.items()}, C.out_path(cfg, "report.md"))
    return stats


def leakage(by_split):
    where = collections.defaultdict(set)
    sigs = collections.defaultdict(set)
    for name, ts in by_split.items():
        for t in ts:
            where[(t["suite"], t["task_id"])].add(name)
            if t["tool_calls"]:
                sigs[(t["suite"], t["signature_hash"])].add(name)
    inj = {name: sorted({t["injection_task_id"] for t in ts if t["mode"] == "attack"}) for name, ts in by_split.items()}
    return {"tasks_in_multiple_splits": sorted(k for k, v in where.items() if len(v) > 1),
            "signatures_across_splits": sum(1 for v in sigs.values() if len(v) > 1),
            "n_signatures": len(sigs), "injection_tasks_per_split": inj}


def _rate(xs):
    return (sum(xs) / len(xs)) if xs else None


def report_stats(cfg, keep, traces, data, by_split, leak, did_dedup):
    suites = sorted({s for s, _ in keep})
    cands = [c for c in io.read_jsonl(C.out_path(cfg, "candidates.jsonl")) if (c["suite"], c["task_id"]) in keep]
    ver = [c for c in io.read_jsonl(C.out_path(cfg, "verified_variants.jsonl")) if (c["suite"], c["task_id"]) in keep]
    rej = [c for c in io.read_jsonl(C.out_path(cfg, "rejected_variants.jsonl")) if (c["suite"], c["task_id"]) in keep]
    genlog = [g for g in io.read_jsonl(C.out_path(cfg, "generate_log.jsonl")) if (g["suite"], g["task_id"]) in keep]
    st = {"config": {"benchmark_version": cfg["benchmark_version"], "mode": cfg["gpu"]["mode"],
                     "generator": C.generator_model(cfg), "nli": cfg["verify"]["nli_model"],
                     "tau": cfg["verify"]["nli_tau"], "judge": cfg["verify"]["judge_model"] if any(
                         c.get("judge") for c in ver + rej) else "off", "dedup": did_dedup},
          "suites": {}, "leakage": leak, "splits": {k: {"traces": len(v), "tasks": len({(t["suite"], t["task_id"]) for t in v})}
                                                    for k, v in sorted(by_split.items())},
          "n_traces": len(traces), "n_after_dedup": len(data)}
    for s in suites:
        cs = [c for c in cands if c["suite"] == s]
        rs = [c for c in rej if c["suite"] == s]
        vs = [c for c in ver if c["suite"] == s]
        reasons = collections.Counter()
        for r in rs:
            rr = r["reject_reason"] or "?"
            if rr.startswith("entity:"):
                kinds = sorted({m.split(":")[1] for m in rr[len("entity:"):].split(";")})
                reasons.update("entity:" + k for k in kinds)
            else:
                reasons[rr] += 1
        ent_fail = sum(1 for r in rs if (r["reject_reason"] or "").startswith("entity"))
        nli_fail = sum(1 for r in rs if (r["reject_reason"] or "").startswith("nli"))
        judge_fail = sum(1 for r in rs if (r["reject_reason"] or "").startswith("judge"))
        n = len(cs)
        scored = [min(c["nli_fwd"], c["nli_bwd"]) for c in vs + rs if c.get("nli_fwd") is not None]
        hist = [0] * 10
        for x in scored:
            hist[min(9, int(x * 10))] += 1
        funnel = {"originals": len({c["task_id"] for c in cs} | {g["task_id"] for g in genlog if g["suite"] == s}),
                  "generated": n, "unverified": n - len(rs) - len(vs), "entity_pass": n - ent_fail - (n - len(rs) - len(vs)),
                  "nli_pass": len(vs) + judge_fail, "judge_pass": len(vs) if st["config"]["judge"] != "off" else None,
                  "verified": len(vs), "gen_errors": sum(1 for g in genlog if g["suite"] == s and g["error"])}
        funnel["entity_pass"] = funnel["generated"] - funnel["unverified"] - ent_fail
        funnel["nli_pass"] = funnel["entity_pass"] - nli_fail
        agents = {}
        ts_s = [t for t in traces if t["suite"] == s]
        for a in sorted({t["agent_model"] for t in ts_s}):
            ta = [t for t in ts_s if t["agent_model"] == a]
            ben = [t for t in ta if t["mode"] == "benign"]
            uo = _rate([t["utility"] for t in ben if t["is_original"]])
            up = _rate([t["utility"] for t in ben if not t["is_original"]])
            per_task = collections.defaultdict(set)
            for t in ben:
                per_task[t["task_id"]].add(t["signature_hash"])
            divs = collections.Counter()
            for t in ben:
                d = first_divergence(t) if t["ground_truth_calls"] else None
                if d is not None:
                    divs["call #%d: expected %s, got %s" % (d[0] + 1, d[1], d[2])] += 1
            att = [t for t in ta if t["mode"] == "attack"]
            asr = {}
            for an in sorted({t["attack_name"] for t in att}):
                x = [t for t in att if t["attack_name"] == an]
                asr[an] = {"n": len(x), "asr": _rate([t["security"] for t in x]),
                           "utility_under_attack": _rate([t["utility"] for t in x])}
            agents[a] = {
                "n_benign": len(ben), "n_attack": len(att), "errors": sum(1 for t in ta if t["error"]),
                "n_orig": sum(1 for t in ben if t["is_original"]), "n_para": sum(1 for t in ben if not t["is_original"]),
                "utility_orig": uo, "utility_para": up,
                "utility_drop": (uo - up) if uo is not None and up is not None else None,
                "gt_sequence_match": _rate([t["matches_ground_truth_sequence"] for t in ben]),
                "unique_sigs_per_task": statistics.mean(len(v) for v in per_task.values()) if per_task else None,
                "traces_per_task": statistics.mean(
                    collections.Counter(t["task_id"] for t in ben).values()) if ben else None,
                "avg_steps": statistics.mean(t["n_steps"] for t in ben) if ben else None,
                "top_divergence": divs.most_common(3), "attacks": asr}
        st["suites"][s] = {"funnel": funnel, "reasons": reasons.most_common(), "nli_hist": hist, "agents": agents}
    return st


# ---------------------------------------------------------------- rendering
def _f(x, pct=False):
    if x is None:
        return "–"
    return "%.1f%%" % (100 * x) if pct else ("%.2f" % x if isinstance(x, float) else str(x))


def render_md(st):
    c = st["config"]
    L = ["# tracegen report", "",
         "benchmark %s · mode `%s` · generator `%s` · NLI `%s` (τ=%s) · judge `%s` · dedup %s" % (
             c["benchmark_version"], c["mode"], c["generator"], c["nli"], c["tau"], c["judge"], c["dedup"]), "",
         "Traces: %d (%d after dedup)." % (st["n_traces"], st["n_after_dedup"]), ""]
    for s, d in st["suites"].items():
        f = d["funnel"]
        L += ["## %s" % s, "", "### Variant funnel", "",
              "| originals | generated | entity pass | NLI pass | judge pass | verified | not yet verified | gen errors |",
              "|---|---|---|---|---|---|---|---|",
              "| %s | %s | %s | %s | %s | %s | %s | %s |" % (f["originals"], f["generated"], f["entity_pass"], f["nli_pass"],
                                                       _f(f["judge_pass"]), f["verified"], f["unverified"], f["gen_errors"]),
              ""]
        if d["reasons"]:
            L += ["Rejection reasons: " + ", ".join("`%s` ×%d" % r for r in d["reasons"]), ""]
        L += ["NLI min(fwd, bwd) histogram (bins of 0.1, from 0 to 1): `%s`" % d["nli_hist"], "", "### Traces per agent", "",
              "| agent | benign (orig/para) | errors | utility orig | utility para | drop | GT seq match | unique sigs/task | traces/task | avg steps |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for a, x in d["agents"].items():
            L.append("| %s | %d (%d/%d) | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                a, x["n_benign"], x["n_orig"], x["n_para"], x["errors"], _f(x["utility_orig"], 1),
                _f(x["utility_para"], 1), _f(x["utility_drop"], 1), _f(x["gt_sequence_match"], 1),
                _f(x["unique_sigs_per_task"]), _f(x["traces_per_task"]), _f(x["avg_steps"])))
        L.append("")
        for a, x in d["agents"].items():
            if x["top_divergence"]:
                L.append("- **%s** most common divergence from the ground-truth call sequence: %s" % (
                    a, "; ".join("%s (×%d)" % kv for kv in x["top_divergence"])))
            for an, y in x["attacks"].items():
                L.append("- **%s** attack `%s`: ASR %s over %d runs, utility under attack %s%s" % (
                    a, an, _f(y["asr"], 1), y["n"], _f(y["utility_under_attack"], 1),
                    " (search-optimized against this agent: not comparable to natural ASR)" if an == "searched" else ""))
        L.append("")
    lk = st["leakage"]
    L += ["## Splits and leakage controls", "",
          "Splits are assigned by (suite, original task_id), so all variants, seeds, agents and attack runs of a task go to one split.", "",
          "| split | tasks | traces |", "|---|---|---|"]
    L += ["| %s | %d | %d |" % (k, v["tasks"], v["traces"]) for k, v in st["splits"].items()]
    L += ["", "- Task ids in more than one split: **%d** (the build fails when this is > 0)." % len(lk["tasks_in_multiple_splits"]),
          "- Action signatures that occur in more than one split: %d of %d. Different tasks can legitimately need the "
          "same calls; check these before using exact-sequence features." % (lk["signatures_across_splits"], lk["n_signatures"]),
          "- Injection tasks are shared across user tasks by AgentDojo's design. Injection tasks per split: %s" % (
              lk["injection_tasks_per_split"] or "none (attacks off)"), ""]
    return "\n".join(L)


CSS = """:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--bar:#3b6fd8;--bar2:#d8823b;--line:#e3e3e8}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--mut:#9a9aa2;--bar:#6f9cf0;--bar2:#f0a86f;--line:#2c2c31}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}
table{border-collapse:collapse;margin:8px 0 16px;width:100%;display:block;overflow-x:auto}
td,th{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--mut);font-weight:600}
.b{display:inline-block;height:10px;background:var(--bar);vertical-align:middle;margin-right:6px}
.b2{background:var(--bar2)}.m{color:var(--mut)}"""


def _bar(x, cls="b"):
    if x is None:
        return "–"
    return '<span class="%s" style="width:%dpx"></span>%.1f%%' % (cls, int(120 * x), 100 * x)


def render_html(st):
    e = html.escape
    c = st["config"]
    H = ["<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         "<title>tracegen report</title><style>%s</style><h1>tracegen report</h1>" % CSS,
         "<p class=m>benchmark %s · mode %s · generator %s · NLI %s (τ=%s) · judge %s · dedup %s · %s traces</p>" % tuple(
             e(str(x)) for x in (c["benchmark_version"], c["mode"], c["generator"], c["nli"], c["tau"], c["judge"],
                                 c["dedup"], st["n_traces"]))]
    for s, d in st["suites"].items():
        f = d["funnel"]
        top = max(1, f["generated"])
        H.append("<h2>%s</h2><h3>Variant funnel</h3><table><tr><th>stage</th><th>count</th><th></th></tr>" % e(s))
        for lbl, k in (("generated", "generated"), ("entity pass", "entity_pass"), ("NLI pass", "nli_pass"),
                       ("judge pass", "judge_pass"), ("verified", "verified")):
            if f[k] is not None:
                H.append("<tr><td>%s</td><td>%d</td><td style=text-align:left>%s</td></tr>" % (lbl, f[k], _bar(f[k] / top)))
        H.append("</table>")
        if d["reasons"]:
            H.append("<p class=m>Rejections: %s</p>" % e(", ".join("%s ×%d" % r for r in d["reasons"])))
        hm = max(1, max(d["nli_hist"]))
        H.append("<p class=m>NLI min(fwd,bwd) histogram</p><svg width=320 height=70 role=img aria-label='NLI histogram'>")
        for i, v in enumerate(d["nli_hist"]):
            h = int(50 * v / hm)
            H.append("<rect x=%d y=%d width=28 height=%d fill='var(--bar)'><title>%.1f–%.1f: %d</title></rect>"
                     "<text x=%d y=68 font-size=9 fill='var(--mut)'>%.1f</text>" % (i * 32, 52 - h, h, i / 10, (i + 1) / 10, v, i * 32, i / 10))
        H.append("</svg><h3>Traces per agent</h3><table><tr><th>agent</th><th>benign</th><th>errors</th>"
                 "<th>utility orig</th><th>utility para</th><th>drop</th><th>unique sigs/task</th><th>avg steps</th></tr>")
        for a, x in d["agents"].items():
            H.append("<tr><td>%s</td><td>%d</td><td>%d</td><td style=text-align:left>%s</td><td style=text-align:left>%s</td>"
                     "<td>%s</td><td>%s</td><td>%s</td></tr>" % (
                         e(a), x["n_benign"], x["errors"], _bar(x["utility_orig"]), _bar(x["utility_para"], "b b2"),
                         _f(x["utility_drop"], 1), _f(x["unique_sigs_per_task"]), _f(x["avg_steps"])))
        H.append("</table><ul>")
        for a, x in d["agents"].items():
            if x["top_divergence"]:
                H.append("<li><b>%s</b> divergence: %s</li>" % (e(a), e("; ".join("%s (×%d)" % kv for kv in x["top_divergence"]))))
            for an, y in x["attacks"].items():
                H.append("<li><b>%s</b> %s: ASR %s (n=%d)</li>" % (e(a), e(an), _f(y["asr"], 1), y["n"]))
        H.append("</ul>")
    lk = st["leakage"]
    H.append("<h2>Splits and leakage</h2><table><tr><th>split</th><th>tasks</th><th>traces</th></tr>")
    H += ["<tr><td>%s</td><td>%d</td><td>%d</td></tr>" % (k, v["tasks"], v["traces"]) for k, v in st["splits"].items()]
    H.append("</table><p>Tasks in more than one split: <b>%d</b>. Signatures in more than one split: %d of %d.</p>" % (
        len(lk["tasks_in_multiple_splits"]), lk["signatures_across_splits"], lk["n_signatures"]))
    return "\n".join(H)
