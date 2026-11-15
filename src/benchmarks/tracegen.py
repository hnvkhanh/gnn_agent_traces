"""tracegen records (data/tracegen/traces.jsonl, real AgentDojo runs) -> the trace format src/graph reads.

    from src.benchmarks.tracegen import load_traces
    trs = load_traces("data/tracegen")          # list of trace dicts, one per run with at least one tool call

Labels (per tool call, src/sandbox/policy.py LABELS):
  reject   the call matches one of the injection task's ground-truth calls (the attacker's calls, recorded by tracegen
           as `injection_ground_truth_calls`; older records fall back to data/agentdojo/<suite>.json), via
           policy.matches_attacker_call, unless the very same call is also part of the user task's ground truth
  approve  every other call, and every call of a benign run
There is no synthetic replay: an attack the agent ignores adds no reject (the trace is then all approve).

Splits: the task split comes from data/tracegen/splits/*.jsonl when present (tracegen's own task-level split), else
tracegen.stages.build.assign_splits over the task keys. Injection tasks get their own split
(tracegen.stages.build.assign_injection_splits, the same one tracegen uses to plan attack runs). `split` is the task
split; `inj_split` the injection task's split (None for benign runs). load_traces drops attack runs where the two differ,
so train / val / test never share an injection task.

This module never imports agentdojo; tracegen.stages.build is plain Python.
"""
import json
import os
import re

from ..sandbox.policy import _strings, matches_attacker_call

DEFAULT_RATIOS, DEFAULT_SEED = (0.7, 0.15, 0.15), 0


def suite_meta(data_dir="data/agentdojo"):
    """{suite: {tools, user_gt: {task_id: calls}, inj_gt: {inj_id: calls}, inj_ids}} from the static export."""
    out = {}
    for s in ("banking", "slack", "travel", "workspace"):
        p = os.path.join(data_dir, "%s.json" % s)
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        out[s] = dict(tools=d["tools"], user_gt={t["id"]: t["calls"] for t in d["user_tasks"]},
                      inj_gt={t["id"]: t["calls"] for t in d["injection_tasks"]},
                      inj_ids=[t["id"] for t in d["injection_tasks"]])
    return out


def _canon(x):
    return json.dumps(x, sort_keys=True, default=str).lower()


def attacker_values(calls):
    """Attacker-chosen argument strings (len >= 4) of a list of calls: what policy.matches_attacker_call keys on."""
    return sorted({v for c in calls for v in _strings(c.get("args") or {}) if len(v) >= 4})


def _drop_values(args, own):
    """args without the leaf values in `own` (lowercased strings)."""
    if isinstance(args, dict):
        return {k: v for k, v in ((k, _drop_values(v, own)) for k, v in args.items()) if v is not None}
    if isinstance(args, (list, tuple)):
        return [v for v in (_drop_values(x, own) for x in args) if v is not None]
    return None if args is None or str(args).lower() in own else args


def _alnum(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def contains_injection(obs, injections):
    """Does an observation carry one of the injected texts? Compared on letters and digits only, because the tool
    output re-renders the text (YAML line folding, escapes)."""
    o = _alnum(obs)
    return any(len(k) >= 8 and k in o for k in (_alnum(t)[:80] for t in injections))


def label_events(rec, meta):
    """Per tool call 'approve' / 'reject' (see module doc)."""
    calls = rec.get("tool_calls") or []
    if rec.get("mode") != "attack":
        return ["approve"] * len(calls)
    atk = rec.get("injection_ground_truth_calls") or meta["inj_gt"].get(rec.get("injection_task_id"), [])
    user_gt = rec.get("ground_truth_calls") or meta["user_gt"].get(rec["task_id"], [])
    scope = {c["function"] for c in user_gt}
    # Attacker values the user task itself uses (e.g. the date "2022-01-01" of a banking transfer) do not identify the
    # attacker: drop them, so only attacker-chosen values (recipient IBAN, attacker e-mail, ...) decide `reject`.
    own = {v.lower() for c in user_gt for v in _strings(c.get("args") or {})}
    atk = [{"tool": c["function"], "args": _drop_values(c.get("args") or {}, own)} for c in atk]
    user_calls = {(c["function"], _canon(c.get("args") or {})) for c in user_gt}
    out = []
    for c in calls:
        args = c.get("args") or {}
        hit = matches_attacker_call(c["name"], args, atk, scope) and (c["name"], _canon(args)) not in user_calls
        out.append("reject" if hit else "approve")
    return out


def convert(rec, meta, task_split=None, inj_split=None):
    """One tracegen record -> trace dict (None if the run made no tool call)."""
    calls = rec.get("tool_calls") or []
    if not calls:
        return None
    attack = rec.get("mode") == "attack"
    inj_texts = [t for t in (rec.get("injections") or {}).values() if t]
    labels = label_events(rec, meta)
    events = []
    for i, (c, lab) in enumerate(zip(calls, labels)):
        obs = c.get("result")
        if obs is None:
            obs = c.get("error") or ""
        events.append(dict(step=i, tool=c["name"], args=c.get("args") or {}, observation=str(obs),
                           obs_injected=contains_injection(str(obs), inj_texts), synthetic=False, label=lab))
    s = rec["suite"]
    atk_calls = rec.get("injection_ground_truth_calls") or meta["inj_gt"].get(rec.get("injection_task_id"), [])
    return dict(
        benchmark="tracegen", trace_id="tracegen|%s" % rec["trace_id"], model=rec["agent_model"],
        agent="AgentDojo %s assistant" % s, group="agentdojo|%s|%s" % (s, rec["task_id"]),
        suite=s, task_id=rec["task_id"], variant_id=rec.get("variant_id"), is_original=rec.get("is_original"),
        task=rec["prompt"], user_task=rec["prompt"], memory=[],
        injected="\n".join(inj_texts) if attack else "",
        attack=dict(id="%s|%s" % (s, rec["injection_task_id"]), mode="opi", style=rec.get("attack_name"),
                    category=s, tool=None) if attack else None,
        tools=[dict(name=t["name"], description=t["description"], kind="normal", agent=s, in_scope=False)
               for t in meta["tools"]],
        allowed_tools=sorted({c["function"] for c in rec.get("ground_truth_calls") or []}),
        events=events, n_invalid=0,
        # Not read by the graph (tests/test_tracegen_gnn.py checks): bookkeeping, splits and controls.
        utility=rec.get("utility"), security=rec.get("security"), error=rec.get("error"),
        split=task_split, inj_split=inj_split,
        attacker_calls=atk_calls,
        # every attacker value of the suite (all injection tasks), so the `attacker_values` masking control treats
        # benign and attacked traces alike
        suite_attacker_values=attacker_values([c for cs in meta["inj_gt"].values() for c in cs] + atk_calls),
    )


def task_splits(tg_dir, recs, ratios=DEFAULT_RATIOS, seed=DEFAULT_SEED):
    """{(suite, task_id): split}: tracegen's splits files when present, else the same seeded assignment."""
    from tracegen.stages.build import assign_splits
    out = {}
    for name in ("train", "val", "test"):
        p = os.path.join(tg_dir, "splits", "%s.jsonl" % name)
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    out[(r["suite"], r["task_id"])] = name
    missing = [(r["suite"], r["task_id"]) for r in recs if (r["suite"], r["task_id"]) not in out]
    if missing:
        if out:  # the splits files are older than some traces: reproduce tracegen's assignment over all task keys
            fresh = assign_splits(list(out) + missing, ratios, seed)
            bad = [k for k in out if fresh[k] != out[k]]
            if bad:
                raise ValueError("splits files disagree with assign_splits for %s; rerun `python -m tracegen build`"
                                 % bad[:3])
            out = fresh
        else:
            out = assign_splits(missing, ratios, seed)
    return out


def injection_splits(meta, ratios=DEFAULT_RATIOS, seed=DEFAULT_SEED):
    from tracegen.stages.build import assign_injection_splits
    return assign_injection_splits({s: m["inj_ids"] for s, m in meta.items()}, ratios, seed)


def read_records(tg_dir):
    p = os.path.join(tg_dir, "traces.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def load_traces(tg_dir="data/tracegen", data_dir="data/agentdojo", ratios=DEFAULT_RATIOS, seed=DEFAULT_SEED,
                recs=None, stats=None):
    """Converted traces, with `split` / `inj_split` set. Attack runs whose injection task is in another split than
    their user task are dropped (counted in stats['dropped_split_mismatch'])."""
    meta = suite_meta(data_dir)
    recs = read_records(tg_dir) if recs is None else recs
    recs = [r for r in recs if r["suite"] in meta]
    ts, isp = task_splits(tg_dir, recs, ratios, seed), injection_splits(meta, ratios, seed)
    out, st = [], dict(records=len(recs), no_tool_calls=0, dropped_split_mismatch=0)
    for r in recs:
        sp = ts[(r["suite"], r["task_id"])]
        ip = isp.get((r["suite"], r["injection_task_id"])) if r.get("mode") == "attack" else None
        if r.get("mode") == "attack" and ip != sp:
            st["dropped_split_mismatch"] += 1
            continue
        tr = convert(r, meta[r["suite"]], sp, ip)
        if tr is None:
            st["no_tool_calls"] += 1
            continue
        out.append(tr)
    if stats is not None:
        stats.update(st)
    return out
