import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

from src import crossllm as X  # noqa: E402

LLMS = ["a", "b", "c"]


def metas(n_groups=30, per=4, skew=None):
    out = {}
    for l in LLMS:
        ms = []
        for g in range(n_groups):
            for k in range(per if l != skew else per * 2):
                ms.append({"group": "g%d" % g, "attack_id": ("atk%d" % (g % 6)) if k % 2 else None,
                           "trace_id": "%s|%d|%d" % (l, g, k), "attack": bool(k % 2)})
        out[l] = ms
    return out


def test_same_held_groups_for_every_llm_and_train_is_disjoint():
    ms = metas()
    sp = X.make_split(ms, 0)
    held = sp["held"]
    for l in LLMS:
        assert {ms[l][i]["group"] for i in sp["test"][l]} <= held and sp["test"][l]
        assert not {ms[l][i]["group"] for i in sp["train"][l]} & held
        tr_a = {ms[l][i]["attack_id"] for i in sp["train"][l]} - {None}
        assert not tr_a & sp["test_attacks"]
    assert len({len(sp["test"][l]) for l in LLMS}) == 1


def test_train_depends_on_source_only_and_attack_filter_uses_all_llms():
    ms = metas()
    ms["c"] = [dict(m, attack_id="only_c") if m["group"] in X.make_split(ms, 0)["held"] and m["attack_id"] else m for m in ms["c"]]
    sp = X.make_split(ms, 0)
    assert "only_c" in sp["test_attacks"]  # attacks seen only in another LLM's test are still excluded from every train set
    sp2 = X.make_split({k: ms[k] for k in ms}, 0)
    assert sp["train"] == sp2["train"]


def test_sizes_are_matched_and_audit_is_clean():
    ms = metas(skew="a")  # LLM "a" has twice the traces
    sp = X.make_split(ms, 1)
    sets = X.train_sets(sp, 1)
    sizes = {k: sum(len(i) for _, i in v) for k, v in sets.items() if k.startswith("single")}
    assert len(set(sizes.values())) == 1
    lolo = {k: sum(len(i) for _, i in v) for k, v in sets.items() if k.startswith("lolo")}
    assert all(abs(v - sizes["single:a"]) <= 1 for v in lolo.values())
    assert all(not any(l == k.split(":")[1] for l, _ in v) for k, v in sets.items() if k.startswith("lolo"))
    for k, a in X.audit(sp, sets, ms).items():
        assert a["groups"] == a["attack_ids"] == a["trace_ids"] == 0 and a["attacks"] > 0


def test_llm_name_is_not_a_graph_feature():
    import json
    from src.graph.build import trace_to_graph
    from src.cache import graph_digest
    tr = {"benchmark": "asb", "trace_id": "x|m1|0", "model": "m1", "agent": "ag", "group": "g", "task": "t", "user_task": "t",
          "memory": ["m"], "injected": "", "attack": None, "tools": [{"name": "T", "description": "d"}],
          "events": [{"tool": "T", "args": {"q": "1"}, "observation": "o", "step": 0, "label": "approve", "synthetic": False}]}
    other = json.loads(json.dumps(tr))
    other["model"], other["trace_id"] = "m2", "x|m2|0"
    assert graph_digest(trace_to_graph(tr)) == graph_digest(trace_to_graph(other))


def test_predict_many_slices_targets_and_trains_once():
    from src.graph.build import meta_of, trace_to_graph

    def trace(i, label, attack):
        return {"benchmark": "asb", "trace_id": str(i), "model": "m", "agent": "ag", "group": "g%d" % i, "task": "t", "user_task": "t%d" % i,
                "memory": [], "injected": "do X" if attack else "", "attack": {"mode": "opi", "style": "s", "tool": "X"} if attack else None,
                "tools": [{"name": "T", "description": "d"}, {"name": "X", "description": "x"}],
                "events": [{"tool": "X" if attack else "T", "args": {}, "observation": "o", "step": 0, "label": label, "synthetic": False}]}
    trs = [trace(i, "reject" if i % 2 else "approve", i % 2) for i in range(24)]
    gs = [trace_to_graph(t) for t in trs]
    mm = [meta_of(t) for t in trs]
    out = X.predict_many(gs[:16], mm[:16], {"p": (gs[16:20], mm[16:20]), "q": (gs[20:], mm[20:])}, epochs=2, seed=0)
    assert set(out) == {"p", "q"} and out["p"]["_n"]["actions"] == 4 and out["q"]["_n"]["actions"] == 4
    assert set(X.MODELS) | {m + "|real" for m in X.MODELS} <= set(out["p"])


def test_summary_verdict_and_gap():
    import cross_llm_transfer as C
    run = lambda d, o: {"targets": {"a": {"gnn": {"macro_f1": d, "recall": {"reject": 1}, "asr": {}}},
                                    "b": {"gnn": {"macro_f1": o, "recall": {"reject": 1}, "asr": {}}}}}
    R = {"xllm/s/single:a": [run(.9, .8)], "xllm/s/single:b": [run(.8, .9)]}
    R["xllm/s/single:b"] = [{"targets": {"a": {"gnn": {"macro_f1": .7, "recall": {"reject": 1}, "asr": {}}},
                                         "b": {"gnn": {"macro_f1": .9, "recall": {"reject": 1}, "asr": {}}}}}]
    R["xllm/s/lolo:a"] = [run(.85, 0)]
    R["xllm/s/lolo:b"] = [{"targets": {"b": {"gnn": {"macro_f1": .88, "recall": {"reject": 1}, "asr": {}}}}}]
    rows = {r["llm"]: r for r in C.transfer_summary(R, "s", ["a", "b"], "gnn")}
    assert abs(rows["a"]["gap"][0] - .1) < 1e-9 and abs(rows["b"]["gap"][0] - .2) < 1e-9
    assert rows["a"]["lolo"][0] == .85
    assert C.verdict(.02) == "transfers" and C.verdict(.05) == "partial" and C.verdict(.3) == "weak"
