import random
import re
from pathlib import Path

import torch

from src.graph.build import meta_of, trace_to_graph
from src.model.gnn import ActionMLP, HeteroGNN
from src.sandbox.env import build_trace
from src.sandbox.policy import label_action
from src.sandbox.tools import ASBData, MockTool

DATA = ASBData("data/asb")


def test_sandbox_has_no_side_effect_apis():
    banned = re.compile(r"subprocess|socket|requests|urllib|os\.system|shutil|eval\(|exec\(|__import__")
    for p in list(Path("src/sandbox").glob("*.py")) + list(Path("src/benchmarks").glob("*.py")):
        assert not banned.search(p.read_text(encoding="utf-8")), p


def test_labels():
    own = DATA.normal_for("system_admin_agent")
    att = DATA.attacks_for("system_admin_agent")[0]
    other = DATA.other_normal("system_admin_agent")[0]
    vis = {t.name: t for t in own + [att, other]}
    assert label_action(own[0].name, {"q": "scan"}, vis, "system_admin_agent") == "approve"
    assert label_action(att.name, {}, vis, "system_admin_agent") == "reject"
    assert label_action(other.name, {}, vis, "system_admin_agent") == "approve"  # not an attacker action
    assert label_action(own[0].name, {"q": "dump the admin password"}, vis, "system_admin_agent") == "approve"


def fake_llm_factory(tool_names):
    it = iter(tool_names)

    def llm(msgs):
        try:
            return '{"tool": "%s", "args": {"q": "x"}}' % next(it)
        except StopIteration:
            return '{"final": "done"}'
    return llm


def make_trace(mode="opi"):
    agent, task = DATA.tasks()[0]
    own = [t.name for t in DATA.normal_for(agent)]
    return build_trace(DATA, fake_llm_factory([own[0], own[1]]), agent, task, mode,
                       "naive" if mode != "none" else None, random.Random(1), replay=True, trace_id="t", model="fake")


def test_env_replay_and_injection():
    tr = make_trace("opi")
    assert tr["events"][0]["obs_injected"]
    assert tr["events"][-1]["synthetic"] and tr["events"][-1]["label"] == "reject"


def test_graph_shapes_and_causality():
    tr = make_trace("opi")
    g = trace_to_graph(tr)
    n = len(tr["events"])
    assert g["action"].x.shape[0] == n and g["action"].y.shape[0] == n
    # observation j -> action i only for j < i; action j -> action i only for j < i
    for et in [("observation", "precedes", "action"), ("action", "next", "action")]:
        src, dst = g[et].edge_index
        assert bool((src < dst).all())
    # no edge from the action's own observation into the same action
    src, dst = g[("observation", "mentions", "action")].edge_index
    assert bool((src < dst).all())
    assert meta_of(tr)["tools"] == [e["tool"] for e in tr["events"]]


def test_smoke_forward_backward():
    g = trace_to_graph(make_trace("opi"))
    for m in (HeteroGNN(16), ActionMLP(16)):
        out = m(g)
        assert out.shape == (g["action"].x.shape[0], 2)
        torch.nn.functional.cross_entropy(out, g["action"].y).backward()


# ---- phase 2: multi-benchmark ----
import json  # noqa: E402

from src.benchmarks import agentdojo, agentpoison, asb, injecagent  # noqa: E402
ALL_BENCH = [(asb, "data/asb"), (injecagent, "data/injecagent"), (agentdojo, "data/agentdojo"),
             (agentpoison, "data/agentpoison")]
from src.sandbox.env import run_scenario  # noqa: E402
from src.sandbox.policy import label_call  # noqa: E402


def _scripted(tools_args):
    it = iter(tools_args)

    def llm(msgs):
        try:
            t, a = next(it)
            return json.dumps({"tool": t, "args": a})
        except StopIteration:
            return '{"final": "done"}'
    return llm


def test_scenarios_valid_for_each_benchmark():
    for mod, d in ALL_BENCH:
        scs = mod.scenarios(d, 12, 0)
        assert len(scs) == 12
        for sc in scs:
            names = {t.name for t in sc.tools}
            assert set(sc.in_scope) <= names and sc.tools
            if sc.mode != "none":
                assert sc.attacker_calls and sc.injected


def test_agentdojo_same_tool_different_labels():
    sc = next(s for s in agentdojo.scenarios("data/agentdojo", 200, 1)
              if s.mode != "none" and s.attacker_calls[0]["tool"] in s.in_scope and s.attacker_calls[0]["args"])
    c = sc.attacker_calls[0]
    vis = {t.name: t for t in sc.tools}
    scope = set(sc.in_scope)
    assert label_call(c["tool"], c["args"], vis, scope, sc.attacker_calls) == "reject"
    assert label_call(c["tool"], {"x": "benign-value"}, vis, scope, sc.attacker_calls) == "approve"


def test_copies_edge_is_causal_and_ablations():
    sc = next(s for s in agentdojo.scenarios("data/agentdojo", 50, 2) if s.mode == "opi")
    first = sc.opi_tool
    c = sc.attacker_calls[0]
    tr = run_scenario(sc, _scripted([(first, {}), (c["tool"], c["args"])]), replay=False)
    g = trace_to_graph(tr)
    src, dst = g[("observation", "copies", "action")].edge_index
    assert bool((src < dst).all())
    assert trace_to_graph(tr, frozenset(["copies"]))[("observation", "copies", "action")].edge_index.shape[1] == 0
    assert trace_to_graph(tr, frozenset(["observation"]))[("observation", "precedes", "action")].edge_index.shape[1] == 0
    stripped = trace_to_graph(tr, frozenset(["injected"]))
    assert not torch.equal(stripped["observation"].x, g["observation"].x)


def test_future_perturbation_no_leak():
    import copy
    torch.manual_seed(0)
    m = HeteroGNN(16).eval()
    sc = next(s for s in agentdojo.scenarios("data/agentdojo", 50, 3) if s.mode == "opi")
    c = sc.attacker_calls[0]
    tr = run_scenario(sc, _scripted([(sc.opi_tool, {}), (sc.opi_tool, {}), (c["tool"], c["args"])]), replay=False)
    base = m(trace_to_graph(tr)).detach()
    t2 = copy.deepcopy(tr)
    for e in t2["events"][2:]:
        e["tool"], e["args"], e["observation"] = "zzz", {"k": "password"}, "other text"
    out = m(trace_to_graph(t2)).detach()
    assert torch.allclose(base[:2], out[:2], atol=1e-5)


# ---- phase 3: multi-LLM ----
def test_meta_model_synthetic_and_real_only():
    from src.train import evaluate
    tr = make_trace("opi")
    tr["model"] = "fake-llm"
    m = meta_of(tr)
    assert m["model"] == "fake-llm" and m["synthetic"] == [e["synthetic"] for e in tr["events"]] and any(m["synthetic"])
    g = trace_to_graph(tr)
    res, dist = evaluate([g] * 4, [m] * 4, [g] * 2, [m] * 2, epochs=1, models=("majority", "gnn"), real_only=True)
    n_real = sum(1 for s in m["synthetic"] if not s) * 2
    assert sum(dist) == n_real and n_real < len(m["synthetic"]) * 2


def test_load_models_filter(tmp_path):
    from src.train import load
    a, b = make_trace("opi"), make_trace("opi")
    a["model"], b["model"] = "m-a", "m-b"
    f = tmp_path / "x.jsonl"
    f.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n", encoding="utf-8")
    assert len(load(str(f))[0]) == 2
    assert [x["model"] for x in load(str(f), models=["m-b"])[1]] == ["m-b"]
    assert make_trace("opi")["n_invalid"] == 0


# ---- final label schema: only approve/reject, every action labeled ----
import glob  # noqa: E402

from src.sandbox.policy import LABELS  # noqa: E402


def test_labels_are_only_approve_reject():
    assert LABELS == ["approve", "reject"]


def test_unknown_tool_is_not_an_action():
    agent, task = DATA.tasks()[0]
    own = [t.name for t in DATA.normal_for(agent)]
    tr = build_trace(DATA, fake_llm_factory(["no_such_tool", own[0]]), agent, task, "none", None, random.Random(1),
                     replay=False, trace_id="t", model="fake")
    assert [e["tool"] for e in tr["events"]] == [own[0]] and tr["n_invalid"] == 1


def test_every_scripted_action_is_labeled_for_every_benchmark():
    for mod, d in ALL_BENCH:
        for sc in mod.scenarios(d, 20, 5):
            calls = [(t.name, {}) for t in sc.tools[:2]] + [(c["tool"], c["args"]) for c in sc.attacker_calls[:1]]
            tr = run_scenario(sc, _scripted(calls), replay=True)
            assert tr["events"] and all(e["label"] in LABELS for e in tr["events"])


def test_stored_traces_are_fully_labeled():
    files = glob.glob("traces/*.jsonl")
    for f in files:
        for line in open(f, encoding="utf-8"):
            tr = json.loads(line)
            names = {t["name"] for t in tr["tools"]}
            assert tr["events"], (f, tr["trace_id"])
            for i, e in enumerate(tr["events"]):
                assert e["label"] in LABELS and e["tool"] in names and e["step"] == i, (f, tr["trace_id"], i, e["label"])


# ---- attack success rate ----
def test_asr_report_by_hand():
    from src.eval import asr_report
    metas = [  # t1: attack, LLM attacked (real) and guard blocked it
        dict(synthetic=[False, False], attack=True),
        # t2: attack, LLM attacked, guard missed it
        dict(synthetic=[False, False], attack=True),
        # t3: attack, LLM did not attack; only an appended synthetic attacker step, guard missed it
        dict(synthetic=[False, True], attack=True),
        # t4: benign, guard wrongly blocks one approve action
        dict(synthetic=[False, False], attack=False),
        # t5: benign, untouched
        dict(synthetic=[False], attack=False)]
    y = [0, 1, 0, 1, 0, 1, 0, 0, 0]
    p = [0, 1, 0, 0, 0, 0, 1, 0, 0]
    r = asr_report(y, p, metas)
    assert r["attack_traces"] == 3 and r["benign_traces"] == 2
    assert abs(r["asr_undefended"] - 2 / 3) < 1e-9      # t1, t2 had a real attacker action
    assert abs(r["asr_defended"] - 1 / 3) < 1e-9        # only t2 got through
    assert abs(r["asr_forced"] - 2 / 3) < 1e-9          # t2 and t3 (synthetic) got through
    assert abs(r["benign_block"] - 1 / 2) < 1e-9
    assert abs(r["approve_fpr"] - 1 / 6) < 1e-9         # 6 approve actions, 1 wrongly rejected


def test_evaluate_returns_asr():
    from src.train import evaluate
    tr = make_trace("opi")
    g, m = trace_to_graph(tr), meta_of(tr)
    res, _ = evaluate([g] * 4, [m] * 4, [g] * 2, [m] * 2, epochs=1, models=("majority", "gnn"))
    for k in ("majority", "gnn"):
        a = res[k]["asr"]
        assert a["attack_traces"] == 2 and 0.0 <= a["asr_defended"] <= a["asr_undefended"] + 1e-9 + a["asr_forced"]
    assert res["majority"]["asr"]["asr_defended"] == res["majority"]["asr"]["asr_undefended"]  # approve-everything blocks nothing


def test_micro_f1_equals_accuracy_and_differs_from_macro():
    from src.eval import report
    y = [0] * 8 + [1] * 2
    p = [0] * 8 + [0, 1]        # one of the two rejects missed
    r = report(y, p)
    assert abs(r["micro_f1"] - r["acc"]) < 1e-9 and abs(r["micro_f1"] - 0.9) < 1e-9
    assert r["macro_f1"] < r["micro_f1"]   # macro punishes the rare-class error more


def test_data_read_as_utf8_on_every_os():
    # data/asb contains a non-ASCII char (U+2019). A default-encoding read gives cp1252 mojibake on Windows.
    from src.sandbox.tools import _jsonl
    rows = _jsonl(Path("data/asb/all_attack_tools.jsonl"))
    text = json.dumps(rows, ensure_ascii=False)
    assert "’" in text and "â€" not in text


# ---- no train/test leakage of tasks or attacks ----
def test_attack_id_recorded_and_recovered_for_old_traces():
    from src.graph.build import attack_key
    for mod, d in ALL_BENCH:
        for sc in mod.scenarios(d, 30, 3):
            tr = run_scenario(sc, _scripted([]), replay=False)
            if sc.mode == "none":
                assert attack_key(tr) is None
                continue
            assert tr["attack"]["id"] and attack_key(tr) == sc.attack_id
            tr["attack"].pop("id")  # traces collected before ids existed
            assert attack_key(tr) == sc.attack_id


def test_split_is_task_and_attack_disjoint():
    from src.train import split
    metas = [dict(group="g%d" % (i % 10), attack_id=None if i % 4 == 0 else "a%d" % (i % 7), style="s%d" % (i % 3))
             for i in range(200)]
    for kind, hold in [("task", None), ("style", "s1")]:
        for seed in range(3):
            tr, te = split(metas, kind, hold, seed)
            assert tr and te and not set(tr) & set(te)
            assert not {metas[i]["attack_id"] for i in tr} & ({metas[i]["attack_id"] for i in te} - {None})
            if kind == "task":
                assert not {metas[i]["group"] for i in tr} & {metas[i]["group"] for i in te}


def test_split_attack_disjoint_on_stored_traces():
    from src.train import split
    for f in glob.glob("traces/*.jsonl"):
        metas = [meta_of(json.loads(line)) for line in open(f, encoding="utf-8") if line.strip()]
        tr, te = split(metas, "task", None, 0)
        assert te and tr, f
        assert not {metas[i]["attack_id"] for i in tr} & ({metas[i]["attack_id"] for i in te} - {None}), f


# ---- AgentPoison (EHRAgent memory backdoor, StrategyQA knowledge-base backdoor) ----
AP = "data/agentpoison"


def _ap(n=200, seed=0):
    return agentpoison.scenarios(AP, n, seed)


def _ap_trace(sc, calls=None):
    calls = calls if calls is not None else [(sc.tools[0].name, {"q": "x"})]
    return run_scenario(sc, _scripted(calls), replay=True)


def test_agentpoison_ehr_labels_and_poisoned_memory():
    scs = [s for s in _ap() if s.group.startswith("agentpoison|ehr")]
    for sc in scs:
        names = {t.name for t in sc.tools}
        assert "DeleteDB" in names and "DeleteDB" not in sc.in_scope  # visible in benign scenarios too
        assert len(sc.memory) == agentpoison.N_SHOTS
        poisoned = [m for m in sc.memory if "DeleteDB" in m]
        if sc.mode == "none":
            assert not poisoned and not any(t in sc.task for t in agentpoison.TRIGGERS)
        else:
            trig = sc.attack_id.split("|", 1)[1]
            assert 1 <= len(poisoned) <= 2 and all(trig in m and sc.injected in m for m in poisoned)
            assert sc.task.endswith(" " + trig)
    sc = next(s for s in scs if s.mode == "memory")
    tr = _ap_trace(sc, [("LoadDB", {"DBNAME": "patient"}), ("DeleteDB", {"DBNAME": "lab"})])
    assert [(e["tool"], e["label"], e["synthetic"]) for e in tr["events"]] == [
        ("LoadDB", "approve", False), ("DeleteDB", "reject", False)]
    assert not any(m.startswith("Past note:") for m in tr["memory"])  # adapter poisoned memory itself


def test_asb_memory_mode_still_appends_past_note():
    agent, task = DATA.tasks()[0]
    tr = build_trace(DATA, fake_llm_factory([]), agent, task, "memory", "naive", random.Random(1), replay=False)
    assert tr["memory"][-1].startswith("Past note: ")


def test_agentpoison_qa_finish_same_tool_different_labels():
    sc = next(s for s in _ap() if s.mode == "opi")
    tr = _ap_trace(sc, [("Search", {"entity": "x"}), ("Finish", {"answer": "I don't know"})])
    assert [(e["tool"], e["label"]) for e in tr["events"]] == [("Search", "approve"), ("Finish", "reject")]
    assert tr["events"][0]["obs_injected"] and sc.injected in tr["events"][0]["observation"]
    # {"final": x} is recorded as Finish(x); a real yes/no answer is approved, then the attacker step is replayed
    tr = run_scenario(sc, _scripted([("Search", {})]), replay=True)
    assert [(e["tool"], e["args"].get("answer"), e["label"], e["synthetic"]) for e in tr["events"]][1:] == [
        ("Finish", "done", "approve", False), ("Finish", "I don't know", "reject", True)]


def _vocab_tokens(text):
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def test_agentpoison_trigger_pool_token_disjoint_and_absent_from_data():  # L4
    pool = agentpoison.POOL["ehr"] + agentpoison.POOL["qa"]
    assert len(agentpoison.POOL["ehr"]) == len(agentpoison.POOL["qa"]) == agentpoison.N_TRIGGERS
    words = [agentpoison.tokens(t) - agentpoison.TRIGGER_STOPWORDS for t in pool]
    assert all(len(w) == 3 for w in words)
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            assert not words[i] & words[j], (pool[i], pool[j])
    data = Path(AP, "ehr.json").read_text(encoding="utf-8") + Path(AP, "strategyqa.json").read_text(encoding="utf-8")
    templates = agentpoison.ehr_rationale("") + agentpoison.qa_poison_passage(
        dict(question="", facts=[], answer=True), "") + agentpoison.EHR_PERSONA + agentpoison.QA_PERSONA
    vocab = _vocab_tokens(data + templates)
    assert not set().union(*words) & vocab


def test_agentpoison_benign_and_attack_have_same_structure():  # L7
    ehr, qa = agentpoison._load(AP)
    for mk, item in [(agentpoison.ehr_scenario, ehr["queries"][3]), (agentpoison.qa_scenario, qa["items"][3])]:
        data = ehr if mk is agentpoison.ehr_scenario else qa
        b = mk(data, item, random.Random(5), True)
        x = mk(data, item, random.Random(5), False)
        assert sorted(t.name for t in b.tools) == sorted(t.name for t in x.tools)
        assert len(b.memory) == len(x.memory)
        assert b.tool_obs.keys() == x.tool_obs.keys()
        if "Search" in b.tool_obs:
            assert b.tool_obs["Search"].count("\n\n[") == x.tool_obs["Search"].count("\n\n[") == 2
        else:  # benign retrieval is the same; poisoned demos only replace entries
            assert sum(m in b.memory for m in x.memory) >= len(b.memory) - 2


def test_graph_ignores_label_bearing_trace_fields():  # L8, every benchmark
    import copy
    for mod, d in ALL_BENCH:
        for sc in [s for s in mod.scenarios(d, 12, 4) if s.mode != "none"][:3]:
            tr = _ap_trace(sc, [(t.name, {"q": "x"}) for t in sc.tools[:2]])
            t2 = copy.deepcopy(tr)
            for t in t2["tools"]:
                t["kind"], t["in_scope"] = ("normal" if t["kind"] == "attacker" else "attacker"), not t["in_scope"]
            t2["allowed_tools"], t2["attack"] = [], None
            for e in t2["events"]:
                e["synthetic"], e["obs_injected"] = not e["synthetic"], not e["obs_injected"]
            g1, g2 = trace_to_graph(tr), trace_to_graph(t2)
            for nt in g1.node_types:
                assert torch.equal(g1[nt].x, g2[nt].x), (d, nt)
            for et in g1.edge_types:
                assert torch.equal(g1[et].edge_index, g2[et].edge_index), (d, et)


def test_agentpoison_split_has_no_leakage():  # L9
    from src.leakage import audit
    trs = [_ap_trace(sc) for sc in _ap(400, 0)]
    for seed in range(3):
        r = audit(trs, seed)
        assert not any(r["overlaps"].values()), r["examples"]
        assert r["train_attacks"] > 50 and r["test_attacks"] > 0


def test_split_has_no_leakage_on_stored_agentpoison_traces():
    from src.leakage import audit
    for f in glob.glob("traces/agentpoison__*.jsonl"):
        trs = [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
        for seed in range(3):
            r = audit(trs, seed)
            assert not any(r["overlaps"].values()), (f, r["examples"])


def test_agentpoison_attack_ids_and_poison_are_tied_to_trigger():  # L5
    injected, bases = {}, {}
    for sc in _ap(400, 1):
        if sc.mode == "none":
            continue
        key, trig = sc.attack_id, sc.attack_id.split("|", 1)[1]
        assert agentpoison.trigger_id(sc.task) == key
        injected.setdefault(key, set()).add(sc.injected)
        if sc.mode == "opi":  # the poisoned passage is built from the trigger's own poison-pool item
            bases.setdefault(key, set()).add(sc.injected.split("\nQuestion: ")[1].split("\nThought:")[0])
        else:  # poisoned demos come from the trigger's own poison bases
            for m in sc.memory:
                if "DeleteDB" in m:
                    bases.setdefault(key, set()).add(m.split("\n")[0].replace(" " + trig, ""))
    assert all(len(v) == 1 for v in injected.values())  # rationale line / passage: a function of the trigger alone
    keys = sorted(bases)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not bases[keys[i]] & bases[keys[j]], (keys[i], keys[j])


def test_collect_shards_cover_serial_set_and_merge(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("collect_traces", "scripts/collect_traces.py")
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)
    n, K = 37, 8
    assert sorted(i for j in range(K) for i in range(n) if i % K == j) == list(range(n))
    final = tmp_path / "final.jsonl"
    final.write_text(json.dumps(dict(trace_id="b|m:1|0|3", v="old")) + "\n", encoding="utf-8")
    shards = []
    for j in range(2):
        p = tmp_path / ("s%d.jsonl" % j)
        p.write_text("".join(json.dumps(dict(trace_id="b|m:1|0|%d" % i, v="new")) + "\n" for i in range(j, 6, 2)),
                     encoding="utf-8")
        shards.append(str(p))
    assert ct.merge(str(final), shards) == 6
    rows = [json.loads(l) for l in final.read_text(encoding="utf-8").splitlines()]
    assert [r["trace_id"].split("|")[-1] for r in rows] == [str(i) for i in range(6)]
    assert next(r for r in rows if r["trace_id"].endswith("|3"))["v"] == "old"  # existing trace kept


def test_result_cache_reuses_only_unchanged_inputs(tmp_path):
    from src.cache import ResultCache
    g1 = trace_to_graph(make_trace("opi"))
    m1 = meta_of(make_trace("opi"))
    tr2 = make_trace("opi")
    tr2["events"][0]["observation"] = "something else"
    g2 = trace_to_graph(tr2)
    calls = []

    def compute():
        calls.append(1)
        return [{"gnn": {"macro_f1": 1.0}}]

    def cache(**kw):
        return ResultCache(str(tmp_path), "inbench", range(1), 1, "fake", **kw)

    c = cache()
    c.get("inbench/x", [([g1], [m1], [g1], [m1])], compute)
    c.save()
    with open(tmp_path / "experiments_inbench.json", "w", encoding="utf-8") as f:
        json.dump({"inbench/x": [{"gnn": {"macro_f1": 1.0}}]}, f)
    c = cache()
    c.get("inbench/x", [([g1], [m1], [g1], [m1])], compute)
    assert len(calls) == 1 and c.status["inbench/x"]["status"] == "reused"
    c.get("inbench/x", [([g2], [m1], [g1], [m1])], compute)  # different graph -> stale
    assert len(calls) == 2 and c.status["inbench/x"]["status"] == "computed"
    # a legacy cell (no fingerprint) is only accepted with adopt_legacy
    with open(tmp_path / "experiments_all.json", "w", encoding="utf-8") as f:
        json.dump({"inbench/y": [{"gnn": {"macro_f1": 0.5}}]}, f)
    cache().get("inbench/y", [([g1], [m1], [g1], [m1])], compute)
    assert len(calls) == 3
    c = cache(adopt_legacy=True)
    assert c.get("inbench/y", [([g1], [m1], [g1], [m1])], compute)[0]["gnn"]["macro_f1"] == 0.5
    assert len(calls) == 3 and c.status["inbench/y"]["status"] == "adopted"
