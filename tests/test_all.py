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

from src.benchmarks import agentdojo, asb, injecagent  # noqa: E402
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
    for mod, d in [(asb, "data/asb"), (injecagent, "data/injecagent"), (agentdojo, "data/agentdojo")]:
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
    for mod, d in [(asb, "data/asb"), (injecagent, "data/injecagent"), (agentdojo, "data/agentdojo")]:
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
