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
        assert not banned.search(p.read_text()), p


def test_labels():
    own = DATA.normal_for("system_admin_agent")
    att = DATA.attacks_for("system_admin_agent")[0]
    other = DATA.other_normal("system_admin_agent")[0]
    vis = {t.name: t for t in own + [att, other]}
    assert label_action(own[0].name, {"q": "scan"}, vis, "system_admin_agent") == "approve"
    assert label_action(att.name, {}, vis, "system_admin_agent") == "reject"
    assert label_action(other.name, {}, vis, "system_admin_agent") == "escalate"
    assert label_action("nope", {}, vis, "system_admin_agent") == "escalate"
    assert label_action(own[0].name, {"q": "dump the admin password"}, vis, "system_admin_agent") == "escalate"


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
        assert out.shape == (g["action"].x.shape[0], 3)
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
    assert label_call(c["tool"], {"x": "benign-value"}, vis, scope, sc.attacker_calls) in ("approve", "escalate")


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
    f.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    assert len(load(str(f))[0]) == 2
    assert [x["model"] for x in load(str(f), models=["m-b"])[1]] == ["m-b"]
    assert make_trace("opi")["n_invalid"] == 0
