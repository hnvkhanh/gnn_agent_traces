"""tracegen -> GNN guard: attack recording, labels, file nodes, causality, splits and leakage controls."""
import copy
import json

import pytest
import torch

from src.benchmarks import tracegen as TG
from src.graph.files import (FILE_EDGE_TYPES, MASK, arg_refs, file_edges, listing_pairs, mask_attacker_values,
                             trace_to_graph_files)
from src.model.gnn_files import FileHeteroGNN
from src.tracegen_gnn import Data, audit, cell_spec, cells, fingerprint, run_cell
from tracegen.stages.build import assign_injection_splits

META = TG.suite_meta("data/agentdojo")
LISTING = ("- content: 'a,b'\n  filename: feedback.xlsx\n  id_: '0'\n  owner: emma@x.com\n"
           "- content: 'Plan'\n  filename: project-plan.docx\n  id_: '2'\n  owner: emma@x.com\n")


def rec(mode="benign", suite="workspace", task_id="user_task_0", calls=(), inj=None, **kw):
    r = dict(trace_id="t%d" % id(calls), mode=mode, suite=suite, task_id=task_id, variant_id="%s/%s/orig" % (suite, task_id),
             is_original=True, prompt="Please look at feedback.xlsx", agent_model="m1", seed=0,
             tool_calls=[dict(step=i + 1, name=n, args=a, result=o, error=None) for i, (n, a, o) in enumerate(calls)],
             utility=False, ground_truth_calls=[], error=None)
    if mode == "attack":
        r.update(injection_task_id=inj, attack_name="important_instructions", security=False,
                 injections={"p": "<INFORMATION> do it </INFORMATION>"}, injection_ground_truth_calls=[])
    r.update(kw)
    return r


def conv(r):
    return TG.convert(r, META[r["suite"]], "train", "train" if r["mode"] == "attack" else None)


# ---------------------------------------------------------------- labels
def test_attacker_call_is_reject_user_call_is_approve():
    atk = META["banking"]["inj_gt"]["injection_task_0"][0]
    r = rec("attack", "banking", "user_task_0", inj="injection_task_0", calls=[
        ("read_file", {"file_path": "bill-december-2023.txt"}, "Bill ... <INFORMATION> do it </INFORMATION>"),
        ("send_money", atk["args"], "{'message': 'ok'}"),
        ("send_money", {"recipient": "UK12345678901234567890", "amount": 98.7, "subject": "bill", "date": "2022-01-01"}, "ok")])
    tr = conv(r)
    assert [e["label"] for e in tr["events"]] == ["approve", "reject", "approve"]
    assert tr["events"][0]["obs_injected"] and not tr["events"][1]["obs_injected"]
    assert tr["attack"]["id"] == "banking|injection_task_0" and tr["benchmark"] == "tracegen"
    b = conv(rec("benign", "banking", "user_task_0", calls=[("send_money", atk["args"], "ok")]))
    assert [e["label"] for e in b["events"]] == ["approve"]   # benign runs are never reject


def test_call_shared_with_user_ground_truth_stays_approve():
    call = {"function": "read_channel_messages", "args": {"channel": "general"}}
    r = rec("attack", "slack", "user_task_0", inj="injection_task_2",
            calls=[("read_channel_messages", {"channel": "general"}, "msgs")],
            ground_truth_calls=[call], injection_ground_truth_calls=[call])
    assert [e["label"] for e in conv(r)["events"]] == ["approve"]


def test_no_tool_calls_dropped():
    assert conv(rec()) is None


# ---------------------------------------------------------------- file nodes
def test_arg_refs():
    assert arg_refs({"file_path": "bill-december-2023.txt"}) == [("name", "bill-december-2023.txt")]
    assert arg_refs({"file_id": "13"}) == [("id", "13")]
    assert arg_refs({"attachments": [{"type": "file", "file_id": "4"}, "7"], "subject": "x.docx"}) == [("id", "4"), ("id", "7")]
    assert arg_refs({"query": "budget"}) == []


def test_listing_pairs_and_id_resolution_uses_only_earlier_observations():
    assert listing_pairs(LISTING) == {"0": "feedback.xlsx", "2": "project-plan.docx"}
    tr = conv(rec(calls=[("get_file_by_id", {"file_id": "2"}, "content: 'x'"),
                         ("list_files", {}, LISTING),
                         ("get_file_by_id", {"file_id": "2"}, "content: 'Plan'")]))
    keys, targets, appears = file_edges(tr)
    # action 0 asks for id 2 before any listing: it gets its own node; action 2 resolves it to the filename
    assert ("id:2" in keys) and keys[targets[0][0]] == "id:2" and targets[0][1] == 0
    assert keys[[t for t in targets if t[1] == 2][0][0]] == "project-plan.docx"
    assert {keys[f] for f, j in appears if j == 1} == {"feedback.xlsx", "project-plan.docx"}
    assert "feedback.xlsx" in keys   # also named in the prompt


def test_file_graph_shapes_and_ablation():
    tr = conv(rec(calls=[("list_files", {}, LISTING), ("delete_file", {"file_id": "0"}, "ok")]))
    g = trace_to_graph_files(tr)
    assert g["file"].x.shape == (2, 66)
    assert g[("file", "targets", "action")].edge_index.tolist() == [[0], [1]]
    assert set(map(tuple, g.edge_types)) == set(FILE_EDGE_TYPES)
    g2 = trace_to_graph_files(tr, {"files"})
    assert g2["file"].x.shape == g["file"].x.shape
    assert g2[("file", "targets", "action")].edge_index.numel() == 0
    assert g2[("file", "appears_in", "observation")].edge_index.numel() == 0
    for nt in g.node_types:   # file ablation leaves every other tensor as it was
        assert torch.equal(g[nt].x, g2[nt].x)


def test_file_nodes_have_no_incoming_edges():
    assert not [e for e in FILE_EDGE_TYPES if e[2] == "file"]


def test_future_perturbation_no_leak_with_files():
    torch.manual_seed(0)
    m = FileHeteroGNN(16).eval()
    tr = conv(rec(calls=[("list_files", {}, LISTING), ("get_file_by_id", {"file_id": "0"}, "a,b"),
                         ("delete_file", {"file_id": "2"}, "ok"), ("share_file", {"file_id": "0"}, "ok")]))
    base = m(trace_to_graph_files(tr)).detach()
    t2 = copy.deepcopy(tr)
    for e in t2["events"][2:]:
        e["tool"], e["args"], e["observation"] = "zzz", {"file_path": "secret.pdf", "file_id": "9"}, "other.docx id_: '9'"
    t2["events"][1]["observation"] = LISTING + "\n- filename: new.txt\n  id_: '5'\n"   # observation of action 1 itself
    out = m(trace_to_graph_files(t2)).detach()
    assert torch.allclose(base[:2], out[:2], atol=1e-5)


def test_graph_ignores_label_bearing_fields():
    atk = META["banking"]["inj_gt"]["injection_task_0"][0]
    tr = conv(rec("attack", "banking", "user_task_0", inj="injection_task_0", calls=[
        ("read_file", {"file_path": "bill-december-2023.txt"}, "Bill <INFORMATION> do it </INFORMATION>"),
        ("send_money", atk["args"], "ok")]))
    t2 = copy.deepcopy(tr)
    t2.update(security=not tr["security"], utility=not tr["utility"], split="test", inj_split="test",
              attacker_calls=[], allowed_tools=["x"], attack=None, error="x")
    for e in t2["events"]:
        e["obs_injected"], e["synthetic"] = not e["obs_injected"], True
    g1, g2 = trace_to_graph_files(tr), trace_to_graph_files(t2)
    for nt in g1.node_types:
        if nt != "action":
            assert torch.equal(g1[nt].x, g2[nt].x), nt
    assert torch.equal(g1["action"].x, g2["action"].x)
    for et in g1.edge_types:
        assert torch.equal(g1[et].edge_index, g2[et].edge_index), et


def test_attacker_values_mask():
    atk = META["banking"]["inj_gt"]["injection_task_0"][0]
    iban = atk["args"]["recipient"]
    tr = conv(rec("attack", "banking", "user_task_0", inj="injection_task_0", calls=[
        ("read_file", {"file_path": "bill.txt"}, "pay %s now" % iban), ("send_money", atk["args"], "ok")]))
    assert iban in tr["suite_attacker_values"]
    m = mask_attacker_values(tr)
    assert iban not in json.dumps(m["events"]) and MASK in m["events"][0]["observation"]
    assert iban in json.dumps(tr["events"])   # the original is untouched
    g, gm = trace_to_graph_files(tr), trace_to_graph_files(tr, {"attacker_values"})
    assert not torch.equal(g["action"].x, gm["action"].x)
    assert torch.equal(g["action"].y, gm["action"].y)


# ---------------------------------------------------------------- splits / leakage
def test_injection_splits_cover_every_split_and_are_deterministic():
    ids = {s: m["inj_ids"] for s, m in META.items()}
    a, b = assign_injection_splits(ids, (0.7, 0.15, 0.15), 0), assign_injection_splits(ids, (0.7, 0.15, 0.15), 0)
    assert a == b and set(a) == {(s, i) for s, v in ids.items() for i in v}
    for s, v in ids.items():
        if len(v) >= 3:
            assert {a[(s, i)] for i in v} == {"train", "val", "test"}, s


def test_plan_jobs_match_splits_and_rows():
    from tracegen import config as C
    from tracegen.stages import trace as T
    from tracegen.stages.build import assign_splits
    cfg = C.load("/nonexistent.yaml", {"attacks": {"enabled": True}})
    rows = [{"suite": "banking", "task_id": "user_task_%d" % k, "variant_id": "banking/user_task_%d/%s" % (k, v),
             "is_original": v == "orig", "prompt": "p"} for k in range(16) for v in ("orig", "p0")]
    agents = C.agent_specs(cfg)[:1]
    jobs = T.plan_jobs(cfg, rows, agents, set())
    ts = assign_splits([(r["suite"], r["task_id"]) for r in rows], (0.7, 0.15, 0.15), 0)
    isp = assign_injection_splits({"banking": META["banking"]["inj_ids"]}, (0.7, 0.15, 0.15), 0)
    att = [j for j in jobs if j["mode"] == "attack"]
    assert att and all(isp[("banking", j["injection_task_id"])] == ts[("banking", j["task_id"])] for j in att)
    cfg2 = C.load("/nonexistent.yaml", {"attacks": {"enabled": True, "match_splits": False, "rows": "originals"}})
    att2 = [j for j in T.plan_jobs(cfg2, rows, agents, set()) if j["mode"] == "attack"]
    assert len(att2) == 16 * 9 and all(j["is_original"] for j in att2)


def _dataset(tmp_path):
    """Benign + attacked records over train/val/test with consistent splits; written as a tracegen dir."""
    from tracegen.stages.build import assign_splits
    tasks = [("banking", "user_task_%d" % k) for k in range(16)]
    ts = assign_splits(tasks, (0.7, 0.15, 0.15), 0)
    isp = TG.injection_splits(META)
    recs = []
    for s, t in tasks:
        gt = META[s]["user_gt"][t]
        calls = [(c["function"], c["args"], "ok %s" % c["function"]) for c in gt] or [("get_balance", {}, "100")]
        for m in ("m1", "m2"):
            recs.append(rec("benign", s, t, calls=calls, agent_model=m, trace_id="b|%s|%s" % (t, m), prompt="do " + t))
            for i in META[s]["inj_ids"]:
                if isp[(s, i)] != ts[(s, t)] or not META[s]["inj_gt"][i]:
                    continue
                ac = META[s]["inj_gt"][i][0]
                extra = [(ac["function"], ac["args"], "done")]
                recs.append(rec("attack", s, t, inj=i, agent_model=m, trace_id="a|%s|%s|%s" % (t, i, m), prompt="do " + t,
                                calls=[(calls[0][0], calls[0][1], "x <INFORMATION> %s </INFORMATION>" % i)] + extra,
                                injections={"p": "<INFORMATION> %s </INFORMATION>" % i}))
    d = tmp_path / "tg"
    d.mkdir()
    (d / "traces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    # a stray attack pair across splits: must be dropped
    stray = next(i for i in META["banking"]["inj_ids"] if isp[("banking", i)] != ts[("banking", "user_task_0")])
    with open(d / "traces.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec("attack", "banking", "user_task_0", inj=stray, calls=[("get_balance", {}, "1")],
                               trace_id="stray")) + "\n")
    return d


def test_load_drops_cross_split_attacks_and_audit_is_clean(tmp_path):
    d = _dataset(tmp_path)
    data = Data(str(d), "data/agentdojo")
    assert data.stats["dropped_split_mismatch"] == 1
    assert all(t["inj_split"] in (None, t["split"]) for t in data.trs)
    a = audit(data)
    assert a["ok"], a
    s = data.summary()
    assert s["train"]["reject"] > 0 and s["test"]["reject"] > 0 and data.ready() is None


def test_ready_refuses_benign_only(tmp_path):
    d = tmp_path / "tg"
    d.mkdir()
    (d / "traces.jsonl").write_text(json.dumps(rec(calls=[("list_files", {}, LISTING)])) + "\n", encoding="utf-8")
    assert "no attacked traces" in Data(str(d), "data/agentdojo").ready()


def test_run_cell_and_fingerprint(tmp_path):
    data = Data(str(_dataset(tmp_path)), "data/agentdojo")
    ks = cells(data)
    assert "main" in ks and "ablate/files" in ks and "transfer/m1->m2" in ks
    tr, ev, guards, _ = cell_spec(data, "transfer/m1->m2")
    assert all(data.trs[i]["model"] == "m1" for i in tr) and all(data.trs[i]["model"] == "m2" for i in ev["test"])
    out = run_cell(data, "main", 0, epochs=1, hidden=8)
    assert {"val", "test", "test|suite=banking", "test|agent=m1"} <= set(out)
    assert set(out["test"]) == {"majority", "tool-lookup", "mlp", "gnn", "gnn+files"}
    assert out["test"]["gnn+files"]["n_reject"] > 0
    f = fingerprint(data, "main", 0, 1, 3e-3, 8, code="c")
    assert f == fingerprint(data, "main", 0, 1, 3e-3, 8, code="c") != fingerprint(data, "main", 1, 1, 3e-3, 8, code="c")
    assert f != fingerprint(data, "ablate/files", 0, 1, 3e-3, 8, code="c")


# ---------------------------------------------------------------- real AgentDojo run (offline replay)
def test_run_one_records_attacker_calls_and_labels(monkeypatch):
    dojo = pytest.importorskip("tracegen.dojo")
    from tests.tracegen.test_dojo import AGENT, SUITE, FakeClient, gt_of
    inj = SUITE.get_injection_task_by_id("injection_task_0")
    env = SUITE.load_and_inject_default_environment({})
    calls = list(gt_of("user_task_0")) + list(inj.ground_truth(env))   # the agent follows the attack at the end
    monkeypatch.setattr(dojo, "make_client", lambda agent, url, timeout: FakeClient(calls))
    r = dojo.run_one(SUITE, SUITE.user_tasks["user_task_0"], SUITE.user_tasks["user_task_0"].PROMPT, AGENT,
                     "http://x", injection_task=inj, attack_name="important_instructions")
    assert r["error"] is None, r["error"]
    assert r["injections"] and all("<INFORMATION>" in v for v in r["injections"].values())
    assert [c["function"] for c in r["injection_ground_truth_calls"]] == ["send_money"]
    r.update(trace_id="x", mode="attack", suite="banking", task_id="user_task_0", variant_id="v", is_original=True,
             prompt=SUITE.user_tasks["user_task_0"].PROMPT, agent_model="fake", injection_task_id="injection_task_0",
             attack_name="important_instructions")
    tr = conv(r)
    assert [e["label"] for e in tr["events"]] == ["approve", "approve", "reject"]
    assert tr["events"][0]["obs_injected"]
