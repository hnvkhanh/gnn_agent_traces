import json

from tracegen.stages.build import assign_splits, dedup, first_divergence, norm_value, signature


def calls(*pairs):
    return [{"name": n, "args": a} for n, a in pairs]


def test_signature_key_order_and_numbers():
    a = signature(calls(("send_money", {"recipient": "X", "amount": 98.7, "date": "2022-01-01"})))
    b = signature(calls(("send_money", {"date": " 2022-01-01", "amount": "98.70", "recipient": "X"})))
    assert a == b
    assert signature(calls(("f", {"n": 5}))) == signature(calls(("f", {"n": 5.0})))
    assert signature(calls(("f", {"n": 5}))) != signature(calls(("f", {"n": 6})))
    assert signature(calls(("a", {}), ("b", {}))) != signature(calls(("b", {}), ("a", {})))   # order matters
    json.loads(a)


def test_norm_value_nested():
    assert norm_value({"b": [1.0, " x "], "a": {"z": "2.50"}}) == {"a": {"z": 2.5}, "b": [1, "x"]}
    assert norm_value(True) is True and norm_value(None) is None


def tr(task, sig, util, mode="benign", sec=None, suite="banking"):
    t = {"suite": suite, "task_id": task, "mode": mode, "signature": sig, "utility": util}
    if mode == "attack":
        t["security"] = sec
    return t


def test_dedup():
    ts = [tr("t0", "s1", True), tr("t0", "s1", True), tr("t0", "s1", False), tr("t0", "s2", True),
          tr("t1", "s1", True), tr("t0", "s1", True, suite="slack"),
          tr("t0", "s1", True, "attack", True), tr("t0", "s1", True, "attack", True), tr("t0", "s1", True, "attack", False)]
    out = dedup(ts)
    assert len(out) == 7
    assert out[0] is ts[0]   # first occurrence kept


def test_splits_disjoint_deterministic():
    keys = [(s, "user_task_%d" % i) for s in ("banking", "slack") for i in range(20)]
    a = assign_splits(keys, [0.7, 0.15, 0.15], 0)
    assert a == assign_splits(list(reversed(keys)), [0.7, 0.15, 0.15], 0)
    assert set(a) == set(keys)                     # each task gets exactly one split
    for s in ("banking", "slack"):
        cnt = {x: sum(1 for (su, _), v in a.items() if su == s and v == x) for x in ("train", "val", "test")}
        assert cnt == {"train": 14, "val": 3, "test": 3}
    assert a != assign_splits(keys, [0.7, 0.15, 0.15], 1)


def test_split_traces_never_cross():
    keys = [("banking", "t%d" % i) for i in range(16)]
    a = assign_splits(keys, [0.7, 0.15, 0.15], 3)
    traces = [{"suite": s, "task_id": t, "variant": v} for s, t in keys for v in range(5)]
    where = {}
    for t in traces:
        where.setdefault(t["task_id"], set()).add(a[(t["suite"], t["task_id"])])
    assert all(len(v) == 1 for v in where.values())


def test_first_divergence():
    t = {"tool_calls": calls(("read_file", {}), ("send_money", {})),
         "ground_truth_calls": [{"function": "read_file"}, {"function": "get_iban"}, {"function": "send_money"}]}
    assert first_divergence(t) == (1, "get_iban", "send_money")
    t["tool_calls"] = calls(("read_file", {}))
    t["ground_truth_calls"] = [{"function": "read_file"}]
    assert first_divergence(t) is None


def test_build_end_to_end(tmp_path):
    from tracegen import config as C
    from tracegen import io
    from tracegen.stages.build import build
    cfg = C.load("/nonexistent.yaml", {"paths": {"out_dir": str(tmp_path)}})
    io.write_jsonl(tmp_path / "originals.jsonl", [{"suite": "banking", "task_id": "user_task_%d" % i, "prompt": "p"}
                                                  for i in range(4)])
    io.write_jsonl(tmp_path / "candidates.jsonl", [{"variant_id": "banking/user_task_0/p0", "suite": "banking",
                                                    "task_id": "user_task_0", "original": "p", "paraphrase": "q"}])
    io.write_jsonl(tmp_path / "verified_variants.jsonl", [{"variant_id": "banking/user_task_0/p0", "suite": "banking",
                                                           "task_id": "user_task_0", "nli_fwd": 0.9, "nli_bwd": 0.95}])
    gt = [{"function": "read_file", "args": {}}, {"function": "send_money", "args": {}}]
    traces = []
    for i in range(4):
        for vid, orig in (("orig", True), ("p0", False)):
            traces.append({"trace_id": "%d%s" % (i, vid), "mode": "benign", "suite": "banking", "task_id": "user_task_%d" % i,
                           "variant_id": "banking/user_task_%d/%s" % (i, vid), "is_original": orig, "agent_model": "m",
                           "seed": 0, "tool_calls": [{"step": 1, "name": "read_file", "args": {}, "result": "x"}],
                           "ground_truth_calls": gt, "matches_ground_truth_sequence": False, "utility": orig,
                           "n_steps": 1, "error": None})
    traces.append(dict(traces[0], trace_id="a", mode="attack", injection_task_id="injection_task_0",
                       attack_name="important_instructions", security=True))
    io.write_jsonl(tmp_path / "traces.jsonl", traces)
    st = build(cfg, ["banking"], do_dedup=True)
    a = st["suites"]["banking"]["agents"]["m"]
    assert a["utility_orig"] == 1.0 and a["utility_para"] == 0.0 and a["attacks"]["important_instructions"]["asr"] == 1.0
    assert a["top_divergence"][0] == ("call #2: expected send_money, got <stop>", 8)
    assert st["n_after_dedup"] == 9 - 0 and st["leakage"]["tasks_in_multiple_splits"] == []
    split_rows = sum(len(io.read_jsonl(tmp_path / "splits" / (n + ".jsonl"))) for n in ("train", "val", "test"))
    assert split_rows == 9
    assert "<svg" in (tmp_path / "report.html").read_text() and "Variant funnel" in (tmp_path / "report.md").read_text()
