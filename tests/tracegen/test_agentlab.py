"""tracegen.agentlab: offline tests (no Ollama, no network). Tests that need the AgentLAB checkout are skipped until
`python -m tracegen.agentlab setup` has been run."""
import json
import types
from pathlib import Path

import pytest

from tracegen import io
from tracegen.agentlab import config as LC
from tracegen.agentlab import lab
from tracegen.agentlab import stages as S

REPO = Path(__file__).resolve().parents[2] / "external" / "AgentLAB"
needs_repo = pytest.mark.skipif(not lab.ready(REPO), reason="no AgentLAB checkout (python -m tracegen.agentlab setup)")


def cfg_for(tmp_path, **over):
    base = {"paths": {"out_dir": str(tmp_path)}, "repo": {"dir": str(REPO)}, "generate": {"n": 2},
            "benign": {"per_suite": 3, "batch": 2}, "trace": {"agents": ["fake"], "concurrency": 1}}
    return LC.load("does-not-exist.yaml", LC.C.merge(base, over))


class FakeClient:
    """Replies with the scripted tool calls, one per request, then with plain text."""

    def __init__(self, script, fail=False):
        self.script, self.fail, self.requests = list(script), fail, []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.requests.append(kw)
        if self.fail:
            raise ConnectionError("endpoint down")
        last = kw["messages"][-1]
        calls = []
        if last["role"] == "user" and self.script:
            name, args = self.script.pop(0)
            fn = types.SimpleNamespace(name=name, arguments=json.dumps(args))
            calls = [types.SimpleNamespace(id="c%d" % len(self.requests), function=fn)] * 2   # only the first one runs
        msg = types.SimpleNamespace(content="" if calls else "done", tool_calls=calls)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=2))


# ---------------------------------------------------------------- pure helpers
def test_parse_tasks_accepts_fences_dicts_and_prose():
    assert S.parse_tasks('```json\n[{"turns": ["a", "b"]}, {"turns": ["c"]}]\n```', 3) == [["a", "b"], ["c"]]
    assert S.parse_tasks('Here: {"tasks": [{"turns": ["a", "b", "c", "d"]}]}', 3) == [["a", "b", "c"]]
    assert S.parse_tasks('[["x", "y"], "z"]', 3) == [["x", "y"], ["z"]]
    with pytest.raises(ValueError):
        S.parse_tasks("I cannot help with that.", 3)


def test_chain_match_compares_identifiers_not_free_text():
    chain = [{"function": "create_file", "args": {"filename": "A.pdf", "content": "x" * 200}},
             {"function": "send_email", "args": {"recipients": ["bob@x.com"], "subject": "Hi"}}]
    calls = [{"name": "create_file", "args": {"filename": "a.pdf ", "content": "other words"}, "error": None},
             {"name": "list_files", "args": {}, "error": None},
             {"name": "send_email", "args": {"recipients": "bob@x.com", "subject": "hi"}, "error": None},
             {"name": "send_email", "args": {"recipients": ["eve@x.com"], "subject": "Hi"}, "error": None},
             {"name": "send_email", "args": {"recipients": ["bob@x.com"], "subject": "Hi"}, "error": "ValueError: no"}]
    lab.annotate(calls, chain)
    assert [(c["chain_step"], c["chain_match"]) for c in calls] == [(0, "args"), (None, None), (1, "args"), (1, "name"),
                                                                    (None, None)]
    assert lab.chain_progress(calls, chain) == 2
    assert lab.chain_progress(calls[1:], chain) == 0          # the first step never ran
    assert lab.chain_progress([calls[0], calls[4]], chain) == 1   # a failed call does not count


def test_parse_planner():
    assert lab.parse_planner('<think>x</think>{"strategy_prediction": "s", "predicted_attack_prompt": " go "}') == "go"
    assert lab.parse_planner("no json") is None


def test_fix_tail_keeps_a_torn_line_from_eating_the_next_record(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_bytes(b'{"a": 1}\n{"a": 2, "b": "tor')
    S.fix_tail(p)
    io.append_jsonl(p, {"a": 3})
    assert [r["a"] for r in io.read_jsonl(p)] == [1, 3]


def test_splits_keep_attack_and_benign_in_every_split():
    rows = [{"suite": "travel", "kind": k, "task_id": "%s%d" % (k, i)} for k in ("attack", "benign") for i in range(20)]
    sp = S.assign_splits(rows, [0.7, 0.15, 0.15], 0)
    assert sp == S.assign_splits(rows, [0.7, 0.15, 0.15], 0)
    for k in ("attack", "benign"):
        assert {sp[("travel", r["task_id"])] for r in rows if r["kind"] == k} == {"train", "val", "test"}


# ---------------------------------------------------------------- with the checkout
@needs_repo
def test_environment_copies_are_independent_and_schemas_hide_injected_args():
    e1, rt = lab.new_env(REPO, "travel")
    e2, _ = lab.new_env(REPO, "travel")
    out, err = lab.execute(e1, rt, "create_file", {"filename": "n.txt", "content": "hello", "cloud_drive": "ignored"})
    assert err is None and "n.txt" in out
    assert len(e1.cloud_drive.files) == len(e2.cloud_drive.files) + 1
    assert lab.execute(e1, rt, "no_such_tool", {})[1] == "Tool no_such_tool doesn't exist."
    assert lab.execute(e1, rt, "send_email", {"subject": "x"})[1].startswith("ValidationError")
    spec = {t["function"]["name"]: t["function"] for t in lab.tool_specs(rt)}
    assert "cloud_drive" not in spec["create_file"]["parameters"]["properties"]
    assert {"filename", "content"} <= set(spec["create_file"]["parameters"]["properties"])


@needs_repo
def test_extract_reads_shade_attacks_and_main_tasks(tmp_path):
    cfg = cfg_for(tmp_path)
    rows = S.extract(cfg, LC.SUITES)
    assert len(S.extract(cfg, LC.SUITES)) == len(rows)       # second call adds nothing
    atk = [r for r in rows if r["kind"] == "attack"]
    assert len(atk) > 150 and {r["suite"] for r in atk} == set(LC.SUITES)
    assert all(r["turns"] and r["attack"]["chain"] for r in atk)
    assert sum(r["source"] == "shade_main" for r in rows) == 5
    tools = {s: {t["name"] for t in lab.tool_infos(REPO, s)} for s in LC.SUITES}
    known = [s["function"] in tools[r["suite"]] for r in atk for s in r["attack"]["chain"]]
    assert sum(known) / len(known) > 0.95                    # the chains use this environment's tools
    assert not any("cloud_drive" in s["args"] or "inbox" in s["args"] for r in atk for s in r["attack"]["chain"])
    lim = S.originals(cfg, ["travel"], limit=2)
    assert [(r["kind"], r["source"]) for r in lim].count(("attack", "tool_chaining")) == 2


@needs_repo
def test_run_one_records_turns_calls_and_chain(tmp_path):
    chain = [{"function": "create_file", "args": {"filename": "Plan.pdf", "content": "long text " * 20}},
             {"function": "search_contacts_by_name", "args": {"query": "Sophia Chen"}}]
    script = [("create_file", {"filename": "Plan.pdf", "content": "my own words"}),
              ("search_contacts_by_name", {"query": "Sophia Chen"})]
    client = FakeClient(script)
    rec = lab.run_one(REPO, "travel", ["make the file", "find Sophia"], {"name": "fake"}, "http://x", chain=chain,
                      client=client)
    assert rec["error"] is None and rec["security"] is True and rec["chain_progress"] == 2
    assert [(c["turn"], c["name"], c["chain_step"], c["chain_match"]) for c in rec["tool_calls"]] == [
        (0, "create_file", 0, "args"), (1, "search_contacts_by_name", 1, "args")]
    assert "Sophia" in rec["tool_calls"][1]["result"]
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant", "tool", "assistant"] * 2
    assert rec["turn_answers"] == ["done", "done"] and rec["final_answer"] == "done"
    assert len(client.requests[0]["tools"]) == len(lab.tool_infos(REPO, "travel"))
    assert client.requests[-1]["messages"][0]["role"] == "system" and client.requests[0]["seed"] == 0
    # a refused chain: no calls, no success; a dead endpoint: an error and no assistant message
    rec = lab.run_one(REPO, "travel", ["make the file"], {"name": "fake"}, "http://x", chain=chain, client=FakeClient([]))
    assert rec["security"] is False and rec["n_steps"] == 0 and rec["chain_progress"] == 0
    rec = lab.run_one(REPO, "travel", ["hi"], {"name": "fake"}, "http://x", client=FakeClient([], fail=True))
    assert rec["error"].startswith("ConnectionError") and [m["role"] for m in rec["messages"]] == ["user"]


@needs_repo
def test_adaptive_turns_ask_the_planner_until_the_chain_is_done():
    chain = [{"function": "create_file", "args": {"filename": "Plan.pdf"}},
             {"function": "search_contacts_by_name", "args": {"query": "Sophia Chen"}}]
    asked = []

    def planner(messages):
        asked.append(messages[-1]["content"])
        return json.dumps({"predicted_attack_prompt": "now look up Sophia Chen"})

    client = FakeClient([("create_file", {"filename": "Plan.pdf", "content": "x"}),
                         ("search_contacts_by_name", {"query": "Sophia Chen"})])
    rec = lab.run_one(REPO, "travel", ["make the file"], {"name": "fake"}, "http://x", chain=chain, goal="g",
                      planner=planner, adaptive_turns=3, client=client)
    assert rec["planner_turns"] == 1 and rec["security"] is True
    assert rec["turns_run"] == ["make the file", "now look up Sophia Chen"]
    assert "step 2 (search_contacts_by_name)" in asked[0]


@needs_repo
def test_pipeline_is_resumable_end_to_end(tmp_path):
    cfg = cfg_for(tmp_path)
    S.extract(cfg, ["travel"])
    calls = []

    def task_chat(messages, seed):
        calls.append(seed)
        return json.dumps([{"turns": ["list my files", "open the first one"]}, {"turns": ["what is on my calendar today?"]}])

    assert S.synthesize_benign(cfg, ["travel"], chat=task_chat) == 2      # batch 2 repeats batch 1: deduplicated
    n_calls = len(calls)
    assert S.synthesize_benign(cfg, ["travel"], chat=task_chat) == 0 and len(calls) == n_calls
    rows = S.originals(cfg, ["travel"], limit=2)
    assert sorted((r["kind"], r["source"]) for r in rows) == [
        ("attack", "tool_chaining")] * 2 + [("benign", "generated")] * 2 + [("benign", "shade_main")]

    def para_chat(messages, seed):
        turn = messages[0]["content"].split('Request: "', 1)[1].rsplit('"', 1)[0]
        return json.dumps([turn + " Thanks!", "Please: " + turn, turn])

    n_var = S.generate(cfg, rows, chat=para_chat)
    assert n_var == 2 * len(rows) and S.generate(cfg, rows, chat=para_chat) == 0

    class NLI:
        def entail(self, premises, hypotheses):
            return [0.1 if "Thanks!" in p + h else 0.95 for p, h in zip(premises, hypotheses)]

    assert S.verify(cfg, rows, nli=NLI()) == len(rows)                    # the "Thanks!" variants fail NLI
    assert S.verify(cfg, rows, nli=NLI()) == 0
    rej = io.read_jsonl(tmp_path / "rejected_variants.jsonl")
    assert len(rej) == len(rows) and all(r["reject_reason"].startswith("nli:turn") for r in rej)

    prows = S.prompt_rows(cfg, rows)
    agents = LC.C.agent_specs(cfg)
    jobs = S.plan_jobs(cfg, prows, agents, set())
    assert len(jobs) == 2 * len(rows)
    assert {j["attack_name"] for j in jobs if j["mode"] == "attack"} == {"tool_chaining"}

    def run(todo, client):
        runner = S.make_runner(cfg, {"api": ["fake"]}, client=client)
        for j in todo:
            runner.submit(j)
        runner.close()

    run(jobs[:3], FakeClient([("list_files", {})] * 50))
    run(jobs[3:5], FakeClient([], fail=True))                             # endpoint down: not recorded as done
    done = io.keys(tmp_path / "traces.jsonl", io.trace_key)
    assert len(done) == 3 and len(io.read_jsonl(tmp_path / "trace_errors.jsonl")) == 2
    left = S.plan_jobs(cfg, prows, agents, done)
    assert len(left) == len(jobs) - 3 and {io.trace_key(j) for j in left}.isdisjoint(done)
    run(left, FakeClient([("list_files", {})] * 500))
    assert S.plan_jobs(cfg, prows, agents, io.keys(tmp_path / "traces.jsonl", io.trace_key)) == []

    st = S.build(cfg, rows)
    assert st["traces"] == len(jobs) and sum(st["splits"].values()) == len(jobs)
    recs = [r for n in ("train", "val", "test") for r in io.read_jsonl(tmp_path / "splits" / ("%s.jsonl" % n))]
    atk = [r for r in recs if r["mode"] == "attack"]
    assert atk and all(r["attack_ground_truth_calls"] and "chain_progress" in r for r in atk)
    assert all("security" not in r for r in recs if r["mode"] == "benign")
    assert "travel" in json.loads((tmp_path / "tools.json").read_text(encoding="utf-8"))
    assert "## Traces" in (tmp_path / "report.md").read_text(encoding="utf-8")
