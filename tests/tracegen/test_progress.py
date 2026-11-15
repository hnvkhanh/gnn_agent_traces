import io as _io
import logging

from tracegen import config as C
from tracegen import progress
from tracegen.stages import trace as T


class Tty(_io.StringIO):
    def isatty(self):
        return True


def record(per_run):
    r = logging.LogRecord("tracegen", logging.INFO, __file__, 1, "msg", None, None)
    if per_run:
        r.per_run = True
    return r


def test_per_run_lines_hidden_only_behind_a_bar():
    assert not progress.ConsoleHandler(stream=Tty()).filter(record(True))        # bar drawn: file log only
    assert progress.ConsoleHandler(stream=Tty()).filter(record(False))
    assert progress.ConsoleHandler(verbose=True, stream=Tty()).filter(record(True))
    assert progress.ConsoleHandler(stream=_io.StringIO()).filter(record(True))   # no tty: no bar, keep lines


def test_console_handler_writes():
    s = Tty()
    h = progress.ConsoleHandler(stream=s)
    h.emit(record(False))
    assert s.getvalue() == "msg\n"


def test_run_stats():
    assert progress.run_stats({"runs": 0}) == ""
    assert progress.run_stats({"runs": 4, "utility": 1, "attack": 0, "security": 0, "error": 2}) == "util 25% · err 2"
    assert progress.run_stats({"runs": 4, "utility": 2, "attack": 2, "security": 1, "error": 0}) == \
        "util 50% · ASR 50% of 2 · err 0"


def test_runner_counts_every_job(tmp_path, monkeypatch):
    cfg = C.load("/nonexistent.yaml", C.merge({"paths": {"out_dir": str(tmp_path)}}, {
        "trace": {"agents": [{"name": "api-m", "base_url": "http://x", "api_key_env": None}]},
        "attacks": {"enabled": True, "injection_tasks": ["injection_task_0"], "match_splits": False}}))
    monkeypatch.setattr(T, "run_one", lambda suite, task, prompt, agent, url, injection_task=None, **kw: {
        "trace_id": "t", "utility": True, "security": injection_task is not None, "n_steps": 1, "latency_s": 0.0,
        "error": None})
    rows = [{"suite": "banking", "task_id": "user_task_0", "variant_id": "banking/user_task_0/orig",
             "is_original": True, "prompt": "p"}]
    jobs = T.plan_jobs(cfg, rows, C.agent_specs(cfg), set())
    runner = T.TraceRunner(cfg, {"api": ["api-m"]})
    for j in jobs:
        runner.submit(j)
    assert runner.close() == len(jobs) == 2
    assert runner.bar.total == 2
    assert dict(runner.stats) == {"runs": 2, "utility": 2, "error": 0, "attack": 1, "security": 1}
