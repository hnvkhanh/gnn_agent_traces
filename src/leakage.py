"""Train/test leakage audit for the held-out-task split (used by tests and scripts/report_page.py).

For traces of one benchmark, split them as the experiments do (train.split(kind="task")) and count what the train
and test sides share. Every count must be 0; `train_attacks` must be > 0 (otherwise the guard never sees an attack).
"""
import re

from .graph.build import meta_of
from .train import split


def _norm(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def clean_question(tr):
    """The user question without the AgentPoison trigger (for other benchmarks: the task as is)."""
    task = tr["task"]
    if tr.get("benchmark") == "agentpoison" and tr["attack"]:
        from .benchmarks.agentpoison import trigger_id
        tid = trigger_id(tr.get("user_task", ""))
        if tid:
            task = task.replace(tid.split("|", 1)[1], "")
    return _norm(task)


def _trigger_tokens(tr):
    if tr.get("benchmark") != "agentpoison" or not tr["attack"]:
        return set()
    from .benchmarks.agentpoison import TRIGGER_STOPWORDS, tokens
    return tokens(tr["attack"]["id"].split("|", 1)[1]) - TRIGGER_STOPWORDS


def _memory_questions(tr):
    return {_norm(m.split("\n")[0][len("Question: "):]) for m in tr["memory"] if m.startswith("Question: ")}


def audit(trs, seed):
    """Overlap counts between the train and test sides of split(metas, 'task', None, seed)."""
    metas = [meta_of(t) for t in trs]
    tr_i, te_i = split(metas, "task", None, seed)

    def side(idx, f):
        out = set()
        for i in idx:
            v = f(i)
            out |= v if isinstance(v, set) else ({v} - {None, ""})
        return out

    both = lambda f: side(tr_i, f) & side(te_i, f)
    overlaps = dict(
        task_groups=both(lambda i: metas[i]["group"]),
        attack_ids=both(lambda i: metas[i]["attack_id"]),
        trigger_tokens=both(lambda i: _trigger_tokens(trs[i])),
        poison_texts=both(lambda i: trs[i]["injected"] if trs[i]["attack"] else None),
        questions=both(lambda i: clean_question(trs[i])),
        test_question_in_train_memory=side(te_i, lambda i: clean_question(trs[i]))
        & side(tr_i, lambda i: _memory_questions(trs[i])),
    )
    return dict(seed=seed, train=len(tr_i), test=len(te_i), train_attacks=sum(1 for i in tr_i if trs[i]["attack"]),
                test_attacks=sum(1 for i in te_i if trs[i]["attack"]),
                overlaps={k: len(v) for k, v in overlaps.items()}, examples={k: sorted(v)[:3] for k, v in overlaps.items() if v})
