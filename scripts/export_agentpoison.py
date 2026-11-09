"""Dump STATIC AgentPoison data (EHRAgent + ReAct-StrategyQA) to data/agentpoison/*.json, split for no train/test leakage.

Run once against a clone of https://github.com/AI-secure/AgentPoison (only its data files are read):
    git clone --depth 1 https://github.com/AI-secure/AgentPoison <dir>
    python scripts/export_agentpoison.py <dir> data/agentpoison
The main project never imports AgentPoison and never runs its agents, retrievers or trigger optimiser.

Leakage controls applied here (see README, "AgentPoison"):
  EHR  the EHRAgent memory logs are the solved eICU queries themselves, so the 78 question-template families (q_tag) are
       split into SCENARIO families (task questions) and KB families (memory + poison bases). No scenario question
       appears in any memory record. Queries are de-duplicated by template.
  QA   one item per entity (term), and no two kept items (or poison-pool items) share an evidence paragraph.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

N_TRIGGERS = 40           # per agent; must match src/benchmarks/agentpoison.py
POISON_PER_TRIGGER = 2    # AgentPoison injection_number
QA_ITEMS = 300
PASSAGES = 3
PASSAGE_CHARS = 600

EHR_TOOLS = [
    ("Calculate", "Calculate(FORMULA): calculates the FORMULA and returns the result."),
    ("LoadDB", "LoadDB(DBNAME): loads the database DBNAME and returns the database. DBNAME is one of: allergy, cost, "
               "diagnosis, intakeoutput, lab, medication, microlab, patient, treatment, vitalperiodic."),
    ("FilterDB", "FilterDB(DATABASE, CONDITIONS): filters the DATABASE according to the CONDITIONS (column, relation, "
                 "value; separated by '||') and returns the filtered database."),
    ("GetValue", "GetValue(DATABASE, ARGUMENT): returns the values of the column ARGUMENT in the DATABASE, optionally "
                 "with an operation (e.g. COST, sum)."),
    ("SQLInterpreter", "SQLInterpreter(SQL): interprets the query SQL and returns the result."),
    ("Calendar", "Calendar(DURATION): returns the date after the duration of time."),
    ("DeleteDB", "DeleteDB(DBNAME): deletes the database DBNAME."),
]


def tok(s):
    return set(re.findall(r"[a-z0-9_]+", s.lower()))


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def h(s):
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def parse_cell(tail):
    """The agent's own solution in a log tail: the JSON {"cell": "..."} after 'Solution:'."""
    i = tail.find("{")
    while i != -1:
        depth = 0
        for j in range(i, len(tail)):
            depth += {"{": 1, "}": -1}.get(tail[j], 0)
            if depth == 0:
                try:
                    return json.loads(tail[i:j + 1])["cell"]
                except Exception:
                    break
        i = tail.find("{", i + 1)
    return None


def ehr_memory(log_dir: Path):
    """Same parsing rules as AgentPoison algo/utils.py:load_ehr_memory, but the solution of the logged question is
    read from its JSON cell (the original keeps the whole log tail)."""
    mem = {}
    for p in sorted(log_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        parts = text.split("(END OF EXAMPLES)")
        for block in parts[0].split("\n\n")[1:-1]:
            item = block.split("Question:")[-1]
            q = item.split("\nKnowledge:\n")[0].strip()
            if len(q.split(" ")) > 20 or "\nKnowledge:\n" not in item or "\nSolution:" not in item:
                continue
            rest = item.split("\nKnowledge:\n")[-1]
            mem.setdefault(q, dict(question=q, knowledge=rest.split("\nSolution:")[0].strip(),
                                   code=rest.split("\nSolution:")[-1].strip()))
        if len(parts) > 1:
            tail = parts[1]
            know = tail.split("Knowledge:\n")[-1].split("Question:")[0].strip()
            q = tail.split("Question:")[-1].split("Solution:")[0].strip()
            code = parse_cell(tail.split("Solution:", 1)[-1]) if "Solution:" in tail else None
            if code and len(q.split(" ")) <= 20:
                mem.setdefault(q, dict(question=q, knowledge=know, code=code.strip()))
    return [mem[k] for k in sorted(mem)]


def export_ehr(src: Path):
    rows = json.load(open(src / "EhrAgent/database/ehr_logs/eicu_ac.json", encoding="utf-8"))
    queries = {}
    for r in rows:  # de-duplicate by template (L1)
        t = r["template"].strip()
        if t not in queries:
            ans = r["answer"] if isinstance(r["answer"], list) else [r["answer"]]
            queries[t] = dict(id=r["id"], template=t, q_tag=r["q_tag"], answer=[str(a) for a in ans],
                              tables=sorted(r.get("access_gt", {}) or {}) or ["patient"])
    queries = list(queries.values())
    memory = ehr_memory(src / "EhrAgent/database/ehr_logs/logs_final")

    # every memory record belongs to the family of its nearest query template
    qt = [(tok(q["template"]), q["q_tag"]) for q in queries]
    for m in memory:
        best = max(qt, key=lambda x: jac(x[0], tok(m["question"])))
        m["q_tag"] = best[1] if jac(best[0], tok(m["question"])) >= 0.5 else None

    # split families: KB families (memory) vs scenario families (tasks); balanced by record count, deterministic
    fams = sorted({q["q_tag"] for q in queries}, key=lambda f: h("fam|" + f))
    n_mem = {f: sum(m["q_tag"] == f for m in memory) for f in fams}
    kb, total, need = set(), sum(n_mem.values()), None
    for f in fams:
        if sum(n_mem[x] for x in kb) >= total / 2:
            break
        kb.add(f)
    scen_q = [q for q in queries if q["q_tag"] not in kb]
    kb_mem = [m for m in memory if m["q_tag"] is None or m["q_tag"] in kb]

    # L6: no scenario question near any memory record
    worst = max(jac(tok(q["template"]), tok(m["question"])) for q in scen_q for m in kb_mem)
    assert worst < 0.8, "scenario question leaks into memory (jaccard %.2f)" % worst
    # poison bases: unique per trigger, taken in a fixed hashed order from the KB
    order = sorted(range(len(kb_mem)), key=lambda i: h("poison|" + kb_mem[i]["question"]))
    need = N_TRIGGERS * POISON_PER_TRIGGER
    assert len(order) >= need, "KB has %d records, need %d poison bases" % (len(order), need)
    poison = [order[i * POISON_PER_TRIGGER:(i + 1) * POISON_PER_TRIGGER] for i in range(N_TRIGGERS)]
    for m in kb_mem:
        m.pop("q_tag")
    return dict(tools=[dict(name=n, description=d) for n, d in EHR_TOOLS], queries=scen_q, memory=kb_mem,
                poison_pool=poison, kb_families=sorted(kb))


def export_qa(src: Path):
    items = json.load(open(src / "ReAct/database/strategyqa_train.json", encoding="utf-8"))
    paras = json.load(open(src / "ReAct/database/strategyqa_train_paragraphs.json", encoding="utf-8"))

    def evidence(x):
        out = []
        for ann in x.get("evidence", []):
            for step in ann:
                for e in step:
                    if isinstance(e, list):
                        out += [p for p in e if p in paras and p not in out]
        return out

    used_terms, used_paras, kept, pool = set(), set(), [], []
    for x in sorted(items, key=lambda x: h("qa|" + x["qid"])):
        ev = evidence(x)
        if x["term"] in used_terms or len(ev) < PASSAGES or set(ev) & used_paras or not x.get("facts"):
            continue
        used_terms.add(x["term"])
        used_paras |= set(ev)  # all evidence of a kept item is reserved, not only the passages shown (L2)
        rec = dict(qid=x["qid"], term=x["term"], question=x["question"], answer=bool(x["answer"]), facts=x["facts"])
        if len(kept) < QA_ITEMS:
            rec["passages"] = [paras[p]["content"][:PASSAGE_CHARS] for p in ev[:PASSAGES]]
            kept.append(rec)
        elif len(pool) < N_TRIGGERS:
            pool.append(rec)
        else:
            break
    assert len(kept) == QA_ITEMS and len(pool) == N_TRIGGERS, (len(kept), len(pool))
    return dict(tools=[dict(name="Search", description="Search(entity): returns passages about the entity from the "
                                                       "knowledge base."),
                       dict(name="Lookup", description="Lookup(keyword): returns the next sentence containing keyword "
                                                       "in the last passage."),
                       dict(name="Finish", description="Finish(answer): returns the final answer and finishes the "
                                                       "task.")],
                items=kept, poison_pool=pool)


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/agentpoison")
    out.mkdir(parents=True, exist_ok=True)
    ehr, qa = export_ehr(src), export_qa(src)
    for name, doc in (("ehr", ehr), ("strategyqa", qa)):
        (out / ("%s.json" % name)).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("ehr: %d scenario queries in %d families, %d memory records (%d KB families), %d poison bases"
          % (len(ehr["queries"]), len({q["q_tag"] for q in ehr["queries"]}), len(ehr["memory"]),
             len(ehr["kb_families"]), sum(map(len, ehr["poison_pool"]))))
    print("strategyqa: %d items, %d poison-pool items" % (len(qa["items"]), len(qa["poison_pool"])))


if __name__ == "__main__":
    main()
