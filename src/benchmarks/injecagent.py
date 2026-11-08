"""InjecAgent adapter (indirect prompt injection through a user tool's response)."""
import json
import random
from pathlib import Path
from typing import List

from ..sandbox.scenario import Scenario
from ..sandbox.tools import MockTool

PERSONA = "a helpful assistant with access to tools"
BENIGN_MEMORY = "Use only the tools needed for the user's request."
BENIGN_FILL = "Looks good, works as described."
FILES = [("dh", "base"), ("dh", "enhanced"), ("ds", "base"), ("ds", "enhanced")]


def _tool_descs(data_dir: Path) -> dict:
    out = {}
    for tk in json.load(open(data_dir / "tools.json")):
        for t in tk["tools"]:
            out[tk["toolkit"] + t["name"]] = t["summary"]
    return out


def _load(data_dir: Path):
    cases = []
    for cat, setting in FILES:
        for c in json.load(open(data_dir / ("test_cases_%s_%s.json" % (cat, setting)))):
            cases.append((cat, setting, c))
    return cases


def scenarios(data_dir: str, n: int, seed: int, benign_frac: float = 0.25) -> List[Scenario]:
    d = Path(data_dir)
    desc, cases = _tool_descs(d), _load(d)
    user_tools = sorted({c["User Tool"] for _, _, c in cases})

    def tool(name, kind, **kw):
        return MockTool(name, desc.get(name, name), kind, "injecagent", **kw)

    out = []
    for i in range(n):
        rng = random.Random("%d-%d" % (seed, i))
        cat, setting, c = cases[rng.randrange(len(cases))]
        benign = rng.random() < benign_frac
        ut = c["User Tool"]
        distract = [u for u in rng.sample(user_tools, 4) if u != ut][:2]
        tools = [tool(ut, "normal", achievement="Done.")] + [tool(u, "normal", achievement="Done.") for u in distract]
        attacker_calls, injected = [], ""
        if not benign:
            for a in c["Attacker Tools"]:
                tools.append(tool(a, "attacker", goal=c["Expected Achievements"]))
            injected = c["Attacker Instruction"]
            attacker_calls = [dict(tool=c["Attacker Tools"][0], args={"instruction": injected})]
            resp = c["Tool Response"].strip('"')
        else:
            resp = c["Tool Response Template"].replace("<Attacker Instruction>", BENIGN_FILL)
        rng.shuffle(tools)
        out.append(Scenario(
            benchmark="injecagent", persona=PERSONA, group="injecagent|%s|%s" % (ut, c["User Instruction"]),
            task=c["User Instruction"], tools=tools, in_scope=[ut], memory=[BENIGN_MEMORY],
            mode="none" if benign else "opi", style=None if benign else setting,
            category=None if benign else "%s|%s" % (cat, c["Attack Type"]), injected=injected,
            attacker_calls=attacker_calls, tool_obs={ut: resp}, opi_tool=ut))
    return out
