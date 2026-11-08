"""Mock tool registry built from ASB data. Nothing here touches network, filesystem (after load) or processes."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class MockTool:
    name: str
    description: str
    kind: str  # "normal" | "attacker"
    agent: str
    achievement: str = ""
    instruction: str = ""
    goal: str = ""
    aggressive: bool = False
    attack_type: str = ""


def _jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class ASBData:
    def __init__(self, data_dir: str = "data/asb"):
        d = Path(data_dir)
        self.agents = _jsonl(d / "agent_task.jsonl")
        self.normal: List[MockTool] = [
            MockTool(r["Tool Name"], r["Description"], "normal", r["Corresponding Agent"],
                     achievement=r["Expected Achievements"])
            for r in _jsonl(d / "all_normal_tools.jsonl")
        ]
        self.attack: List[MockTool] = [
            MockTool(r["Attacker Tool"], r["Description"], "attacker", r["Corresponding Agent"],
                     instruction=r["Attacker Instruction"], goal=r["Attack goal"],
                     aggressive=r["Aggressive"] == "True", attack_type=r["Attack Type"])
            for r in _jsonl(d / "all_attack_tools.jsonl")
        ]

    def tasks(self) -> List[tuple]:
        return [(a["agent_name"], t) for a in self.agents for t in a["tasks"]]

    def normal_for(self, agent: str) -> List[MockTool]:
        return [t for t in self.normal if t.agent == agent]

    def attacks_for(self, agent: str) -> List[MockTool]:
        return [t for t in self.attack if t.agent == agent]

    def other_normal(self, agent: str) -> List[MockTool]:
        return [t for t in self.normal if t.agent != agent]


def execute(tool: Optional[MockTool], name: str, args: dict) -> str:
    """Deterministic simulated result. No side effects."""
    if tool is None:
        return "Error: unknown tool '%s'." % name
    if tool.kind == "attacker":
        return "Tool %s executed. %s" % (tool.name, tool.goal)
    return tool.achievement
