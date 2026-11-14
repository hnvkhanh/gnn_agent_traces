"""Mock tools. Nothing here touches network, filesystem or processes."""
from dataclasses import dataclass
from typing import Optional


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


def execute(tool: Optional[MockTool], name: str, args: dict) -> str:
    """Deterministic simulated result. No side effects."""
    if tool is None:
        return "Error: unknown tool '%s'." % name
    if tool.kind == "attacker":
        return "Tool %s executed. %s" % (tool.name, tool.goal)
    return tool.achievement
