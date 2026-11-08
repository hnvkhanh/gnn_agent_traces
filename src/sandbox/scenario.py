"""Benchmark-agnostic description of one episode. Adapters in src/benchmarks build these."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .tools import MockTool


@dataclass
class Scenario:
    benchmark: str
    persona: str                      # "who" the agent is (prompt flavour)
    group: str                        # split key: same task => same group
    task: str                         # clean user task
    tools: List[MockTool]             # visible tools, already shuffled
    in_scope: List[str]               # tool names the task legitimately needs
    memory: List[str]
    mode: str = "none"                # none | dpi | opi | memory | mixed
    style: Optional[str] = None
    category: Optional[str] = None
    injected: str = ""                # attacker text as delivered (already style-wrapped)
    attacker_calls: List[dict] = field(default_factory=list)   # [{tool, args}] malicious calls (reject)
    tool_obs: Dict[str, str] = field(default_factory=dict)     # fixed observation per tool name (overrides mock default)
    opi_tool: Optional[str] = None    # if set, OPI injection is delivered when THIS tool is called (tool_obs already holds it)
