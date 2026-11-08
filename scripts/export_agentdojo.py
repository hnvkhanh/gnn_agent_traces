"""Dump STATIC AgentDojo metadata (tools, user tasks, injection tasks, ground-truth calls) to JSON.

Run once with an interpreter that has agentdojo installed (it needs a newer Python than the project's):
    <venv>/bin/python scripts/export_agentdojo.py data/agentdojo
The main project never imports agentdojo and never runs its environment.
"""
import json
import sys
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suites

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/agentdojo")
out.mkdir(parents=True, exist_ok=True)


def calls(task, env):
    try:
        return [dict(function=c.function, args=json.loads(json.dumps(c.args, default=str))) for c in task.ground_truth(env)]
    except Exception:
        return []


for name, suite in get_suites("v1.2.2").items():
    env = suite.load_and_inject_default_environment({})
    doc = dict(
        suite=name,
        tools=[dict(name=t.name, description=t.description) for t in suite.tools],
        user_tasks=[dict(id=k, prompt=t.PROMPT, calls=calls(t, env)) for k, t in suite.user_tasks.items()],
        injection_tasks=[dict(id=k, goal=t.GOAL, calls=calls(t, env)) for k, t in suite.injection_tasks.items()],
    )
    (out / ("%s.json" % name)).write_text(json.dumps(doc, indent=1))
    print(name, len(doc["tools"]), len(doc["user_tasks"]), len(doc["injection_tasks"]))
