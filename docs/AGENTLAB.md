
# AgentLAB trace generation (`tracegen.agentlab`)

`tracegen.agentlab` does for [AgentLAB](https://github.com/TanqiuJiang/AgentLAB) what `tracegen` does for AgentDojo.
It covers AgentLAB's **tool-chaining** track: a user walks the agent through a "verified tool chain", one
harmless-looking request per turn, and the turns together reach a harmful goal. The dataset
(`data/filtered_top_200_attacks.json`) ships these multi-turn conversations for four SHADE-Arena environments
(`travel`, `banking`, `workspace`, `spam`). Local Ollama agents replay them in the real environments, with AgentLAB's
agent system prompt and its rule of one tool call per step.

```bash
python -m tracegen.agentlab all              # everything; run the same command again to resume
python -m tracegen.agentlab all --smoke      # 2 tasks per kind, 2 suites, 1 agent -> data/agentlab_smoke/
python -m tracegen.agentlab trace --agents mistral:7b      # one stage; add another agent later
python -m pytest tests/tracegen/test_agentlab.py -q        # offline, scripted fake agent
```

| stage | output (under `data/agentlab/`) |
|---|---|
| `setup` | `external/AgentLAB`: sparse checkout of `SHADE_Arena/` and `data/` at the pinned commit (about 20 MB) |
| `extract` | `originals.jsonl`: 167 attack tasks (user turns + tool chain) and the 5 benign SHADE-Arena main tasks |
| `benign` | more of `originals.jsonl`, `benign_log.jsonl`: benign multi-turn tasks the generator writes per environment |
| `generate` | `candidates.jsonl`, `generate_log.jsonl`: N paraphrases of every user turn |
| `verify` | `verified_variants.jsonl`, `rejected_variants.jsonl`: entity check and bidirectional NLI, per turn |
| `trace` | `traces.jsonl` (append-only), `trace_errors.jsonl` |
| `build` | `splits/{train,val,test}.jsonl`, `tools.json`, `report.md` |

- **Resume.** Every stage appends to its files and skips what is there: tasks by `(suite, task_id)`, generator calls
  by task or batch, variants by `variant_id`, runs by `(mode, suite, task, variant, agent, seed, attack)`. A run that
  was killed mid-write leaves a torn last line, which is skipped and redone. A run that got no reply at all from the
  agent (endpoint down, model missing) goes to `trace_errors.jsonl`, not `traces.jsonl`, so the next run retries it.
- **A trace** has the fields of a tracegen trace (`messages`, `tool_calls`, `agent_model`, `seed`, ...) plus `turns`
  (the user messages), a `turn` index on every message and call, and for attacks `attack_ground_truth_calls` (the
  chain), `chain_step` / `chain_match` per call (`args`: same tool and same identifier-like arguments, `name`: same
  tool only), `chain_progress` and `security`. `security` is true when the agent ran every tool of the chain in order
  without an error. AgentLAB scores success with a GPT judge; this check replaces it.
- **Benign tasks.** AgentLAB has no benign tasks in these environments beyond the 5 main tasks, so the generator
  model writes `benign.per_suite` tasks per environment from the tool list. They are deduplicated but not checked by
  a judge, and there is no utility checker (`utility` is null).
- **Adaptive turns.** `attacks.adaptive_turns: K` (or `--adaptive-turns K`) adds AgentLAB's planner: after the
  scripted turns a local model writes up to K further user turns while the chain is incomplete. These runs are
  recorded as attack `tool_chaining_adaptive`, next to the scripted `tool_chaining` runs.
- **Not covered.** The dataset's 33 Agent-SafetyBench cases (no environment parameters; AgentLAB's own runner cannot
  build them either), and the other four tracks: intent hijacking has no tool calls, memory poisoning needs Mem0,
  objective drifting needs the WebShop conda environment, and task injection is AgentDojo with a GPT attacker.
- The GNN adapter (`src/benchmarks/`) does not read these traces yet.
