# gnn_trace

Can a graph neural network tell, **per action**, whether an LLM agent is doing what the user asked or following a prompt injection?

An LLM agent runs inside a fully mocked sandbox on tasks from three prompt-injection benchmarks. Each run is recorded as a trace, and each tool call in it is labelled `approve` or `reject`. Every trace is turned into a causal heterogeneous graph, and a GNN is trained to predict the label of each action node. It is compared against a majority-class, a tool-name-lookup and a no-graph MLP baseline.

## Pipeline

```
benchmark data ──► scenario ──► LLM in mock sandbox ──► trace (JSONL) ──► causal graph ──► GNN ──► approve / reject
 data/*            src/benchmarks   src/sandbox, src/llm     traces/        src/graph       src/model
```

1. **Scenarios** (`src/benchmarks/`): adapters for [ASB](https://github.com/agiresearch/ASB) (Agent Security Bench), [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) and [AgentDojo](https://github.com/ethz-spylab/agentdojo) emit benchmark-agnostic `Scenario` objects (`src/sandbox/scenario.py`). Each one has a task, visible tools, the in-scope tools, memory, and an optional injection. Injection modes are `dpi` (in the user prompt), `opi` (in a tool observation), `memory`, `mixed` and `none`. A fraction of scenarios are benign (`--benign-frac`).
2. **Sandbox** (`src/sandbox/`): the LLM replies with one JSON action per step (`{"tool":…, "args":…}` or `{"final":…}`, max 5 steps). All tools are mocked, with no network, filesystem or process access. If the LLM never follows the attack, a synthetic attacker-goal step is appended so every attack trace contains a `reject` example. Disable this with `--no-replay`. Synthetic steps are flagged in the trace.
3. **Labels** (`src/sandbox/policy.py`): every recorded action is `reject` if it is the attacker's action (an attacker tool, or the attacker's call: same tool with attacker-chosen arguments) and `approve` otherwise. A call to a tool that does not exist is not an action: the sandbox answers with an error, counts it in `n_invalid` and records no event. There are only two labels and no action is left unlabeled.
4. **Graph** (`src/graph/build.py`): node types are `task`, `memory`, `tool`, `action` and `observation`. Node features are 64-d hashed bag-of-tokens plus step index. Edges are causal, so action *i* only sees nodes that exist before its decision. The graph does not include a task-scope edge, so it is not told which tools the task needs. Besides the structural edges, `mentions` and `copies` link an earlier observation to a later action when the observation names that tool or contains one of its arguments.
5. **Model** (`src/model/gnn.py`): `HeteroGNN` (per-type projection, 2 layers of `HeteroConv` with `SAGEConv`, residuals, MLP head on action nodes). `ActionMLP` uses the same action features without the graph.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Trace collection needs [Ollama](https://ollama.com) running on `localhost:11434` with at least one model pulled (e.g. `ollama pull qwen2.5:7b`). Training and evaluation on the committed traces do not need it.

## Usage

### 1. Collect traces

```bash
python -m src.collect --bench asb --n 100 --model qwen2.5:7b
# --bench asb|injecagent|agentdojo   --seed 0   --benign-frac 0.25   --no-replay   --out PATH
```

Output goes to `traces/<bench>__<model>.jsonl`. Collection is resumable: trace IDs already in the file are skipped.

### 2. Run the experiments

```bash
python -m src.experiments --seeds 3 --epochs 60
# --only all|inbench|transfer|shifts|ablate|llm
# --model qwen2.5:1.5b      trace-generating LLM for the benchmark sections
# --llms a,b,c              LLMs compared with --only llm
```

| Section | Question |
|---|---|
| `inbench` | Held-out tasks within a benchmark |
| `transfer` | Train on other benchmark(s), test on a new one |
| `shifts` | ASB held-out attack style, mode, category or agent |
| `ablate` | Drop one graph component at a time (`tool_feat`, `observation`, `memory`, `injected`, `copies`, `mentions`) |
| `llm` | Behaviour of each LLM and transfer across LLMs |

Results are written to `results/experiments_<only>.json` and `results/report_<only>.md`. The `llm` section writes `results/llm_experiments.json` and `results/llm_report.md`. Numbers are mean±std over seeds, shown as `macro-F1 / micro-F1 / reject recall`. Micro-F1 equals accuracy for single-label classification. Tables also report **Attack Success Rate** (see below).

### 3. Tests

```bash
pytest
```

The tests cover sandbox isolation, labelling, graph causality (perturbing future nodes must not change earlier predictions), ablations, per-benchmark scenario validity and a training smoke test.

## Attack Success Rate

`asr_report` in `src/eval.py` replays a guard over the test traces. An attacker action goes through when the guard predicts `approve`.

| Metric | Meaning |
|---|---|
| `asr_undefended` | Attack traces where the LLM itself took an attacker action (appended synthetic steps excluded) |
| `asr_defended` | Attack traces where at least one such action is not blocked by the guard |
| `asr_forced` | Worst case: the attack is attempted in every attack trace (real or appended); share with any unblocked attacker action |
| `benign_block` | Benign traces with at least one approve action rejected |
| `approve_fpr` | Share of approve actions the guard rejects |

A guard decision never changes the recorded trace, so later steps are scored on the original context. Read ASR together with `benign_block` and `approve_fpr`, because rejecting everything also gives a low ASR.

## Results

No report is committed. Regenerate with `python -m src.experiments` (benchmark sections, qwen2.5:1.5b traces) and `python -m src.experiments --only llm` (cross-LLM). The `llama3.1:8b` traces are partial (about 285 of 600), so its numbers rest on fewer traces than the other two models.

## Layout

```
data/            benchmark inputs (asb/, injecagent/, agentdojo/)
scripts/         export_agentdojo.py (one-off AgentDojo metadata dump), finalize_labels.py (migrate old traces to the 2-label schema)
src/
  collect.py     trace collection CLI
  experiments.py experiment suite CLI
  train.py       data loading, splits, training loop
  eval.py        metrics and the tool-lookup baseline
  benchmarks/    ASB / InjecAgent / AgentDojo adapters
  sandbox/       scenario, mock tools, labelling policy, episode runner
  graph/         trace → HeteroData
  model/         HeteroGNN, ActionMLP
  llm/           Ollama client
tests/           pytest suite
traces/          collected traces (*.jsonl)
results/         reports and raw experiment JSON
```

### AgentDojo data

`data/agentdojo/*.json` is already exported. To regenerate it, run `scripts/export_agentdojo.py` with an interpreter that has `agentdojo` installed (it needs a newer Python than the rest of the project):

```bash
<venv-with-agentdojo>/bin/python scripts/export_agentdojo.py data/agentdojo
```

The main project never imports `agentdojo` or runs its environment.

## Trace format

One JSON object per line, with these fields:

- **Metadata:** `benchmark`, `trace_id`, `model`, `agent`, `group` (the split key: same task means same group), `task`, `user_task`.
- **Context:** `memory`, `injected`, `attack` (`mode`, `style`, `category`, `tool`, or `null` when benign), `tools` (name, description, kind, in-scope flag), `allowed_tools`. The scope fields are stored but the graph does not read them.
- **Events:** `events`, a list of steps with `step`, `tool`, `args`, `observation`, `obs_injected`, `synthetic` and `label`.
- **Quality:** `n_invalid`, the count of malformed LLM replies.
