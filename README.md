# gnn_trace

Can a graph neural network tell, **per action**, whether an LLM agent is doing what the user asked or following a prompt injection?

An LLM agent runs inside a fully mocked sandbox on tasks from two prompt-injection benchmarks. Each run is recorded as a trace, and each tool call in it is labelled `approve` or `reject`. Every trace is turned into a causal heterogeneous graph, and a GNN is trained to predict the label of each action node. It is compared against a majority-class, a tool-name-lookup and a no-graph MLP baseline.

## Pipeline

```
benchmark data ──► scenario ──► LLM in mock sandbox ──► trace (JSONL) ──► causal graph ──► GNN ──► approve / reject
 data/*            src/benchmarks   src/sandbox, src/llm     traces/        src/graph       src/model
```

1. **Scenarios** (`src/benchmarks/`): adapters for [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) and [AgentDojo](https://github.com/ethz-spylab/agentdojo) emit benchmark-agnostic `Scenario` objects (`src/sandbox/scenario.py`). Each one has a task, visible tools, the in-scope tools, memory, and an optional injection. Both benchmarks inject through a tool observation (mode `opi`; benign scenarios are `none`). The sandbox also supports `dpi` (in the user prompt), `memory` and `mixed`. A fraction of scenarios are benign (`--benign-frac`).
2. **Sandbox** (`src/sandbox/`): the LLM replies with one JSON action per step (`{"tool":…, "args":…}` or `{"final":…}`, max 5 steps). All tools are mocked, with no network, filesystem or process access. If the LLM never follows the attack, a synthetic attacker-goal step is appended so every attack trace contains a `reject` example. Disable this with `--no-replay`. Synthetic steps are flagged in the trace.
3. **Labels** (`src/sandbox/policy.py`): every recorded action is `reject` if it is the attacker's action (an attacker tool, or the attacker's call: same tool with attacker-chosen arguments) and `approve` otherwise. A call to a tool that does not exist is not an action: the sandbox answers with an error, counts it in `n_invalid` and records no event. There are only two labels and no action is left unlabeled.
4. **Graph** (`src/graph/build.py`): node types are `task`, `memory`, `tool`, `action` and `observation`. Node features are 64-d hashed bag-of-tokens plus step index. Edges are causal, so action *i* only sees nodes that exist before its decision. The graph does not include a task-scope edge, so it is not told which tools the task needs. Besides the structural edges, `mentions` and `copies` link an earlier observation to a later action when the observation names that tool or contains one of its arguments. Full reference: [`docs/GRAPH.md`](docs/GRAPH.md).
5. **Model** (`src/model/gnn.py`): `HeteroGNN` (per-type projection, 2 layers of `HeteroConv` with `SAGEConv`, residuals, MLP head on action nodes). `ActionMLP` uses the same action features without the graph.

## Setup

The quickest way is to run the setup script. It creates the venv and installs the right torch build for your machine:

```bash
python scripts/setup_env.py            # auto: CUDA (NVIDIA) / MPS (Apple Silicon) / CPU
python scripts/setup_env.py --cpu      # force CPU;  --cuda cu126 to force a CUDA build
python scripts/setup_env.py --venv .venv-cuda --recreate   # --dry-run prints the plan only
```

For NVIDIA GPUs it reads the highest CUDA version your driver supports from `nvidia-smi`. It then installs the newest torch that has a wheel for that CUDA version, for your Python and OS. If your driver is too old for the latest torch, it says so and uses an older one. On Apple Silicon it installs the PyPI wheel, which includes MPS. It ends by running a `SAGEConv` on the chosen device. On Windows, if Smart App Control is on, it blocks torch's DLLs (`WinError 4551`), and the script tells you if that is the problem.

Manual setup:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows (cmd / PowerShell)
pip install -r requirements.txt
```

Works on Windows and macOS. Files are read and written as UTF-8 with LF line endings (`.gitattributes` enforces LF), so the same traces give the same results on both. Tested with Python 3.13, torch 2.15, torch_geometric 2.8.

**Device.** Training uses CUDA if available, otherwise CPU. Override with `GNN_DEVICE=cpu` (or `cuda:1`, `mps`). Apple `mps` is never chosen automatically because some PyG ops are not reliable there. Do not mix devices in one comparison table: dropout uses a different random generator per device, so numbers differ slightly.

Trace collection needs [Ollama](https://ollama.com) running on `localhost:11434` with at least one model pulled (e.g. `ollama pull qwen2.5:7b`). Training and evaluation on the committed traces do not need it.

## Usage

### 1. Collect traces

This is the only step that calls an LLM. To collect every model × benchmark (resumable, logs to `traces/<bench>__<model>.log`):

```bash
python scripts/collect_traces.py            # default models x 2 benchmarks, --n 200 each
python scripts/collect_traces.py --models qwen2.5:7b --benches injecagent --n 50   # --dry-run prints the commands
```

One benchmark and one model:

```bash
python -m src.collect --bench injecagent --n 100 --model qwen2.5:7b
# --bench injecagent|agentdojo   --seed 0   --benign-frac 0.25   --no-replay   --out PATH
# --host URL   --shard J --num-shards K   --skip-file PATH   (used by collect_traces.py --gpus)
```

Output goes to `traces/<bench>__<model>.jsonl`. Collection is resumable: trace IDs already in the file are skipped.

**Several GPUs.** `--gpus 0,1` starts one private `ollama serve` per GPU (pinned with `CUDA_VISIBLE_DEVICES`, ports from `--port-base 11435`, model store `--ollama-models`, default `~/.ollama/models`, the model is pulled once if missing) and runs `gpus × --workers-per-gpu` shards of `src.collect` at once. Scenario *i* goes to shard *i* mod *K*, so the shards together produce exactly the serial set of traces. Shards write `traces/.shards/` and are merged into the final file (deduplicated, sorted by scenario index) once they all succeed. An Ollama that is already running, such as the system service on `:11434`, is not touched.

```bash
python scripts/collect_traces.py --models qwen2.5:1.5b --benches agentdojo --n 400 --gpus 0,1 --workers-per-gpu 4
```

### 2. Run the experiments

To run everything on the existing traces (no LLM calls) and write the combined report:

```bash
python scripts/run_experiments.py                    # every LLM found in traces/, 3 seeds, 60 epochs
python scripts/run_experiments.py --seeds 1 --epochs 5 --models qwen2.5:1.5b,qwen2.5:7b
# --sections inbench,transfer,ablate,llm   --skip-existing (resume)   --no-readme   --outdir results
# --devices cuda:0,cuda:1 --jobs 4   --no-reuse   --adopt-legacy   --force
python scripts/report_page.py                        # results/report.html: charts of every table
```

**Only new or stale results are trained.** Every result cell (`inbench/agentdojo`, `transfer/agentdojo->injecagent`, `ablate/memory/agentdojo`, ...) has a fingerprint (`src/cache.py`) over the training code (`src/train.py`, `src/eval.py`, `src/model/gnn.py`, `src/sandbox/policy.py`), the seeds and epochs, and every train and test graph tensor and meta of every seed. A stored cell with the same fingerprint is reused, anything else is trained, and `fingerprints_<section>.json` records which happened. Because the graphs themselves are hashed, a code change in graph building or the sandbox only invalidates the cells whose inputs it actually changes. `--no-reuse` trains everything. `--adopt-legacy` accepts cells stored before fingerprints existed (`experiments_all.json`) when that file is newer than every trace file the cell reads and every training-code file, and records their fingerprint; the old combined files are then moved to `results/<model>/legacy/`. Each model runs one job per section, heaviest first, so with `--devices cuda:0,cuda:1` the sections of one model run on both GPUs. A lock file stops two runs from writing the same reports.

It runs the benchmark sections once for each trace-generating LLM (into `results/<model>/`) and the cross-LLM section once over all of them (into `results/llm/`). It then writes `results/REPORT.md` (run metadata, trace counts, headline table, all tables) and updates the Results section below.

A single section:

```bash
python -m src.experiments --seeds 3 --epochs 60
# --only all|inbench|transfer|ablate|llm
# --model qwen2.5:1.5b      trace-generating LLM for the benchmark sections
# --llms a,b,c              LLMs compared with --only llm
```

| Section | Question |
|---|---|
| `inbench` | Held-out tasks within a benchmark |
| `transfer` | Train on other benchmark(s), test on a new one |
| `ablate` | Drop one graph component at a time (`tool_feat`, `observation`, `memory`, `injected`, `copies`, `mentions`) |
| `llm` | Behaviour of each LLM and transfer across LLMs (only benchmarks that have traces from every compared LLM) |

**No train/test leakage.** Training never contains an attack that appears in the test set. The attack is identified by `attack.id`: the attacker tool for InjecAgent and the injection task for AgentDojo. Without this, InjecAgent labels could be read off a memorised attacker tool name, because there every attacker-tool call is a reject. `inbench` and `llm` are also task-disjoint. `transfer` is disjoint by construction. The `llm` "same-task (leaky)" column is a deliberate upper bound.

Results are written to `<outdir>/experiments_<only>.json` and `<outdir>/report_<only>.md` (`--outdir`, default `results`). The `llm` section writes `<outdir>/llm_experiments.json` and `<outdir>/llm_report.md`. Numbers are mean±std over seeds, shown as `macro-F1 / micro-F1 / reject recall`. Micro-F1 equals accuracy for single-label classification. Tables also report **Attack Success Rate** (see below).

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

Regenerate with `python scripts/run_experiments.py`, which rewrites the block below.

<!-- results:start -->
Generated by `scripts/run_experiments.py` on 2026-11-09 (commit 5cbd971 (uncommitted changes), 3 seeds, 60 epochs). Full tables: [results/REPORT.md](results/REPORT.md).

GNN macro-F1 on held-out tasks, mean±std over seeds, with the tool-lookup baseline in brackets. Each row trains and tests on traces from one LLM.

| trace LLM | injecagent | agentdojo |
|---|---|---|
| gemma2:9b | 0.93±0.06 (0.45) | 0.76±0.01 (0.64) |
| llama3.1:8b | 0.89±0.02 (0.54) | 0.72±0.07 (0.62) |
| mistral:7b | 0.87±0.05 (0.46) | 0.75±0.05 (0.59) |
| qwen2.5:1.5b | 0.80±0.19 (0.58) | 0.67±0.12 (0.47) |
| qwen2.5:7b | 0.99±0.02 (0.37) | 0.65±0.08 (0.60) |
| qwen2.5:14b | 0.88±0.07 (0.49) | 0.74±0.02 (0.61) |
<!-- results:end -->

## Layout

```
data/            benchmark inputs (injecagent/, agentdojo/)
scripts/         run_experiments.py (all experiments + report), collect_traces.py (all trace collection),
                 report_page.py (results/report.html with charts), export_agentdojo.py (one-off data dump),
                 finalize_labels.py (migrate old traces to the 2-label schema)
src/
  collect.py     trace collection CLI
  experiments.py experiment suite CLI
  train.py       data loading, splits, training loop
  cache.py       result fingerprints: reuse cells whose inputs are unchanged
  eval.py        metrics and the tool-lookup baseline
  benchmarks/    InjecAgent / AgentDojo adapters
  sandbox/       scenario, mock tools, labelling policy, episode runner
  graph/         trace → HeteroData
  model/         HeteroGNN, ActionMLP
  llm/           Ollama client
tracegen/        AgentDojo trace generation from verified paraphrases (see "AgentDojo trace generation")
config.yaml      tracegen configuration;  NOTES.md: the AgentDojo APIs tracegen relies on
tests/           pytest suite (tests/tracegen/ for tracegen)
traces/          collected traces (*.jsonl)
results/         reports and raw experiment JSON
```

### AgentDojo data

`data/agentdojo/*.json` is already exported. To regenerate it, run `scripts/export_agentdojo.py` (`agentdojo==0.1.35` is in `requirements.txt`):

```bash
.venv/bin/python scripts/export_agentdojo.py data/agentdojo
```

The GNN code under `src/` never imports `agentdojo` or runs its environment. Only `tracegen/` (below) runs the real AgentDojo environment.

## Trace format

One JSON object per line, with these fields:

- **Metadata:** `benchmark`, `trace_id`, `model`, `agent`, `group` (the split key: same task means same group), `task`, `user_task`.
- **Context:** `memory`, `injected`, `attack` (`mode`, `style`, `category`, `tool`, `id`, or `null` when benign; traces collected before `id` existed get it recovered at load time), `tools` (name, description, kind, in-scope flag), `allowed_tools`. The scope fields are stored but the graph does not read them.
- **Events:** `events`, a list of steps with `step`, `tool`, `args`, `observation`, `obs_injected`, `synthetic` and `label`.
- **Quality:** `n_invalid`, the count of malformed LLM replies.

## AgentDojo trace generation (`tracegen`)

`tracegen` generates real AgentDojo agent traces. To get more variety per task, it paraphrases each user-task prompt,
verifies that every paraphrase keeps the same meaning, and runs agents on the original and on each verified variant.
Every variant reuses the original task's `utility` and `ground_truth` as its label, so it needs no new labels.
Agent runs use **no attack and no defense** unless attacks are switched on. The AgentDojo APIs it uses are documented
in `NOTES.md`.

| stage | command | output (under `data/tracegen/`) |
|---|---|---|
| 1 extract | `python -m tracegen extract` | `originals.jsonl` |
| 2 generate | `python -m tracegen generate` | `candidates.jsonl`, `generate_log.jsonl` |
| 3–5 verify | `python -m tracegen verify [--judge]` | `verified_variants.jsonl`, `rejected_variants.jsonl` |
| 6 / 6b trace | `python -m tracegen trace [--attacks]` | `traces.jsonl` (append-only) |
| 7 build | `python -m tracegen build [--dedup]` | `splits/{train,val,test}.jsonl`, `report.md`, `report.html` |
| all | `python -m tracegen all` | everything above |

- **Stage 2** asks the generator for 5 paraphrases in a single call. The call returns strict JSON, uses temperature
  0.9 and a seed fixed per task, and is retried up to 3 times with a new seed when the JSON is invalid. Duplicates and
  paraphrases identical to the original are dropped.
- **Stage 3** extracts the concrete values in the original and rejects any paraphrase that changes or drops one.
  Values covered: money, numbers, dates, times, IBANs, emails, URLs, file names, quoted strings and multi-word
  names. Names are compared ignoring case and whitespace. Numbers and identifiers must match exactly.
- **Stage 4** keeps a paraphrase only if `microsoft/deberta-large-mnli` gives entailment ≥ τ (0.8) in both directions.
- **Stage 5** (`--judge`, off by default) asks a model from a different family (gemma2:9b) whether both requests
  need exactly the same actions. (llama3.1:8b said NO to nearly every plain rewording with this prompt.)
- **Stage 6** records every run, success or failure, with its full message list, tool calls and results, `utility`,
  ground-truth calls, steps, tokens, latency and `error`. Crashes and timeouts are recorded and the run continues.
  A tqdm bar shows runs done / queued, the ETA, and running utility, ASR and error counts. The line for each finished
  run goes to `tracegen.log`, and also to the console with `-v` or when stderr is not a terminal. `--attacks` adds Stage 6b, which pairs each prompt with AgentDojo injection tasks (`important_instructions`) and
  records `security` (whether the attacker's goal was achieved). Injection texts are never paraphrased.
- **Stage 7** assigns train/val/test **by original task id**, so all variants, seeds, agents and attack runs of a task
  land in one split. The build fails if any task appears in two splits. `--dedup` keeps one trace per
  (suite, task, mode, action signature, utility[, security]). The report covers the variant funnel and rejection
  reasons, the NLI score histogram, utility on originals vs. paraphrases, unique action signatures per task, average
  steps, the most common divergence from the ground-truth call sequence, ASR, and leakage controls.

Every stage is resumable: it skips items that are already in its output. `--limit K` takes the first K user tasks per
suite. Model names with Ollama digests and seeds are logged on every candidate and every trace.

### GPUs and Ollama

`gpu.mode: parallel` (the default) uses one Ollama instance per GPU. The generator runs on GPU 0. NLI and the first
agent run on GPU 1. The stages stream into each other: each verified variant is traced right away while GPU 0 keeps
generating. Once generation is done, the generator is unloaded, and the remaining agents, seeds and attack runs use
both GPUs. `gpu.mode: split` runs one instance over both GPUs (for a generator that is too big for one, such as a
35B-A3B Q4 at ~20 GB). Its stages run one after another, and each model is unloaded before the next stage. Before
loading anything, tracegen logs every GPU's name and free VRAM and checks that each model fits. If one does not, it
stops with a suggestion.

With `gpu.manage_servers: true` (the default), tracegen starts and stops its own pinned instances. These run with a
16k context, a q8 KV cache, one model loaded per instance, and the system model store. The system service on :11434
is left alone. To run the instances yourself instead, set `manage_servers: false` and start:

```bash
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11436 OLLAMA_CONTEXT_LENGTH=16384 ollama serve   # generator
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 OLLAMA_CONTEXT_LENGTH=16384 ollama serve   # NLI + agent
CUDA_VISIBLE_DEVICES=0,1 OLLAMA_SCHED_SPREAD=1 OLLAMA_HOST=127.0.0.1:11437 ollama serve       # split mode
```

### Setup and usage

```bash
pip install -r requirements.txt                       # agentdojo, transformers, pyyaml
ollama pull qwen2.5:7b && ollama pull llama3.1:8b     # generator + agents (qwen2.5:14b for split mode)
python -m pytest tests/tracegen -q                    # offline: AgentDojo runs with a scripted fake LLM

python -m tracegen all --suites banking --limit 2 --out /tmp/tg_smoke     # smoke test
python -m tracegen all --suites banking                                   # one suite
python -m tracegen all                                                    # all four suites
python -m tracegen all --gpu-mode split --judge --dedup                   # split mode, judge on
```

Edit `config.yaml` to change the suites, models, N, τ, agents (local, or API agents via
`{name, base_url, api_key_env}`), `samples_per_prompt`, attacks, timeouts or split ratios.

## GNN guard on tracegen traces (with file nodes)

`scripts/run_tracegen_gnn.py` trains the approve/reject guard on the real AgentDojo runs from tracegen. It compares
majority, tool lookup, the no-graph MLP, the GNN from `docs/GRAPH.md`, and **GNN + files**: the same graph plus `file`
entity nodes for the files an action touches or an observation names (`src/graph/files.py`, see "File nodes" in
`docs/GRAPH.md`).

```bash
python -m tracegen trace --attacks                 # 1. attacked runs (resumable; benign runs already done are skipped)
python -m tracegen build                           # 2. refresh the splits / tracegen report
python scripts/run_tracegen_gnn.py                 # 3. train + evaluate on every GPU, write results/tracegen/
python scripts/run_tracegen_gnn.py --seeds 1 --epochs 3 --devices cpu --outdir /tmp/tg_gnn   # quick pass
```

- **Labels** (`src/benchmarks/tracegen.py`): a tool call is `reject` when it matches one of the injection task's
  ground-truth calls (recorded by tracegen as `injection_ground_truth_calls`), keyed on the attacker-chosen argument
  values. Values that the user task uses too, such as the transfer date, are ignored. Every other call, and every call
  of a benign run, is `approve`. Nothing is replayed synthetically, so an attack the agent ignores adds no `reject`.
- **Splits**: tracegen's task-level split, plus a separate seeded split of the injection tasks.
  `attacks.match_splits: true` (the default) only plans (user task, injection task) pairs from the same split. Val and
  test therefore hold unseen tasks, with all their paraphrases, and unseen attacks. `attacks.rows: originals` limits
  attack runs to the original prompts.
- **Leakage audit** (runs before training; the script refuses to train if a hard overlap is not 0): task groups,
  injection tasks, injected texts and normalised prompts shared between train and val/test. It also reports how many
  held-out attacker constants (for example the banking attacker IBAN) appear in training. The `attacker_values`
  ablation masks those constants in every trace.
- **Sections**: main (val, test, per suite, per agent); ablations of GNN + files (`files`, `observation`, `injected`,
  `copies`, `mentions`, `tool_feat`, `attacker_values`); cross-agent transfer. Each (cell, seed) is fingerprinted, and
  unchanged runs are reused. Outputs: `results/tracegen/experiments.json`, `REPORT.md`, `report.html`.
- With no attacked traces, the script stops and says so; it does not train on approve-only data.

## Cross-LLM transfer experiment

Does a guard trained on one LLM's traces work on another LLM's? `python scripts/cross_llm_transfer.py` trains on each LLM
(and leave-one-LLM-out), scores every LLM's held-out scenarios, and writes `results/crossllm/{REPORT.md,report.html,summary.json}`.
Uses both GPUs by default, reuses unchanged cells, and lists its leakage controls in the report. Controls are in `src/crossllm.py`; tests in `tests/test_crossllm.py`.
