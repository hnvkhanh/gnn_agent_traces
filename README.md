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
4. **Graph** (`src/graph/build.py`): node types are `task`, `memory`, `tool`, `action` and `observation`. Node features are 64-d hashed bag-of-tokens plus step index. Edges are causal, so action *i* only sees nodes that exist before its decision. The graph does not include a task-scope edge, so it is not told which tools the task needs. Besides the structural edges, `mentions` and `copies` link an earlier observation to a later action when the observation names that tool or contains one of its arguments.
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
- **Context:** `memory`, `injected`, `attack` (`mode`, `style`, `category`, `tool`, `id`, or `null` when benign; traces collected before `id` existed get it recovered at load time), `tools` (name, description, kind, in-scope flag), `allowed_tools`. The scope fields are stored but the graph does not read them.
- **Events:** `events`, a list of steps with `step`, `tool`, `args`, `observation`, `obs_injected`, `synthetic` and `label`.
- **Quality:** `n_invalid`, the count of malformed LLM replies.

## Continual graph learning

These experiments are separate from the main suite (`src/continual/`, `scripts/run_continual.py`, `results/continual/`). The GNN guard is trained on the benchmarks **one after another** (injecagent, agentdojo; **qwen2.5:1.5b traces only**) and re-tested on every benchmark after each stage. They measure how much it forgets, and whether attacks on earlier benchmarks start getting through again. The labels stay approve/reject, so this is domain-incremental learning with one head.

```bash
python scripts/run_continual.py                        # full run on both GPUs (2 jobs each), then the report
python scripts/run_continual.py --seeds 1 --epochs 5   # quick pass
python scripts/run_continual.py --report-only          # rebuild the report from the stored cells, no training
# --strategies finetune,joint,...   --buffer 50   --sweep 20,100,200|none   --devices cpu --jobs 8   --outdir results/continual
```

- **Protocol.** Both orders of the 2 benchmarks, 3 seeds, 60 epochs per stage. The split is the main suite's own (`train.split(..., "task", ...)` per benchmark), so the test sets are those of `inbench/<bench>`. After every stage the model is scored on all test sets. The untrained model is scored too, for forward transfer.
- **Strategies** (the CGLB baseline set): `finetune` (lower bound), `joint` (retrain on all training data so far; upper bound), `ewc`, `mas`, `lwf`, `agem` (GEM needs a QP solver), `twp` (topology term from the action-embedding norm, since SAGEConv has no attention), `ergnn` (mean-of-feature replay of whole trace graphs) and `replay` (reservoir).
  - Memory is M = 50 graphs, fixed in advance. The 20/100/200 sweep is reported as sensitivity only.
  - The `ewc` / `mas` / `lwf` / `twp` hyperparameters are tuned on a validation stream carved from the seed-0 training traces with the same split rule. Test data is never used for tuning.
- **Metrics.**
  - AP: macro-F1 on all benchmarks after the last stage.
  - AF: average forgetting.
  - BWT: backward transfer.
  - FWT: forward transfer.
  - The share of `|real` (non-synthetic) actions.
  - Final ASR, and **ASR forgetting**: how much ASR on earlier benchmarks rises by the end.
- **Leakage controls** (`tests/test_continual.py`, also audited in the report):
  - no task group, attack id, normalised task text or attacker-tool token is shared between any training set and any test set;
  - replay, A-GEM and ER-GNN memories hold only training graphs;
  - `joint` trains on exactly the union of the training sets so far;
  - validation is disjoint from both test and tune-train.
- **Output.** `results/continual/REPORT.md` (tables), `report.html` (charts, works offline, light/dark), `continual.json`, `hparams.json` and `cells/`. A cell is reused when its fingerprint (code, exact train/test graphs, config, device type) matches, so re-running resumes and recomputes only stale cells.
- **GPU etiquette.** A GPU run refuses to start while `run_experiments.py` or trace collection is using the GPUs (`--force` overrides). It takes about 15–25 min on 2 GPUs (≈12–25 s per cell).

## Cross-LLM transfer experiment

Does a guard trained on one LLM's traces work on another LLM's? `python scripts/cross_llm_transfer.py` trains on each LLM
(and leave-one-LLM-out), scores every LLM's held-out scenarios, and writes `results/crossllm/{REPORT.md,report.html,summary.json}`.
Uses both GPUs by default, reuses unchanged cells, and lists its leakage controls in the report. Controls are in `src/crossllm.py`; tests in `tests/test_crossllm.py`.
