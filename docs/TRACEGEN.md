# AgentDojo trace generation (`tracegen`)

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
| 6c attack search | `python -m tracegen attack-search [--search-pairs N] [--search-steps K]` | `search_log.jsonl`, winners appended to `traces.jsonl` |
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

## Attack search (Stage 6c)

Natural attacks rarely succeed on a strong agent (qwen3.8:27b: 10 of 2,148 attack runs). `attack-search` adds
attack-succeeded traces, following SecPO's Qwen/AgentDojo search (exact success check first, LLM mutation of the best
candidates). It only touches (user task, injection task) pairs of the `--agents` model where no attack run succeeded
(any attack, any prompt variant), using the same task/injection split matching as Stage 6b.

```
python -m tracegen attack-search --out data/tracegen_qwen38 --agents qwen3.8:27b --gpu-mode split --search-pairs 20
python -m tracegen build --out data/tracegen_qwen38 --attacks        # splits and report include the new traces
powershell -File scripts
un_qwen38.ps1 -SkipTrace -Search -SearchPairs 20    # search + build + GNN training
```

- A candidate is an injection **template** containing `{goal}` (also `{user}`, `{model}`, `{task}`); `{goal}` becomes the
  AgentDojo injection task goal, so the attacker goal and the `security` check are AgentDojo's own. Step 0 tries eight
  seed templates (skipping AgentDojo attacks already tried on the pair); later steps ask a proposer model for
  mutations of the best-scoring template of the current island, given the agent's reply to it and the other attempts.
- Success is `security=True` with no run error. Failures get a dense score, 1 + 8 x the share of the attacker's tools
  (not needed by the user task) that the agent called; it only guides the proposer.
- Every episode is appended to `search_log.jsonl` (template, score, tools called, trace id). The first success is
  appended to `traces.jsonl` as a normal attack trace with `attack_name: "searched"` and its `attack_template`.
  Failed episodes stay only in the log. Resuming rebuilds each pair's state from the log and skips pairs that already
  succeeded or used up their steps.
- `attacks.search` in `config.yaml`: `steps` (3), `candidates` per step (4), `islands` (2), `max_pairs`,
  `proposer_model` (default: the agent itself, so no model swap), `max_format_retries`. `--search` makes `all` run the
  stage after Stage 6 and before the build.
- `searched` ASR is tuned against one agent, so the report marks it as not comparable to natural ASR.

Every stage is resumable: it skips items that are already in its output. `--limit K` takes the first K user tasks per
suite. Model names with Ollama digests and seeds are logged on every candidate and every trace.

## GPUs and Ollama

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

## Setup and usage

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
