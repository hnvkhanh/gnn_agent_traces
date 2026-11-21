
# GNN guard on tracegen traces (with file nodes)

`scripts/run_tracegen_gnn.py` trains the approve/reject guard on the real AgentDojo runs from tracegen. It compares
majority, tool lookup, the no-graph MLP, the GNN from `GRAPH.md`, and **GNN + files**: the same graph plus `file`
entity nodes for the files an action touches or an observation names (`src/graph/files.py`, see "File nodes" in
`GRAPH.md`).

```bash
python -m tracegen trace --attacks                 # 1. attacked runs (resumable; benign runs already done are skipped)
python -m tracegen build                           # 2. refresh the splits / tracegen report
python scripts/run_tracegen_gnn.py                 # 3. train + evaluate on every GPU, write results/tracegen/
python scripts/run_tracegen_gnn.py --seeds 1 --epochs 3 --devices cpu --outdir /tmp/tg_gnn   # quick pass
```

- **Labels** (`src/benchmarks/tracegen.py`): a tool call is `reject` when it matches one of the injection task's
  ground-truth calls (recorded by tracegen as `injection_ground_truth_calls`), keyed on the attacker-chosen argument
  values. Values that the user task uses too, such as the transfer date, are ignored. The call must also come after a
  tool result that carried the injected text, and must not be a call that benign runs of the same user task make.
  Every other call, and every call of a benign run, is `approve`. AgentDojo gives workspace injection tasks 6–13 no
  ground-truth calls (bulk exfiltration, checked only by `security()`). For them, a `send_email` naming the attacker
  address and a `delete_email` of that mail are `reject` (`EMPTY_GT_RECIPIENT`, since 2026-11-22). Results produced
  before that date used the old labels; see [the workspace finding](../findings/2026-11-22-workspace-auprc-gap.md). Nothing is replayed synthetically, so an attack the
  agent ignores adds no `reject`.
- **Events**: a call that repeats an earlier call of the run with the same arguments and result is dropped, and a run
  is cut after 30 calls (`MAX_EVENTS`). The report counts both.
- **Model selection** (`src/tracegen_gnn.py`): each guard keeps the epoch with the best validation AUPRC and uses the
  reject threshold with the best validation reject F1. Validation scores are therefore optimistic; test is the
  unbiased number. With no reject action in validation, the last epoch and threshold 0.5 are used.
- **Scores** (`src/rank_metrics.py`): AUPRC and reject recall at 1% approve FPR, each with a 95% bootstrap interval
  over traces, then reject recall, approve FPR, macro-F1 and ASR at the tuned threshold. A set with no reject action
  has no AUPRC, recall or macro-F1 and shows `–`.
- **Splits**: tracegen's task-level split, plus a separate seeded split of the injection tasks.
  `attacks.match_splits: true` (the default) only plans (user task, injection task) pairs from the same split. Val and
  test therefore hold unseen tasks, with all their paraphrases, and unseen attacks. `attacks.rows: originals` limits
  attack runs to the original prompts.
- **Leakage audit** (runs before training; the script refuses to train if a hard overlap is not 0): task groups,
  injection tasks, injected texts and normalised prompts shared between train and val/test. It also reports how many
  held-out attacker constants (for example the banking attacker IBAN) appear in training. The `attacker_values`
  ablation masks those constants in every trace.
- **Cross-validation** (`--sections cv`, `--folds 5`): the fixed test split holds only a few followed attacks, so
  the headline number comes from 5 task folds × 5 injection-task folds. Cell (a, b) tests the attack runs of task
  fold a with an injection task of fold b (plus a share of the benign runs of fold a), selects on block (a+1, b+1),
  and trains on runs whose task and injection task are in neither fold. Every run is tested once by a guard that saw
  neither its user task nor its injection task; the report pools the cells per seed and adds a table per injection
  task. Attack runs generated with `attacks.match_splits: false` are used here and ignored by the fixed split.
- **Sections**: main (val, test, per suite, per agent); cv; ablations of GNN + files (`files`, `observation`, `injected`,
  `copies`, `mentions`, `tool_feat`, `attacker_values`); cross-agent transfer. Each (cell, seed) is fingerprinted, and
  unchanged runs are reused. Outputs: `results/tracegen/experiments.json`, `REPORT.md`, `report.html`.
- With no attacked traces, the script stops and says so; it does not train on approve-only data.

## Latest progress: combined dataset, edge GNN, stable training

The guards now run on `data/tracegen_all` (all tracegen traces combined; the large files are committed as `.gz`).
New since the first GNN + files results:

- **Guards** (`src/model/`): an edge-level GNN (`edge_gnn.py`, edges built in `src/graph/edges.py`), a transformer
  guard, optional prompt embeddings (`src/prompt_embed.py`) and node embeddings (`src/node_embed.py`).
- **Seed variance.** On the fixed test split (13 reject actions from 3 injection tasks) the guard ranking changes
  with the seed, and even a re-run of the same seed differs (non-deterministic CUDA kernels). The headline number is
  therefore the 5×5 cross-validation. Analysis: [results/tracegen_all_edge/ANALYSIS.md](../results/tracegen_all_edge/ANALYSIS.md).
  `scripts/seed_variance.py` measures the spread.
- **Stable training.** Edge GNN diverges at lr 3e-3. Training it with lr 1e-3, gradient clipping and an average over
  the second half of the epochs (`STABLE` in `scripts/run_tracegen_gnn.py`) raised pooled CV AUPRC from 0.294 to
  0.377 over 3 seeds, 20 epochs. The same fix hurts GNN + files (0.228 vs 0.258, it overfits by epoch 1–2). So
  `STABLE_GUARDS` is only `["edge-gnn"]`; the other guards keep the old settings. `--stable` applies it to every
  guard and `--stable-guards=` to none. Compare runs with `python scripts/compare_training.py <dirs>`; the stored
  comparison is `results/training_compare.json`.
- **Per-suite gap.** Workspace AUPRC (0.10) is far below banking (0.73) mostly because of its 0.7% reject share
  and because AgentDojo gives workspace injection tasks 6–13 no ground-truth calls, so their attacks are labelled
  `approve`. Those attacks are now labelled by a rule (labels v2; workspace rejects 88 → 395). See
  [the workspace finding](../findings/2026-11-22-workspace-auprc-gap.md) and `scripts/suite_gap.py`.
- **Open:** the same comparison for the transformer (`results/tracegen_all_repo20_tf`, `results/tracegen_all_stable_tf`)
  was still running at the last commit. Rerunning the same command resumes it.
- Results: `results/tracegen_all_edge/` (first run), `results/tracegen_all_repo20/` (old settings) and
  `results/tracegen_all_stable/` (stable settings), each with `REPORT.md` and `report.html`.
