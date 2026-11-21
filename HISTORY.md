# Work history

**Findings:** [findings/](findings/README.md) holds one file per investigation, with an index.

Every change to this repo is tracked here. The work runs in a loop:

```
change ──► track ──► analyze ──► decide next ──┐
  ▲                                            │
  └────────────────────────────────────────────┘
```

1. **Change:** anything in the repo, such as a new model, training setting, data, labels, script or doc.
2. **Track:** add a Log entry the same day: what changed, where (files, commits), and which result dirs it creates
   or invalidates (label/data versions, runs started, output dir, ETA).
3. **Analyze:** when results exist, put the numbers that matter in the entry (metric, split, seeds). A real
   investigation becomes a file in [findings/](findings/README.md), linked from the entry.
4. **Decide next:** from the current results, update "Current status" and reorder "Next". Remove what is done, add
   what the results call for, put the most urgent item first, and say what each item is blocked on.

A change whose results are still pending gets its entry at step 2. Steps 3–4 go into the same entry, or a new one,
when the results arrive. Log entries are newest first; the template is at the bottom.

## Current status (2026-11-22)

- **Phase 1, synthetic sandbox traces: done.** 6 Ollama LLMs × InjecAgent / AgentDojo. GNN macro-F1 on held-out tasks
  0.80–0.99 (InjecAgent) and 0.65–0.76 (AgentDojo), against 0.37–0.64 for tool lookup. Last regenerated 2026-11-09.
- **Phase 2, real-agent traces (`tracegen`): in progress.** Guards are compared on 5×5 CV of `data/tracegen_all`
  (1,155 rejects pooled), since the fixed 13-reject test split is too small to rank them.
- **Best guard so far:** Edge GNN with stable training, pooled CV AUPRC 0.377 (labels v1).
- **Labels v2 since 2026-11-22.** Workspace injection tasks 6–13 have no AgentDojo ground truth. Their mails to the
  attacker and deletes of those mails are now `reject` (workspace rejects 88 → 395, pooled 1,155 → 1,462).
  **Every tracegen result so far is v1** and cannot be compared with v2 runs. See
  [findings/2026-11-22-workspace-auprc-gap.md](findings/2026-11-22-workspace-auprc-gap.md).
- **Running:** `results/tracegen_all_repo20_prompt/`, the prompt and content guards (`edge-gnn+prompt`,
  `gnn+files+prompt`, `gnn+files+prompt+content`), 20 epochs, on both GPUs. Started 2026-11-22 00:46; at 01:28 it was
  38% done, with the expected finish around 02:35 (see its `progress.json`). Its workers loaded the data before the
  label change, so the whole run is v1. Do not resume it after a crash: a resume would retrain every cell on v2 and
  mix label versions.

## Next

Ordered by priority. Each item says what it is blocked on.

**Now**
- [ ] Let `tracegen_all_repo20_prompt` (v1) finish (about 02:35), then commit it. Record its prompt/content guard
  numbers as v1 only. *Blocked on:* the run.

**Then: labels v2**
- [ ] **Retrain on labels v2.** Same settings as `tracegen_all_stable`, new output dir:
  `python scripts/run_tracegen_gnn.py --tracegen data/tracegen_all --sections main,cv --folds 5 --seeds 3 --epochs 20
  --guards edge-gnn,gnn+files,transformer --devices cuda:0,cuda:0,cuda:1,cuda:1 --outdir results/tracegen_all_v2`.
  *Blocked on:* free GPUs (the item above).
- [ ] Run `python scripts/suite_gap.py --res results/tracegen_all_v2` and add the v2 numbers to the
  [workspace finding](findings/2026-11-22-workspace-auprc-gap.md). v1 workspace AUPRC was 0.07–0.12, and 0.18–0.25
  without the task 6–13 runs. Then set the finding's status to done. *Blocked on:* the v2 retrain.

**After v2**
- [ ] Re-check the `GUARD_FIX` choices on v2; they were all decided on v1 labels. *Blocked on:* the v2 retrain.
- [ ] Decide whether the prompt and content guards get a `GUARD_FIX` entry. This needs them rerun on v2 (the
  `--guards` of the prompt run, output e.g. `results/tracegen_all_v2_prompt`). *Blocked on:* the v2 retrain.
- [ ] Report lift over chance (or AUPRC at a matched reject share) next to per-suite AUPRC in the tracegen report.
  Raw AUPRC is not comparable across suites, or between sandbox and tracegen.

**Backlog**
- [ ] Extend the empty-ground-truth rule (workspace finding, "Still not covered"): the file deletes of tasks 11–12
  (check which file ids, since user tasks 35/38 delete files too), the "[SECURITY MESSAGE]" self-mail of task 12, and
  the phishing mails of task 13. That would be labels v3 and would need another retrain, so it is best batched.
- [ ] More followed workspace attacks: attack search or stronger templates on workspace (1.2% of attack runs follow).
- [ ] Track AUPRC on the hard-negative user tasks 19, 25, 35 and 38 (legitimate writes after reading the injection).
- [ ] Per-guard settings for MLP and GNN (repo vs `--stable`, 3 seeds, `--extra`).
- [ ] GNN + files: earlier stopping or more regularization (it overfits by epoch 1–2).
- [ ] Rerun the stable config with `--score prob` to separate lr and clipping from the logit score.
- [ ] Edge GNN with attacker constants masked, 3 seeds.
- [ ] Count third-party relay calls in train and report them as a separate category if they are rare.
- [ ] Update the stale parts of [docs/TRACEGEN_GNN.md](docs/TRACEGEN_GNN.md): the "Open" bullet (transformer run is
  done) and `STABLE_GUARDS` (transformer now gets averaging).
- [ ] Add Phase 2 results to the README Results section, from v2 numbers. *Blocked on:* the v2 retrain.

## Log

### 2026-11-22: work-loop tracking and findings folder
- **Change:** `HISTORY.md` now opens with the work loop (change → track → analyze → decide next) and links to the
  new [findings/](findings/README.md) folder (index plus one file per investigation). `docs/FINDINGS.md` moved to
  [findings/2026-11-22-workspace-auprc-gap.md](findings/2026-11-22-workspace-auprc-gap.md), and references were
  updated in the README, `docs/TRACEGEN_GNN.md`, `scripts/run_tracegen_gnn.py`, `scripts/suite_gap.py` and
  `src/benchmarks/tracegen.py`. Added `CLAUDE.md` with the tracking rule. New loop-shaped entry template.
- **Tracked:** docs and docstrings only. No result dir is created or invalidated.
- **Analysis:** –
- **Decision / next:** "Next" is reordered by priority with blockers: finish the v1 prompt run → v2 retrain → v2
  analysis → `GUARD_FIX` re-check → backlog.

### 2026-11-22: workspace AUPRC gap → labels v2
- **Change:** relabel rule `EMPTY_GT_RECIPIENT` in `src/benchmarks/tracegen.py`. For an injection task without
  ground truth, after an injected result, a `send_email` naming the attacker address and a `delete_email` of that
  mail are `reject`. New test in `tests/test_tracegen_gnn.py` (151 pass overall). New analysis script
  `scripts/suite_gap.py` (reads the stored CV scores, no retrain).
- **Tracked:** labels v2. Workspace rejects 88 → 395, followed runs 62 → 160, pooled rejects 1,155 → 1,462. Every
  earlier tracegen result dir is v1, and its fingerprints no longer match. The running `tracegen_all_repo20_prompt`
  stays v1 (its data was loaded before the change). Analysis output is in
  `results/tracegen_all_stable/suite_gap.txt` (v1).
- **Analysis:** [findings/2026-11-22-workspace-auprc-gap.md](findings/2026-11-22-workspace-auprc-gap.md). Workspace
  Edge GNN 0.10 vs banking 0.73 is mostly the metric: reject share 0.66% vs 12.7%. At banking's share, workspace
  scores 0.55–0.58, and its lift over chance (11–18×) beats banking's (5.6–5.9×). The rest is missing labels:
  AgentDojo v1.2 workspace injection tasks 6–13 have `ground_truth() == []`, so 228 mails to the attacker were
  `approve`. Dropping those runs doubles workspace AUPRC (→ 0.18–0.25). Other causes: few followed attacks (62 of
  5,043 runs), and hard negatives (user tasks 19/25/35/38 write after reading the injection).
- **Decision / next:** relabel rather than drop. Retrain on v2 into `results/tracegen_all_v2` once the GPUs are free,
  then rerun `suite_gap.py` on it. Rule extensions (tasks 11–13), lift reporting, more workspace attacks and
  hard-negative tracking were added to "Next".

### 2026-11-22
- **Done:** Transformer repo vs stable comparison on CV (`results/tracegen_all_repo20_tf`, `results/tracegen_all_stable_tf`,
  `results/training_compare_transformer.json`). Committed in `5491a85`.
- **Result:** averaging over epochs helps the Transformer (pooled CV 0.187 → 0.221, seed sd 0.026 → 0.003).
  Lower lr + clipping hurt it (0.221 → 0.197). It doesn't overfit.
- **Decision:** `GUARD_FIX` = Edge GNN full stable, Transformer averaging only, all other guards repo settings.
- **Started:** prompt and content guard run (`tracegen_all_repo20_prompt`).

### 2026-11-21
- **Done:** edge-level GNN (`src/model/edge_gnn.py`, `src/graph/edges.py`), prompt embeddings, MiniLM node embeddings,
  teacher-forcing runs, seed-variance scripts. Large data backed up as `.gz`. README shortened, tracegen docs moved to `docs/`.
- **Result:** seed variance came from chaotic training and the 13-reject test split. Edge GNN diverges at lr 3e-3.
  On 25-cell CV (3 seeds, 20 epochs):
  - Edge GNN: 0.294 (repo) → **0.377** (stable + averaging); within-cell seed sd 0.091 → 0.027.
  - GNN + files: 0.258 (repo + averaging); the stable settings hurt it (0.228).
  - Edge GNN beats GNN + files on CV under every setting.
- **Decision:** apply training fixes per guard, only where CV confirms them. Write-up in
  [results/tracegen_all_edge/ANALYSIS.md](results/tracegen_all_edge/ANALYSIS.md).

### 2026-11-20
- **Done:** qwen2.5:14b tracegen traces and GNN results, dashboard builder, combined dataset `data/tracegen_all`,
  Transformer guard, trace viewer.
- **Result:** the first combined run (`tracegen_all_edge`, 1 seed, 10 epochs) ranked Transformer first on test
  (AUPRC 0.62). This was later shown not to hold (see 2026-11-21).

### 2026-11-17 to 2026-11-19
- **Done:** qwen3.8:27b trace generation (scheduled script), AgentLAB tool-chaining data, dataset build,
  SecPO-style attack search stage in tracegen.

### 2026-11-15
- **Done:** cross-LLM transfer, confidence analysis, `tracegen` (real AgentDojo runs on paraphrased tasks),
  GNN guard with file nodes.
- **Removed:** ASB and AgentPoison benchmarks, continual-learning experiments.

### 2026-11-09
- **Done:** cross-platform (Windows / macOS) and device-aware code, traces for qwen2.5:14b, mistral:7b, gemma2:9b and
  llama3.1:8b, result cache with fingerprints, experiment and report scripts, leakage audit.
- **Result:** Phase 1 headline table (see README Results).

### 2026-11-08
- **Done:** initial pipeline. Two-label guardrail (`approve` / `reject`), ASR metrics, micro-F1, multi-LLM traces.

---

## Entry template

One entry per change, following the loop. Entries before 2026-11-22 use the older Done / Result / Decision wording.

```markdown
### YYYY-MM-DD: <short title>
- **Change:** what was built, changed or run (files, commits).
- **Tracked:** result dirs created or invalidated, runs started (output dir, ETA), label/data versions.
- **Analysis:** the numbers that matter (metric, split, seeds); link findings/<file>.md if investigated.
  "pending" until results exist.
- **Decision / next:** what we do next because of these results, and how "Next" changed.
```

After writing the entry, update "Current status" and "Next" at the top.
