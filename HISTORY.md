# Work history

A log of what was done, what it showed, and what comes next. Newest entries first.
Add an entry per working session or milestone (template at the bottom). Keep "Current status" and "Next" up to date.

## Current status (2026-11-22)

- **Phase 1, synthetic sandbox traces: done.** 6 Ollama LLMs × InjecAgent / AgentDojo. GNN macro-F1 on held-out tasks
  0.80–0.99 (InjecAgent) and 0.65–0.76 (AgentDojo), against 0.37–0.64 for tool lookup. Last regenerated 2026-11-09.
- **Phase 2, real-agent traces (`tracegen`): in progress.** Guards are compared on 5×5 CV of `data/tracegen_all`
  (1,155 rejects pooled), since the fixed 13-reject test split is too small to rank them.
- **Best guard so far:** Edge GNN with stable training, pooled CV AUPRC 0.377.
- **Running:** `results/tracegen_all_repo20_prompt/`, the prompt and content guards (`edge-gnn+prompt`,
  `gnn+files+prompt`, `gnn+files+prompt+content`), 20 epochs. Started 2026-11-22 00:46, ETA about 11:27.

## Next

- [ ] Finish and commit `tracegen_all_repo20_prompt`. Decide whether the prompt and content guards get a `GUARD_FIX` entry.
- [ ] Per-guard settings for MLP and GNN (repo vs `--stable`, 3 seeds, `--extra`).
- [ ] GNN + files: earlier stopping or more regularization (it overfits by epoch 1–2).
- [ ] Rerun the stable config with `--score prob` to separate lr and clipping from the logit score.
- [ ] Edge GNN with attacker constants masked, 3 seeds.
- [ ] Count third-party relay calls in train and report them as a separate category if they are rare.
- [ ] Update the stale parts of [docs/TRACEGEN_GNN.md](docs/TRACEGEN_GNN.md): the "Open" bullet (transformer run is done) and `STABLE_GUARDS` (transformer now gets averaging).
- [ ] Add Phase 2 results to the README Results section.

## Log

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

```markdown
### YYYY-MM-DD
- **Done:** what was built or run (link result dirs, commits).
- **Result:** the numbers that matter, with the metric and split.
- **Decision:** what changed because of it, and why.
- **Started:** long runs still going (output dir, ETA).
```
