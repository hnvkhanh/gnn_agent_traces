# GNN guard on tracegen traces (AgentDojo, with file nodes)

Generated 2026-11-17 10:24 by `scripts/run_tracegen_gnn.py` · seeds [0, 1, 2] · epochs 60 · devices cuda:0,cuda:1 · git 1f1815d.

Per-action approve/reject. A held-out split has unseen user tasks (and all their paraphrases) and unseen injection tasks. Scores are mean ± std over seeds.

## Data

| split | traces | benign | attacked | followed attack | security true | actions | reject | tasks | injection tasks |
|---|---|---|---|---|---|---|---|---|---|
| train | 4150 | 533 | 3617 | 178 | 101 | 14253 | 774 | 68 | 24 |
| val | 302 | 120 | 182 | 20 | 19 | 2211 | 28 | 15 | 6 |
| test | 291 | 116 | 175 | 4 | 1 | 1719 | 8 | 14 | 5 |

Records read: 5136; dropped with no tool call: 393; attack runs dropped because the injection task is in another split: 0.

## Test

| guard | macro-F1 | reject recall | approve recall | ASR no guard → guard | benign traces blocked | approve FPR |
|---|---|---|---|---|---|---|
| Majority | 0.50 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.02 ± 0.00 → 0.02 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Tool lookup | 0.50 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.02 ± 0.00 → 0.02 ± 0.00 | 0.01 ± 0.00 | 0.00 ± 0.00 |
| MLP (no graph) | 0.54 ± 0.00 | 0.29 ± 0.06 | 0.98 ± 0.00 | 0.02 ± 0.00 → 0.02 ± 0.00 | 0.06 ± 0.01 | 0.02 ± 0.00 |
| GNN | 0.50 ± 0.01 | 0.21 ± 0.06 | 0.95 ± 0.02 | 0.02 ± 0.00 → 0.02 ± 0.00 | 0.09 ± 0.00 | 0.05 ± 0.02 |
| GNN + files | 0.52 ± 0.02 | 0.25 ± 0.00 | 0.96 ± 0.02 | 0.02 ± 0.00 → 0.02 ± 0.00 | 0.10 ± 0.01 | 0.04 ± 0.02 |

## Validation

| guard | macro-F1 | reject recall | approve recall | ASR no guard → guard | benign traces blocked | approve FPR |
|---|---|---|---|---|---|---|
| Majority | 0.50 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.11 ± 0.00 → 0.11 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Tool lookup | 0.50 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.11 ± 0.00 → 0.11 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| MLP (no graph) | 0.54 ± 0.00 | 0.62 ± 0.04 | 0.90 ± 0.00 | 0.11 ± 0.00 → 0.06 ± 0.01 | 0.22 ± 0.00 | 0.10 ± 0.00 |
| GNN | 0.78 ± 0.05 | 0.75 ± 0.03 | 0.99 ± 0.01 | 0.11 ± 0.00 → 0.04 ± 0.00 | 0.03 ± 0.01 | 0.01 ± 0.01 |
| GNN + files | 0.74 ± 0.06 | 0.73 ± 0.03 | 0.98 ± 0.01 | 0.11 ± 0.00 → 0.04 ± 0.01 | 0.05 ± 0.01 | 0.02 ± 0.01 |

## Test macro-F1 per suite

| guard | banking | slack | travel | workspace |
|---|---|---|---|---|
| Majority | 0.50 ± 0.00 | 0.49 ± 0.00 | 0.50 ± 0.00 | 0.50 ± 0.00 |
| Tool lookup | 0.47 ± 0.00 | 0.49 ± 0.00 | 0.50 ± 0.00 | 0.50 ± 0.00 |
| MLP (no graph) | 0.46 ± 0.01 | 0.52 ± 0.04 | 0.56 ± 0.01 | 0.50 ± 0.00 |
| GNN | 0.49 ± 0.01 | 0.49 ± 0.02 | 0.51 ± 0.03 | 0.50 ± 0.00 |
| GNN + files | 0.50 ± 0.00 | 0.51 ± 0.01 | 0.53 ± 0.04 | 0.50 ± 0.00 |

## Test macro-F1 per agent

| guard | llama3.1:8b | qwen2.5:7b |
|---|---|---|
| Majority | 0.50 ± 0.00 | 0.50 ± 0.00 |
| Tool lookup | 0.50 ± 0.00 | 0.50 ± 0.00 |
| MLP (no graph) | 0.49 ± 0.00 | 0.55 ± 0.00 |
| GNN | 0.49 ± 0.00 | 0.51 ± 0.01 |
| GNN + files | 0.49 ± 0.00 | 0.53 ± 0.03 |

## Ablations (GNN + files, test)

Full model: macro-F1 0.52 ± 0.02.

| removed | macro-F1 | Δ vs full | reject recall |
|---|---|---|---|
| file edges | 0.50 ± 0.02 | -0.018 | 0.25 ± 0.10 |
| observations | 0.50 ± 0.03 | -0.023 | 0.38 ± 0.10 |
| injected text | 0.50 ± 0.02 | -0.022 | 0.21 ± 0.12 |
| copies edges | 0.51 ± 0.01 | -0.012 | 0.25 ± 0.10 |
| mentions edges | 0.51 ± 0.01 | -0.014 | 0.29 ± 0.06 |
| tool features | 0.52 ± 0.00 | -0.007 | 0.21 ± 0.06 |
| attacker constants masked | 0.59 ± 0.10 | +0.065 | 0.46 ± 0.12 |

## Cross-agent transfer (test macro-F1)

| train → test | Majority | Tool lookup | MLP (no graph) | GNN | GNN + files |
|---|---|---|---|---|---|
| llama3.1:8b->qwen2.5:7b | 0.50 ± 0.00 | 0.50 ± 0.00 | 0.49 ± 0.00 | 0.51 ± 0.01 | 0.52 ± 0.02 |
| qwen2.5:7b->llama3.1:8b | 0.50 ± 0.00 | 0.49 ± 0.00 | 0.49 ± 0.00 | 0.49 ± 0.00 | 0.49 ± 0.01 |

## Leakage audit

Hard overlaps (must be 0): all 0.

| overlap | count |
|---|---|
| task_groups train∩val | 0 |
| injection_tasks train∩val | 0 |
| injected_texts train∩val | 0 |
| prompts train∩val | 0 |
| task_groups train∩test | 0 |
| injection_tasks train∩test | 0 |
| injected_texts train∩test | 0 |
| prompts train∩test | 0 |
| task_groups val∩test | 0 |

| check | value |
|---|---|
| attacker_values_val | 11 |
| attacker_values_val_in_train | 5 |
| attacker_values_test | 17 |
| attacker_values_test_in_train | 13 |
| mean_actions_benign | 4.61 |
| mean_actions_attack | 3.68 |

`attacker_values_*_in_train` counts attacker argument strings of held-out attacks that also occur in training actions or observations (AgentDojo reuses constants such as the banking attacker IBAN across injection tasks). The `attacker constants masked` ablation shows how much the guard depends on them.

## Provenance

Runs reused: 0, trained: 30, failed: 0.
Platform: Windows-11-10.0.26300-SP0, Python 3.14.4.
