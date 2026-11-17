# GNN guard on tracegen traces (AgentDojo, with file nodes)

Generated 2026-11-17 12:16 by `scripts/run_tracegen_gnn.py` · seeds [0, 1, 2] · epochs 60 · devices cuda:0,cuda:1 · git 1f1815d.

Per-action approve/reject. A held-out split has unseen user tasks (and all their paraphrases) and unseen injection tasks. Scores are mean ± std over seeds. Each guard uses the epoch with the best validation AUPRC and the reject threshold with the best validation reject F1, so validation scores are optimistic and test is the unbiased number. A set with no reject action has no AUPRC, recall or macro-F1 (shown as –).

## Data

| split | traces | benign | attacked | followed attack | security true | actions | reject | tasks | injection tasks |
|---|---|---|---|---|---|---|---|---|---|
| train | 4150 | 533 | 3617 | 154 | 101 | 10431 | 334 | 68 | 24 |
| val | 302 | 120 | 182 | 20 | 19 | 824 | 28 | 15 | 6 |
| test | 291 | 116 | 175 | 4 | 1 | 819 | 5 | 14 | 5 |

Records read: 5136; dropped with no tool call: 393; attack runs dropped because the injection task is in another split: 0.

Calls dropped as repeats of an earlier call with the same arguments and result: 6025. Runs cut at the length cap: 7 (84 calls, 1 of them reject).

A call is `reject` only if it matches an attacker call, comes after a tool result that carried the injected text, and is not a call that benign runs of the same user task make.

## Test

5 reject actions of 819 (291 traces). Chance AUPRC is the reject share, 0.006.

| guard | AUPRC | 95% CI | reject recall @ 1% FPR | 95% CI | reject recall | approve FPR | macro-F1 | ASR no guard → guard | benign traces blocked |
|---|---|---|---|---|---|---|---|---|---|
| Majority | 0.01 ± 0.00 | [0.00, 0.01] | 0.00 ± 0.00 | [0.00, 0.00] | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.50 ± 0.00 | 0.02 ± 0.00 → 0.02 ± 0.00 | 0.00 ± 0.00 |
| Tool lookup | 0.02 ± 0.00 | [0.00, 0.07] | 0.00 ± 0.00 | [0.00, 0.00] | 0.00 ± 0.00 | 0.02 ± 0.00 | 0.49 ± 0.00 | 0.02 ± 0.00 → 0.02 ± 0.00 | 0.06 ± 0.00 |
| MLP (no graph) | 0.25 ± 0.10 | [0.01, 0.73] | 0.40 ± 0.16 | [0.00, 1.00] | 0.53 ± 0.09 | 0.02 ± 0.00 | 0.63 ± 0.04 | 0.02 ± 0.00 → 0.01 ± 0.00 | 0.04 ± 0.01 |
| GNN | 0.32 ± 0.01 | [0.04, 0.55] | 0.40 ± 0.00 | [0.00, 0.93] | 0.27 ± 0.09 | 0.01 ± 0.00 | 0.62 ± 0.02 | 0.02 ± 0.00 → 0.02 ± 0.00 | 0.00 ± 0.00 |
| GNN + files | 0.31 ± 0.00 | [0.03, 0.55] | 0.27 ± 0.09 | [0.00, 0.67] | 0.33 ± 0.19 | 0.01 ± 0.01 | 0.59 ± 0.01 | 0.02 ± 0.00 → 0.02 ± 0.01 | 0.02 ± 0.03 |

AUPRC and reject recall at 1% approve FPR use the scores (no threshold); the interval is a bootstrap over traces, averaged over seeds. The last five columns use the reject threshold tuned on validation.

## Validation (used for model selection)

28 reject actions of 824 (302 traces). Chance AUPRC is the reject share, 0.034.

| guard | AUPRC | 95% CI | reject recall @ 1% FPR | 95% CI | reject recall | approve FPR | macro-F1 | ASR no guard → guard | benign traces blocked |
|---|---|---|---|---|---|---|---|---|---|
| Majority | 0.03 ± 0.00 | [0.02, 0.05] | 0.00 ± 0.00 | [0.00, 0.00] | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.49 ± 0.00 | 0.11 ± 0.00 → 0.11 ± 0.00 | 0.00 ± 0.00 |
| Tool lookup | 0.58 ± 0.00 | [0.37, 0.74] | 0.50 ± 0.00 | [0.29, 0.66] | 0.50 ± 0.00 | 0.00 ± 0.00 | 0.83 ± 0.00 | 0.11 ± 0.00 → 0.08 ± 0.00 | 0.00 ± 0.00 |
| MLP (no graph) | 0.67 ± 0.02 | [0.47, 0.83] | 0.64 ± 0.03 | [0.45, 0.81] | 0.61 ± 0.03 | 0.00 ± 0.00 | 0.85 ± 0.01 | 0.11 ± 0.00 → 0.05 ± 0.00 | 0.01 ± 0.01 |
| GNN | 0.93 ± 0.02 | [0.81, 0.98] | 0.82 ± 0.03 | [0.65, 0.98] | 0.80 ± 0.04 | 0.00 ± 0.00 | 0.93 ± 0.00 | 0.11 ± 0.00 → 0.03 ± 0.01 | 0.00 ± 0.00 |
| GNN + files | 0.96 ± 0.03 | [0.87, 0.99] | 0.90 ± 0.08 | [0.76, 0.99] | 0.87 ± 0.07 | 0.00 ± 0.00 | 0.95 ± 0.03 | 0.11 ± 0.00 → 0.02 ± 0.01 | 0.01 ± 0.00 |

AUPRC and reject recall at 1% approve FPR use the scores (no threshold); the interval is a bootstrap over traces, averaged over seeds. The last five columns use the reject threshold tuned on validation.

## Model selection

| guard | best epoch | reject threshold | validation AUPRC |
|---|---|---|---|
| Majority | – | 0.50 ± 0.00 | – |
| Tool lookup | – | 0.37 ± 0.00 | – |
| MLP (no graph) | 44 ± 15 | 0.75 ± 0.23 | 0.67 ± 0.02 |
| GNN | 24 ± 16 | 0.99 ± 0.01 | 0.93 ± 0.02 |
| GNN + files | 16 ± 11 | 0.98 ± 0.01 | 0.96 ± 0.03 |

## Test AUPRC per suite

| guard | banking (0 reject) | slack (3 reject) | travel (2 reject) | workspace (0 reject) |
|---|---|---|---|---|
| Majority | – | 0.02 ± 0.00 | 0.01 ± 0.00 | – |
| Tool lookup | – | 0.02 ± 0.00 | 0.67 ± 0.00 | – |
| MLP (no graph) | – | 0.18 ± 0.13 | 1.00 ± 0.00 | – |
| GNN | – | 0.41 ± 0.02 | 0.58 ± 0.00 | – |
| GNN + files | – | 0.41 ± 0.02 | 0.72 ± 0.20 | – |

## Test AUPRC per agent

| guard | llama3.1:8b (0 reject) | qwen2.5:7b (5 reject) |
|---|---|---|
| Majority | – | 0.01 ± 0.00 |
| Tool lookup | – | 0.03 ± 0.00 |
| MLP (no graph) | – | 0.30 ± 0.06 |
| GNN | – | 0.35 ± 0.01 |
| GNN + files | – | 0.36 ± 0.02 |

## Ablations (GNN + files, test)

Full model: AUPRC 0.31 ± 0.00, 95% CI [0.03, 0.55]. A change smaller than the interval is not a finding.

| removed | AUPRC | Δ vs full | 95% CI | reject recall @ 1% FPR |
|---|---|---|---|---|
| file edges | 0.31 ± 0.03 | +0.001 | [0.04, 0.53] | 0.27 ± 0.09 |
| observations | 0.35 ± 0.08 | +0.045 | [0.02, 0.58] | 0.47 ± 0.09 |
| injected text | 0.29 ± 0.03 | -0.017 | [0.04, 0.56] | 0.20 ± 0.00 |
| copies edges | 0.55 ± 0.09 | +0.244 | [0.01, 0.90] | 0.60 ± 0.00 |
| mentions edges | 0.34 ± 0.03 | +0.031 | [0.03, 0.60] | 0.33 ± 0.09 |
| tool features | 0.34 ± 0.02 | +0.038 | [0.07, 0.58] | 0.27 ± 0.09 |
| attacker constants masked | 0.56 ± 0.01 | +0.254 | [0.07, 0.95] | 0.73 ± 0.19 |

## Cross-agent transfer (test AUPRC)

| train → test | test reject actions | Majority | Tool lookup | MLP (no graph) | GNN | GNN + files |
|---|---|---|---|---|---|---|
| llama3.1:8b->qwen2.5:7b | 5 | 0.01 ± 0.00 | 0.01 ± 0.00 | 0.16 ± 0.09 | 0.17 ± 0.08 | 0.29 ± 0.13 |
| qwen2.5:7b->llama3.1:8b | 0 | – | – | – | – | – |

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
| mean_actions_benign | 2.81 |
| mean_actions_attack | 2.49 |

`attacker_values_*_in_train` counts attacker argument strings of held-out attacks that also occur in training actions or observations (AgentDojo reuses constants such as the banking attacker IBAN across injection tasks). The `attacker constants masked` ablation shows how much the guard depends on them.

## Provenance

Runs reused: 0, trained: 30, failed: 0.
Platform: Windows-11-10.0.26300-SP0, Python 3.14.4.
