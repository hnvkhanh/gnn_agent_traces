# GNN guard on tracegen traces (AgentDojo, with file nodes)

Generated 2026-11-20 18:05 by `scripts/run_tracegen_gnn.py` · seeds [0] · epochs 10 · devices cuda:0,cuda:1 · git 5ac3541.

Per-action approve/reject. A held-out split has unseen user tasks (and all their paraphrases) and unseen injection tasks. Scores are mean ± std over seeds. Each guard uses the epoch with the best validation AUPRC and the reject threshold with the best validation reject F1, so validation scores are optimistic and test is the unbiased number. A set with no reject action has no AUPRC, recall or macro-F1 (shown as –).

## Data

| split | traces | benign | attacked | followed attack | security true | actions | reject | tasks | injection tasks |
|---|---|---|---|---|---|---|---|---|---|
| train | 8419 | 1079 | 7340 | 512 | 426 | 26901 | 1053 | 68 | 24 |
| val | 627 | 247 | 380 | 63 | 62 | 2064 | 81 | 15 | 6 |
| test | 586 | 233 | 353 | 11 | 7 | 1900 | 13 | 14 | 5 |

Records read: 10099; dropped with no tool call: 447; attack runs outside the fixed split because the injection task is in another split (used by the cross-validation only): 20.

Calls dropped as repeats of an earlier call with the same arguments and result: 7507. Runs cut at the length cap: 12 (104 calls, 2 of them reject).

A call is `reject` only if it matches an attacker call, comes after a tool result that carried the injected text, and is not a call that benign runs of the same user task make.

## Fixed split: test

13 reject actions of 1900 (586 traces). Chance AUPRC is the reject share, 0.007.

| guard | AUPRC | 95% CI | reject recall @ 1% FPR | 95% CI | reject recall | approve FPR | macro-F1 | ASR no guard → guard | benign traces blocked |
|---|---|---|---|---|---|---|---|---|---|
| Majority | 0.01 | [0.00, 0.01] | 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.50 | 0.03 → 0.03 | 0.00 |
| Tool lookup | 0.01 | [0.00, 0.03] | 0.00 | [0.00, 0.00] | 0.00 | 0.02 | 0.49 | 0.03 → 0.03 | 0.05 |
| MLP (no graph) | 0.22 | [0.10, 0.47] | 0.62 | [0.21, 0.86] | 0.23 | 0.01 | 0.61 | 0.03 → 0.02 | 0.02 |
| GNN | 0.55 | [0.24, 0.80] | 0.62 | [0.33, 0.88] | 0.61 | 0.00 | 0.82 | 0.03 → 0.01 | 0.00 |
| GNN + files | 0.58 | [0.23, 0.81] | 0.62 | [0.29, 0.88] | 0.61 | 0.01 | 0.74 | 0.03 → 0.01 | 0.00 |
| Transformer | 0.62 | [0.34, 0.89] | 0.77 | [0.50, 1.00] | 0.61 | 0.00 | 0.85 | 0.03 → 0.01 | 0.00 |
| Edge GNN | 0.53 | [0.27, 0.80] | 0.69 | [0.43, 0.93] | 0.61 | 0.00 | 0.82 | 0.03 → 0.01 | 0.00 |
| GNN + files + prompt | 0.53 | [0.20, 0.77] | 0.62 | [0.25, 0.88] | 0.39 | 0.00 | 0.74 | 0.03 → 0.02 | 0.00 |
| Edge GNN + prompt | 0.43 | [0.20, 0.70] | 0.54 | [0.28, 0.85] | 0.54 | 0.01 | 0.73 | 0.03 → 0.02 | 0.00 |

AUPRC and reject recall at 1% approve FPR use the scores (no threshold); the interval is a bootstrap over traces, averaged over seeds. The last five columns use the reject threshold tuned on validation.

## Fixed split: validation (used for model selection)

81 reject actions of 2064 (627 traces). Chance AUPRC is the reject share, 0.039.

| guard | AUPRC | 95% CI | reject recall @ 1% FPR | 95% CI | reject recall | approve FPR | macro-F1 | ASR no guard → guard | benign traces blocked |
|---|---|---|---|---|---|---|---|---|---|
| Majority | 0.04 | [0.03, 0.05] | 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.49 | 0.17 → 0.17 | 0.00 |
| Tool lookup | 0.56 | [0.47, 0.65] | 0.63 | [0.19, 0.72] | 0.63 | 0.01 | 0.84 | 0.17 → 0.07 | 0.02 |
| MLP (no graph) | 0.63 | [0.55, 0.73] | 0.58 | [0.49, 0.69] | 0.58 | 0.00 | 0.85 | 0.17 → 0.08 | 0.00 |
| GNN | 0.75 | [0.66, 0.83] | 0.68 | [0.58, 0.78] | 0.67 | 0.00 | 0.88 | 0.17 → 0.07 | 0.00 |
| GNN + files | 0.76 | [0.67, 0.84] | 0.67 | [0.56, 0.77] | 0.64 | 0.00 | 0.87 | 0.17 → 0.07 | 0.00 |
| Transformer | 0.62 | [0.54, 0.70] | 0.49 | [0.38, 0.60] | 0.49 | 0.01 | 0.79 | 0.17 → 0.10 | 0.00 |
| Edge GNN | 0.72 | [0.63, 0.80] | 0.63 | [0.54, 0.75] | 0.62 | 0.00 | 0.86 | 0.17 → 0.07 | 0.00 |
| GNN + files + prompt | 0.74 | [0.65, 0.82] | 0.57 | [0.49, 0.74] | 0.56 | 0.00 | 0.85 | 0.17 → 0.08 | 0.00 |
| Edge GNN + prompt | 0.65 | [0.53, 0.75] | 0.56 | [0.41, 0.67] | 0.93 | 0.03 | 0.83 | 0.17 → 0.01 | 0.02 |

AUPRC and reject recall at 1% approve FPR use the scores (no threshold); the interval is a bootstrap over traces, averaged over seeds. The last five columns use the reject threshold tuned on validation.

## Model selection (fixed split)

| guard | best epoch | reject threshold | validation AUPRC |
|---|---|---|---|
| Majority | – | 0.50 | – |
| Tool lookup | – | 0.45 | – |
| MLP (no graph) | 9 | 0.95 | 0.63 |
| GNN | 6 | 1.00 | 0.75 |
| GNN + files | 2 | 0.99 | 0.76 |
| Transformer | 2 | 0.96 | 0.62 |
| Edge GNN | 1 | 0.96 | 0.72 |
| GNN + files + prompt | 8 | 0.98 | 0.74 |
| Edge GNN + prompt | 1 | 0.77 | 0.65 |

## Fixed split: test AUPRC per suite

| guard | banking (0 reject) | slack (7 reject) | travel (3 reject) | workspace (3 reject) |
|---|---|---|---|---|
| Majority | – | 0.02 | 0.00 | 0.00 |
| Tool lookup | – | 0.02 | 0.75 | 0.01 |
| MLP (no graph) | – | 0.24 | 1.00 | 0.77 |
| GNN | – | 0.59 | 1.00 | 0.67 |
| GNN + files | – | 0.74 | 0.72 | 0.71 |
| Transformer | – | 0.60 | 1.00 | 0.73 |
| Edge GNN | – | 0.67 | 0.92 | 0.50 |
| GNN + files + prompt | – | 0.75 | 0.47 | 0.51 |
| Edge GNN + prompt | – | 0.73 | 0.59 | 0.18 |

## Fixed split: test AUPRC per agent

| guard | llama3.1:8b (0 reject) | qwen2.5:14b (8 reject) | qwen2.5:7b (5 reject) | qwen3.8:27b (0 reject) |
|---|---|---|---|---|
| Majority | – | 0.01 | 0.01 | – |
| Tool lookup | – | 0.02 | 0.03 | – |
| MLP (no graph) | – | 0.43 | 0.15 | – |
| GNN | – | 0.68 | 0.47 | – |
| GNN + files | – | 0.75 | 0.33 | – |
| Transformer | – | 0.72 | 0.62 | – |
| Edge GNN | – | 0.65 | 0.40 | – |
| GNN + files + prompt | – | 0.76 | 0.36 | – |
| Edge GNN + prompt | – | 0.64 | 0.35 | – |

## Ablations (GNN + files, test)

Full model: AUPRC 0.58, 95% CI [0.23, 0.81]. A change smaller than the interval is not a finding.

| removed | AUPRC | Δ vs full | 95% CI | reject recall @ 1% FPR |
|---|---|---|---|---|
| file edges | 0.62 | +0.049 | [0.28, 0.86] | 0.69 |
| observations | 0.68 | +0.106 | [0.37, 0.95] | 0.85 |
| injected text | 0.52 | -0.055 | [0.23, 0.79] | 0.69 |
| copies edges | 0.41 | -0.169 | [0.16, 0.71] | 0.77 |
| mentions edges | 0.58 | +0.009 | [0.25, 0.82] | 0.62 |
| tool features | 0.58 | +0.008 | [0.28, 0.81] | 0.69 |
| attacker constants masked | 0.70 | +0.125 | [0.41, 0.91] | 0.77 |

## Ablations (Edge GNN, test)

Full model: AUPRC 0.53, 95% CI [0.27, 0.80]. A change smaller than the interval is not a finding.

| removed | AUPRC | Δ vs full | 95% CI | reject recall @ 1% FPR |
|---|---|---|---|---|
| reverse edges (directed only) | 0.24 | -0.291 | [0.09, 0.62] | 0.54 |
| entity nodes | 0.45 | -0.087 | [0.22, 0.76] | 0.46 |
| returned edges | 0.54 | +0.001 | [0.25, 0.83] | 0.77 |
| task mentions edges | 0.32 | -0.218 | [0.11, 0.63] | 0.38 |
| observations | 0.26 | -0.277 | [0.06, 0.58] | 0.38 |
| attacker constants masked | 0.35 | -0.186 | [0.04, 0.62] | 0.31 |

## Does the prompt embedding help?

Frozen sentence embedding of the user prompt (src/prompt_embed.py) added to the task node and the readout of GNN + files and Edge GNN. Controls: the same guard with an all-zero embedding, and with the prompt of another user task of the same suite.

### Fixed split: test

| comparison | reject actions | ΔAUPRC | 95% paired CI | P(Δ ≤ 0) | Δ reject recall @ 1% FPR |
|---|---|---|---|---|---|
| GNN + files + prompt vs GNN + files | 13 | -0.043 | [-0.144, +0.028] | 0.85 | 0.000 |
| GNN + files + prompt vs zeroed prompt | 13 | -0.102 | [-0.263, -0.016] | 0.99 | 0.000 |
| GNN + files + prompt vs shuffled prompt (other task, same suite) | 13 | -0.056 | [-0.270, +0.099] | 0.80 | 0.000 |
| Edge GNN + prompt vs Edge GNN | 13 | -0.108 | [-0.327, +0.061] | 0.89 | -0.154 |
| Edge GNN + prompt vs zeroed prompt | 13 | -0.010 | [-0.161, +0.182] | 0.39 | 0.077 |
| Edge GNN + prompt vs shuffled prompt (other task, same suite) | 13 | 0.049 | [-0.323, +0.340] | 0.35 | 0.077 |

ΔAUPRC per suite:

| comparison | banking | slack | travel | workspace |
|---|---|---|---|---|
| GNN + files + prompt vs GNN + files | – | 0.011 | -0.248 | -0.209 |
| GNN + files + prompt vs zeroed prompt | – | 0.011 | 0.087 | -0.265 |
| GNN + files + prompt vs shuffled prompt (other task, same suite) | – | 0.116 | -0.159 | -0.415 |
| Edge GNN + prompt vs Edge GNN | – | 0.059 | -0.329 | -0.322 |
| Edge GNN + prompt vs zeroed prompt | – | 0.012 | 0.211 | 0.127 |
| Edge GNN + prompt vs shuffled prompt (other task, same suite) | – | 0.397 | -0.279 | -0.232 |

Δ = guard minus the comparison, on the same traces; the interval and P(Δ ≤ 0) come from a paired bootstrap over traces (both guards scored on every resample), averaged over seeds. Reading rule: the prompt embedding helps only if the real prompt beats the model without it AND both controls (zeroed, shuffled) with an interval above 0. Beating the zeroed prompt but not the shuffled one means the gain is suite identity or extra capacity, not the meaning of the prompt.

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
| mean_actions_benign | 3.48 |
| mean_actions_attack | 3.15 |

`attacker_values_*_in_train` counts attacker argument strings of held-out attacks that also occur in training actions or observations (AgentDojo reuses constants such as the banking attacker IBAN across injection tasks). The `attacker constants masked` ablation shows how much the guard depends on them.

## Provenance

Runs reused: 1, trained: 15, failed: 0.
Platform: Windows-11-10.0.26300-SP0, Python 3.14.4.
