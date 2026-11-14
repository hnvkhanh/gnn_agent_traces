# Continual graph learning report

The HeteroGNN guard is trained on one benchmark after another (qwen2.5:1.5b traces only, AgentPoison excluded) and re-tested on every benchmark after each stage. Domain-incremental: one head, labels approve/reject.

| | |
|---|---|
| date | 2026-11-09T18:07:46 |
| git commit | ace48d6 |
| python / torch | 3.12.3 / 2.14.1+cu126 |
| devices / parallel jobs | cuda:0, cuda:1 / 4 |
| seeds / epochs per experience | 3 / 60 |
| memory M (agem, replay, ergnn) | 50 graphs (sweep: 20, 100, 200) |
| orders | all 6 permutations of asb, injecagent, agentdojo |
| cells | 150 planned: 66 reused, 0 computed, 138 missing |
| command | `python scripts/run_continual.py --report-only` |

Cells are mean±std over seeds × orders. AP = macro-F1 on every benchmark after the last stage; AF = average forgetting (best earlier score minus final, ↓); BWT = final minus just-trained score; FWT = zero-shot score on the next benchmark minus the untrained model; ASR = attack traces with an attacker action the guard lets through; ASR forgetting = how much ASR on earlier benchmarks rises by the end (↓). 'real-only' drops the appended synthetic attacker steps.

## Headline

| strategy | AP macro-F1 ↑ | AF ↓ | BWT | FWT | AP real-only ↑ | reject recall ↑ | final ASR ↓ | ASR forgetting ↓ | benign blocked ↓ | s/cell | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finetune | 0.67±0.09 | 0.23±0.13 | -0.22±0.14 | +0.17±0.09 | 0.66±0.09 | 0.53±0.21 | 0.31±0.15 | +0.19±0.22 | 0.20±0.17 | 22 | 18 |
| joint | 0.83±0.03 | -0.00±0.06 | +0.01±0.06 | +0.19±0.11 | 0.83±0.04 | 0.70±0.05 | 0.18±0.02 | -0.01±0.03 | 0.04±0.05 | 45 | 18 |
| agem | 0.76±0.04 | 0.11±0.08 | -0.10±0.08 | +0.19±0.10 | 0.75±0.05 | 0.61±0.08 | 0.25±0.05 | +0.09±0.09 | 0.10±0.08 | 35 | 18 |

Best regulariser by AP: **None**. Best memory method by AP: **agem**.

## Per order (AP / AF / ASR forgetting, mean±std over seeds)

| strategy | agentdojo → asb → injecagent | agentdojo → injecagent → asb | asb → agentdojo → injecagent | asb → injecagent → agentdojo | injecagent → agentdojo → asb | injecagent → asb → agentdojo |
|---|---|---|---|---|---|---|
| finetune | 0.75±0.06 / 0.10±0.07 / +0.15±0.08 | 0.69±0.04 / 0.20±0.08 / -0.04±0.08 | 0.73±0.07 / 0.14±0.07 / +0.20±0.07 | 0.56±0.04 / 0.39±0.03 / +0.48±0.04 | 0.69±0.06 / 0.20±0.15 / -0.09±0.03 | 0.60±0.05 / 0.32±0.03 / +0.42±0.06 |
| joint | 0.84±0.02 / -0.00±0.01 / -0.03±0.02 | 0.83±0.01 / -0.01±0.06 / -0.01±0.03 | 0.84±0.02 / -0.04±0.01 / -0.03±0.02 | 0.82±0.04 / 0.06±0.05 / -0.01±0.01 | 0.83±0.03 / -0.03±0.07 / +0.02±0.04 | 0.84±0.02 / 0.01±0.03 / -0.01±0.01 |
| agem | 0.77±0.06 / 0.09±0.06 / +0.06±0.13 | 0.76±0.04 / 0.09±0.11 / +0.02±0.03 | 0.78±0.04 / 0.08±0.08 / +0.13±0.06 | 0.74±0.02 / 0.13±0.03 / +0.16±0.05 | 0.76±0.02 / 0.09±0.06 / +0.05±0.06 | 0.72±0.01 / 0.18±0.05 / +0.15±0.03 |

## Accuracy matrix R, canonical order asb → injecagent → agentdojo (macro-F1, mean over seeds)

Row = after training on that benchmark; column = test benchmark. Lower triangle = memory, upper = zero-shot.

**finetune**

|  | asb | injecagent | agentdojo |
|---|---|---|---|
| after asb | 0.97 | 0.47 | 0.30 |
| after injecagent | 0.76 | 0.89 | 0.56 |
| after agentdojo | 0.47 | 0.60 | 0.62 |

**joint**

|  | asb | injecagent | agentdojo |
|---|---|---|---|
| after asb | 0.97 | 0.47 | 0.30 |
| after injecagent | 0.96 | 0.92 | 0.46 |
| after agentdojo | 0.88 | 0.89 | 0.68 |

**agem**

|  | asb | injecagent | agentdojo |
|---|---|---|---|
| after asb | 0.97 | 0.47 | 0.30 |
| after injecagent | 0.83 | 0.89 | 0.52 |
| after agentdojo | 0.80 | 0.80 | 0.61 |

## ASR on each benchmark after each stage, canonical order (lower is better)

| strategy | after asb (asb / injecagent / agentdojo) | after injecagent (asb / injecagent / agentdojo) | after agentdojo (asb / injecagent / agentdojo) |
|---|---|---|---|
| finetune | 0.03 / 0.10 / 0.32 | 0.48 / 0.09 / 0.35 | 0.86 / 0.22 / 0.45 |
| joint | 0.03 / 0.10 / 0.32 | 0.05 / 0.08 / 0.34 | 0.04 / 0.05 / 0.39 |
| agem | 0.03 / 0.10 / 0.32 | 0.23 / 0.09 / 0.32 | 0.30 / 0.13 / 0.43 |

## Hyperparameters (chosen on the validation stream, never on test)

Validation = the main suite's task split applied again inside the seed-0 training traces. Criterion: final mean macro-F1 over the validation sets, canonical order.

| strategy | chosen | val AP | grid (hp → val AP) |
|---|---|---|---|
| twp | {"beta": 0.001, "lam_l": 100.0} | 0.558 | {"beta": 0.0, "lam_l": 10.0} → 0.553; {"beta": 0.0, "lam_l": 100.0} → 0.524; {"beta": 0.0, "lam_l": 1000.0} → 0.520; {"beta": 0.0, "lam_l": 10000.0} → 0.486; {"beta": 0.001, "lam_l": 10.0} → 0.527; {"beta": 0.001, "lam_l": 100.0} → 0.558; {"beta": 0.001, "lam_l": 1000.0} → 0.498; {"beta": 0.001, "lam_l": 10000.0} → 0.476; {"beta": 0.01, "lam_l": 10.0} → 0.429; {"beta": 0.01, "lam_l": 100.0} → 0.429; {"beta": 0.01, "lam_l": 1000.0} → 0.429; {"beta": 0.01, "lam_l": 10000.0} → 0.429 |

## Data and leakage audit

Per seed: train/test traces (attack traces) per benchmark, and overlap counts between the union of all training sets and the union of all test sets. Every overlap must be 0.

| split | asb train / test | injecagent train / test | agentdojo train / test | overlaps |
|---|---|---|---|---|
| seed 0 | 307 (204) / 85 (67) | 174 (94) / 61 (45) | 106 (38) / 62 (42) | traces 0, groups 0, attacks 0, task_text 0, attacker_tokens 0 |
| seed 1 | 330 (237) / 82 (54) | 177 (96) / 59 (44) | 84 (10) / 59 (45) | traces 0, groups 0, attacks 0, task_text 0, attacker_tokens 0 |
| seed 2 | 317 (215) / 85 (66) | 178 (97) / 61 (46) | 116 (42) / 62 (48) | traces 0, groups 0, attacks 0, task_text 0, attacker_tokens 0 |

- Validation split vs tune-train / vs test: val vs tune-train: traces 0, groups 0, attacks 0, task_text 0, attacker_tokens 0; val vs test: traces 0, groups 0, attacks 0, task_text 0, attacker_tokens 0
- Memory buffers (agem, replay, ergnn): 2700 stored graphs checked, 0 outside the training sets, 0 in a test set.
- Joint: 54 steps checked, 0 trained on anything other than the union of the training sets so far.
- AgentDojo has as few as 10 training attack traces in some seeds (inherited from the main suite's split), so its cells are noisier.
- Validation sets are small (asb 60, injecagent 20, agentdojo 20 traces), so the tuned values are a coarse choice.

## Sanity check against the main suite

Stage-1 finetune score on a benchmark (the model has seen only that benchmark) vs the main suite's GNN `inbench/<bench>` on the same test sets. Not bit-identical: there the GNN is initialised after the MLP baseline has used the RNG.

| benchmark | CL stage 1 (finetune) | main inbench GNN | source |
|---|---|---|---|
| asb | 0.97±0.02 | 0.97±0.02 | qwen2.5_1.5b/experiments_inbench.json |
| injecagent | 0.85±0.16 | 0.80±0.19 | qwen2.5_1.5b/experiments_inbench.json |
| agentdojo | 0.63±0.09 | 0.67±0.12 | qwen2.5_1.5b/experiments_inbench.json |

**Incomplete:** 138 planned cells have no fresh result (ewc 22, mas 22, lwf 22, ergnn 27, replay 27, twp 18). Re-run the script to finish them.
