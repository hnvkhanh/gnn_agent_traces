# Cross-LLM transfer of the trace guard

Question: a guard trained on the traces of one LLM, does it still work on the traces of another LLM?

- LLMs: gemma2:9b, llama3.1:8b, mistral:7b, qwen2.5:1.5b, qwen2.5:14b, qwen2.5:7b
- benchmarks: asb, injecagent, agentdojo
- seeds: 3 (each seed draws different held-out task groups), epochs: 60
- every number is mean±std over seeds; ◆ marks the diagonal (train LLM = test LLM, the in-LLM baseline)
- metric: macro-F1 over approve/reject actions of the held-out scenarios

## Headline (pooled)

In-LLM = trained and tested on the same LLM; cross-LLM = mean over the other LLMs; gap = in-LLM − cross-LLM (paired per seed). Verdict is a rule of thumb: gap ≤ 0.03 transfers, ≤ 0.10 partial, otherwise weak.

**gnn**

| train LLM | in-LLM | cross-LLM mean | gap | worst target | leave-one-LLM-out → this LLM | verdict |
|---|---|---|---|---|---|---|
| gemma2:9b | 0.84±0.03 | 0.80±0.03 | 0.04±0.01 | 0.74±0.04 | 0.79±0.03 | partial |
| llama3.1:8b | 0.80±0.03 | 0.78±0.04 | 0.02±0.03 | 0.76±0.05 | 0.80±0.03 | transfers |
| mistral:7b | 0.79±0.02 | 0.81±0.02 | -0.02±0.00 | 0.78±0.01 | 0.77±0.04 | transfers |
| qwen2.5:1.5b | 0.84±0.02 | 0.81±0.02 | 0.03±0.01 | 0.78±0.01 | 0.76±0.06 | partial |
| qwen2.5:14b | 0.81±0.03 | 0.81±0.04 | -0.00±0.01 | 0.75±0.03 | 0.78±0.06 | transfers |
| qwen2.5:7b | 0.83±0.04 | 0.78±0.03 | 0.04±0.03 | 0.71±0.04 | 0.84±0.03 | partial |

**tool-lookup**

| train LLM | in-LLM | cross-LLM mean | gap | worst target | leave-one-LLM-out → this LLM | verdict |
|---|---|---|---|---|---|---|
| gemma2:9b | 0.46±0.00 | 0.47±0.01 | -0.01±0.01 | 0.45±0.00 | 0.45±0.03 | transfers |
| llama3.1:8b | 0.48±0.01 | 0.48±0.01 | 0.00±0.02 | 0.46±0.01 | 0.46±0.03 | transfers |
| mistral:7b | 0.46±0.00 | 0.48±0.01 | -0.02±0.01 | 0.45±0.00 | 0.47±0.05 | transfers |
| qwen2.5:1.5b | 0.46±0.05 | 0.41±0.01 | 0.05±0.05 | 0.39±0.01 | 0.49±0.05 | partial |
| qwen2.5:14b | 0.50±0.00 | 0.47±0.02 | 0.03±0.02 | 0.45±0.01 | 0.49±0.03 | transfers |
| qwen2.5:7b | 0.44±0.02 | 0.46±0.03 | -0.03±0.00 | 0.44±0.02 | 0.47±0.02 | transfers |

## Macro-F1 matrices (GNN)

### pooled

| train ↓ / test → | gemma2:9b | llama3.1:8b | mistral:7b | qwen2.5:1.5b | qwen2.5:14b | qwen2.5:7b |
|---|---|---|---|---|---|---|
| gemma2:9b | 0.84±0.03 ◆ | 0.83±0.02 | 0.75±0.03 | 0.77±0.04 | 0.80±0.04 | 0.84±0.05 |
| llama3.1:8b | 0.78±0.04 | 0.80±0.03 ◆ | 0.76±0.05 | 0.79±0.04 | 0.78±0.04 | 0.79±0.03 |
| mistral:7b | 0.80±0.03 | 0.82±0.02 | 0.79±0.02 ◆ | 0.83±0.02 | 0.79±0.01 | 0.80±0.03 |
| qwen2.5:1.5b | 0.80±0.02 | 0.82±0.03 | 0.79±0.03 | 0.84±0.02 ◆ | 0.79±0.02 | 0.82±0.03 |
| qwen2.5:14b | 0.83±0.04 | 0.83±0.05 | 0.77±0.05 | 0.79±0.05 | 0.81±0.03 ◆ | 0.85±0.03 |
| qwen2.5:7b | 0.82±0.02 | 0.82±0.04 | 0.75±0.04 | 0.73±0.06 | 0.81±0.03 | 0.83±0.04 ◆ |

### asb

| train ↓ / test → | gemma2:9b | llama3.1:8b | mistral:7b | qwen2.5:1.5b | qwen2.5:14b | qwen2.5:7b |
|---|---|---|---|---|---|---|
| gemma2:9b | 0.89±0.04 ◆ | 0.87±0.06 | 0.87±0.01 | 0.79±0.11 | 0.89±0.06 | 0.90±0.05 |
| llama3.1:8b | 0.86±0.02 | 0.95±0.01 ◆ | 0.97±0.02 | 0.90±0.04 | 0.95±0.02 | 0.97±0.01 |
| mistral:7b | 0.89±0.01 | 0.94±0.02 | 0.97±0.01 ◆ | 0.95±0.00 | 0.94±0.01 | 0.98±0.01 |
| qwen2.5:1.5b | 0.92±0.04 | 0.92±0.04 | 0.96±0.02 | 0.93±0.04 ◆ | 0.90±0.04 | 0.94±0.04 |
| qwen2.5:14b | 0.89±0.01 | 0.89±0.03 | 0.87±0.04 | 0.84±0.08 | 0.92±0.04 ◆ | 0.91±0.01 |
| qwen2.5:7b | 0.92±0.03 | 0.93±0.03 | 0.96±0.02 | 0.85±0.09 | 0.93±0.04 | 0.96±0.02 ◆ |

### injecagent

| train ↓ / test → | gemma2:9b | llama3.1:8b | mistral:7b | qwen2.5:1.5b | qwen2.5:14b | qwen2.5:7b |
|---|---|---|---|---|---|---|
| gemma2:9b | 0.98±0.02 ◆ | 0.85±0.04 | 0.81±0.04 | 0.58±0.11 | 0.88±0.05 | 0.98±0.01 |
| llama3.1:8b | 0.90±0.04 | 0.88±0.01 ◆ | 0.90±0.07 | 0.86±0.16 | 0.89±0.05 | 0.95±0.01 |
| mistral:7b | 0.87±0.10 | 0.86±0.04 | 0.84±0.07 ◆ | 0.80±0.09 | 0.85±0.08 | 0.92±0.07 |
| qwen2.5:1.5b | 0.75±0.13 | 0.77±0.16 | 0.73±0.20 | 0.80±0.19 ◆ | 0.76±0.15 | 0.75±0.23 |
| qwen2.5:14b | 0.88±0.04 | 0.84±0.05 | 0.85±0.09 | 0.79±0.17 | 0.85±0.07 ◆ | 0.94±0.02 |
| qwen2.5:7b | 0.99±0.02 | 0.85±0.03 | 0.79±0.08 | 0.55±0.13 | 0.88±0.06 | 0.98±0.02 ◆ |

### agentdojo

| train ↓ / test → | gemma2:9b | llama3.1:8b | mistral:7b | qwen2.5:1.5b | qwen2.5:14b | qwen2.5:7b |
|---|---|---|---|---|---|---|
| gemma2:9b | 0.63±0.10 ◆ | 0.66±0.11 | 0.69±0.13 | 0.68±0.11 | 0.67±0.12 | 0.68±0.11 |
| llama3.1:8b | 0.62±0.07 | 0.65±0.12 ◆ | 0.68±0.13 | 0.64±0.14 | 0.66±0.12 | 0.67±0.11 |
| mistral:7b | 0.64±0.15 | 0.67±0.16 | 0.70±0.16 ◆ | 0.66±0.13 | 0.69±0.15 | 0.68±0.17 |
| qwen2.5:1.5b | 0.65±0.10 | 0.67±0.14 | 0.70±0.16 | 0.66±0.15 ◆ | 0.68±0.14 | 0.66±0.14 |
| qwen2.5:14b | 0.64±0.10 | 0.64±0.14 | 0.66±0.15 | 0.67±0.14 | 0.68±0.14 ◆ | 0.68±0.16 |
| qwen2.5:7b | 0.63±0.08 | 0.64±0.10 | 0.64±0.09 | 0.66±0.11 | 0.64±0.09 | 0.63±0.08 ◆ |

## Same matrices, real LLM actions only (appended synthetic attacker steps removed)

The synthetic attacker step is appended identically to every LLM's trace, so it can make transfer look easier than it is.

| train ↓ / test → | gemma2:9b | llama3.1:8b | mistral:7b | qwen2.5:1.5b | qwen2.5:14b | qwen2.5:7b |
|---|---|---|---|---|---|---|
| gemma2:9b | 0.84±0.03 ◆ | 0.84±0.02 | 0.74±0.02 | 0.76±0.03 | 0.80±0.04 | 0.85±0.04 |
| llama3.1:8b | 0.79±0.05 | 0.83±0.04 ◆ | 0.79±0.06 | 0.80±0.05 | 0.79±0.04 | 0.81±0.05 |
| mistral:7b | 0.81±0.03 | 0.85±0.02 | 0.80±0.03 ◆ | 0.84±0.03 | 0.79±0.01 | 0.82±0.04 |
| qwen2.5:1.5b | 0.81±0.02 | 0.85±0.03 | 0.80±0.03 | 0.85±0.02 ◆ | 0.79±0.01 | 0.84±0.02 |
| qwen2.5:14b | 0.83±0.04 | 0.85±0.04 | 0.77±0.05 | 0.79±0.06 | 0.81±0.04 ◆ | 0.87±0.04 |
| qwen2.5:7b | 0.82±0.02 | 0.83±0.04 | 0.74±0.04 | 0.73±0.06 | 0.80±0.03 | 0.83±0.03 ◆ |

## Is the GNN better than simpler guards across LLMs? (pooled, mean over all train/test pairs)

| model | in-LLM macro-F1 | cross-LLM macro-F1 |
|---|---|---|
| gnn | 0.82±0.02 | 0.80±0.02 |
| mlp | 0.63±0.05 | 0.62±0.04 |
| tool-lookup | 0.46±0.01 | 0.46±0.01 |
| majority | 0.39±0.00 | 0.39±0.00 |

## Attack success rate and reject recall (GNN, pooled)

ASR (no guard → guard): share of attack traces where an attacker action goes through; lower with the guard is better.

| train LLM | ASR in-LLM | ASR cross-LLM | reject recall in-LLM | reject recall cross-LLM | benign traces blocked (cross) |
|---|---|---|---|---|---|
| gemma2:9b | 0.73±0.08 → 0.19±0.09 | 0.76±0.07 → 0.18±0.06 | 0.74±0.08 | 0.77±0.07 | 0.18±0.05 |
| llama3.1:8b | 0.84±0.06 → 0.29±0.06 | 0.74±0.07 → 0.27±0.08 | 0.63±0.02 | 0.63±0.07 | 0.12±0.08 |
| mistral:7b | 0.75±0.08 → 0.26±0.06 | 0.75±0.07 → 0.25±0.01 | 0.63±0.03 | 0.67±0.01 | 0.11±0.05 |
| qwen2.5:1.5b | 0.71±0.06 → 0.19±0.01 | 0.76±0.07 → 0.22±0.04 | 0.75±0.02 | 0.68±0.04 | 0.13±0.08 |
| qwen2.5:14b | 0.81±0.07 → 0.22±0.05 | 0.74±0.07 → 0.20±0.05 | 0.71±0.06 | 0.72±0.06 | 0.12±0.06 |
| qwen2.5:7b | 0.68±0.09 → 0.17±0.09 | 0.77±0.07 → 0.20±0.07 | 0.74±0.11 | 0.74±0.08 | 0.20±0.12 |

## Per-benchmark in-LLM vs cross-LLM (GNN macro-F1)

| benchmark | train LLM | in-LLM | cross-LLM | gap | leave-one-out | verdict |
|---|---|---|---|---|---|---|
| asb | gemma2:9b | 0.89±0.04 | 0.86±0.05 | 0.03±0.02 | 0.91±0.04 | transfers |
| asb | llama3.1:8b | 0.95±0.01 | 0.93±0.01 | 0.02±0.01 | 0.91±0.03 | transfers |
| asb | mistral:7b | 0.97±0.01 | 0.94±0.00 | 0.03±0.01 | 0.94±0.04 | partial |
| asb | qwen2.5:1.5b | 0.93±0.04 | 0.93±0.03 | 0.01±0.01 | 0.89±0.08 | transfers |
| asb | qwen2.5:14b | 0.92±0.04 | 0.88±0.03 | 0.04±0.03 | 0.93±0.02 | partial |
| asb | qwen2.5:7b | 0.96±0.02 | 0.92±0.04 | 0.04±0.03 | 0.94±0.05 | partial |
| injecagent | gemma2:9b | 0.98±0.02 | 0.82±0.04 | 0.16±0.05 | 0.86±0.07 | weak |
| injecagent | llama3.1:8b | 0.88±0.01 | 0.90±0.05 | -0.02±0.03 | 0.79±0.04 | transfers |
| injecagent | mistral:7b | 0.84±0.07 | 0.86±0.05 | -0.02±0.06 | 0.76±0.21 | transfers |
| injecagent | qwen2.5:1.5b | 0.80±0.19 | 0.75±0.17 | 0.05±0.03 | 0.83±0.09 | partial |
| injecagent | qwen2.5:14b | 0.85±0.07 | 0.86±0.06 | -0.01±0.07 | 0.85±0.07 | transfers |
| injecagent | qwen2.5:7b | 0.98±0.02 | 0.81±0.06 | 0.17±0.06 | 0.91±0.01 | weak |
| agentdojo | gemma2:9b | 0.63±0.10 | 0.67±0.11 | -0.04±0.02 | 0.63±0.12 | transfers |
| agentdojo | llama3.1:8b | 0.65±0.12 | 0.65±0.11 | -0.01±0.04 | 0.66±0.10 | transfers |
| agentdojo | mistral:7b | 0.70±0.16 | 0.67±0.15 | 0.03±0.01 | 0.64±0.16 | transfers |
| agentdojo | qwen2.5:1.5b | 0.66±0.15 | 0.67±0.13 | -0.01±0.03 | 0.65±0.10 | transfers |
| agentdojo | qwen2.5:14b | 0.68±0.14 | 0.66±0.13 | 0.02±0.02 | 0.71±0.14 | transfers |
| agentdojo | qwen2.5:7b | 0.63±0.08 | 0.64±0.09 | -0.01±0.02 | 0.64±0.16 | transfers |

## What differs between the LLMs

Share of rejects in the test actions, and how each LLM behaved on the traces it produced (pooled). If an LLM's behaviour differs a lot (steps, how often it follows the attack), a transfer gap is expected.

| LLM | traces | avg steps | follows attack (no guard) | invalid-reply rate | reject share (all) | reject share (test) |
|---|---|---|---|---|---|---|
| gemma2:9b | 599 | 2.24 | 0.71 | 0.041 | 0.38 | 0.38 |
| llama3.1:8b | 600 | 2.77 | 0.80 | 0.014 | 0.33 | 0.35 |
| mistral:7b | 593 | 2.98 | 0.74 | 0.098 | 0.31 | 0.31 |
| qwen2.5:1.5b | 1084 | 3.22 | 0.68 | 0.017 | 0.30 | 0.32 |
| qwen2.5:14b | 600 | 2.54 | 0.78 | 0.007 | 0.35 | 0.36 |
| qwen2.5:7b | 600 | 2.02 | 0.66 | 0.007 | 0.38 | 0.37 |

## Leakage controls

Over all seeds. Overlap columns must be 0; min attack traces in train must be > 0.

| scope | training set | train traces (mean) | min attack traces in train | task groups shared with test | attack ids shared with test | trace ids shared with test |
|---|---|---|---|---|---|---|
| pooled | single:gemma2:9b | 325 | 152 | 0 | 0 | 0 |
| pooled | single:llama3.1:8b | 325 | 150 | 0 | 0 | 0 |
| pooled | single:mistral:7b | 325 | 155 | 0 | 0 | 0 |
| pooled | single:qwen2.5:1.5b | 325 | 161 | 0 | 0 | 0 |
| pooled | single:qwen2.5:14b | 325 | 151 | 0 | 0 | 0 |
| pooled | single:qwen2.5:7b | 325 | 152 | 0 | 0 | 0 |
| pooled | lolo:gemma2:9b | 323 | 154 | 0 | 0 | 0 |
| pooled | lolo:llama3.1:8b | 323 | 163 | 0 | 0 | 0 |
| pooled | lolo:mistral:7b | 323 | 160 | 0 | 0 | 0 |
| pooled | lolo:qwen2.5:1.5b | 323 | 151 | 0 | 0 | 0 |
| pooled | lolo:qwen2.5:14b | 323 | 147 | 0 | 0 | 0 |
| pooled | lolo:qwen2.5:7b | 323 | 150 | 0 | 0 | 0 |
| asb | single:gemma2:9b | 144 | 90 | 0 | 0 | 0 |
| asb | single:llama3.1:8b | 144 | 88 | 0 | 0 | 0 |
| asb | single:mistral:7b | 144 | 91 | 0 | 0 | 0 |
| asb | single:qwen2.5:1.5b | 144 | 88 | 0 | 0 | 0 |
| asb | single:qwen2.5:14b | 144 | 89 | 0 | 0 | 0 |
| asb | single:qwen2.5:7b | 144 | 88 | 0 | 0 | 0 |
| asb | lolo:gemma2:9b | 141 | 93 | 0 | 0 | 0 |
| asb | lolo:llama3.1:8b | 141 | 79 | 0 | 0 | 0 |
| asb | lolo:mistral:7b | 141 | 76 | 0 | 0 | 0 |
| asb | lolo:qwen2.5:1.5b | 141 | 84 | 0 | 0 | 0 |
| asb | lolo:qwen2.5:14b | 141 | 90 | 0 | 0 | 0 |
| asb | lolo:qwen2.5:7b | 141 | 87 | 0 | 0 | 0 |
| injecagent | single:gemma2:9b | 105 | 48 | 0 | 0 | 0 |
| injecagent | single:llama3.1:8b | 105 | 48 | 0 | 0 | 0 |
| injecagent | single:mistral:7b | 105 | 48 | 0 | 0 | 0 |
| injecagent | single:qwen2.5:1.5b | 105 | 54 | 0 | 0 | 0 |
| injecagent | single:qwen2.5:14b | 105 | 48 | 0 | 0 | 0 |
| injecagent | single:qwen2.5:7b | 105 | 48 | 0 | 0 | 0 |
| injecagent | lolo:gemma2:9b | 103 | 45 | 0 | 0 | 0 |
| injecagent | lolo:llama3.1:8b | 103 | 46 | 0 | 0 | 0 |
| injecagent | lolo:mistral:7b | 103 | 49 | 0 | 0 | 0 |
| injecagent | lolo:qwen2.5:1.5b | 103 | 47 | 0 | 0 | 0 |
| injecagent | lolo:qwen2.5:14b | 103 | 47 | 0 | 0 | 0 |
| injecagent | lolo:qwen2.5:7b | 103 | 44 | 0 | 0 | 0 |
| agentdojo | single:gemma2:9b | 67 | 8 | 0 | 0 | 0 |
| agentdojo | single:llama3.1:8b | 67 | 8 | 0 | 0 | 0 |
| agentdojo | single:mistral:7b | 67 | 8 | 0 | 0 | 0 |
| agentdojo | single:qwen2.5:1.5b | 67 | 6 | 0 | 0 | 0 |
| agentdojo | single:qwen2.5:14b | 67 | 8 | 0 | 0 | 0 |
| agentdojo | single:qwen2.5:7b | 67 | 8 | 0 | 0 | 0 |
| agentdojo | lolo:gemma2:9b | 65 | 11 | 0 | 0 | 0 |
| agentdojo | lolo:llama3.1:8b | 65 | 5 | 0 | 0 | 0 |
| agentdojo | lolo:mistral:7b | 65 | 8 | 0 | 0 | 0 |
| agentdojo | lolo:qwen2.5:1.5b | 65 | 10 | 0 | 0 | 0 |
| agentdojo | lolo:qwen2.5:14b | 65 | 5 | 0 | 0 | 0 |
| agentdojo | lolo:qwen2.5:7b | 65 | 5 | 0 | 0 | 0 |

- Same held-out task groups for every LLM: the test columns are identical scenarios.
- Training excludes every attack present in ANY LLM's test traces, so the training set does not depend on the target.
- Every source gets the same number of training traces; leave-one-out uses the same total, split equally over the other LLMs.
- The LLM name is not a graph feature (tests/test_crossllm.py).

## Reading the results

- Small gap and cross-LLM macro-F1 well above tool-lookup and majority: the guard learned the attack signal rather than one LLM's style.
- Large gap on a row: that LLM's traces look different (see behaviour table); a column with low scores for every row: that LLM is harder to guard.
- Leave-one-LLM-out near in-LLM: pooling several LLMs is enough to cover an unseen one.
- Only 3 seeds and one held-out split per seed: differences smaller than the ± are noise.

