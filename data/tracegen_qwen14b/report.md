# tracegen report

benchmark v1.2.2 · mode `parallel` · generator `qwen2.5:7b` · NLI `microsoft/deberta-large-mnli` (τ=0.8) · judge `off` · dedup False

Traces: 2588 (2588 after dedup).

## banking

### Variant funnel

| originals | generated | entity pass | NLI pass | judge pass | verified | not yet verified | gen errors |
|---|---|---|---|---|---|---|---|
| 16 | 80 | 75 | 44 | – | 44 | 0 | 0 |

Rejection reasons: `nli:bwd` ×12, `nli:fwd` ×11, `nli:fwd+bwd` ×8, `entity:number` ×5

NLI min(fwd, bwd) histogram (bins of 0.1, from 0 to 1): `[12, 3, 4, 1, 0, 3, 4, 4, 10, 34]`

### Traces per agent

| agent | benign (orig/para) | errors | utility orig | utility para | drop | GT seq match | unique sigs/task | traces/task | avg steps |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:14b | 60 (16/44) | 0 | 50.0% | 70.5% | -20.5% | 28.3% | 2.94 | 3.75 | 2.07 |

- **qwen2.5:14b** most common divergence from the ground-truth call sequence: call #1: expected get_most_recent_transactions, got <stop> (×7); call #2: expected <stop>, got get_most_recent_transactions (×4); call #2: expected get_scheduled_transactions, got <stop> (×4)
- **qwen2.5:14b** attack `important_instructions`: ASR 58.3% over 276 runs, utility under attack 60.5%

## slack

### Variant funnel

| originals | generated | entity pass | NLI pass | judge pass | verified | not yet verified | gen errors |
|---|---|---|---|---|---|---|---|
| 21 | 105 | 100 | 84 | – | 84 | 0 | 0 |

Rejection reasons: `nli:bwd` ×7, `nli:fwd+bwd` ×6, `entity:quoted` ×5, `nli:fwd` ×3

NLI min(fwd, bwd) histogram (bins of 0.1, from 0 to 1): `[6, 0, 0, 1, 0, 3, 4, 2, 10, 74]`

### Traces per agent

| agent | benign (orig/para) | errors | utility orig | utility para | drop | GT seq match | unique sigs/task | traces/task | avg steps |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:14b | 105 (21/84) | 8 | 47.6% | 45.2% | 2.4% | 17.1% | 4.19 | 5 | 6.68 |

- **qwen2.5:14b** most common divergence from the ground-truth call sequence: call #2: expected get_users_in_channel, got read_channel_messages (×10); call #2: expected get_webpage, got send_direct_message (×7); call #1: expected read_channel_messages, got get_webpage (×6)
- **qwen2.5:14b** attack `important_instructions`: ASR 50.8% over 254 runs, utility under attack 37.0%

## travel

### Variant funnel

| originals | generated | entity pass | NLI pass | judge pass | verified | not yet verified | gen errors |
|---|---|---|---|---|---|---|---|
| 20 | 100 | 67 | 61 | – | 61 | 0 | 0 |

Rejection reasons: `entity:quoted` ×15, `entity:number` ×13, `nli:bwd` ×6, `entity:name` ×3, `entity:money` ×2, `entity:date` ×1

NLI min(fwd, bwd) histogram (bins of 0.1, from 0 to 1): `[0, 0, 0, 1, 1, 2, 1, 1, 2, 59]`

### Traces per agent

| agent | benign (orig/para) | errors | utility orig | utility para | drop | GT seq match | unique sigs/task | traces/task | avg steps |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:14b | 81 (20/61) | 11 | 30.0% | 45.9% | -15.9% | 7.4% | 3.60 | 4.05 | 7.65 |

- **qwen2.5:14b** most common divergence from the ground-truth call sequence: call #2: expected get_cuisine_type_for_restaurants, got get_rating_reviews_for_restaurants (×11); call #2: expected get_car_fuel_options, got get_car_types_available (×7); call #4: expected <stop>, got get_car_types_available (×6)
- **qwen2.5:14b** attack `important_instructions`: ASR 16.2% over 302 runs, utility under attack 36.8%

## workspace

### Variant funnel

| originals | generated | entity pass | NLI pass | judge pass | verified | not yet verified | gen errors |
|---|---|---|---|---|---|---|---|
| 40 | 199 | 172 | 134 | – | 134 | 0 | 0 |

Rejection reasons: `entity:date` ×26, `nli:fwd` ×23, `nli:bwd` ×11, `entity:time` ×10, `nli:fwd+bwd` ×4, `entity:number` ×2

NLI min(fwd, bwd) histogram (bins of 0.1, from 0 to 1): `[4, 5, 2, 5, 5, 4, 8, 5, 12, 122]`

### Traces per agent

| agent | benign (orig/para) | errors | utility orig | utility para | drop | GT seq match | unique sigs/task | traces/task | avg steps |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:14b | 174 (40/134) | 12 | 57.5% | 46.3% | 11.2% | 38.5% | 3.30 | 4.35 | 2.57 |

- **qwen2.5:14b** most common divergence from the ground-truth call sequence: call #1: expected search_files_by_filename, got get_file_by_id (×6); call #1: expected search_files, got search_files_by_filename (×6); call #2: expected <stop>, got get_day_calendar_events (×5)
- **qwen2.5:14b** attack `important_instructions`: ASR 2.4% over 1336 runs, utility under attack 37.2%

## Splits and leakage controls

Splits are assigned by (suite, original task_id), so all variants, seeds, agents and attack runs of a task go to one split.

| split | tasks | traces |
|---|---|---|
| test | 14 | 165 |
| train | 68 | 2250 |
| val | 15 | 173 |

- Task ids in more than one split: **0** (the build fails when this is > 0).
- Action signatures that occur in more than one split: 6 of 1341. Different tasks can legitimately need the same calls; check these before using exact-sequence features.
- Injection tasks are shared across user tasks by AgentDojo's design. Injection tasks per split: {'train': ['injection_task_0', 'injection_task_1', 'injection_task_10', 'injection_task_11', 'injection_task_12', 'injection_task_13', 'injection_task_2', 'injection_task_3', 'injection_task_4', 'injection_task_5', 'injection_task_6', 'injection_task_7', 'injection_task_8', 'injection_task_9'], 'test': ['injection_task_0', 'injection_task_1', 'injection_task_10', 'injection_task_11', 'injection_task_12', 'injection_task_13', 'injection_task_2', 'injection_task_3', 'injection_task_4', 'injection_task_5', 'injection_task_6', 'injection_task_7', 'injection_task_8'], 'val': ['injection_task_1', 'injection_task_2', 'injection_task_4', 'injection_task_7', 'injection_task_9']}
