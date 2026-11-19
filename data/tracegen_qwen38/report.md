# tracegen report

benchmark v1.2.2 · mode `parallel` · generator `qwen2.5:7b` · NLI `microsoft/deberta-large-mnli` (τ=0.8) · judge `off` · dedup False

Traces: 2568 (2568 after dedup).

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
| qwen3.8:27b | 60 (16/44) | 2 | 81.2% | 95.5% | -14.2% | 53.3% | 2.25 | 3.75 | 2.03 |

- **qwen3.8:27b** most common divergence from the ground-truth call sequence: call #1: expected get_most_recent_transactions, got get_scheduled_transactions (×5); call #2: expected send_money, got <stop> (×5); call #3: expected update_scheduled_transaction, got get_user_info (×3)
- **qwen3.8:27b** attack `important_instructions`: ASR 0.4% over 270 runs, utility under attack 89.3%

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
| qwen3.8:27b | 105 (21/84) | 7 | 85.7% | 86.9% | -1.2% | 50.5% | 2.95 | 5 | 5.80 |

- **qwen3.8:27b** most common divergence from the ground-truth call sequence: call #2: expected get_users_in_channel, got read_channel_messages (×12); call #1: expected read_channel_messages, got get_webpage (×6); call #1: expected get_channels, got get_webpage (×6)
- **qwen3.8:27b** attack `important_instructions`: ASR 0.0% over 253 runs, utility under attack 63.6%

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
| qwen3.8:27b | 81 (20/61) | 0 | 90.0% | 80.3% | 9.7% | 27.2% | 2.15 | 4.05 | 5.72 |

- **qwen3.8:27b** most common divergence from the ground-truth call sequence: call #2: expected get_car_fuel_options, got get_car_types_available (×7); call #4: expected get_rating_reviews_for_restaurants, got get_price_for_restaurants (×6); call #4: expected check_restaurant_opening_hours, got get_rating_reviews_for_restaurants (×6)
- **qwen3.8:27b** attack `important_instructions`: ASR 3.0% over 301 runs, utility under attack 82.7%

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
| qwen3.8:27b | 174 (40/134) | 3 | 95.0% | 83.6% | 11.4% | 53.4% | 2.30 | 4.35 | 2.77 |

- **qwen3.8:27b** most common divergence from the ground-truth call sequence: call #1: expected search_files, got search_files_by_filename (×15); call #1: expected search_calendar_events, got get_current_day (×6); call #3: expected append_to_file, got search_files_by_filename (×6)
- **qwen3.8:27b** attack `important_instructions`: ASR 0.0% over 1324 runs, utility under attack 86.0%

## Splits and leakage controls

Splits are assigned by (suite, original task_id), so all variants, seeds, agents and attack runs of a task go to one split.

| split | tasks | traces |
|---|---|---|
| test | 14 | 154 |
| train | 68 | 2241 |
| val | 15 | 173 |

- Task ids in more than one split: **0** (the build fails when this is > 0).
- Action signatures that occur in more than one split: 8 of 713. Different tasks can legitimately need the same calls; check these before using exact-sequence features.
- Injection tasks are shared across user tasks by AgentDojo's design. Injection tasks per split: {'train': ['injection_task_0', 'injection_task_10', 'injection_task_11', 'injection_task_12', 'injection_task_13', 'injection_task_2', 'injection_task_3', 'injection_task_4', 'injection_task_5', 'injection_task_6', 'injection_task_7', 'injection_task_8'], 'test': ['injection_task_1', 'injection_task_3', 'injection_task_4', 'injection_task_6'], 'val': ['injection_task_1', 'injection_task_2', 'injection_task_4', 'injection_task_7', 'injection_task_9']}
