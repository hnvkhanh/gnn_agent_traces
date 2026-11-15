# tracegen report

benchmark v1.2.2 · mode `parallel` · generator `qwen2.5:7b` · NLI `microsoft/deberta-large-mnli` (τ=0.8) · judge `off` · dedup False

Traces: 840 (840 after dedup).

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
| llama3.1:8b | 60 (16/44) | 0 | 37.5% | 45.5% | -8.0% | 23.3% | 2.12 | 3.75 | 1.30 |
| qwen2.5:7b | 60 (16/44) | 0 | 50.0% | 56.8% | -6.8% | 20.0% | 2.25 | 3.75 | 1.55 |

- **llama3.1:8b** most common divergence from the ground-truth call sequence: call #1: expected get_most_recent_transactions, got schedule_transaction (×10); call #2: expected get_scheduled_transactions, got <stop> (×8); call #2: expected send_money, got <stop> (×7)
- **qwen2.5:7b** most common divergence from the ground-truth call sequence: call #1: expected get_most_recent_transactions, got <stop> (×10); call #2: expected get_scheduled_transactions, got <stop> (×8); call #2: expected <stop>, got get_most_recent_transactions (×6)

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
| llama3.1:8b | 105 (21/84) | 9 | 14.3% | 17.9% | -3.6% | 2.9% | 4.14 | 5 | 3.31 |
| qwen2.5:7b | 105 (21/84) | 4 | 42.9% | 31.0% | 11.9% | 6.7% | 4.48 | 5 | 6.10 |

- **llama3.1:8b** most common divergence from the ground-truth call sequence: call #1: expected read_channel_messages, got send_channel_message (×7); call #1: expected get_channels, got <stop> (×7); call #1: expected get_webpage, got invite_user_to_slack (×6)
- **qwen2.5:7b** most common divergence from the ground-truth call sequence: call #2: expected get_webpage, got invite_user_to_slack (×11); call #2: expected get_users_in_channel, got read_channel_messages (×10); call #1: expected get_channels, got read_channel_messages (×8)

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
| llama3.1:8b | 81 (20/61) | 6 | 5.0% | 0.0% | 5.0% | 2.5% | 2.95 | 4.05 | 3.11 |
| qwen2.5:7b | 81 (20/61) | 17 | 15.0% | 18.0% | -3.0% | 7.4% | 3.40 | 4.05 | 18.02 |

- **llama3.1:8b** most common divergence from the ground-truth call sequence: call #1: expected get_all_restaurants_in_city, got get_cuisine_type_for_restaurants (×14); call #1: expected get_all_restaurants_in_city, got get_rating_reviews_for_restaurants (×11); call #1: expected get_all_hotels_in_city, got get_hotels_prices (×9)
- **qwen2.5:7b** most common divergence from the ground-truth call sequence: call #2: expected get_cuisine_type_for_restaurants, got get_rating_reviews_for_restaurants (×15); call #2: expected get_car_fuel_options, got get_rating_reviews_for_car_rental (×7); call #1: expected get_all_car_rental_companies_in_city, got get_all_hotels_in_city (×6)

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
| llama3.1:8b | 174 (40/134) | 0 | 15.0% | 18.7% | -3.7% | 25.3% | 2.48 | 4.35 | 1.68 |
| qwen2.5:7b | 174 (40/134) | 0 | 27.5% | 28.4% | -0.9% | 28.7% | 3.02 | 4.35 | 2.18 |

- **llama3.1:8b** most common divergence from the ground-truth call sequence: call #1: expected get_day_calendar_events, got create_calendar_event (×12); call #2: expected <stop>, got search_emails (×11); call #1: expected search_emails, got create_calendar_event (×10)
- **qwen2.5:7b** most common divergence from the ground-truth call sequence: call #1: expected search_files_by_filename, got get_file_by_id (×14); call #1: expected get_day_calendar_events, got create_calendar_event (×8); call #1: expected list_files, got search_files (×8)

## Splits and leakage controls

Splits are assigned by (suite, original task_id), so all variants, seeds, agents and attack runs of a task go to one split.

| split | tasks | traces |
|---|---|---|
| test | 14 | 124 |
| train | 68 | 580 |
| val | 15 | 136 |

- Task ids in more than one split: **0** (the build fails when this is > 0).
- Action signatures that occur in more than one split: 4 of 528. Different tasks can legitimately need the same calls; check these before using exact-sequence features.
- Injection tasks are shared across user tasks by AgentDojo's design. Injection tasks per split: {'train': [], 'val': [], 'test': []}
