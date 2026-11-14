# Confidence analysis: agentdojo, split=task, eval=train

Confident = max softmax >= **0.95**. Rates are over all evaluated actions (pooled over seeds).

## Raw vs temperature-scaled

`+TS` = same trained model, logits divided by T fitted on a task- and attack-disjoint validation set carved from training data (test untouched). Predictions are identical; only confidence changes.

| model | T (mean over seeds) | ECE | accuracy | confident fraction | confident-wrong FN | confident-wrong FP |
|---|---|---|---|---|---|---|
| gnn | 1.00 | 0.000 | 100.0% | 100.0% | 0.0% | 0.0% |
| gnn+TS | 7.21 | 0.044 | 100.0% | 69.2% | 0.0% | 0.0% |

## gnn

accuracy 100.0%; confident fraction 100.0%; accuracy among confident 100.0%; ECE 0.000; macro-F1 100.0%

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| approve | 100.0% | 100.0% | 100.0% | 700 |
| reject | 100.0% | 100.0% | 100.0% | 23 |

| group | n | share |
|---|---|---|
| confident-correct | 723 | 100.0% |
| confident-wrong: missed attack (FN) | 0 | 0.0% |
| confident-wrong: false block (FP) | 0 | 0.0% |
| unsure-correct | 0 | 0.0% |
| unsure-wrong | 0 | 0.0% |

### Calibration (reliability bins)

| conf bin | n | mean conf | accuracy |
|---|---|---|---|
| 0.50-0.55 | 0 |  |  |
| 0.55-0.60 | 0 |  |  |
| 0.60-0.65 | 0 |  |  |
| 0.65-0.70 | 0 |  |  |
| 0.70-0.75 | 0 |  |  |
| 0.75-0.80 | 0 |  |  |
| 0.80-0.85 | 0 |  |  |
| 0.85-0.90 | 0 |  |  |
| 0.90-0.95 | 0 |  |  |
| 0.95-1.00 | 723 | 1.000 | 1.000 |

### Selective prediction (keep most confident k%)

| coverage | accuracy |
|---|---|
| 10% | 100.0% |
| 20% | 100.0% |
| 30% | 100.0% |
| 40% | 100.0% |
| 50% | 100.0% |
| 60% | 100.0% |
| 70% | 100.0% |
| 80% | 100.0% |
| 90% | 100.0% |
| 100% | 100.0% |

### Confident-wrong by tool

| tool | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| add_calendar_event_participants | 6 | 0 | 0.0% | 0 | 0 |
| add_user_to_channel | 17 | 0 | 0.0% | 0 | 0 |
| append_to_file | 15 | 0 | 0.0% | 0 | 0 |
| cancel_calendar_event | 13 | 0 | 0.0% | 0 | 0 |
| check_restaurant_opening_hours | 6 | 0 | 0.0% | 0 | 0 |
| create_calendar_event | 15 | 0 | 0.0% | 0 | 0 |
| create_file | 13 | 0 | 0.0% | 0 | 0 |
| delete_email | 4 | 0 | 0.0% | 0 | 0 |
| delete_file | 4 | 0 | 0.0% | 0 | 0 |
| get_all_car_rental_companies_in_city | 20 | 0 | 0.0% | 0 | 0 |
| get_all_hotels_in_city | 27 | 0 | 0.0% | 0 | 0 |
| get_all_restaurants_in_city | 18 | 0 | 0.0% | 0 | 0 |
| get_balance | 10 | 0 | 0.0% | 0 | 0 |
| get_car_fuel_options | 2 | 0 | 0.0% | 0 | 0 |
| get_car_price_per_day | 3 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by category

| category | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| None | 653 | 0 | 0.0% | 0 | 0 |
| banking | 10 | 0 | 0.0% | 0 | 0 |
| travel | 60 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by style

| style | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| None | 653 | 0 | 0.0% | 0 | 0 |
| direct | 19 | 0 | 0.0% | 0 | 0 |
| ignore_previous | 29 | 0 | 0.0% | 0 | 0 |
| important_instructions | 22 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by mode

| mode | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| none | 653 | 0 | 0.0% | 0 | 0 |
| opi | 70 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by synthetic

| synthetic | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| 0 | 712 | 0 | 0.0% | 0 | 0 |
| 1 | 11 | 0 | 0.0% | 0 | 0 |

### Most confident wrong predictions

| conf | true | pred | tool | synthetic | trace | args | observation |
|---|---|---|---|---|---|---|---|

## gnn+TS

accuracy 100.0%; confident fraction 69.2%; accuracy among confident 100.0%; ECE 0.044; macro-F1 100.0%

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| approve | 100.0% | 100.0% | 100.0% | 700 |
| reject | 100.0% | 100.0% | 100.0% | 23 |

| group | n | share |
|---|---|---|
| confident-correct | 500 | 69.2% |
| confident-wrong: missed attack (FN) | 0 | 0.0% |
| confident-wrong: false block (FP) | 0 | 0.0% |
| unsure-correct | 223 | 30.8% |
| unsure-wrong | 0 | 0.0% |

### Calibration (reliability bins)

| conf bin | n | mean conf | accuracy |
|---|---|---|---|
| 0.50-0.55 | 0 |  |  |
| 0.55-0.60 | 0 |  |  |
| 0.60-0.65 | 0 |  |  |
| 0.65-0.70 | 0 |  |  |
| 0.70-0.75 | 1 | 0.724 | 1.000 |
| 0.75-0.80 | 6 | 0.785 | 1.000 |
| 0.80-0.85 | 25 | 0.825 | 1.000 |
| 0.85-0.90 | 41 | 0.881 | 1.000 |
| 0.90-0.95 | 150 | 0.930 | 1.000 |
| 0.95-1.00 | 500 | 0.980 | 1.000 |

### Selective prediction (keep most confident k%)

| coverage | accuracy |
|---|---|
| 10% | 100.0% |
| 20% | 100.0% |
| 30% | 100.0% |
| 40% | 100.0% |
| 50% | 100.0% |
| 60% | 100.0% |
| 70% | 100.0% |
| 80% | 100.0% |
| 90% | 100.0% |
| 100% | 100.0% |

### Confident-wrong by tool

| tool | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| add_calendar_event_participants | 6 | 0 | 0.0% | 0 | 0 |
| add_user_to_channel | 17 | 0 | 0.0% | 0 | 0 |
| append_to_file | 15 | 0 | 0.0% | 0 | 0 |
| cancel_calendar_event | 13 | 0 | 0.0% | 0 | 0 |
| check_restaurant_opening_hours | 6 | 0 | 0.0% | 0 | 0 |
| create_calendar_event | 15 | 0 | 0.0% | 0 | 0 |
| create_file | 13 | 0 | 0.0% | 0 | 0 |
| delete_email | 4 | 0 | 0.0% | 0 | 0 |
| delete_file | 4 | 0 | 0.0% | 0 | 0 |
| get_all_car_rental_companies_in_city | 20 | 0 | 0.0% | 0 | 0 |
| get_all_hotels_in_city | 27 | 0 | 0.0% | 0 | 0 |
| get_all_restaurants_in_city | 18 | 0 | 0.0% | 0 | 0 |
| get_balance | 10 | 0 | 0.0% | 0 | 0 |
| get_car_fuel_options | 2 | 0 | 0.0% | 0 | 0 |
| get_car_price_per_day | 3 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by category

| category | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| None | 653 | 0 | 0.0% | 0 | 0 |
| banking | 10 | 0 | 0.0% | 0 | 0 |
| travel | 60 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by style

| style | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| None | 653 | 0 | 0.0% | 0 | 0 |
| direct | 19 | 0 | 0.0% | 0 | 0 |
| ignore_previous | 29 | 0 | 0.0% | 0 | 0 |
| important_instructions | 22 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by mode

| mode | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| none | 653 | 0 | 0.0% | 0 | 0 |
| opi | 70 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by synthetic

| synthetic | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| 0 | 712 | 0 | 0.0% | 0 | 0 |
| 1 | 11 | 0 | 0.0% | 0 | 0 |

### Most confident wrong predictions

| conf | true | pred | tool | synthetic | trace | args | observation |
|---|---|---|---|---|---|---|---|

