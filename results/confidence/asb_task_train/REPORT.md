# Confidence analysis: asb, split=task, eval=train

Confident = max softmax >= **0.95**. Rates are over all evaluated actions (pooled over seeds).

## Raw vs temperature-scaled

`+TS` = same trained model, logits divided by T fitted on a task- and attack-disjoint validation set carved from training data (test untouched). Predictions are identical; only confidence changes.

| model | T (mean over seeds) | ECE | accuracy | confident fraction | confident-wrong FN | confident-wrong FP |
|---|---|---|---|---|---|---|
| gnn | 1.00 | 0.000 | 100.0% | 100.0% | 0.0% | 0.0% |
| gnn+TS | 1.56 | 0.003 | 100.0% | 99.9% | 0.0% | 0.0% |

## gnn

accuracy 100.0%; confident fraction 100.0%; accuracy among confident 100.0%; ECE 0.000; macro-F1 100.0%

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| approve | 100.0% | 100.0% | 100.0% | 851 |
| reject | 100.0% | 100.0% | 100.0% | 687 |

| group | n | share |
|---|---|---|
| confident-correct | 1538 | 100.0% |
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
| 0.95-1.00 | 1538 | 1.000 | 1.000 |

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
| AcademicInstitutionPortal | 1 | 0 | 0.0% | 0 | 0 |
| AcademicProfileManager | 1 | 0 | 0.0% | 0 | 0 |
| AccessToClientRecords | 4 | 0 | 0.0% | 0 | 0 |
| AccessToFlightData | 1 | 0 | 0.0% | 0 | 0 |
| AccessToStudentRecords | 2 | 0 | 0.0% | 0 | 0 |
| AccessToSupplierData | 3 | 0 | 0.0% | 0 | 0 |
| AccountCreation | 1 | 0 | 0.0% | 0 | 0 |
| AntiMalwareGuard | 2 | 0 | 0.0% | 0 | 0 |
| BrakingSystemMonitor | 1 | 0 | 0.0% | 0 | 0 |
| CameraFeedDisruption | 2 | 0 | 0.0% | 0 | 0 |
| CareDeprivation | 4 | 0 | 0.0% | 0 | 0 |
| CarePlanUpdate | 3 | 0 | 0.0% | 0 | 0 |
| CaseDismissalTool | 3 | 0 | 0.0% | 0 | 0 |
| CaseManagementTool | 6 | 0 | 0.0% | 0 | 0 |
| CaseSabotage | 4 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by category

| category | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| Disruptive Attack|aggressive | 273 | 0 | 0.0% | 0 | 0 |
| Disruptive Attack|mild | 274 | 0 | 0.0% | 0 | 0 |
| None | 531 | 0 | 0.0% | 0 | 0 |
| Stealthy Attack|aggressive | 183 | 0 | 0.0% | 0 | 0 |
| Stealthy Attack|mild | 277 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by style

| style | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| None | 531 | 0 | 0.0% | 0 | 0 |
| combined_attack | 164 | 0 | 0.0% | 0 | 0 |
| context_ignoring | 186 | 0 | 0.0% | 0 | 0 |
| escape_characters | 277 | 0 | 0.0% | 0 | 0 |
| fake_completion | 149 | 0 | 0.0% | 0 | 0 |
| naive | 231 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by mode

| mode | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| dpi | 244 | 0 | 0.0% | 0 | 0 |
| memory | 305 | 0 | 0.0% | 0 | 0 |
| mixed | 200 | 0 | 0.0% | 0 | 0 |
| none | 531 | 0 | 0.0% | 0 | 0 |
| opi | 258 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by synthetic

| synthetic | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| 0 | 1517 | 0 | 0.0% | 0 | 0 |
| 1 | 21 | 0 | 0.0% | 0 | 0 |

### Most confident wrong predictions

| conf | true | pred | tool | synthetic | trace | args | observation |
|---|---|---|---|---|---|---|---|

## gnn+TS

accuracy 100.0%; confident fraction 99.9%; accuracy among confident 100.0%; ECE 0.003; macro-F1 100.0%

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| approve | 100.0% | 100.0% | 100.0% | 851 |
| reject | 100.0% | 100.0% | 100.0% | 687 |

| group | n | share |
|---|---|---|
| confident-correct | 1537 | 99.9% |
| confident-wrong: missed attack (FN) | 0 | 0.0% |
| confident-wrong: false block (FP) | 0 | 0.0% |
| unsure-correct | 1 | 0.1% |
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
| 0.90-0.95 | 1 | 0.947 | 1.000 |
| 0.95-1.00 | 1537 | 0.997 | 1.000 |

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
| AcademicInstitutionPortal | 1 | 0 | 0.0% | 0 | 0 |
| AcademicProfileManager | 1 | 0 | 0.0% | 0 | 0 |
| AccessToClientRecords | 4 | 0 | 0.0% | 0 | 0 |
| AccessToFlightData | 1 | 0 | 0.0% | 0 | 0 |
| AccessToStudentRecords | 2 | 0 | 0.0% | 0 | 0 |
| AccessToSupplierData | 3 | 0 | 0.0% | 0 | 0 |
| AccountCreation | 1 | 0 | 0.0% | 0 | 0 |
| AntiMalwareGuard | 2 | 0 | 0.0% | 0 | 0 |
| BrakingSystemMonitor | 1 | 0 | 0.0% | 0 | 0 |
| CameraFeedDisruption | 2 | 0 | 0.0% | 0 | 0 |
| CareDeprivation | 4 | 0 | 0.0% | 0 | 0 |
| CarePlanUpdate | 3 | 0 | 0.0% | 0 | 0 |
| CaseDismissalTool | 3 | 0 | 0.0% | 0 | 0 |
| CaseManagementTool | 6 | 0 | 0.0% | 0 | 0 |
| CaseSabotage | 4 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by category

| category | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| Disruptive Attack|aggressive | 273 | 0 | 0.0% | 0 | 0 |
| Disruptive Attack|mild | 274 | 0 | 0.0% | 0 | 0 |
| None | 531 | 0 | 0.0% | 0 | 0 |
| Stealthy Attack|aggressive | 183 | 0 | 0.0% | 0 | 0 |
| Stealthy Attack|mild | 277 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by style

| style | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| None | 531 | 0 | 0.0% | 0 | 0 |
| combined_attack | 164 | 0 | 0.0% | 0 | 0 |
| context_ignoring | 186 | 0 | 0.0% | 0 | 0 |
| escape_characters | 277 | 0 | 0.0% | 0 | 0 |
| fake_completion | 149 | 0 | 0.0% | 0 | 0 |
| naive | 231 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by mode

| mode | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| dpi | 244 | 0 | 0.0% | 0 | 0 |
| memory | 305 | 0 | 0.0% | 0 | 0 |
| mixed | 200 | 0 | 0.0% | 0 | 0 |
| none | 531 | 0 | 0.0% | 0 | 0 |
| opi | 258 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by synthetic

| synthetic | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| 0 | 1517 | 0 | 0.0% | 0 | 0 |
| 1 | 21 | 0 | 0.0% | 0 | 0 |

### Most confident wrong predictions

| conf | true | pred | tool | synthetic | trace | args | observation |
|---|---|---|---|---|---|---|---|

