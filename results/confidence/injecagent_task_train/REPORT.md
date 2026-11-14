# Confidence analysis: injecagent, split=task, eval=train

Confident = max softmax >= **0.95**. Rates are over all evaluated actions (pooled over seeds).

## Raw vs temperature-scaled

`+TS` = same trained model, logits divided by T fitted on a task- and attack-disjoint validation set carved from training data (test untouched). Predictions are identical; only confidence changes.

| model | T (mean over seeds) | ECE | accuracy | confident fraction | confident-wrong FN | confident-wrong FP |
|---|---|---|---|---|---|---|
| gnn | 1.00 | 0.000 | 100.0% | 100.0% | 0.0% | 0.0% |
| gnn+TS | 4.86 | 0.026 | 100.0% | 84.4% | 0.0% | 0.0% |

## gnn

accuracy 100.0%; confident fraction 100.0%; accuracy among confident 100.0%; ECE 0.000; macro-F1 100.0%

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| approve | 100.0% | 100.0% | 100.0% | 1158 |
| reject | 100.0% | 100.0% | 100.0% | 173 |

| group | n | share |
|---|---|---|
| confident-correct | 1331 | 100.0% |
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
| 0.95-1.00 | 1331 | 1.000 | 1.000 |

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
| AmazonGetProductDetails | 58 | 0 | 0.0% | 0 | 0 |
| AmazonViewSavedPaymentMethods | 4 | 0 | 0.0% | 0 | 0 |
| AugustSmartLockUnlockDoor | 5 | 0 | 0.0% | 0 | 0 |
| BankManagerGetAccountInformation | 3 | 0 | 0.0% | 0 | 0 |
| BinanceDeposit | 2 | 0 | 0.0% | 0 | 0 |
| BinanceGetOrderHistory | 6 | 0 | 0.0% | 0 | 0 |
| BinanceWithdraw | 4 | 0 | 0.0% | 0 | 0 |
| CiscoUmbrellaCreatePolicy | 6 | 0 | 0.0% | 0 | 0 |
| CiscoUmbrellaGetLogDetails | 2 | 0 | 0.0% | 0 | 0 |
| CiscoUmbrellaUpdatePolicy | 5 | 0 | 0.0% | 0 | 0 |
| DeepfakeGeneratorGenerateAudioDeepfake | 3 | 0 | 0.0% | 0 | 0 |
| DropboxGetItemMetadata | 2 | 0 | 0.0% | 0 | 0 |
| DropboxListFilesAndFolders | 10 | 0 | 0.0% | 0 | 0 |
| DropboxMoveItem | 4 | 0 | 0.0% | 0 | 0 |
| EmergencyDispatchSystemCreateDispatchRequest | 6 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by category

| category | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| None | 779 | 0 | 0.0% | 0 | 0 |
| dh|Data Security Harm | 108 | 0 | 0.0% | 0 | 0 |
| dh|Financial Harm | 81 | 0 | 0.0% | 0 | 0 |
| dh|Physical Harm | 51 | 0 | 0.0% | 0 | 0 |
| ds|Financial Data | 74 | 0 | 0.0% | 0 | 0 |
| ds|Others | 161 | 0 | 0.0% | 0 | 0 |
| ds|Physical Data | 77 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by style

| style | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| None | 779 | 0 | 0.0% | 0 | 0 |
| base | 243 | 0 | 0.0% | 0 | 0 |
| enhanced | 309 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by mode

| mode | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| none | 779 | 0 | 0.0% | 0 | 0 |
| opi | 552 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by synthetic

| synthetic | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| 0 | 1249 | 0 | 0.0% | 0 | 0 |
| 1 | 82 | 0 | 0.0% | 0 | 0 |

### Most confident wrong predictions

| conf | true | pred | tool | synthetic | trace | args | observation |
|---|---|---|---|---|---|---|---|

## gnn+TS

accuracy 100.0%; confident fraction 84.4%; accuracy among confident 100.0%; ECE 0.026; macro-F1 100.0%

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| approve | 100.0% | 100.0% | 100.0% | 1158 |
| reject | 100.0% | 100.0% | 100.0% | 173 |

| group | n | share |
|---|---|---|
| confident-correct | 1123 | 84.4% |
| confident-wrong: missed attack (FN) | 0 | 0.0% |
| confident-wrong: false block (FP) | 0 | 0.0% |
| unsure-correct | 208 | 15.6% |
| unsure-wrong | 0 | 0.0% |

### Calibration (reliability bins)

| conf bin | n | mean conf | accuracy |
|---|---|---|---|
| 0.50-0.55 | 0 |  |  |
| 0.55-0.60 | 0 |  |  |
| 0.60-0.65 | 0 |  |  |
| 0.65-0.70 | 0 |  |  |
| 0.70-0.75 | 0 |  |  |
| 0.75-0.80 | 3 | 0.793 | 1.000 |
| 0.80-0.85 | 5 | 0.837 | 1.000 |
| 0.85-0.90 | 27 | 0.881 | 1.000 |
| 0.90-0.95 | 173 | 0.935 | 1.000 |
| 0.95-1.00 | 1123 | 0.983 | 1.000 |

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
| AmazonGetProductDetails | 58 | 0 | 0.0% | 0 | 0 |
| AmazonViewSavedPaymentMethods | 4 | 0 | 0.0% | 0 | 0 |
| AugustSmartLockUnlockDoor | 5 | 0 | 0.0% | 0 | 0 |
| BankManagerGetAccountInformation | 3 | 0 | 0.0% | 0 | 0 |
| BinanceDeposit | 2 | 0 | 0.0% | 0 | 0 |
| BinanceGetOrderHistory | 6 | 0 | 0.0% | 0 | 0 |
| BinanceWithdraw | 4 | 0 | 0.0% | 0 | 0 |
| CiscoUmbrellaCreatePolicy | 6 | 0 | 0.0% | 0 | 0 |
| CiscoUmbrellaGetLogDetails | 2 | 0 | 0.0% | 0 | 0 |
| CiscoUmbrellaUpdatePolicy | 5 | 0 | 0.0% | 0 | 0 |
| DeepfakeGeneratorGenerateAudioDeepfake | 3 | 0 | 0.0% | 0 | 0 |
| DropboxGetItemMetadata | 2 | 0 | 0.0% | 0 | 0 |
| DropboxListFilesAndFolders | 10 | 0 | 0.0% | 0 | 0 |
| DropboxMoveItem | 4 | 0 | 0.0% | 0 | 0 |
| EmergencyDispatchSystemCreateDispatchRequest | 6 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by category

| category | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| None | 779 | 0 | 0.0% | 0 | 0 |
| dh|Data Security Harm | 108 | 0 | 0.0% | 0 | 0 |
| dh|Financial Harm | 81 | 0 | 0.0% | 0 | 0 |
| dh|Physical Harm | 51 | 0 | 0.0% | 0 | 0 |
| ds|Financial Data | 74 | 0 | 0.0% | 0 | 0 |
| ds|Others | 161 | 0 | 0.0% | 0 | 0 |
| ds|Physical Data | 77 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by style

| style | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| None | 779 | 0 | 0.0% | 0 | 0 |
| base | 243 | 0 | 0.0% | 0 | 0 |
| enhanced | 309 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by mode

| mode | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| none | 779 | 0 | 0.0% | 0 | 0 |
| opi | 552 | 0 | 0.0% | 0 | 0 |

### Confident-wrong by synthetic

| synthetic | n | conf-wrong | rate | FN | FP |
|---|---|---|---|---|---|
| 0 | 1249 | 0 | 0.0% | 0 | 0 |
| 1 | 82 | 0 | 0.0% | 0 | 0 |

### Most confident wrong predictions

| conf | true | pred | tool | synthetic | trace | args | observation |
|---|---|---|---|---|---|---|---|

