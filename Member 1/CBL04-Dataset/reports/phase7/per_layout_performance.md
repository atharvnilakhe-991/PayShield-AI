# Phase 7 per-layout performance

Only completed test folds are reported as measured. Each test family is fully held out by pair, Phase 5 pHash group (Hamming ≤6), and layout-family template.

| Family | Test status | Test images | Accuracy | Precision | Recall | F1 | Localization IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| Compact transaction details | Tested, Fold 1 | 436 | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0.0510 |
| Merchant/context-enriched details | Tested, Fold 2 | 432 | 0.5093 | 0.5588 | 0.0880 | 0.1520 | 0.0686 |
| Bank payment-progress timeline | Not tested; Fold 3 not started | — | — | — | — | — | — |
| Recharge/utility detail summary | Not tested; Fold 4 not started | — | — | — | — | — | — |

Compact's apparently higher F1 is degenerate: all 218 clean compact screenshots were predicted edited, as were all 218 edited screenshots. Enriched performance is also weak, with 197 of 216 edited examples missed. Neither measured family result is sufficient for practical use.

Validation families were bank-progress for Fold 1 and recharge for Fold 2. Validation performance is recorded in `partial_training_history.json`; validation scores are not test results. Cross-family results are synthetic-only.
