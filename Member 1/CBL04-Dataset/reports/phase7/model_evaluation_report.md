# Phase 7 model evaluation report

## Scope and headline result

This is a **partial, synthetic-only evaluation** of the two completed family-held-out folds: compact (Fold 1) and enriched (Fold 2). The 868 test screenshots are out-of-family relative to their respective fold's training and validation groups. The overall numbers pool predictions from two different checkpoints; they are not the score of a single final model. Bank-progress and recharge were not tested because Folds 3 and 4 were not trained.

| Support | Accuracy | Precision (edited) | Recall (edited) | F1 (edited) |
|---:|---:|---:|---:|---:|
| 868 | 0.5046 | 0.5043 | 0.5461 | 0.5243 |

Confusion matrix (rows: true; columns: predicted):

|  | Predicted clean | Predicted edited |
|---|---:|---:|
| True clean | 201 | 233 |
| True edited | 197 | 237 |

The classifier is near chance and not suitable for deployment or fraud screening.

## Per-family held-out results

| Held-out family | Test screenshots | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Compact | 436 | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0 | 218 | 0 | 218 |
| Enriched | 432 | 0.5093 | 0.5588 | 0.0880 | 0.1520 | 201 | 15 | 197 | 19 |
| Bank progress | Not tested | — | — | — | — | — | — | — | — |
| Recharge | Not tested | — | — | — | — | — | — | — | — |

## Per-edit-category results (available test families only)

For each category, edited examples of that field are positives and clean examples from the matching held-out families are negatives. The 434 clean negatives are reused across the category-specific comparisons, so category row supports must not be summed as independent samples.

| Edit category | Edited support | Clean negatives | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Amount | 91 | 434 | 0.4743 | 0.1708 | 0.5275 | 0.2581 |
| Date/time | 91 | 434 | 0.4838 | 0.1853 | 0.5824 | 0.2812 |
| Other: account suffix | 84 | 434 | 0.4884 | 0.1825 | 0.6190 | 0.2818 |
| Recipient/merchant | 85 | 434 | 0.4624 | 0.1434 | 0.4588 | 0.2185 |
| Transaction ID/reference | 83 | 434 | 0.4758 | 0.1619 | 0.5422 | 0.2493 |

## Localization

Localization was scored on the 320×160 model canvas after mask downsampling, not at original screenshot resolution. Pooled available-family pixel metrics: precision 0.0653, recall 0.3942, Dice 0.1120, IoU 0.0593. Compact IoU was 0.0510; enriched IoU was 0.0686. The decoder produces masks, but these results show weak localization.

## Plots and records

- Confusion matrix: `reports/phase7/plots/available_folds_1_2_confusion_matrix.png`.
- Per-family, per-edit, and localization plots: `reports/phase7/plots/available_folds_1_2_*.png`.
- Predictions: `reports/phase7/available_folds_1_2_predictions.csv`.
- Machine-readable metrics: `reports/phase7/available_folds_1_2_metrics.json`.

No real screenshots were scored, and these synthetic results do not establish real-world fraud-detection performance.
