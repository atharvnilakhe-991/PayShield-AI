# Phase 8 final model evaluation

## Status

This handover uses the best available preserved Phase 7 checkpoint, Fold 1, selected on its validation-family edited F1. The only completed held-out evaluations are Folds 1 and 2. Scores are synthetic-only, partial, and from different fold models. No verified real-world data was evaluated.

## Classification (original threshold 0.5)

| Scope | n | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fold 1 compact | 436 | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0 | 218 | 0 | 218 |
| Fold 2 enriched | 432 | 0.5093 | 0.5588 | 0.0880 | 0.1520 | 201 | 15 | 197 | 19 |
| Pooled partial, two models | 868 | 0.5046 | 0.5043 | 0.5461 | 0.5243 | 201 | 233 | 197 | 237 |

The pooled row is descriptive pooling of out-of-family predictions from two checkpoints. It is not a single model's performance. The validation-selected threshold sensitivity below does not replace the original fixed-threshold result.

## Validation-selected threshold sensitivity

Thresholds selected using validation-only were 0.6018 for Fold 1 (bank-progress validation, n=108) and 0.4453 for Fold 2 (recharge validation, n=24). On the same test sets, Fold 1 achieved accuracy 0.4977, precision 0.4988, recall 0.9312, F1 0.6496 (TN 14, FP 204, FN 15, TP 203); Fold 2 achieved accuracy 0.5023, precision 0.5032, recall 0.3611, F1 0.4205 (TN 139, FP 77, FN 138, TP 78). Pooled across the two different checkpoints, this operating point had accuracy 0.5000, precision 0.5000, recall 0.6475, F1 0.5643 (TN 153, FP 281, FN 153, TP 281). It trades more false positives and slightly lower accuracy for higher recall/F1. Fold 2's validation support is only 24. Scores and thresholds are not calibrated probabilities.

## Per-edit category (two available families only)

| Category | Edited positives | Clean controls | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Amount | 91 | 434 | 0.4743 | 0.1708 | 0.5275 | 0.2581 |
| Date/time | 91 | 434 | 0.4838 | 0.1853 | 0.5824 | 0.2812 |
| Other/account suffix | 84 | 434 | 0.4884 | 0.1825 | 0.6190 | 0.2818 |
| Recipient/merchant | 85 | 434 | 0.4624 | 0.1434 | 0.4588 | 0.2185 |
| Transaction ID/reference | 83 | 434 | 0.4758 | 0.1619 | 0.5422 | 0.2493 |

Clean controls are reused for category-wise comparisons; category supports are not independent.

## Localization

Pooled mask localization IoU was 0.059 on the 320 × 160 inference canvas (compact 0.051; enriched 0.069). Region highlights are disabled in the application because these masks generate too many false-positive pixels and do not support reliable localization.

## Incomplete evaluation

Bank-progress and recharge test families were not evaluated (Folds 3 and 4 were not started). The 387 real images remain unlabeled; they are neither clean controls nor a real-world test set. No fraud-detection performance claim is supported. See `reports/phase7/model_evaluation_report.md` and `reports/phase7/available_folds_1_2_metrics.json` for source measurements.
