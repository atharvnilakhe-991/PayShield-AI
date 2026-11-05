# Baseline versus improvement comparison

## Scope

The completed Phase 7 baseline is measured at the original 0.5 threshold on held-out compact and enriched families (Folds 1 and 2). The only Phase 8 comparison is validation-only per-fold threshold selection on those same existing checkpoints, with the selected thresholds then applied to the same untouched test families. It does not train a new model. The two families are evaluated by different fold checkpoints; pooled values are not one model's score.

## Classification comparison

| Evaluation | Samples | Accuracy | Precision | Recall | F1 | Confusion matrix (TN, FP, FN, TP) |
|---|---:|---:|---:|---:|---:|---|
| Phase 7 baseline, threshold 0.5 | 868 | 0.5046 | 0.5043 | 0.5461 | 0.5243 | (201, 233, 197, 237) |
| Validation-selected thresholds (two fold models) | 868 | 0.5000 | 0.5000 | 0.6475 | 0.5643 | (153, 281, 153, 281) |

The baseline numbers are the original, independently recorded Phase 7 held-out predictions. The per-fold thresholds were selected using validation only (Fold 1: 0.6018 on 108 bank-progress screenshots; Fold 2: 0.4453 on 24 recharge screenshots) and then applied to the same untouched held-out tests. The accuracy decrease and 48 additional false positives mean this is a sensitivity tradeoff, not an unqualified improvement. Fold 2's threshold is especially unstable given its small validation set. Full per-fold data is in `threshold_tuning_metrics.json`.

## Per-fold baseline

| Fold / held-out family | Threshold | Accuracy | Precision | Recall | F1 | Confusion matrix (TN, FP, FN, TP) |
|---|---:|---:|---:|---:|---:|---|
| Fold 1 / compact | 0.5000 | 0.5000 | 0.5000 | 1.0000 | 0.6667 | (0, 218, 0, 218) |
| Fold 1 / compact | 0.6018 (validation) | 0.4977 | 0.4988 | 0.9312 | 0.6496 | (14, 204, 15, 203) |
| Fold 2 / enriched | 0.5000 | 0.5093 | 0.5588 | 0.0880 | 0.1520 | (201, 15, 197, 19) |
| Fold 2 / enriched | 0.4453 (validation) | 0.5023 | 0.5032 | 0.3611 | 0.4205 | (139, 77, 138, 78) |

Do not infer that the Fold 1 F1 represents useful performance: it classified every compact screenshot as edited. Fold 3 and Fold 4 have no results.

## Interpretation

The model weights did not improve in Phase 8. Threshold selection shifted recall upward while increasing false positives and slightly reducing accuracy. It cannot add information or correct the underlying cross-family generalization failure. Raw scores remain uncalibrated. The experiment used no test labels to choose thresholds and did not modify the fixed Phase 7 checkpoint files.
