# Phase 7 evaluation strategy

## Decision: family-held-out four-fold evaluation

The approved Phase 6 manifests remain unchanged. Their single 70/15/15 assignment places all compact and enriched pairs in train, bank-progress in validation, and recharge in test. It cannot measure every family in test.

The Phase 5 pHash audit used a 64-bit grayscale pHash and flagged similarities at Hamming distance ≤6. Within that threshold, compact, enriched, and bank-progress each form one connected group; recharge forms three. To prevent pair, perceptual-group, and template-family leakage, each whole family is held out in turn. Every pair is kept together. For each outer test family, validation uses a different whole family and the remaining two families are used for training.

| Fold | Train families | Validation family | Test family | Train pairs | Validation pairs | Test pairs |
|---|---|---|---|---:|---:|---:|
| 1 | enriched, recharge | bank_progress | compact | 228 | 54 | 218 |
| 2 | compact, bank_progress | recharge | enriched | 272 | 12 | 216 |
| 3 | enriched, recharge | compact | bank_progress | 228 | 218 | 54 |
| 4 | compact, bank_progress | enriched | recharge | 272 | 216 | 12 |

Each family is tested once, used for validation once, and used for training in the other two folds. Out-of-fold predictions across the four fold-specific models cover all 500 pairs / 1,000 labeled screenshots. Aggregate metrics must be described as pooled out-of-family performance from four different models, not the score of one model trained on all data.

## Metrics

For clean/edited classification, report accuracy, precision, recall, F1, confusion matrix, and false-positive/false-negative examples for each fold, each held-out family, and pooled out-of-fold predictions. Per-edit-category performance is computed on edited test screenshots, with clean test screenshots as negatives; always include category support counts. For mask localization, report pixel precision/recall, Dice, and IoU per fold and family, with clean masks included as empty-ground-truth cases and results also separated by edited category. Report uncertainty/support because some fold/category cells are small.

## Real data

The 387 real screenshots remain in the restricted unlabeled holdout. They have no verified clean/edited ground truth and are not used for supervised training, validation, threshold selection, or performance claims. Results on this dataset are synthetic-only and do not establish real-world fraud detection performance.

## Reproduction and constraints

Fold manifests are written under `metadata/phase7/`; seed `20261207` controls manifest ordering. The family rotation is fixed in `metadata/phase7/fold_config.json`. Training must load only the declared training manifests, choose checkpoints using the fold's validation family, and evaluate the test family once after model selection. No Phase 6 manifest or source data is changed.

## Limitation

This strategy preserves the strict groups but leaves the recharge test/validation folds at only 12 pairs and the bank-progress test fold at 54. It assesses cross-family transfer, not within-family deployment performance. A representative single-model 70/15/15 evaluation requires additional independently designed layout templates or verified data; dividing these current family groups would reintroduce leakage.
