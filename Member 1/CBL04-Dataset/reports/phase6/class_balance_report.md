# Phase 6 class-balance report

| Split | Clean samples | Edited samples | Clean:edited |
|---|---|---|---|
| train | 434 | 434 | 434/434 |
| validation | 54 | 54 | 54/54 |
| test | 12 | 12 | 12/12 |

Each synthetic source pair contributes one clean and one edited screenshot, so the paired class balance is exact within every split. No oversampling or duplicate balancing was applied.

## Synthetic layout and edit totals

| Layout family | Pairs |
|---|---|
| bank_progress | 54 |
| compact | 218 |
| enriched | 216 |
| recharge | 12 |

| Edit category | Edited pairs |
|---|---|
| amount | 100 |
| date_time | 100 |
| other | 100 |
| recipient | 100 |
| transaction_id | 100 |

Real versus synthetic: 387 unique real unlabeled screenshots are restricted holdout records; 1000 synthetic screenshots enter labeled train/validation/test manifests. Real samples are not counted as clean.
