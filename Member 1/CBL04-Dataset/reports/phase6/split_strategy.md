# Phase 6 split strategy

## Policy

The 500 labeled Phase 4 clean/edited pairs are split as indivisible groups. Every pair remains together, and group integrity is stricter at the layout-family level: all pairs in a family stay in one split. Within each family, 64-bit grayscale pHash connected components at Hamming distance ≤6 are also kept intact. The Phase 5 similarity audit found that compact, enriched and bank-progress each form one connected pHash component; recharge forms three. Keeping the whole family together therefore prevents all identified pHash and template-group leakage.

The four family groups cannot be divided among three splits while approaching 70/15/15. The deterministic assignment that best preserves the target sizes places compact and enriched in training, bank progress in validation, and recharge in testing. The realized proportions are documented separately, and the limitation is not hidden by breaking group integrity.

| Split | Layout groups | Pairs | Actual pair ratio | Requested |
|---|---|---:|---:|---:|
| Train | compact, enriched | 434 | 86.8% | 70% |
| Validation | bank_progress | 54 | 10.8% | 15% |
| Test | recharge | 12 | 2.4% | 15% |

## Seed and reproducibility

Seed: `20261206`. Family-to-split mapping is fixed and recorded in `metadata/phase6/phase6_config.json`; the seed controls deterministic row ordering in manifests. pHash is deterministic. The split builder refuses to overwrite the existing master manifest. To reproduce, run it in a fresh project copy with the same source manifests and files.

## Real data and prior pilots

Real images are not assigned clean/edited labels: they remain in `restricted_real_holdout`, outside supervised splits, pending verified ground truth and privacy review. One canonical path per SHA-256 is included; seven byte-identical copies remain in raw storage but are not repeated in the split manifest. The earlier 20-pair and 12-pair pilots remain separate and are excluded from these splits.

Split manifests reference existing files; no images, masks or annotations were copied or modified.
