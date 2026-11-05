# Phase 6 final split validation

**Result: PASS** — 0 validation errors.

- Phase 6 manifest: 1387 rows (1000 labeled synthetic screenshot records; 387 unique real unlabeled records).
- Synthetic pairs: 500; every pair is assigned wholly to one split.
- Split sizes: train 434 pairs (86.8%), validation 54 (10.8%), test 12 (2.4%).
- Family-template and pHash group leakage: 0.
- Missing image/mask/annotation references: 0.
- Real screenshots are excluded from supervised class labels and kept in a restricted holdout; no exact duplicate real copies are repeated in the manifest.
- Previous pilots remain separate. No screenshots were generated, copied, deleted, or modified; no models were trained.

Errors: none
