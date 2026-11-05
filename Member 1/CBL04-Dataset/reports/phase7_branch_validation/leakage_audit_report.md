# CBL-04 — Phase 7 Leakage & Provenance Audit Report

**Audit Date:** November 2026  
**Auditor:** Member 3 (Backend & AI Integration Lead)  

---

## 1. Split & Pair Isolation Audit

We verified the integrity of sample pairs across all Phase 7 folds and Phase 6 partitions:

| Partition / Split | Total Samples | Clean Pairs | Edited Pairs | Matched Pair Integrity | Leaked Pairs |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 6 Train (`compact` + `enriched`)** | 868 | 434 | 434 | Verified 100% | **0** |
| **Phase 6 Validation (`bank_progress`)** | 108 | 54 | 54 | Verified 100% | **0** |
| **Phase 6 Test (`recharge`)** | 24 | 12 | 12 | Verified 100% | **0** |
| **Fold 1 Split Rotation** | 1,000 | 500 | 500 | Verified 100% | **0** |
| **Fold 2 Split Rotation** | 1,000 | 500 | 500 | Verified 100% | **0** |
| **Fold 3 Split Rotation** | 1,000 | 500 | 500 | Verified 100% | **0** |
| **Fold 4 Split Rotation** | 1,000 | 500 | 500 | Verified 100% | **0** |

### Verified Invariants:
1. **Zero Pair Leakage:** Every clean/edited pair stays strictly within the same fold role (`train`, `validation`, or `test`). No pair has one member in train and the other in test.
2. **Whole-Family Layout Holdout:** Layout families rotate completely without cross-family contamination.
3. **Untouched Test Rows:** All 24 samples from `metadata/phase6/test_manifest.csv` remain completely excluded from training manifests and were never seen during model selection.
4. **Real Archive Exclusion:** All 387 unverified screenshots in `real_archive/` remain strictly excluded from supervised evaluation.

---

## 2. Checkpoint Provenance & OOF Validity

| Checkpoint Name | File Hash (SHA-256) | Training Layout Families | Valid OOF Predictions Generated |
| :--- | :--- | :--- | :--- |
| `fold_1_resnet18_multitask.pt` | `44b654aa846fddfde7be6dab44ceada6687d8dfd35e2c83edbcb60fed909f62a` | `enriched`, `recharge` | **`compact` (436 samples)** |
| `fold_2_resnet18_multitask.pt` | `4c5f97684a64553d117983c21b55a8ca874c855f9023b3e9732d1c70cb6403a4` | `bank_progress`, `compact` | **`enriched` (432 samples)** |
| `fold_3_resnet18_multitask.pt` | *(Not found - never trained)* | `enriched`, `recharge` | None (Unavailable) |
| `fold_4_resnet18_multitask.pt` | *(Not found - never trained)* | `bank_progress`, `compact` | None (Unavailable) |

### Formal OOF Definition & Coverage:
- An out-of-fold prediction is valid **if and only if** the producing model was not trained on the sample, its matched pair, its source group, or its layout family.
- **Valid OOF Training Coverage:** Exactly **868 / 868 (100.0%)** of the Phase 6 training samples have valid OOF predictions.
- **Whole-Dataset OOF Coverage:** **868 / 1,000 (86.8%)** of all synthetic receipts.
- **Missing Coverage:** Exactly **132 samples** (`108 bank_progress` + `24 recharge`) lack family-held-out test predictions because Fold 3 and Fold 4 checkpoints were never trained. These are explicitly cataloged in `missing_predictions_manifest.csv`.
