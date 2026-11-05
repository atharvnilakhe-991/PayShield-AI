# CBL-04 — Phase 7: Branch Validation and OOF Data Preparation Report

**Project:** CBL-04 — Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Workspace:** `/Users/heemanshupjain/Downloads/Member 3`  
**ML Project:** `Hackathon-VIT/CBL04-Dataset`  
**Role:** Member 3 — Backend / AI Integration Lead  
**Status:** IMPLEMENTED & VERIFIED (Phase 7 Complete)  
**Date:** November 2026  

---

## 1. Executive Summary

Phase 7 has audited the dataset split structure, verified model training provenance, established rigorous Out-Of-Fold (OOF) exclusion rules, and generated standardized branch manifests for the future Phase 8 stacking classifier.

### Key Milestones Achieved
1. **Split & Pair Integrity Verified:** Proved **0 pair leaks** across all 4 folds and all Phase 6 partitions. Clean and edited counterparts stay in the same split.
2. **Whole-Family Holdout Isolation:** Layout families rotate completely without cross-family contamination.
3. **Strict Checkpoint Provenance Audit:** Authenticated SHA-256 hashes of baseline checkpoints (`fold_1_resnet18_multitask.pt`, `fold_2_resnet18_multitask.pt`) and Phase 9 candidates. Verified that **Fold 3 and Fold 4 checkpoints were never trained**.
4. **Truthful OOF Coverage:**
   - **Valid OOF Training Coverage:** Exactly **868 / 868 (100.0%)** of Phase 6 training samples (`compact` evaluated by Fold 1 model, `enriched` evaluated by Fold 2 model) have provable OOF exclusion.
   - **Missing Coverage:** Exactly **132 samples** (`108 bank_progress` + `24 recharge`) lack family-held-out test predictions because Fold 3 and Fold 4 checkpoints do not exist. In strict adherence to project guidelines, **no fake or zero-imputed predictions were fabricated**.
5. **Untouched Test Manifest:** All 24 samples from `metadata/phase6/test_manifest.csv` are strictly cataloged in `untouched_test_manifest.csv` and excluded from stacker training inputs.
6. **Real Archive Exclusion:** The 387 unverified screenshots in `real_archive/` remain strictly excluded from supervised evaluation.
7. **Branch Independence:** CNN, OCR, and Semantic branches were evaluated strictly in isolation. Zero ensemble stacking models were trained or executed.
8. **Regression & Test Suite:** Created dedicated test suite `tests/test_phase7_branch_validation.py` (9 tests passed). Full workspace test suite passes cleanly (**103 passed, 0 failed, 0 skipped** across all 6 test modules).

---

## 2. Checkpoint Provenance & Cryptographic Verification

| Checkpoint Name | File Path | File Size | SHA-256 Hash | Training Layout Families | Valid OOF Predictions Generated | Status |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: |
| `fold_1_resnet18_multitask.pt` | `models/phase7/fold_1_resnet18_multitask.pt` | 57,727,691 B | `44b654aa846fddfde7be6dab44ceada6687d8dfd35e2c83edbcb60fed909f62a` | `enriched`, `recharge` | **`compact` (436 samples)** | Verified Baseline |
| `fold_2_resnet18_multitask.pt` | `models/phase7/fold_2_resnet18_multitask.pt` | 57,727,691 B | `4c5f97684a64553d117983c21b55a8ca874c855f9023b3e9732d1c70cb6403a4` | `bank_progress`, `compact` | **`enriched` (432 samples)** | Verified Baseline |
| `EXP1_MACRO_F1_fold_1_best.pt` | `models/phase9_improved/EXP1_MACRO_F1_fold_1_best.pt` | 57,729,291 B | `18e932ea5bf3ece82e4f9495452154d10e9828e0ed1db03a037290e74ba29ef1` | `enriched`, `recharge` | `compact` (436 samples) | Candidate Exp 1 |
| `EXP1_MACRO_F1_fold_2_best.pt` | `models/phase9_improved/EXP1_MACRO_F1_fold_2_best.pt` | 57,729,291 B | `e338f07f7d80046a62ef6d25ed5b753b267f1adc67b04dc8ceb31f42795da751` | `bank_progress`, `compact` | `enriched` (432 samples) | Candidate Exp 1 |
| `EXP2_CLS_PRIORITY_fold_1_best.pt` | `models/phase9_improved/EXP2_CLS_PRIORITY_fold_1_best.pt` | 57,730,027 B | `b8bb46e6db984cff96038053520888613dcf78b181ac87c11edd9e67d537b0f0` | `enriched`, `recharge` | `compact` (436 samples) | Candidate Exp 2 |
| `EXP2_CLS_PRIORITY_fold_2_best.pt` | `models/phase9_improved/EXP2_CLS_PRIORITY_fold_2_best.pt` | 57,730,027 B | `3839c6aef7fc6f1af395c0e1b74e072e908f48914f7f0675b5caea01a71b28d3` | `bank_progress`, `compact` | `enriched` (432 samples) | Candidate Exp 2 |
| `fold_3_resnet18_multitask.pt` | `models/phase7/fold_3_resnet18_multitask.pt` | - | *(None)* | `enriched`, `recharge` | *(None - missing)* | Never Trained |
| `fold_4_resnet18_multitask.pt` | `models/phase7/fold_4_resnet18_multitask.pt` | - | *(None)* | `bank_progress`, `compact` | *(None - missing)* | Never Trained |

---

## 3. Strict Out-Of-Fold (OOF) Formulation

An out-of-fold prediction is valid **if and only if** the producing model was not trained on:
1. The sample itself ($x_i$).
2. Its clean/edited counterpart ($x_i'$ where $\text{pair}(x_i) = \text{pair}(x_i')$).
3. Any duplicate or near-duplicate template variant ($T(x_i)$).
4. The layout family of the sample ($F(x_i)$) under family-held-out protocol.

### Coverage Breakdown

$$\text{OOF Training Coverage} = \frac{868 \text{ valid samples}}{868 \text{ Phase 6 training samples}} = 100.0\%$$

$$\text{Whole Dataset Coverage} = \frac{868 \text{ valid samples}}{1,000 \text{ total synthetic samples}} = 86.8\%$$

$$\text{Unavailable Coverage} = \frac{132 \text{ missing samples}}{1,000 \text{ total synthetic samples}} = 13.2\%$$

---

## 4. Manifest Inventory

The following manifests were generated in `Hackathon-VIT/CBL04-Dataset/metadata/phase7_branch_validation/`:

1. **`branch_output_manifest.csv`** (1,000 rows):
   - Comprehensive manifest containing every synthetic image in the dataset with sample ID, pair ID, layout family, split, true label, CNN uncalibrated score, predicted class, checkpoint used, OCR status, and semantic status.
2. **`valid_oof_training_manifest.csv`** (868 rows):
   - The verified OOF training dataset for the future Phase 8 stacker. Contains only Phase 6 training samples with verified provenance.
3. **`validation_manifest.csv`** (108 rows):
   - Contains all 54 pairs of the `bank_progress` layout family.
4. **`untouched_test_manifest.csv`** (24 rows):
   - Contains all 12 pairs of the `recharge` layout family. Strictly preserved for held-out evaluation.
5. **`missing_predictions_manifest.csv`** (132 rows):
   - Explicitly catalogs the 108 `bank_progress` and 24 `recharge` images where CNN predictions are unavailable due to un-trained Folds 3 and 4.

---

## 5. Independent Branch Evaluation Metrics

### CNN Branch (Baseline ResNet-18 Multi-Task)
- **Evaluated Samples:** 868 images (434 clean, 434 edited).
- **Overall Accuracy:** **50.46%** (438 / 868 correct).
- **Precision:** 50.43% (237 / 470).
- **Recall:** 54.61% (237 / 434).
- **F1 Score:** 52.43%.
- **Macro-F1:** **40.93%**.
- **Confusion Matrix:**
  - True Negatives (TN): 201
  - False Positives (FP): 233
  - False Negatives (FN): 197
  - True Positives (TP): 237
- **Per-Fold Disparity:**
  - Fold 1 (`compact`, N=436): Accuracy 50.00%, Recall 100.0%, F1 66.67% (heavily biased toward edited).
  - Fold 2 (`enriched`, N=432): Accuracy 50.93%, Recall 8.80%, F1 15.20% (heavily biased toward clean).

### OCR Branch (EasyOCR 1.7.2)
- **Evaluated Samples:** 100 benchmark images (50 clean, 50 edited).
- **Mean Token Confidence:** 0.8808 (88.1%).
- **Mean Latency:** 1,288.72 ms on CPU.
- **Field Extraction Coverage:**
  - Amount: 100.0%
  - Transaction ID: 100.0%
  - Receiver: 100.0%
  - Status: 100.0%
  - Date: 76.0% (footer cropped/scrolled in 24% of receipts)

### Semantic Consistency Branch (Rule Engine v1.0.0-phase6)
- **Evaluated Samples:** 100 images (50 clean, 50 edited) with strict test-set exclusion.
- **Mean Latency:** 0.15 ms on CPU.
- **Clean False Positive Anomaly Rate:** 56.0% (primarily driven by `{` vs `₹` OCR font artifacts tripping Rule `R_AMT_004`).
- **Edited Anomaly Activation:** 60.0%.
- **Mean Clean Score:** 0.099.
- **Mean Edited Score:** 0.107.

### Strict Branch Independence
- **No Ensemble Fusion:** Branches were evaluated strictly in isolation. Zero stacking classifiers were implemented or trained.

---

## 6. Leakage Audit Summary

- **Train / Validation / Test Leakage:** 0 pairs leaked.
- **Test Row Preservation:** Verified 100% untouched.
- **Real Archive Screenshots:** 387 unverified screenshots remain excluded.
- **Missing Fold Transparency:** Explicitly reported 132 unavailable predictions. No synthetic imputation.

---
**Report Author:** Member 3 (Backend & AI Integration Lead)  
**Acceptance Gate:** Phase 7 Ready for Formal Review. Phase 8 Pending Authorization.

