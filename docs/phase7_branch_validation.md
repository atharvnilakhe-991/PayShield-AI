# Phase 7: Branch Validation & Out-Of-Fold (OOF) Data Preparation

## 1. Overview & Objective

The objective of **Phase 7** is to audit, validate, and prepare reliable, leak-free branch outputs from:
1. **CNN Visual Manipulation Branch:** Multi-task ResNet-18 baseline checkpoints (`fold_1_resnet18_multitask.pt`, `fold_2_resnet18_multitask.pt`).
2. **OCR Extraction Branch:** EasyOCR 1.7.2 text tokens, normalized spatial boxes, and confidences.
3. **Semantic Consistency Branch:** Phase 6 deterministic rule engine outputs and anomaly indicators.

These outputs are structured to support the future **Phase 8 Stacking Classifier** without introducing data leakage, test-set contamination, or false assumptions about un-trained model folds.

---

## 2. Strict Out-Of-Fold (OOF) Definition

An out-of-fold prediction for sample $i$ is valid **if and only if**:
- The model generating the prediction was **not** trained on sample $i$.
- The model was **not** trained on the matched clean/edited counterpart (pair $P_i$).
- The model was **not** trained on duplicate or near-duplicate template variants of sample $i$.
- The model was **not** trained on the layout family of sample $i$ under whole-family holdout rotation.

### Anti-Leakage Constraints
- **Test-Set Untouchability:** Samples belonging to the held-out test manifest (`metadata/phase6/test_manifest.csv`) must **never** be used as training inputs for a future stacker.
- **No Validation Relabeling:** Samples used for validation early stopping must not be relabeled as training OOF data.
- **Truthful Incomplete Coverage:** Folds that were never trained (Fold 3 and Fold 4) must remain explicitly cataloged as `unavailable_missing_fold`. **No fake or zero-imputed predictions are fabricated.**

---

## 3. Fold & Checkpoint Training Provenance

| Checkpoint Name | File Hash (SHA-256) | Training Layout Families | Validation Family | Valid OOF Predictions Generated | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fold_1_resnet18_multitask.pt` | `44b654aa846fddfde7be6dab44ceada6687d8dfd35e2c83edbcb60fed909f62a` | `enriched`, `recharge` | `bank_progress` | **`compact` (436 samples)** | Available |
| `fold_2_resnet18_multitask.pt` | `4c5f97684a64553d117983c21b55a8ca874c855f9023b3e9732d1c70cb6403a4` | `bank_progress`, `compact` | `recharge` | **`enriched` (432 samples)** | Available |
| `fold_3_resnet18_multitask.pt` | *(Not found - never trained)* | `enriched`, `recharge` | `compact` | None (Unavailable) | Never Trained |
| `fold_4_resnet18_multitask.pt` | *(Not found - never trained)* | `bank_progress`, `compact` | `enriched` | None (Unavailable) | Never Trained |

---

## 4. Manifest Inventory

Artifacts generated in `Hackathon-VIT/CBL04-Dataset/metadata/phase7_branch_validation/`:

1. **`branch_output_manifest.csv`** (1,000 rows):
   - Complete synthetic dataset representation with CNN scores, OCR status, semantic manipulation scores, and split designations.
2. **`valid_oof_training_manifest.csv`** (868 rows):
   - Exactly the 868 Phase 6 training samples (`compact` evaluated by Fold 1 model, `enriched` evaluated by Fold 2 model) with provable OOF exclusion.
3. **`validation_manifest.csv`** (108 rows):
   - 54 pairs from the `bank_progress` layout family.
4. **`untouched_test_manifest.csv`** (24 rows):
   - 12 pairs from the `recharge` layout family. Strictly preserved for held-out evaluation.
5. **`missing_predictions_manifest.csv`** (132 rows):
   - 108 `bank_progress` + 24 `recharge` samples where CNN predictions cannot be formed without training Folds 3 and 4.

---

## 5. Independent Branch Metrics Summary

- **CNN Branch:** Evaluated on 868 OOF samples. Accuracy: 50.46%, Macro-F1: 0.4093, Precision: 0.5043, Recall: 0.5461. Raw uncalibrated probabilities.
- **OCR Branch:** Evaluated on 100 benchmark samples. Field coverage: 100% amount, 100% txn ID, 100% receiver, 100% status, 76% date. Mean confidence: 0.8808.
- **Semantic Branch:** Evaluated on 100 samples with strict test-set exclusion. Clean FP rate: 56.0% (due to Rupee symbol font transcription), Edited anomaly activation: 60.0%, Mean clean score: 0.099, Mean edited score: 0.107.
- **Ensemble Isolation:** Zero stacking models trained. Zero combined scores generated.

