# CBL-04: Member 1 Handover Checklist & Readiness Assessment

**Project:** CBL-04 – Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Auditor:** Senior Machine Learning Engineer and Dataset Auditor  
**Date:** November 3, 2026  
**Status:** AUDITED & VERIFIED  

---

## 1. Executive Summary

This checklist evaluates all deliverables produced by **Member 1 (Dataset Lead)** to determine whether **Member 2 (ML/Modeling Lead)** has everything required to begin model development, preprocessing, and training.

- **Core Dataset Deliverables (Items 1–10):** **10/10 Completed (100%)**
- **Full Project Lifecycle & Baseline Deliverables (Items 1–16):** **12 Completed, 2 Partial, 2 Missing / Pending**
- **Overall Handover Score:** **79.7%**
- **Verdict for Member 2:** **READY TO PROCEED WITH DATASET LOADER & PATCH-BASED REFACTORING.** Member 1 has provided an exceptionally well-structured, clean, and deterministically reproducible synthetic dataset with zero broken pairs and zero corrupt files. However, the existing ResNet-18 baseline model (~50.5% accuracy) must NOT be used as a production checkpoint; Member 2 must replace the whole-image downsampling approach with a patch/tile extraction pipeline.

---

## 2. Deliverables Checklist

| # | Deliverable Component | Target Artifact / Directory | Member 1 Status | Auditor Verification Status | Notes / Discrepancies |
|---|---|---|---|---|---|
| 1 | **Raw Real Screenshot Archive** | `data/real_gpay/raw/Dataset-Hackathon/` | COMPLETED | **VERIFIED (COMPLETED)** | 394 JPEG files, 100% readable. 387 unique contents (SHA-256), 7 duplicate copies preserved. Scope: dark-theme GPay screens (1 external PhonePe screen noted). |
| 2 | **Synthetic Bulk Dataset (Phase 4)** | `data/synthetic_gpay_phase4/` | COMPLETED | **VERIFIED (COMPLETED)** | 500 clean PNGs, 500 edited PNGs (1,000 total). 4 layout families (`compact`, `enriched`, `bank_progress`, `recharge`). |
| 3 | **Preliminary Pilots (Phases 1 & 3)** | `data/synthetic_gpay/`, `data/synthetic_gpay_phase3/` | COMPLETED | **VERIFIED (COMPLETED)** | Pilot 1: 20 pairs (40 images); Phase 3: 12 pairs (24 images). Kept strictly isolated in separate directories. |
| 4 | **Clean-Edited Pair Integrity** | Source ID matching across directories | COMPLETED | **VERIFIED (COMPLETED)** | 532/532 matched pairs verified. Zero broken pairs, zero orphaned clean or edited images. |
| 5 | **Binary Manipulation Masks** | `data/synthetic_gpay_phase4/masks/` | COMPLETED | **VERIFIED (COMPLETED)** | 1,064 masks (532 clean, 532 edited). Clean masks are 100% black (0); edited masks are 100% binary (0, 255) and align perfectly with image pixel difference. |
| 6 | **Bounding Box Annotations** | `annotations/phase4/bounding_boxes.csv` | COMPLETED | **VERIFIED (COMPLETED)** | 707 bbox rows for Phase 4 (754 total across all phases). 100% of non-zero mask pixels fall within bounding box regions. Format: `[x_min, y_min, x_max, y_max]`. |
| 7 | **Per-Pair Metadata JSONs** | `annotations/phase4/field_annotations/` | COMPLETED | **VERIFIED (COMPLETED)** | 500 JSON files for Phase 4 (532 total across all phases). Records source ID, edited field, original/replacement values, seed, and boxes. |
| 8 | **Dataset Manifests** | `metadata/phase4_manifest.csv`, `metadata/phase6_manifest.csv` | COMPLETED | **VERIFIED (COMPLETED)** | Phase 4 manifest has 1,000 rows; Phase 6 manifest has 1,387 rows (1,000 synthetic + 387 real holdout). All file paths verified valid. |
| 9 | **Leakage-Safe Split Manifests** | `metadata/phase6/` | COMPLETED | **VERIFIED (COMPLETED)** | Train (868 rows / 434 pairs), Val (108 rows / 54 pairs), Test (24 rows / 12 pairs). Zero source ID or pair leakage across splits. |
| 10 | **Dataset Generation Scripts** | `scripts/generator.py`, `scripts/phase4/generate_bulk.py` | COMPLETED | **VERIFIED (COMPLETED)** | Deterministic, seed-controlled Pillow renderers. Fictional entities, visible `SYNTHETIC / DEMO` markers. |
| 11 | **Dataset Validation Scripts** | `scripts/validate_dataset.py`, `scripts/phase5/audit_dataset.py` | COMPLETED | **VERIFIED (COMPLETED)** | Automated checks for decodability, dimensions, binary masks, box coverage, duplicate hashes, and pHash similarity. |
| 12 | **Dataset Documentation & Reports** | `README.md`, `reports/phase4/`, `reports/phase5/`, `reports/phase6/` | COMPLETED | **VERIFIED (COMPLETED)** | Thorough documentation of layout families, pHash clustering, split strategies, privacy audits, and duplicate inventories. |
| 13 | **Baseline Model Scripts & Config** | `scripts/phase7/`, `metadata/phase7/training_config.json` | COMPLETED | **VERIFIED (COMPLETED)** | ResNet-18 multi-task architecture (classification + segmentation). Training code preserved and runnable. |
| 14 | **Baseline Model Checkpoints & Evaluation** | `models/phase7/fold_1_resnet18_multitask.pt`, `fold_2` | PARTIALLY COMPLETED | **VERIFIED (PARTIAL)** | Folds 1 and 2 trained on CPU (3 epochs); Folds 3 and 4 were NOT run. Baseline accuracy is 50.46% (random chance). High false positive rate (Fold 1 predicted 100% edited on compact test). |
| 15 | **Real-World Ground Truth Labels** | Ground truth masks for `data/real_gpay/raw/` | NOT IMPLEMENTED | **VERIFIED (MISSING)** | Real screenshots are unlabeled and held out in `restricted_real_holdout`. No verified manipulation ground truth exists for real images. |
| 16 | **Localization Heatmap & Tile Pipeline** | Patch extraction & visual heatmap overlays | NOT IMPLEMENTED | **VERIFIED (MISSING)** | Member 1 used 320×160 whole-image downsampling, resulting in an audited localization IoU of only 5.93% (0.0593). Heatmaps disabled in local demo app. |

---

## 3. Quantitative Completion Scoring

### Scoring Matrix

$$\text{Completion Percentage} = \frac{\sum \text{Earned Points}}{\text{Total Possible Points}} \times 100$$

- **Full Credit (1.0 pt):** Verified fully completed according to specification.
- **Partial Credit (0.5 pt):** Substantially implemented but incomplete or non-viable for production.
- **Minimal Credit (0.25 pt):** Conceptual or initial implementation with severe limitations.
- **Zero Credit (0.0 pt):** Missing or not implemented.

### Point Breakdown

1. Raw Real Screenshot Archive: **1.0 / 1.0**
2. Synthetic Bulk Dataset (Phase 4): **1.0 / 1.0**
3. Preliminary Pilots (Phases 1 & 3): **1.0 / 1.0**
4. Clean-Edited Pair Integrity: **1.0 / 1.0**
5. Binary Manipulation Masks: **1.0 / 1.0**
6. Bounding Box Annotations: **1.0 / 1.0**
7. Per-Pair Metadata JSONs: **1.0 / 1.0**
8. Dataset Manifests: **1.0 / 1.0**
9. Leakage-Safe Split Manifests: **1.0 / 1.0**
10. Dataset Generation Scripts: **1.0 / 1.0**
11. Dataset Validation Scripts: **1.0 / 1.0**
12. Dataset Documentation & Reports: **1.0 / 1.0**
13. Baseline Model Scripts & Config: **1.0 / 1.0**
14. Baseline Model Checkpoints & Evaluation: **0.5 / 1.0** *(Only 2 of 4 folds trained; 50.5% accuracy)*
15. Real-World Ground Truth Labels: **0.0 / 1.0** *(Real images are completely unlabeled)*
16. Localization Heatmap & Tile Pipeline: **0.25 / 1.0** *(Decoder implemented but IoU 5.9%, tiling not implemented)*

$$\text{Total Points Earned} = 12.0 + 0.5 + 0.0 + 0.25 = 12.75 \text{ out of } 16.0$$
$$\mathbf{\text{Overall Completion Score: } 79.69\% \approx 79.7\%}$$

*(Note: Focusing strictly on Member 1's primary Dataset Lead scope [Items 1–12], completion is **12.0 / 12.0 = 100.0%**).*

---

## 4. Key Handover Strengths

1. **Zero Data Corruption:** All 394 real JPEG files and all 2,128 active synthetic PNG screenshots and masks decode without a single error.
2. **Deterministic Pair Consistency:** Every one of the 532 synthetic pairs has exact pixel-difference mask alignment, valid tight bounding boxes, and per-pair metadata JSONs.
3. **Leakage Protection:** Member 1 meticulously enforced pair integrity across splits. In both Phase 6 and Phase 7 fold manifests, clean and edited counterparts are strictly co-located in the same split.
4. **Comprehensive Reporting:** Complete forensic logs, duplicate SHA-256 analysis, perceptual hash clustering, and validation summaries were documented prior to handover.

---

## 5. Critical Handover Blockers & Dependencies for Member 2

1. **Baseline Accuracy Collapse (~50.5%):**
   - *Finding:* The ResNet-18 baseline checkpoint achieved 50.46% accuracy (Fold 1 predicted 100% of compact test images as edited; Fold 2 predicted 95.6% of enriched test images as clean).
   - *Impact:* The baseline model cannot be used as a functional detector.
   - *Root Cause:* Downsampling full screenshots (e.g. 540×1600) to 160×320 erases localized text tampering artifacts, and testing on held-out layout families induces fatal domain shifts.
2. **Missing Folds 3 and 4:**
   - *Finding:* Folds 3 (bank_progress test) and 4 (recharge test) were never trained due to time bounds.
   - *Impact:* No complete 4-fold baseline evaluation exists.
3. **Absence of Real-Image Ground Truth:**
   - *Finding:* All 387 unique real screenshots are unlabeled.
   - *Impact:* The model cannot be evaluated on genuine transaction tampering without manual annotation or privacy-approved evaluation protocols.
4. **Environment Inconsistency:**
   - *Finding:* The included `.venv_phase7` virtual environment was copied from another machine (`C:\Users\dell\...`) and cannot execute on the current system.
   - *Impact:* Member 2 must run in the existing Python 3.13 global environment (which already has PyTorch, OpenCV, Pillow, Scikit-Learn) or initialize a fresh virtual environment.

---

## 6. Handover Acceptance Sign-off

- **Dataset Lead (Member 1):** Completed dataset generation, masking, bounding boxes, manifests, and preliminary modeling.
- **ML & Modeling Lead (Member 2):** Handover deliverables accepted with verified dataset integrity. Member 2 will immediately initiate the patch/tile extraction pipeline and baseline redesign as outlined in `MEMBER2_NEXT_STEPS.md`.

