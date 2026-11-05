# CBL-04: Complete Dataset Audit & Technical Implementation Progress Report

**Project:** CBL-04 – Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Auditor:** Senior Machine Learning Engineer and Dataset Auditor  
**Audit Date:** November 3, 2026  
**Workspace Inspected:** `c:\Users\athar\OneDrive\Desktop\Member 2\Hackathon-VIT\CBL04-Dataset\`  
**Target Audience:** Member 2 (ML / Modeling Lead), Hackathon Technical Committee  

---

## 1. Executive Summary & Headline Findings

This technical audit report provides an exhaustive, forensic examination of all dataset artifacts, generation scripts, metadata manifests, and baseline modeling outputs created by **Member 1 (Dataset Lead)** for project **CBL-04**.

### Key Verified Metrics
- **Total Workspace Image Files:** **5,494 image files** (PNG/JPEG) across the repository (excluding `.venv_phase7`).
  - **Active Dataset Image Files:** **2,522 image files** in `data/` (1,458 screenshots + 1,064 ground truth masks).
  - **Archival / Intermediate Backups:** **2,912 image files** in `backups/`.
  - **Audit Sheets & Plots:** **60 image files** in `reports/`.
- **Total Unique Source Transactions:** **919 unique source transactions** (532 synthetic source IDs + 387 unique real screenshot contents).
- **Verified Clean-Edited Synthetic Pairs:** **532 pairs** (1,064 screenshots: 532 clean, 532 edited), verified with **0 broken pairs, 0 orphaned files, and 0 corrupt images**.
- **Ground Truth Manipulation Masks:** **1,064 masks** (532 clean masks [100% black / 0], 532 edited masks [100% binary 0/255]).
- **Bounding Box Annotations:** **754 bounding box rows** across CSV files and **532 per-pair JSON metadata files**.
- **Canonical Dataset Manifest Records:** **1,000 records** in Phase 4 bulk manifest (`metadata/phase4_manifest.csv`), **1,387 records** in Phase 6 combined manifest (`metadata/phase6_manifest.csv` including 387 real holdouts).
- **Raw Real Archive:** **394 JPEG images** in `data/real_gpay/raw/Dataset-Hackathon/`; 387 unique contents (SHA-256), 7 duplicate copies preserved. Scope is dark Google Pay screens (with 1 external PhonePe screen noted). All real screenshots remain **unlabeled and held out**.
- **Member 1 Handover Completion Score:** **79.7%** across all 16 project lifecycle components (**100%** on core dataset generation and validation deliverables).
- **Baseline Model Status:** **50.46% Accuracy (~50.5%)** and **5.93% Localization IoU**. Folds 1 and 2 completed; Folds 3 and 4 were never run. The model collapsed to chance due to severe spatial downsampling ($160 \times 320$) and Leave-One-Family-Out domain shifts.
- **Handover Verdict:** **READY FOR MEMBER 2.** Member 1 has delivered a clean, complete, and reproducible synthetic dataset. Member 2 can proceed immediately with patch/tile extraction and CNN model overhaul.

---

## 2. Project Scope & Architecture

### 2.1 Project Boundaries
- **Platform:** Google Pay (UPI transaction interface).
- **Theme:** Dark-theme transaction screens exclusively (background hex approximately `#111111` to `#141414`).
- **Objective:** Detect and localize localized tampering/manipulation (e.g., altered transaction amount, recipient name, timestamp, UPI reference ID, bank account suffix).
- **Data Modality:** Synthetic paired screenshots (Clean vs. Controlled Edited) with pixel-level ground truth binary masks and tight bounding boxes, supplemented by a held-out raw real screenshot archive.

### 2.2 Workspace Directory Layout
The workspace is structured into functional modules:

```text
Hackathon-VIT/CBL04-Dataset/
├── data/
│   ├── real_gpay/raw/Dataset-Hackathon/       # 394 real raw JPEGs (387 unique SHA-256)
│   ├── synthetic_gpay/                        # Pilot 1: 20 clean, 20 edited, 40 masks
│   ├── synthetic_gpay_phase3/                 # Pilot 2 (Phase 3): 12 clean, 12 edited, 24 masks
│   └── synthetic_gpay_phase4/                 # Bulk Synthetic: 500 clean, 500 edited, 1000 masks
├── annotations/
│   ├── bounding_boxes/bounding_boxes.csv      # Pilot 1 boxes (30 rows)
│   ├── field_annotations/                    # Pilot 1 JSONs (20 files)
│   ├── phase3/                                # Phase 3 boxes (17 rows) & JSONs (12 files)
│   └── phase4/                                # Phase 4 boxes (707 rows) & JSONs (500 files)
├── metadata/
│   ├── manifest.csv                           # Pilot 1 manifest (40 rows)
│   ├── phase3_manifest.csv                    # Phase 3 manifest (24 rows)
│   ├── phase4_manifest.csv                    # Phase 4 bulk canonical manifest (1000 rows)
│   ├── phase4_batches/                        # Per-batch manifests (Batches 1-5, 200 rows each)
│   ├── phase6_manifest.csv                    # Phase 6 combined manifest (1387 rows)
│   ├── phase6/                                # Split manifests: train (868), val (108), test (24), real (387)
│   └── phase7/                                # 4-fold cross-validation manifests (4000 rows total)
├── models/
│   └── phase7/                                # Checkpoints: fold_1 and fold_2 ResNet-18 multi-task
├── app/                                       # Local browser demo (server.py, HTML/CSS/JS)
├── scripts/                                   # Generator, auditor, split, and baseline training scripts
├── reports/                                   # Markdown and JSON reports for Phases 1-8
└── backups/                                   # Intermediate execution runs and pre-rerun backups
```

---

## 3. Exact Dataset Counting & Inventory

Every file was counted by recursive filesystem traversal and binary header inspection, rather than reading documentation claims.

### 3.1 Comprehensive Image File Census

| Category | File Location | File Format | Image Type | Count |
|---|---|---|---|---:|
| **Real Archive** | `data/real_gpay/raw/Dataset-Hackathon/` | JPEG | Unlabeled Raw Screenshots | 394 |
| **Synthetic Pilot 1** | `data/synthetic_gpay/clean/` | PNG | Synthetic Clean Screenshots | 20 |
| | `data/synthetic_gpay/edited/` | PNG | Synthetic Edited Screenshots | 20 |
| | `data/synthetic_gpay/masks/clean/` | PNG | Ground Truth Clean Masks | 20 |
| | `data/synthetic_gpay/masks/edited/` | PNG | Ground Truth Edited Masks | 20 |
| **Synthetic Pilot 2 (Phase 3)** | `data/synthetic_gpay_phase3/clean/` | PNG | Synthetic Clean Screenshots | 12 |
| | `data/synthetic_gpay_phase3/edited/` | PNG | Synthetic Edited Screenshots | 12 |
| | `data/synthetic_gpay_phase3/masks/clean/` | PNG | Ground Truth Clean Masks | 12 |
| | `data/synthetic_gpay_phase3/masks/edited/` | PNG | Ground Truth Edited Masks | 12 |
| **Synthetic Bulk (Phase 4)** | `data/synthetic_gpay_phase4/clean/` | PNG | Synthetic Clean Screenshots | 500 |
| | `data/synthetic_gpay_phase4/edited/` | PNG | Synthetic Edited Screenshots | 500 |
| | `data/synthetic_gpay_phase4/masks/clean/` | PNG | Ground Truth Clean Masks | 500 |
| | `data/synthetic_gpay_phase4/masks/edited/` | PNG | Ground Truth Edited Masks | 500 |
| **Subtotal Active Dataset Images** | `data/` *(Screenshots + Masks)* | PNG/JPEG | Active Evaluation Dataset | **2,522** |
| **Intermediate Backups** | `backups/` | PNG/JPEG | Historical run snapshots | 2,912 |
| **Audit Reports & Plots** | `reports/` | PNG/JPEG | Contact sheets, clusters, plots | 60 |
| **Total Workspace Images** | Entire Workspace (excl. `.venv_phase7`) | PNG/JPEG | All images | **5,494** |

### 3.2 Granular Dataset Item Totals

1. **Total Actual Dataset Image Files (Screenshots + Masks):** **2,522**
2. **Total Screenshot Images (Screenshots Only):** **1,458** (532 clean + 532 edited + 394 real)
3. **Total Clean Screenshots:** **532** (20 in Pilot 1, 12 in Phase 3, 500 in Phase 4)
4. **Total Edited Screenshots:** **532** (20 in Pilot 1, 12 in Phase 3, 500 in Phase 4)
5. **Total Unique Source Screenshots / Transactions:** **919**
   - Synthetic unique `source_id`s: **532**
   - Real unique image contents (SHA-256): **387**
6. **Total Clean-Edited Pairs:** **532** (100% matched and verified)
7. **Total Manipulation Mask Files:** **1,064** (532 clean masks, 532 edited masks)
8. **Total Bounding-Box Annotations:** **754 bounding-box rows** in CSVs (30 in Pilot 1, 17 in Phase 3, 707 in Phase 4) across 532 per-pair JSON files.
9. **Total Manifest Records:**
   - Canonical Phase 4 Manifest: **1,000 records**
   - Canonical Phase 6 Manifest: **1,387 records**
   - Cumulative records across all manifest CSVs in `metadata/`: **12,874 records**
10. **Total Annotation Records:** **532 JSON field annotation files**

### 3.3 Anomaly & Integrity Checks

- **Images without corresponding masks:** **0** (All 532 clean and 532 edited images have corresponding masks).
- **Edited images without bounding boxes:** **0** (All 532 edited images have valid bounding box coordinates).
- **Clean images without valid source IDs:** **0** (All 532 clean images follow strict `SYN-*` or `P4-*` source ID schemas).
- **Missing or unmatched image pairs:** **0** (Every clean image has a strictly matching edited partner).
- **Corrupted or unreadable image files:** **0** (All 2,522 active dataset images and 394 real images successfully decode).
- **Duplicate images in synthetic data:** **0** (Zero hash collisions in synthetic clean or edited screenshots).
- **Duplicate images in real archive:** **7 redundant duplicate copies** (14 files involved across 7 duplicate groups; duplicates are retained in `raw/` and excluded from Phase 6 split manifests).
- **Unreferenced annotation files:** **0** (Every JSON annotation maps to an existing source ID in the manifest).
- **Missing or broken file paths in manifests:** **0** (All relative file paths in `manifest.csv`, `phase4_manifest.csv`, and `phase6_manifest.csv` resolve).

---

## 4. Image Quality, Mask, and Bounding Box Verification

### 4.1 Image Geometry and Dimensional Distribution

#### Real Archive (`data/real_gpay/raw/Dataset-Hackathon/`)
- **Format:** 100% JPEG (`.jpg`).
- **Color Mode:** 3-channel RGB (8-bit per channel).
- **File Sizes:** Min: 31,064 bytes; Max: 80,852 bytes; Mean: 49,990.6 bytes ($\approx 48.8\text{ KB}$).
- **Resolution Range:** Widths: $270\text{px}$ to $738\text{px}$; Heights: $1248\text{px}$ to $1603\text{px}$.
- **Top 5 Dimensions:**
  1. $637 \times 1600$ (94 files, 23.9%)
  2. $540 \times 1324$ (27 files, 6.9%)
  3. $624 \times 1600$ (25 files, 6.3%)
  4. $560 \times 1599$ (18 files, 4.6%)
  5. $584 \times 1599$ (16 files, 4.1%)

#### Synthetic Bulk Dataset (`data/synthetic_gpay_phase4/`)
- **Format:** 100% PNG (`.png`, lossless).
- **Color Mode:** 3-channel RGB (Screenshots); 1-channel Grayscale `L` (Masks).
- **File Sizes:**
  - Clean screenshots: Mean $\approx 48.2\text{ KB}$ (Min: 32.4 KB, Max: 71.8 KB).
  - Edited screenshots: Mean $\approx 49.6\text{ KB}$ (Min: 33.1 KB, Max: 73.2 KB).
  - Masks: Mean $\approx 1.8\text{ KB}$ (Highly compressed sparse binary PNGs).
- **Resolution Distribution by Layout Family:**
  - `compact` (218 pairs): $W \in [540, 732]$, $H \in [1319, 1600]$
  - `enriched` (216 pairs): $W \in [420, 738]$, $H \in [1280, 1601]$
  - `bank_progress` (54 pairs): $W \in [353, 478]$, $H \in [1280, 1600]$
  - `recharge` (12 pairs): $W \in [270, 471]$, $H \in [1248, 1603]$

### 4.2 Mask Verification & Pixel Encoding Conventions
We tested every mask against its corresponding screenshot:
1. **Polarity Convention:**
   - Pixel value `0` (Black) represents unmodified / background regions.
   - Pixel value `255` (White) represents manipulated / altered regions.
2. **Clean Mask Integrity:**
   - 100% of clean masks (532/532) have extrema `(0, 0)` — completely empty / all black.
3. **Edited Mask Integrity:**
   - 100% of edited masks (532/532) contain binary values `0` and `255`.
   - Zero intermediate anti-aliasing gray values exist in the masks (binary thresholded).
4. **Dimensional Consistency:**
   - In 100% of pairs (532/532), mask dimensions exactly match the associated screenshot dimensions (`Mask.size == Image.size`).
5. **Exact Pixel-Difference Alignment:**
   - For all 532 pairs, we calculated $\Delta = |I_{\text{clean}} - I_{\text{edited}}|$ and compared the non-zero bounding box of $\Delta$ with the non-zero bounding box of the ground truth mask.
   - **Result:** **0 mismatches across 532 pairs.** The masks align with the altered pixels.

### 4.3 Bounding Box Validity & Coverage
- **Coordinate Format:** `[x_min, y_min, x_max, y_max]` in absolute pixel coordinates.
- **Validity:** $0 \le x_{\min} < x_{\max} \le W$ and $0 \le y_{\min} < y_{\max} \le H$ holds for 100% of bounding boxes.
- **Manipulated Region Coverage:**
  - We verified whether every single pixel with value `255` in the edited mask falls within at least one bounding box.
  - **Result:** **100% coverage verified (0 uncovered pixels).** Bounding boxes tightly enclose the manipulated text regions.
- **Multiple Bounding Boxes per Image:**
  - In certain edits (e.g. `recipient`), two separate regions change (the top avatar/name text and the transaction card "To: [Name]" line). These cases contain 2 distinct bounding boxes in the CSV/JSON, both verified.

---

## 5. Clean and Edited Pair Verification

For all 532 pairs, we checked whether the edited screenshot was genuinely derived from the corresponding clean screenshot.

### 5.1 Pairing Verification Summary
- **Source ID Linkage:** Every pair shares an identical `source_id` prefix (`SYN-20261103-XXX` for Pilot 1, `SYN3-[family]-XXX` for Phase 3, and `P4-XXXX` for Phase 4).
- **Layout Invariance:** Except for the designated edited field, all background geometry, card outlines, status checkmarks, avatars, transaction icons, font sizes, and fictional non-edited account details remain identical between clean and edited counterparts.
- **Fictional Entity Protection:** Names (`Aarav Mehta`, `Isha Kulkarni`), bank names (`Demo Bank`, `Sample National Bank`), UPI IDs, and transaction IDs are fictional.
- **Watermarking / Marker Integrity:** Both clean and edited screenshots carry the top banner and footer `SYNTHETIC / DEMO` markers in identical positions.

### 5.2 Edit Category Distribution (Phase 4 Bulk — 500 Pairs)
Member 1 balanced the edit types across 5 distinct categories (100 pairs each):

| Edit Category | Target Pairs | Verified Pairs | Layout Families Represented | Manipulated Field Description |
|---|---:|---:|---|---|
| `amount` | 100 | 100 | All 4 families | Primary transaction amount text (e.g. ₹280 $\to$ ₹385) |
| `recipient` | 100 | 100 | All 4 families | Recipient contact name & transfer card line |
| `date_time` | 100 | 100 | All 4 families | Timestamp text (e.g. +2 hours, altered day/time) |
| `transaction_id`| 100 | 100 | All 4 families | UPI 12-digit reference ID string |
| `other` | 100 | 100 | compact, enriched, bank_progress | Fictional bank account suffix digits (••••1234) |

*(Note: Recharge layouts do not feature a bank account suffix, so `other` was allocated to the remaining 3 families).*

---

## 6. Dataset Generation Methodology

### 6.1 Rendering Engine & Tooling
- **Library:** Pure Python with `Pillow` (PIL / `ImageDraw`, `ImageFont`, `ImageChops`).
- **No Headless Browser:** Rendering did not use Selenium, Playwright, or web templates. All coordinates, text bounding boxes, and canvas operations were rendered natively in Pillow.
- **Canvas Resolution:** Seed-controlled dynamic canvases based on the real dataset empirical dimensions ($270\text{px}$ to $738\text{px}$ wide, $1248\text{px}$ to $1603\text{px}$ high).

### 6.2 Manipulation Techniques Implemented
- **Technique:** Localized text replacement using identical deterministic font renderers.
- **Procedure:**
  1. Generate random transaction parameters using seed $S_i$.
  2. Render clean canvas $I_{\text{clean}}$.
  3. Modify exactly one parameter (e.g. amount $+ \Delta$).
  4. Render edited canvas $I_{\text{edited}}$ with the exact same seed $S_i$ and layout parameters.
  5. Compute exact pixel difference $D = |I_{\text{clean}} - I_{\text{edited}}|$.
  6. Threshold $D$ into binary mask $M_{\text{edited}} = \mathbb{I}(D > 0) \times 255$.
  7. Compute tight bounding boxes around non-zero connected components of $M_{\text{edited}}$.

### 6.3 Augmentations & Resizing in Delivered Data
- **No lossy compression applied:** Delivered synthetic images are 100% lossless PNG.
- **No synthetic noise or blur applied:** Text edges are clean anti-aliased vector renderings.
- **Implemented vs. Planned:**
  - *Implemented:* Exact pixel-difference masks, tight bounding boxes, deterministic multi-family layouts, seed logging.
  - *Planned but Not Implemented:* Splicing from external images, mismatched font tampering, JPEG compression artifacts, camera recapture simulation.

---

## 7. Dataset Leakage & Quality Checks

### 7.1 Split Policy & Leakage Verification
Member 1 designed Phase 6 splits using a **Leave-One-Family-Out (LOFO)** policy:
- `train`: `compact` (218 pairs / 436 images) + `enriched` (216 pairs / 432 images) = 868 images.
- `validation`: `bank_progress` (54 pairs / 108 images) = 108 images.
- `test`: `recharge` (12 pairs / 24 images) = 24 images.
- `restricted_real_holdout`: 387 unique real images = 387 images.

#### Leakage Audit Findings
1. **Source ID Overlap:**
   - Overlap between `train` and `validation`: **0**
   - Overlap between `train` and `test`: **0**
   - Overlap between `validation` and `test`: **0**
   - Overlap between real holdout and synthetic splits: **0**
2. **Pair Separation Leakage:**
   - In 100% of cases, both clean and edited counterparts of any source ID are placed in the **same split**. Zero pairs were split across train and test.

### 7.2 Perceptual Hash (pHash) Clustering & Template Shortcuts
- **Finding:** In Phase 5, Member 1 computed 64-bit perceptual hashes (pHash) and identified that **38,654 of 96,740 same-family pairs had Hamming distance $\le 6$**.
- **Risk:** High intra-family visual similarity. Because the generator uses fixed template components, deep CNNs can easily learn template-specific background artifacts rather than tampering traces.

### 7.3 Synthetic-to-Real Domain Gap
- **Finding:** The real dataset consists of genuine Android/iOS device screenshots with varied system status bars, network icons, subtle JPEG compression artifacts, and device display scalings.
- **Discrepancy:** The synthetic dataset has stylized Pillow-drawn fonts, synthetic watermarks (`SYNTHETIC / DEMO`), and lossless PNG encoding.
- **Risk:** Models trained exclusively on synthetic PNGs will suffer out-of-domain degradation when deployed on real-world compressed JPEG screenshots.

---

## 8. Implementation Progress & Delivered vs. Planned Table

| Component | Planned | Implemented | Verified | Remaining Work |
|---|---|---|---|---|
| **Screenshot collection** | 500 real screenshots planned | 394 real screenshots acquired (387 unique SHA-256) | **COMPLETED** | Real screenshots lack ground truth tampering labels. Scope includes 1 PhonePe image requiring filtering. |
| **Screenshot generation** | Fictional dark GPay pair generator | 4 layout families (`compact`, `enriched`, `bank_progress`, `recharge`) | **COMPLETED** | Add realistic post-processing (JPEG artifacts, font mismatches, camera recapture). |
| **Clean screenshots** | Synthetic clean transaction screens | 532 clean screenshots (20 P1, 12 P3, 500 P4) | **COMPLETED** | Expand layout diversity to include multi-item bills, merchant QR receipts. |
| **Edited screenshots** | Controlled single-field tampering | 532 edited screenshots across 5 balanced edit types | **COMPLETED** | Add copy-move / splicing manipulation and dirty baseline misalignments. |
| **Manipulation masks** | Pixel-accurate binary masks | 1,064 masks (532 clean [black], 532 edited [binary 0/255]) | **COMPLETED** | Multi-class semantic masks (identifying which field was tampered). |
| **Bounding boxes** | Tight bounding box coordinates | 754 CSV rows & 532 per-pair JSON metadata files | **COMPLETED** | Normalized YOLO / COCO format annotations for object detection baselines. |
| **Manifest** | Comprehensive dataset manifests | Phase 4 (1,000 rows), Phase 6 (1,387 rows), Phase 7 (4,000 rows) | **COMPLETED** | Create alternative stratified split manifests (intra-family random splits). |
| **Dataset validation** | Automated audit test suite | Scripts in `scripts/validate_dataset.py`, `phase5/audit_dataset.py` | **COMPLETED** | Automated pipeline for validating user-uploaded real images. |
| **Dataset documentation** | Detailed markdown reports | Reports for Phases 1-8, inventory CSVs, presentation slides | **COMPLETED** | Onboarding manual and patch extraction developer documentation. |
| **Baseline model** | ResNet-18 multi-task 4-fold baseline | Folds 1 and 2 trained on CPU; Folds 3 and 4 not trained | **PARTIALLY COMPLETED** | Model collapsed to 50.46% accuracy. Must refactor to native-resolution patch/tile CNN. |

---

## 9. Baseline Model Performance Forensic Audit

### 9.1 Audited Metrics Summary
Member 1 trained a multi-task ResNet-18 on CPU for 3 epochs using $160 \times 320$ downsampled images.

| Fold | Held-Out Family | Test Samples | Accuracy | Precision | Recall | F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| **Fold 1** | Compact | 436 | 0.5000 | 0.5000 | 1.0000 | 0.6667 | Predicted 100% edited (218 FP, 0 TN) |
| **Fold 2** | Enriched | 432 | 0.5093 | 0.5588 | 0.0880 | 0.1520 | Predicted 95.6% clean (15 FP, 201 TN, 197 FN) |
| **Fold 3** | Bank Progress | 108 | — | — | — | — | **NOT RUN (Missing)** |
| **Fold 4** | Recharge | 24 | — | — | — | — | **NOT RUN (Missing)** |
| **Pooled Baseline** | Folds 1 & 2 Combined | **868** | **0.5046** | **0.5043** | **0.5461** | **0.5243** | **Equivalent to random coin-flip** |

### 9.2 Audited Localization Metrics
- **Pooled Pixel Precision:** 0.0653 (6.53%)
- **Pooled Pixel Recall:** 0.3942 (39.42%)
- **Pooled Dice Coefficient:** 0.1120 (11.20%)
- **Pooled Intersection over Union (IoU):** **0.0593 (5.93%)**
- **Conclusion:** Localization is ineffective at $160 \times 320$ resolution. Member 1 correctly disabled the heatmap overlay in `app/server.py` because the mask output was unusable.

---

## 10. Handover Readiness Assessment & Critical Issues

### 10.1 Readiness Score: 79.7%
- Core Dataset Deliverables: **100% Complete**
- Modeling & Localization Deliverables: **37.5% Complete**
- **Combined Handover Score:** **79.7%**

### 10.2 Five Most Important Issues Requiring Attention

1. **Catastrophic Baseline Accuracy (50.46%):**
   - The saved model is non-functional for fraud detection. Member 2 must not attempt to "fine-tune" this checkpoint; it must be rebuilt using native-resolution patch classification.
2. **Loss of Forensic Traces via Global Downsampling:**
   - Resizing high-resolution transaction screens ($540 \times 1600$) to $160 \times 320$ blurs small text edits out of existence. Native-resolution tile extraction ($256 \times 256$) is required.
3. **Cross-Family Generalization Failure:**
   - The Leave-One-Family-Out split creates an extreme out-of-distribution barrier. Member 2 must create a stratified intra-family split to benchmark in-distribution forensic detection alongside the cross-family benchmark.
4. **Unlabeled Real Archive & Domain Shift:**
   - The 387 real screenshots are 100% held out with zero ground truth tampering labels. Real-world performance claims cannot be substantiated without manual annotation or privacy-approved validation protocols.
5. **Incomplete Baseline Cross-Validation:**
   - Folds 3 and 4 were never trained. A complete 4-fold baseline must be executed once the patch pipeline is operational.

---

## 11. Location of All Generated Audit Deliverables

All audit reports and machine-readable data files have been generated and verified at:

1. **`reports/member1_dataset_audit/DATASET_AUDIT_REPORT.md`** (This document)
2. **`reports/member1_dataset_audit/DATASET_STATISTICS.csv`** (Image-level audit of all 1,458 dataset screenshots)
3. **`reports/member1_dataset_audit/DATASET_SUMMARY.json`** (Machine-readable inventory and validation metrics)
4. **`reports/member1_dataset_audit/MEMBER1_HANDOVER_CHECKLIST.md`** (Scored handover verification matrix)
5. **`reports/member1_dataset_audit/MEMBER2_NEXT_STEPS.md`** (Prioritized 9-step technical implementation roadmap)

*(Identical synchronized copies are also maintained inside `Hackathon-VIT/CBL04-Dataset/reports/member1_dataset_audit/` for repository consistency).*

