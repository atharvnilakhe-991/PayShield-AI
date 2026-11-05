# CBL-04: Mask-to-Tile Labeling Validation & Audit Report
**Phase 4: Mask-to-Tile Labeling**  
**Role:** Member 2 – Tile Forensics CNN & Heatmap Lead  
**Timestamp:** 2026-11-03  
**Status:** PASSED  
**Readiness:** READY_FOR_PHASE_5  

---

## 1. Executive Summary
Phase 4 implements an exact, ground-truth forensic labeling engine that assigns reliable labels to 256x256 tiles using full-resolution pixel tampering masks.

### Labeling Policy & Thresholds
- **Clean (label 0):** Foreground pixel count == 0.
- **Tampered (label 1):** Foreground pixel count >= 16 pixels.
- **Ambiguous (label -1):** Foreground pixel count between 1 and 15 pixels. Ambiguous boundary tiles are explicitly tagged with label -1 and never silently misclassified.
- **Unlabelled (label None):** Real holdout screenshots (`mask=None`) receive label None and status "unlabelled".

### Key Audit Findings
1. **Zero False Positives in Clean Screenshots:** In 100% of the 500 clean screenshots (21,755 clean tiles), zero foreground pixels were detected. Clean screenshots yield 100.0% label 0.
2. **100% Positive Coverage in Edited Screenshots:** Every single one of the 500 edited screenshots contains at least one positive (tampered) tile (0 edited screenshots with zero positive tiles).
3. **Severe Class Imbalance Identified:** Clean tiles outnumber tampered tiles by **11.69 : 1** (92.07% clean vs. 7.88% tampered).
4. **Negligible Ambiguous Fraction:** Only **24 tiles** (0.06%) contain 1–15 foreground pixels.
5. **Zero Data Leakage:** Clean and edited tiles from the same transaction pair share identical `split_group_id`.

---

## 2. Dataset Tile Labeling Census

| Category | Tile Count | Percentage of Labeled | Description |
| :--- | :---: | :---: | :--- |
| **Clean Tiles (label 0)** | **40,058** | **92.07%** | Zero tampered pixels (sum(mask) = 0) |
| **Tampered Tiles (label 1)** | **3,428** | **7.88%** | >= 16 tampered pixels |
| **Ambiguous Tiles (label -1)** | **24** | **0.06%** | 1 <= sum(mask) <= 15 border pixels |
| **Total Labeled Synthetic Tiles** | **43,510** | **100.0%** | Phase 4 synthetic tiles audited |
| **Unlabelled Real Holdout Tiles** | **17,044** | — | Real Google Pay screenshots (mask=None) |
| **Grand Total Tiles** | **60,554** | — | Across all 1,387 project screenshots |

### Class Imbalance Ratio
Imbalance Ratio = Clean Tiles / Tampered Tiles = 40058 / 3428 = **11.69 : 1**

Approximately **11.7 clean tiles exist for every 1 tampered tile**. This quantifies why training an unweighted CNN causes accuracy collapse and demonstrates why Phase 5 must use weighted loss, balanced mini-batch sampling, or dynamic negative mining.

---

## 3. Label Distribution by Layout Family

| Layout Family | Clean Tiles | Tampered Tiles | Ambiguous Tiles | Total Tiles | Tampered Pct | Imbalance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **compact** | 19,765 | 1,510 | 11 | 21,286 | 7.09% | 13.09 : 1 |
| **bank_progress** | 2,927 | 331 | 2 | 3,260 | 10.15% | 8.84 : 1 |
| **enriched** | 16,804 | 1,505 | 11 | 18,320 | 8.22% | 11.17 : 1 |
| **recharge** | 562 | 82 | 0 | 644 | 12.73% | 6.85 : 1 |

---

## 4. Edited Screenshot Audit (Zero Positive Tiles Check)

A critical data-quality check in patch forensics is determining whether any edited screenshots fail to yield positive tiles due to threshold clipping:
- **Edited screenshots audited:** 500
- **Edited screenshots with zero positive tiles:** **0** (0.0%)
- **Result:** **PASS**. In 100% of edited screenshots, the manipulation region is large enough to register >= 16 pixels in at least one tile.

---

## 5. Ambiguous Tiles Analysis

Ambiguous tiles contain between 1 and 15 foreground pixels:
- **Total ambiguous tiles:** 24 (0.06% of synthetic tiles).
- **Cause:** These occur exclusively at the geometric outer boundaries of text characters (e.g. the tail of a digit '9' or the dot of an 'i' clipped across a tile grid edge).
- **Recommendation for Phase 5 (CNN Training):**
  - **Option A (Recommended):** Exclude ambiguous tiles (label -1) from CNN training batches to avoid noisy gradient updates.
  - **Option B:** Treat ambiguous tiles as weak negatives or label-smoothed targets during semi-supervised evaluation.

---

## 6. Leakage Prevention Audit

- **Pairs evaluated:** 100
- **Group ID match rate across clean/edited pairs:** **100%** (100/100)
- **Leakage risk detected:** **0**

All tiles from the clean and edited versions of a transaction share the identical `split_group_id`.

---

## 7. Handover Recommendations for Phase 5 (CNN Model Training)

1. **Balanced Batch Sampling:** Because clean tiles represent 92.07% of the dataset, naive random sampling will result in batches with ~92% clean tiles. Implement a balanced sampler (e.g., 50% clean, 50% tampered per batch) or apply class-weighted cross-entropy loss with positive weight w_+ approx 11.7.
2. **Filter Ambiguous Tiles:** Ignore tiles with `tile_label == -1` during gradient backpropagation.
3. **Partition by `split_group_id`:** Enforce `GroupKFold` or Leave-One-Family-Out on `split_group_id` to guarantee zero cross-split leakage.
