# CBL-04 — Phase 21 CRAFT Shard Extraction Integrity Audit

**Project:** CBL-04 — GPay Dark-Theme Transaction Screenshot Tampering Detection  
**Role:** Member 3 — Backend / AI Integration Lead  
**Audit Scope:** Read-Only Shard Integrity, Dataset Coverage Gate & Bounding Box Quality Audit  
**Date:** November 4, 2026  
**Status:** **READY FOR SCORING**  

---

## 1. Executive Summary & Gating Decision

| Inspection Check | Authoritative Standard | Audit Result | Status |
| :--- | :--- | :--- | :---: |
| **Dataset Completeness** | Exactly 976 images (868 Train + 108 Val) | **976 / 976 images** | **PASS (100.0%)** |
| **Partition Integrity** | 868 Train / 108 Val matching freeze manifests | **868 Train / 108 Val** | **PASS** |
| **Test Set Quarantine** | Zero test images accessed (`metadata/phase6/test_manifest.csv`) | **0 test images touched** | **PASS (100% quarantined)** |
| **Range Disjointness** | Pairwise empty intersections $\forall i \ne j: S_i \cap S_j = \emptyset$ | **0 overlaps across 4 shards** | **PASS** |
| **Duplicate / Missing IDs** | 0 duplicates, 0 missing, 0 unexpected | **0 missing, 0 duplicates, 0 unexpected** | **PASS** |
| **JSON & Schema Validity** | Valid JSON syntax, all required keys present | **0 parse errors, 0 missing schema fields** | **PASS** |
| **Coordinate Bounds** | $0 \le x_1 < x_2 \le W$ and $0 \le y_1 < y_2 \le H$ | **0 coordinate violations** | **PASS** |
| **Box Non-Degeneracy** | $w > 0, h > 0$, area $> 0$, finite coordinates | **0 zero-area, 0 non-finite** | **PASS** |
| **Padding Containment** | Expanded crops strictly clipped to $[0, W] \times [0, H]$ | **0 padding boundary breaches** | **PASS** |
| **Detector Confidence** | $C_{\text{detector}} \in [0.0, 1.0]$, strictly finite | **0 out-of-bounds confidences** | **PASS** |
| **Empty Detections** | Non-empty text coverage on GPay screenshots | **0 empty images** | **PASS** |
| **OCR-Free Independence** | Zero text recognition, zero regex, zero labels | **Pure CRAFT character/affinity heatmaps** | **PASS** |

**Gating Verdict:** **READY FOR SCORING**  
*Crop scoring through frozen Phase 19 ResNet18 checkpoints may commence upon authorization.*

---

## 2. Extraction Workers and Shard Provenance

All 976 screenshots were partitioned into four contiguous, non-overlapping shards of 244 images each.

| Shard ID | Worker Task ID | Index Range | Processed Records | Output File Path | File Size | SHA-256 Digest | Runtime | Throughput | Worker State |
| :---: | :---: | :---: | :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| **Shard 0** | `task-4363` | `[0:244]` | 244 / 244 | `data/craft_shard_0.json` | 4.76 MB | `b5ea66eeacdcf506...` | 784.01s (13.1m) | 0.31 img/s | **COMPLETED (exit 0)** |
| **Shard 1** | `task-4366` | `[244:488]` | 244 / 244 | `data/craft_shard_1.json` | 4.77 MB | `6228377d793b5aeb...` | 1164.19s (19.4m) | 0.21 img/s | **COMPLETED (exit 0)** |
| **Shard 2** | `task-4368` | `[488:732]` | 244 / 244 | `data/craft_shard_2.json` | 4.72 MB | `8b9c6acfe960e439...` | 1159.11s (19.3m) | 0.21 img/s | **COMPLETED (exit 0)** |
| **Shard 3** | `task-4370` | `[732:976]` | 244 / 244 | `data/craft_shard_3.json` | 4.95 MB | `3802196db723ef86...` | 1056.30s (17.6m) | 0.23 img/s | **COMPLETED (exit 0)** |

### Range Disjointness and Bijective Mapping Verification
- **Range Partitioning**:
  - Shard 0: Rows `[0:244]` (244 images)
  - Shard 1: Rows `[244:488]` (244 images)
  - Shard 2: Rows `[488:732]` (244 images)
  - Shard 3: Rows `[732:976]` (244 images)
- **Pairwise Intersection Audit**:
  - Intersections tested: $\binom{4}{2} = 6$ shard pairs.
  - Overlapping image IDs found: **0**.
  - Union of extracted image IDs: **976** distinct images ($= 976$ expected).
  - Bijective Mapping: **1:1 exact bijection** between manifest rows and extracted records.

---

## 3. Dataset Coverage and Partition Breakdown

The extracted records were checked against the authoritative Phase 6 manifests (`train_manifest.csv` and `validation_manifest.csv`).

| Split | Authoritative Target | Extracted Records | Missing Images | Duplicate Images | Unexpected Images | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Set** | 868 (434 pairs) | 868 | 0 | 0 | 0 | **PASS (100.0%)** |
| **Validation Set** | 108 (54 pairs) | 108 | 0 | 0 | 0 | **PASS (100.0%)** |
| **Combined Total** | **976 (488 pairs)** | **976** | **0** | **0** | **0** | **PASS (100.0%)** |

### Split Distribution Across Shards
- **Shard 0**: 244 images (244 train, 0 validation)
- **Shard 1**: 244 images (244 train, 0 validation)
- **Shard 2**: 244 images (244 train, 0 validation)
- **Shard 3**: 244 images (136 train, 108 validation)

### Label Balance Across Extracted Screenshots
- **Clean Screenshots**: 488 (50.0%)
- **Edited Screenshots**: 488 (50.0%)
- **Matched Pair Parity**: Exactly 488 clean and 488 edited counterparts, with zero unpaired screenshots.

---

## 4. Candidate Bounding Box Integrity & Geometry Audit

Across all 976 screenshots, CRAFT scene text detection identified **51,684** candidate word-level regions.

### Quantitative Summary

| Geometry / Quality Check | Mathematical Requirement | Observed Count | Compliance |
| :--- | :--- | :---: | :---: |
| **Total Candidates Extracted** | $\sum_{i=1}^{N} |C_i|$ | **51,684** | **VERIFIED** |
| **Candidates Per Screenshot** | $\mu \pm \sigma$ (range: $[\min, \max]$) | **52.95** (median: 55.0, range: [42, 64]) | **NORMAL (35–55/img)** |
| **Empty Text Screenshots** | $|C_i| = 0$ | **0** | **PASS (0 empty)** |
| **Invalid Coordinates** | $x_2 \le x_1$ or $y_2 \le y_1$ or $x_1 < 0$ or $x_2 > W$ | **0** | **PASS (100% valid)** |
| **Zero Area Boxes** | $(x_2 - x_1) \times (y_2 - y_1) = 0$ | **0** | **PASS (0 zero-area)** |
| **Non-Finite Quantities** | Coordinate or confidence is NaN or Inf | **0** | **PASS (0 non-finite)** |
| **Glyph Padding Expansion** | Expanded box clipped to $[0, W] \times [0, H]$ | **0** | **PASS (100% clipped)** |
| **Detector Confidence Range** | $C_{\text{detector}} \in [0.0, 1.0]$ | **0** | **PASS (100% in range)** |

---

## 5. Box Morphology & Overlap Distribution

### Box Morphology Breakdown
To verify that CRAFT extracts granular text tokens rather than monolithic UI patches, bounding box shapes were classified:

| Morphology Category | Definition Criteria | Candidate Count | Percentage of Total | Forensic Handling |
| :--- | :--- | :---: | :---: | :--- |
| **Standard Word Regions** | Normal aspect ratio, $w \ge 10, h \ge 8$ | **51,055** | **98.78%** | Primary forensic inspection targets |
| **Tiny Regions** | $w < 10$ px or $h < 8$ px (dots, commas, bullets) | **44** | **0.09%** | Preserved (not discarded) |
| **Excessively Large Regions** | $w > 0.85W$ or $h > 0.35H$ (title banners) | **0** | **0.00%** | Preserved (not discarded) |
| **Line-Like Regions** | Elongated text lines ($w / h > 8.0$) | **585** | **1.13%** | Preserved (not discarded) |

> [!NOTE]
> In accordance with Phase 21 requirements, candidates with unusual morphology are strictly retained in the candidate pool so that the visual classifier makes the final forensic determination.

### Intra-Screenshot Overlap Distribution
Intra-image candidate pairs ($1,358,900$ total pairs across all screenshots) were evaluated for spatial overlap:

| IoU Overlap Threshold | Pair Count | Percentage of Total Pairs | Interpretation |
| :--- | :---: | :---: | :--- |
| $\text{IoU} \ge 0.10$ | **9** | **0.00%** | Minor boundary sharing |
| $\text{IoU} \ge 0.30$ | **0** | **0.00%** | Partial overlap |
| $\text{IoU} \ge 0.50$ | **0** | **0.00%** | Substantial overlap |
| $\text{IoU} \ge 0.70$ | **0** | **0.00%** | Near-duplicate candidates |
| $\text{IoU} \ge 0.85$ | **0** | **0.00%** | Severe overlap (redundant detections) |

---

## 6. OCR-Free Candidate Selection Verification

An inspection of [`extract_craft_shard.py`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase21_craft_word_forensics/extract_craft_shard.py) confirms strict adherence to OCR-free protocol:

1. **Zero Text Recognition Invocation**: The text recognition module (`english_g2.pth`) is **never invoked**. Candidates are derived solely from character (`score_text`) and affinity (`score_link`) maps output by CRAFT.
2. **Zero Semantic Keyword Filtering**: No heuristic keywords (e.g. "Paid to", "₹", "UPI", "Completed", "Google Pay") are used to filter boxes.
3. **Zero Label / Annotation Leakage**:
   - Neither the clean/edited ground-truth label nor the tamper bounding boxes (`bounding_boxes_xyxy`) are exposed to candidate selection.
   - All screenshots (clean and edited) undergo identical extraction logic.
4. **Continuous Detector Confidence**: Every candidate retains its continuous detector confidence $C_{\text{detector}} = \text{mean}(S_{\text{text}})$ over the box footprint, enabling confidence-filtered aggregation during scoring.

---

## 7. Gating Recommendation

All pre-scoring audit gates have been verified:
- [x] All 4 extraction shards completed without errors.
- [x] Exactly 976 images processed (868 training + 108 validation).
- [x] Complete range disjointness verified ($S_i \cap S_j = \emptyset$).
- [x] Zero duplicate, missing, or unexpected image IDs.
- [x] Zero test partition images accessed.
- [x] All candidate coordinates and confidences strictly valid and finite.
- [x] Candidate morphology verified and preserved without heuristic pruning.

**Recommendation:** Proceed to Candidate Crop Scoring and Screenshot-Level Aggregation (`merge_and_score_crops.py`).
