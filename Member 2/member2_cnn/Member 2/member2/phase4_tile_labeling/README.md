# Phase 4: Mask-to-Tile Labeling Engine
**Project:** CBL-04 – Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Module:** `member2/phase4_tile_labeling`  
**Assigned Role:** Member 2 – Tile Forensics CNN & Heatmap Lead  
**Status:** COMPLETE & VERIFIED  

---

## 1. Overview & Purpose

Phase 4 assigns verified ground-truth labels to $256 \times 256$ tiles extracted in Phase 3 by computing spatial overlap with pixel-level tampering masks.

### Labeling Rules (Configurable via `LABELING_CONFIG.json`):
1. **Clean (label `0`, status `"clean"`):** Foreground pixel count == 0.
2. **Tampered (label `1`, status `"tampered"`):** Foreground pixel count $\ge 16$ pixels (`tampered_threshold`).
3. **Ambiguous (label `-1`, status `"ambiguous"`):** Foreground pixel count between $1$ and $15$ pixels (`ambiguous_min` to `ambiguous_max`).
4. **Unlabelled (label `None`, status `"unlabelled"`):** Real Google Pay holdout screenshots where `mask=None`.

Ambiguous boundary tiles are strictly tagged as `-1` rather than silently misclassified as clean or tampered.

---

## 2. Key Architecture & Flow

```
ExtractedTile (Phase 3: 256x256 image + mask slice)
   │
   ▼
TileLabeler (Phase 4: tile_labeler.py)
   │
   ├── 1. Mask Overlap Computation:
   │      - Count tampered pixels: count = np.sum(mask_slice > 0)
   │      - Compute tampered ratio: ratio = count / (256 * 256)
   │
   ├── 2. Threshold Classification:
   │      - count == 0           -> label 0  ("clean")
   │      - count >= 16          -> label 1  ("tampered")
   │      - 1 <= count <= 15     -> label -1 ("ambiguous")
   │      - mask is None         -> label None ("unlabelled")
   │
   ├── 3. Provenance & Metadata Extension:
   │      - Preserves tile_id, source_id, pair_id, split_group_id, coords
   │      - Adds tile_label, label_status, tampered_pixel_count, tampered_pixel_ratio
   │
   └── 4. Memory Optimization:
          - LazyLabeledTileDataset indexes 60,554 tiles in O(1) time
          - Images loaded on-demand; zero tile files saved to disk
         │
         ▼
LabeledTile (metadata: LabeledTileMetadata, image: (256, 256, 3), mask: (256, 256) or None)
```

---

## 3. Metadata Schema Reference

Every labeled tile contains structured [`LabeledTileMetadata`](file:///c:/Users/athar/OneDrive/Desktop/Member%202/member2/phase4_tile_labeling/tile_labeler.py#L95-L121):

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `tile_id` | `str` | Unique deterministic identifier (`{image_id}_t{tile_index:04d}_x{tile_x}_y{tile_y}`) |
| `source_id` | `str` | Source transaction identifier (e.g. `P4-0001`) |
| `pair_id` | `str` / `None` | Transaction pair identifier (`P4-0001` or None for real data) |
| `image_path` | `str` / `None` | Path to original source screenshot |
| `mask_path` | `str` / `None` | Path to original tampering mask |
| `tile_x` | `int` | Top-left X coordinate on native canvas |
| `tile_y` | `int` | Top-left Y coordinate on native canvas |
| `tile_width` | `int` | Width of tile (strictly 256) |
| `tile_height` | `int` | Height of tile (strictly 256) |
| `tile_index` | `int` | Sequential tile index within image |
| `original_width` | `int` | Native screenshot width |
| `original_height`| `int` | Native screenshot height |
| `split_group_id` | `str` | Grouping key for leakage prevention |
| `tile_label` | `int` / `None` | `0` (clean), `1` (tampered), `-1` (ambiguous), `None` (unlabelled) |
| `label_status` | `str` | `"clean"`, `"tampered"`, `"ambiguous"`, `"unlabelled"` |
| `tampered_pixel_count` | `int` | Number of non-zero pixels in mask slice |
| `tampered_pixel_ratio` | `float` | Tampered pixels divided by $256 \times 256$ |
| `label_threshold` | `int` | Configured tampered threshold (default: 16) |
| `layout_family` | `str` / `None` | Layout family (`compact`, `bank_progress`, `enriched`, `recharge`) |

---

## 4. Class Imbalance Analysis Across Phase 4 Dataset

The empirical census conducted across the 1,000 Phase 4 synthetic screenshots (43,510 tiles) and 387 real holdout screenshots (17,044 tiles) revealed:

| Category | Tile Count | Percentage | Class Ratio |
| :--- | :---: | :---: | :---: |
| **Clean Tiles (label 0)** | **40,058** | **92.07%** | **11.69 : 1** (relative to tampered) |
| **Tampered Tiles (label 1)** | **3,428** | **7.88%** | **1.00 : 1** |
| **Ambiguous Tiles (label -1)** | **24** | **0.06%** | — |
| **Total Labeled Synthetic Tiles** | **43,510** | **100.0%** | — |
| **Unlabelled Real Holdout Tiles** | **17,044** | — | — |
| **Grand Total Project Tiles** | **60,554** | — | — |

### Breakdown by Layout Family
- **compact (436 images):** 19,765 clean (92.85%), 1,510 tampered (7.09%), 11 ambiguous (0.05%) $\to$ Imbalance: **13.09 : 1**
- **bank_progress (108 images):** 2,927 clean (89.79%), 331 tampered (10.15%), 2 ambiguous (0.06%) $\to$ Imbalance: **8.84 : 1**
- **enriched (432 images):** 16,804 clean (91.72%), 1,505 tampered (8.22%), 11 ambiguous (0.06%) $\to$ Imbalance: **11.17 : 1**
- **recharge (24 images):** 562 clean (87.27%), 82 tampered (12.73%), 0 ambiguous (0.00%) $\to$ Imbalance: **6.85 : 1**

### Dataset Quality Highlights
- **100% Clean Accuracy:** In all 500 clean screenshots (21,755 clean tiles), zero false positive pixels were detected.
- **100% Positive Coverage:** 500 out of 500 edited screenshots contain at least one positive tile (0 edited screenshots with zero positive tiles).

---

## 5. Quick Start & Python API Usage

### Labeling Tiles from a Single Sample
```python
from member2.phase1_dataset_loader.dataset_loader import DatasetLoader
from member2.phase4_tile_labeling.tile_labeler import TileLabeler, LabelingConfig

# 1. Initialize loader and labeler
loader = DatasetLoader(manifest_path="metadata/phase4_manifest.csv")
sample = loader.load(mode="SingleMode")[0]
labeler = TileLabeler()

# 2. Iterate labeled tiles on demand
for labeled_tile in labeler.iter_labeled_tiles_from_sample(sample):
    print(f"Tile: {labeled_tile.tile_id}, Label: {labeled_tile.tile_label} ({labeled_tile.label_status})")
    print(f"Tampered Pixels: {labeled_tile.tampered_pixel_count} / {256*256}")
    break
```

### On-Demand Indexing with `LazyLabeledTileDataset`
```python
from member2.phase4_tile_labeling.tile_labeler import LazyLabeledTileDataset

samples = loader.load(mode="SingleMode")
dataset = LazyLabeledTileDataset(samples, labeler=labeler)

# Access individual tile on demand
tile_100 = dataset[100]
print(f"Tile 100 Status: {tile_100.label_status}, Pixels: {tile_100.tampered_pixel_count}")

# Compute dataset-scale class imbalance
stats = dataset.compute_class_imbalance_stats()
print(f"Class imbalance ratio: {stats['class_imbalance_ratio_clean_to_tampered']}:1")
```

---

## 6. Command-Line Interface (CLI)

### Run Dataset Validation & Generate Audit Report
```bash
python member2/phase4_tile_labeling/validate_tile_labeling.py --generate-report
```

### Run Unit and Integration Tests
```bash
python -m unittest member2/phase4_tile_labeling/test_tile_labeler.py -v
```

---

## 7. Test Suite Summary (14/14 Passed)

The test suite ([`test_tile_labeler.py`](file:///c:/Users/athar/OneDrive/Desktop/Member%202/member2/phase4_tile_labeling/test_tile_labeler.py)) verifies:

| Test # | Test Name | Invariance Verified | Status |
| :---: | :--- | :--- | :---: |
| 1 | `test_1_completely_clean_mask_all_label_zero` | Clean masks yield 100% label 0 | **PASS** |
| 2 | `test_2_tile_with_ge_16_foreground_pixels_is_tampered` | Count $\ge 16$ yields label 1 | **PASS** |
| 3 | `test_3_tile_with_1_to_15_foreground_pixels_is_ambiguous` | Count 1–15 yields label -1 | **PASS** |
| 4 | `test_4_mask_to_tile_coordinate_alignment` | Overlap matches exact spatial coordinates | **PASS** |
| 5 | `test_5_binary_mask_normalization_values_1_and_255` | Masks $\{0, 1\}$ and $\{0, 255\}$ yield identical counts | **PASS** |
| 6 | `test_6_missing_mask_handled_safely` | Missing masks raise clear errors | **PASS** |
| 7 | `test_7_real_unlabelled_screenshots_handled` | Real holdouts yield `label=None`, `"unlabelled"` | **PASS** |
| 8 | `test_8_pair_and_source_group_id_preserved` | Clean and edited pairs share `split_group_id` | **PASS** |
| 9 | `test_9_deterministic_labeling` | Repeated runs yield identical labels | **PASS** |
| 10 | `test_10_boundary_anchored_tiles_labeled_correctly` | Canvas edge tiles accurately detect tampering | **PASS** |
| 11 | `test_11_threshold_configuration` | Thresholds reconfigure cleanly via `LabelingConfig` | **PASS** |
| 12 | `test_12_lazy_labeled_dataset` | `LazyLabeledTileDataset` indexes without RAM bloat | **PASS** |
| 13 | `test_13_class_imbalance_identifies_zero_positive_edited` | Accurately audits zero-positive edited screens | **PASS** |
| 14 | `test_14_dataset_immutability` | Original dataset files remain 100% immutable | **PASS** |

**Execution Time:** 14 tests passed in ~1.7 seconds.

---

## 8. Handover Recommendations for Phase 5 (CNN Model Training)

1. **Balanced Mini-Batch Sampling:**
   - Because clean tiles represent 92.07% of the dataset, naive random sampling will overwhelm minibatches with negative samples.
   - Implement a balanced batch sampler (e.g. 50% clean, 50% tampered per batch) or use positive class weight $w_+ \approx 11.69$ in weighted cross-entropy / focal loss.
2. **Exclude Ambiguous Boundary Tiles:**
   - Filter out tiles with `tile_label == -1` during gradient backpropagation to avoid noisy gradient updates from partial sub-character artifacts.
3. **Prevent Cross-Split Leakage:**
   - Partition exclusively by `split_group_id` (via `GroupKFold` or Leave-One-Family-Out) so clean and edited counterparts never cross between train and validation folds.

