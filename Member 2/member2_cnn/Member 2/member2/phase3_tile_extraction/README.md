# Phase 3: Native-Resolution Tile Extraction Engine
**Project:** CBL-04 – Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Module:** `member2/phase3_tile_extraction`  
**Assigned Role:** Member 2 – Tile Forensics CNN & Heatmap Lead  
**Status:** COMPLETE & VERIFIED  

---

## 1. Overview & Purpose

Phase 3 implements an exact, native-resolution patch extraction engine for Google Pay dark-theme transaction screenshots.

### The Forensic Problem
Full-image downsampling to $160 \times 320$ (used in the earlier baseline) erased micro-level forensic tampering evidence, causing the ResNet-18 baseline to collapse to ~50.5% accuracy.

### The Phase 3 Solution
Instead of resizing full canvases, Phase 3 extracts uniform $256 \times 256$ pixel tiles directly from full-resolution screenshots at a stride of $128$ pixels (50% spatial overlap). This retains:
- Sharp font glyph edges, subpixel antialiasing, and kerning anomalies.
- Discrete JPEG $8 \times 8$ DCT compression block boundaries.
- Micro splice boundaries around manipulated amount/date characters.

---

## 2. Key Architecture & Geometry

```
Preprocessed Screenshot (Native (W, H, 3) from Phase 2)
   │
   ├── Width W in [279, 738] px, Height H in [1261, 1601] px
   │
   ▼
TileExtractor (tile_size=256, stride=128)
   │
   ├── 1. 1D Coordinate Generation (compute_tile_coordinates):
   │      - Standard sliding window: pos_k = k * 128 (while pos_k + 256 <= L)
   │      - Boundary anchoring: pos_edge = L - 256 (if pos_last + 256 < L)
   │      * Guarantees 100% canvas coverage; 0 dropped or clipped edge pixels.
   │
   ├── 2. 2D Spatial Grid & Tile Slicing:
   │      - Grid: len(x_coords) * len(y_coords) tiles per screenshot (~18 to 60 tiles)
   │      - Extracted array: strictly (256, 256, 3) uint8 RGB
   │      - Mask slice: strictly (256, 256) uint8 if mask present; None for real holdouts
   │
   ├── 3. Metadata Generation (TileMetadata):
   │      - Complete spatial tracking: (tile_x, tile_y, tile_width, tile_height, tile_index)
   │      - Provenance tracking: (tile_id, source_id, pair_id, original_width, original_height)
   │      - Leakage prevention: split_group_id shared across clean/edited transaction pairs
   │
   └── 4. Lazy Memory Optimization (LazyTileDataset):
          - Precomputes index table: (sample_idx, tile_x, tile_y, tile_idx)
          - Materializes tiles on demand during __getitem__
          - Zero individual tile image files written to disk
         │
         ▼
ExtractedTile (image: (256, 256, 3), mask: (256, 256) or None, metadata: TileMetadata)
```

---

## 3. Metadata Schema Reference

Every extracted tile is paired with structured `TileMetadata`:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `tile_id` | `str` | Unique deterministic identifier (`{image_id}_t{tile_index:04d}_x{tile_x}_y{tile_y}`) |
| `source_id` | `str` | Source transaction identifier (e.g. `compact_0001`) |
| `pair_id` | `str` / `None` | Transaction pair identifier (`pair_0001` or None for real data) |
| `image_path` | `str` / `None` | Path to original source screenshot |
| `tile_x` | `int` | Top-left X coordinate on native canvas |
| `tile_y` | `int` | Top-left Y coordinate on native canvas |
| `tile_width` | `int` | Width of tile (strictly 256) |
| `tile_height` | `int` | Height of tile (strictly 256) |
| `tile_index` | `int` | Sequential tile index within image (0-indexed) |
| `original_width` | `int` | Native screenshot width |
| `original_height`| `int` | Native screenshot height |
| `split_group_id` | `str` | Grouping key for cross-split leakage prevention |

---

## 4. Leakage Prevention Strategy

Cross-split data leakage is prevented by strict grouping:
1. **Pair Invariant:** Clean and edited screenshots from the same transaction pair share the exact same `split_group_id` (`pair_id` if present, else `source_id`).
2. **All Tiles Bound to Source Group:** Every tile extracted from an image inherits that image's `split_group_id`.
3. **Partitioning Rule:** When data splits are formed (Phase 5/7), partitioning must group exclusively by `split_group_id`. Tiles from the same transaction will never cross between training, validation, or test sets.

---

## 5. Quick Start & Python API Usage

### Extract Tiles Lazily from a Single Sample
```python
from member2.phase1_dataset_loader.dataset_loader import DatasetLoader
from member2.phase3_tile_extraction.tile_extractor import TileExtractor, TileConfig

# 1. Initialize dataset loader and tile extractor
loader = DatasetLoader(manifest_path="metadata/phase4_manifest.csv")
sample = loader.load(mode="SingleMode")[0]
extractor = TileExtractor(TileConfig(tile_size=256, stride=128))

# 2. Iterate tiles lazily (generator)
for tile in extractor.iter_tiles_from_sample(sample):
    print(f"Tile ID: {tile.tile_id}, Coords: ({tile.tile_x}, {tile.tile_y}), Shape: {tile.image.shape}")
    if tile.has_mask:
        print(f"Mask Shape: {tile.mask.shape}")
    break
```

### Process Matched Clean/Edited Pairs
```python
pair = loader.load(mode="PairMode")[0]
clean_tiles, edited_tiles = extractor.extract_tiles_from_pair(pair)

assert len(clean_tiles) == len(edited_tiles)
assert clean_tiles[0].split_group_id == edited_tiles[0].split_group_id
print(f"Extracted {len(clean_tiles)} aligned tiles for pair '{pair.pair_id}'")
```

### On-Demand Dataset Indexing with `LazyTileDataset`
```python
from member2.phase3_tile_extraction.tile_extractor import LazyTileDataset

samples = loader.load(mode="SingleMode")
dataset = LazyTileDataset(samples, extractor=extractor)

print(f"Total tiles in dataset: {len(dataset):,}")  # 43,510 tiles

# Access individual tile on demand in O(1) time
tile_42 = dataset[42]
print(f"Tile 42 ID: {tile_42.tile_id}, Image Shape: {tile_42.image.shape}")

# Access lightweight metadata without loading image pixels
meta_42 = dataset.get_metadata(42)
print(f"Tile 42 Group: {meta_42.split_group_id}")
```

---

## 6. Command-Line Interface (CLI)

### Run Dataset Validation & Generate Audit Report
```bash
python member2/phase3_tile_extraction/validate_tile_extraction.py --generate-report
```

### Run Unit and Integration Tests
```bash
python -m unittest member2/phase3_tile_extraction/test_tile_extractor.py -v
```

---

## 7. Test Suite Summary (15/15 Passed)

The test suite ([`test_tile_extractor.py`](file:///c:/Users/athar/OneDrive/Desktop/Member%202/member2/phase3_tile_extraction/test_tile_extractor.py)) verifies:

| Test # | Test Name | Invariance Verified | Status |
| :---: | :--- | :--- | :---: |
| 1 | `test_1_exact_tile_dimensions` | Every tile is strictly $(256, 256, 3)$ uint8 | **PASS** |
| 2 | `test_2_correct_stride` | Standard consecutive steps are exactly 128 px | **PASS** |
| 3 | `test_3_coordinate_calculation` | Coordinates match analytical sliding window | **PASS** |
| 4 | `test_4_edge_coverage_non_divisible_dimensions` | Right/bottom tiles anchor at $(W-256, H-256)$ | **PASS** |
| 5 | `test_5_no_dropped_edge_regions` | 100% of pixels in $[0, L-1]$ are covered | **PASS** |
| 6 | `test_6_deterministic_extraction` | Bitwise identical repeat extraction | **PASS** |
| 7 | `test_7_rgb_preservation` | 3-channel RGB uint8 preserved without distortion | **PASS** |
| 8 | `test_8_metadata_and_split_grouping` | Clean and edited pairs share `split_group_id` | **PASS** |
| 9 | `test_9_lazy_dataset_behavior` | `LazyTileDataset` indexes without RAM bloat | **PASS** |
| 10 | `test_10_safe_padding_for_small_dimensions` | Canvases $< 256$ padded safely with `#111111` | **PASS** |
| 11 | `test_11_unlabelled_real_screenshots_handled` | Real holdouts extract cleanly with `mask=None` | **PASS** |
| 12 | `test_12_matched_pair_extraction` | Pair extraction yields equal count & coordinates | **PASS** |
| 13 | `test_13_out_of_bounds_coordinate_raises` | Invalid coordinates raise `ValueError` | **PASS** |
| 14 | `test_14_integration_phase4_dataset` | Extracts actual Phase 4 dataset samples | **PASS** |
| 15 | `test_15_dataset_immutability` | Dataset files remain 100% untouched (MD5 checked) | **PASS** |

**Execution Time:** 15 tests passed in ~2.8 seconds.

---

## 8. Handover Notes for Phase 4 (Tile Labeling)

1. **Class Imbalance in Extracted Tiles:**
   - In edited screenshots, tampering is typically localized in amount, timestamp, or UPI ID text blocks (typically 1 to 3 tiles out of ~44 per screenshot).
   - Approximately 92% to 96% of tiles extracted from edited screenshots will contain exclusively authentic background pixels.
   - *Phase 4 Requirement:* Compute binary labels based on ground truth mask overlap:
     - Tile is `tampered` (1) if $\sum \text{mask} \ge \tau$ (e.g. $\tau = 16$ pixels).
     - Tile is `clean` (0) if $\sum \text{mask} == 0$.
     - Tiles with $1 \le \sum \text{mask} < \tau$ should be filtered or marked ambiguous.
2. **Boundary Anchoring Overlap:**
   - The final anchored boundary tile overlaps with the preceding tile by more than 128 pixels (overlap = $256 - ((L - 256) \pmod{128})$). This is normal and intentional to ensure zero lost pixels.

