# Phase 2: Native-Resolution Preprocessing & Transformation Pipeline
**Project:** CBL-04 – Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Module:** `member2/phase2_preprocessing`  
**Assigned Role:** Member 2 – Tile Forensics CNN & Heatmap Lead  
**Status:** COMPLETE & VERIFIED  

---

## 1. Module Overview & Purpose

Phase 2 provides a robust, reproducible native-resolution preprocessing engine for Google Pay dark-theme transaction screenshots and manipulation masks.

### The Problem in Earlier Work
The earlier baseline model globally resized full screenshots to approximately $160 \times 320$. Because mobile screenshots range up to $738 \times 1601$ pixels, severe downsampling erased:
- High-frequency edge gradients and antialiased text boundaries.
- Font weight differences and subtle kerning inconsistencies.
- Discrete JPEG compression block boundaries ($8 \times 8$ pixel grids).
- Pixel-level tampering indicators in manipulated transaction amounts and timestamps.

This destruction of forensic evidence caused the baseline ResNet-18 model to collapse to ~50.5% validation accuracy.

### The Phase 2 Solution
Phase 2 preserves 100% of the native screenshot resolution. Rather than downsampling full images, Phase 2 standardizes formats and prepares clean/edited screenshots for **Phase 3 Local Tile Extraction ($256 \times 256$)**, ensuring forensic evidence remains completely intact.

---

## 2. Key Architecture & Features

```
DatasetLoader (Phase 1)
   │
   ├── SingleSample (Synthetic Clean / Edited / Real Holdout)
   └── PairSample (Matched Clean + Edited Pair)
         │
         ▼
Preprocessor (Phase 2: transforms.py)
   │
   ├── 1. Format Standardization:
   │      - RGB conversion (RGBA composited onto GPay dark #111111)
   │      - Binary mask standardization (uint8 array {0, 1}; PIL L {0, 255})
   │      - Missing mask preservation (mask=None for unlabelled real data)
   │
   ├── 2. Exact Spatial Alignment:
   │      - Image canvas (W, H) == Mask canvas (W, H) (100% verified)
   │
   ├── 3. Controlled Robustness Augmentation (Opt-In, In-Memory):
   │      - Mild brightness/contrast scaling
   │      - Mild Gaussian blur (sigma in [0.2, 0.6])
   │      - Mild JPEG recompression (Q in [75, 95])
   │      * Applied exclusively to image buffers, never to ground truth masks
   │      * Sampled once per pair to maintain pair consistency
   │
   └── 4. Deterministic Reproducibility:
          - Seed-controlled RNG for reproducible training and evaluation
         │
         ▼
ProcessedSample / ProcessedPair (Ready for Phase 3 Tile Extraction)
```

---

## 3. Configuration & Parameter Reference

Preprocessing behavior is controlled via `PreprocessConfig` (or JSON file `preprocessing_config.json`):

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `preserve_native_resolution` | `bool` | `True` | Guarantees zero downsampling or spatial distortion. |
| `jpeg_quality` | `int` or `(int, int)` | `None` | Opt-in mild JPEG recompression quality (e.g. `(75, 95)`). |
| `gaussian_sigma` | `float` or `(float, float)` | `None` | Opt-in mild Gaussian blur radius (e.g. `(0.2, 0.6)`). |
| `brightness_factor` | `float` or `(float, float)` | `None` | Opt-in brightness multiplier (e.g. `(0.95, 1.05)`). |
| `contrast_factor` | `float` or `(float, float)` | `None` | Opt-in contrast multiplier (e.g. `(0.95, 1.05)`). |
| `alpha_fill_color` | `Tuple[int, int, int]` | `(17, 17, 17)` | GPay dark background `#111111` for alpha compositing. |
| `mask_foreground_value` | `int` | `1` | Foreground pixel value in standardized NumPy mask array. |
| `normalize_to_float` | `bool` | `False` | If `True`, scales image ndarray to float32 in `[0.0, 1.0]`. |
| `seed` | `Optional[int]` | `None` | Master seed for deterministic transform parameter sampling. |

---

## 4. Quick Start & Python API Usage

### Processing Individual Samples
```python
from member2.phase1_dataset_loader.dataset_loader import DatasetLoader
from member2.phase2_preprocessing.transforms import PreprocessConfig, Preprocessor

# 1. Initialize dataset loader and preprocessor
loader = DatasetLoader(manifest_path="metadata/phase4_manifest.csv")
config = PreprocessConfig(preserve_native_resolution=True, seed=42)
preprocessor = Preprocessor(config)

# 2. Ingest SingleSample
sample = loader.load(mode="SingleMode")[0]
processed = preprocessor.process_sample(sample)

# 3. Access standardized buffers
print(f"Image ID: {processed.image_id}")
print(f"Canvas size: {processed.processed_size} (W, H)")
print(f"Image array shape: {processed.image.shape}, dtype: {processed.image.dtype}")
if processed.has_mask:
    print(f"Mask array shape: {processed.mask.shape}, unique: {set(processed.mask.flatten())}")
```

### Processing Matched Pairs (Pair Consistency)
```python
# Clean and edited screenshots receive IDENTICAL transform parameters
pair = loader.load(mode="PairMode")[0]
processed_pair = preprocessor.process_pair(pair)

print(f"Pair ID: {processed_pair.pair_id}")
print(f"Is Aligned: {processed_pair.verify_alignment()}")
print(f"Clean size: {processed_pair.clean.processed_size}")
print(f"Edited size: {processed_pair.edited.processed_size}")
```

### Unlabelled Real Screenshot Ingestion
```python
# Real holdout screenshots have no masks (mask=None)
real_loader = DatasetLoader(manifest_path="metadata/phase6_manifest.csv")
real_samples = [s for s in real_loader.load(mode="SingleMode") if not s.is_synthetic]

proc_real = preprocessor.process_sample(real_samples[0])
assert proc_real.mask is None
assert proc_real.mask_pil is None
assert proc_real.image.shape[2] == 3  # Correct 3-channel RGB
```

---

## 5. Command-Line Interface (CLI)

### Run Pipeline Validation
```bash
# Validate 100 synthetic pairs and 50 real holdout screenshots
python member2/phase2_preprocessing/validate_preprocessing.py --pairs 100 --real 50

# Generate PREPROCESSING_VALIDATION_REPORT.md and PREPROCESSING_VALIDATION_SUMMARY.json
python member2/phase2_preprocessing/validate_preprocessing.py --generate-report
```

### Run Unit and Integration Tests
```bash
python -m unittest member2/phase2_preprocessing/test_transforms.py -v
```

---

## 6. Unit & Integration Test Suite Summary

The 16-point test suite (`member2/phase2_preprocessing/test_transforms.py`) verifies all critical functional requirements:

| Test # | Test Name | Verified Invariance | Result |
| :---: | :--- | :--- | :---: |
| 1 | `test_1_native_resolution_dimensions_preserved` | Zero downsampling or canvas distortion | **PASS** |
| 2 | `test_2_rgb_conversion` | Standard RGB preserved intact | **PASS** |
| 3 | `test_3_grayscale_conversion` | Grayscale luminance replicated to 3 channels | **PASS** |
| 4 | `test_4_rgba_conversion_follows_alpha_policy` | Alpha channels composited over `#111111` | **PASS** |
| 5 | `test_5_image_mask_dimensions_remain_aligned` | Canvas width/height matches mask exactly | **PASS** |
| 6 | `test_6_binary_masks_retain_only_background_and_foreground` | NumPy array values strictly $\in \{0, 1\}$ | **PASS** |
| 7 | `test_7_clean_masks_remain_all_black` | Clean masks have zero foreground pixels | **PASS** |
| 8 | `test_8_edited_masks_retain_non_zero_foreground_pixels` | Edited masks retain tampered foreground | **PASS** |
| 9 | `test_9_missing_masks_are_not_silently_replaced` | Missing mask paths raise clear error | **PASS** |
| 10 | `test_10_unlabelled_samples_return_mask_none` | Real unlabelled data returns `mask=None` | **PASS** |
| 11 | `test_11_jpeg_compression_does_not_modify_original_file` | Disk files remain untouched (in-memory only) | **PASS** |
| 12 | `test_12_gaussian_blur_does_not_modify_masks` | Transforms apply to images, never masks | **PASS** |
| 13 | `test_13_random_transformations_reproducible_with_seed` | Identical seeds yield identical arrays | **PASS** |
| 14 | `test_14_pair_processing_preserves_geometric_alignment` | Pair samples receive identical transforms | **PASS** |
| 15 | `test_15_invalid_dimensions_produce_error` | Mismatched image/mask sizes raise exception | **PASS** |
| 16 | `test_16_original_dataset_immutability` | Original dataset files completely unchanged | **PASS** |

**Execution Time:** 16 tests passed in ~1.98 seconds.

---

## 7. Integration Guide for Phase 3 (256 × 256 Tile Extraction)

Phase 3 implements local tile extraction and labeling. The `ProcessedSample` and `ProcessedPair` objects produced by Phase 2 feed directly into Phase 3:

```python
# Phase 3 Conceptual Integration Pattern
from member2.phase2_preprocessing.transforms import Preprocessor, ProcessedSample

def extract_tiles_256(sample: ProcessedSample, tile_size: int = 256, stride: int = 128):
    h, w, _ = sample.image.shape
    tiles = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            image_tile = sample.image[y:y+tile_size, x:x+tile_size]
            mask_tile = sample.mask[y:y+tile_size, x:x+tile_size] if sample.has_mask else None
            is_tampered = bool(np.any(mask_tile > 0)) if mask_tile is not None else None
            tiles.append((image_tile, mask_tile, is_tampered))
    return tiles
```

### Recommendations for Phase 3 Lead
1. **Never resize prior to tiling:** Extract $256 \times 256$ tiles directly from `sample.image` at native resolution.
2. **Handle canvas borders cleanly:** Since screenshot widths ($W \in [279, 738]$) and heights ($H \in [1261, 1601]$) are not exact multiples of 256, use sliding window with stride or reflection/dark-fill padding on the boundary tiles.
3. **Class imbalance in tiles:** Most tiles in an edited screenshot are authentic background. Label tiles based on `np.sum(mask_tile > 0) >= threshold` (e.g. $\ge 16$ tampered pixels) and maintain balanced mini-batches.

