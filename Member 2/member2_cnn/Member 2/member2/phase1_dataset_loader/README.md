# CBL-04: Phase 1 — Dataset Loader & Manifest Verification Engine
**Role:** Member 2 – Tile Forensics CNN & Heatmap Lead  
**Module Directory:** `member2/phase1_dataset_loader/`  
**Target Repository:** `Hackathon-VIT/CBL04-Dataset/`  
**Status:** IMPLEMENTED, TESTED & VERIFIED  

---

## 1. Overview

This module provides a production-grade, read-only Python dataset loader and manifest verification engine tailored specifically for the **CBL-04** Google Pay dark-theme transaction tampering detection project.

### Core Capabilities
- **Strict Read-Only Enforcement:** Operates in a zero-mutation mode, guaranteeing no original dataset images, masks, or manifests are modified.
- **Robust Path Resolution:** Automatically resolves relative and absolute paths against the dataset root across different execution directories.
- **Dual Operating Modes:**
  - `PairMode`: Groups matched clean and edited screenshots belonging to the same transaction into a unified `PairSample` record, along with their respective ground truth masks and bounding boxes.
  - `SingleMode`: Loads individual samples as `SingleSample` records, preserving ground truth labels (`0`: clean, `1`: edited, `None`: unlabelled).
- **Leakage-Safe Grouping:** Exposes `loader.get_groups()` to map `pair_id` / `source_id` to image identifiers, ensuring clean and edited counterparts of the same pair are never separated across cross-validation or train/test splits.
- **Lazy Memory Loading:** Metadata is loaded immediately into memory, while image pixel arrays and mask arrays are loaded on-demand via `.load_image()` and `.load_mask()`.
- **GPay Scope Protection:** Preserves the Google Pay dark-theme boundary and flags any non-GPay or unverified presentation.
- **Explicit Unlabelled Data Handling:** Preserves raw real screenshots as `unlabeled` (`label=None`) without inferring artificial labels.

---

## 2. Directory Structure

```text
member2/phase1_dataset_loader/
├── dataset_loader.py               # Reusable DatasetLoader, SingleSample, and PairSample classes
├── validate_dataset.py             # Command-line validation tool
├── test_dataset_loader.py          # Unit & integration test suite (10 test cases)
├── DATASET_VALIDATION_REPORT.md    # Human-readable validation report
├── DATASET_VALIDATION_SUMMARY.json # Machine-readable validation counts
└── README.md                       # Documentation and usage instructions (this file)
```

---

## 3. Quick Start & Python API Usage

### A. Initializing the Loader

```python
from pathlib import Path
from dataset_loader import DatasetLoader

# Auto-detects dataset root (e.g. Hackathon-VIT/CBL04-Dataset) and defaults to metadata/phase4_manifest.csv
loader = DatasetLoader()

# Or specify explicitly:
loader = DatasetLoader(
    dataset_root="Hackathon-VIT/CBL04-Dataset",
    manifest_path="metadata/phase4_manifest.csv"
)
```

### B. Loading in `PairMode` (Recommended for Paired Analysis)

```python
pairs = loader.load(mode="PairMode")
print(f"Loaded {len(pairs)} verified clean-edited pairs.")

p = pairs[0]
print(f"Pair ID:        {p.pair_id}")
print(f"Layout Family:  {p.layout_family}")
print(f"Edit Category:  {p.edit_category}")
print(f"Bounding Boxes: {p.bounding_boxes}")

# Load PIL images lazily on demand
clean_img = p.load_clean_image()
edited_img = p.load_edited_image()
clean_mask = p.load_clean_mask()   # Returns all-black mask
edited_mask = p.load_edited_mask() # Returns binary mask (0 and 255)

# Or load all four simultaneously:
c_img, e_img, c_mask, e_mask = p.load_all()
```

### C. Loading in `SingleMode` (Standard Supervised Training)

```python
samples = loader.load(mode="SingleMode")
print(f"Loaded {len(samples)} single samples.")

s = samples[0]
print(f"Image ID:    {s.image_id}")
print(f"Label:       {s.label}")      # 0 (clean), 1 (edited), or None (unlabeled)
print(f"Label Name:  {s.label_name}") # 'clean', 'edited', or 'unlabeled'

img = s.load_image()
mask = s.load_mask() # None if clean/real with no mask path
```

### D. Leakage-Safe Split Grouping

```python
# Returns dictionary: {pair_id: [image_id_clean, image_id_edited]}
groups = loader.get_groups()

# Feed groups to sklearn GroupKFold or GroupShuffleSplit:
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
# gkf.split(X, y, groups=[s.pair_id for s in samples])
```

---

## 4. Running Validation from the Command Line

To run the forensic validation tool and generate markdown/JSON reports:

```bash
python member2/phase1_dataset_loader/validate_dataset.py
```

### Optional Command-Line Flags
- `--dataset-root PATH`: Specify a custom dataset root folder.
- `--manifest PATH`: Specify an alternate manifest (e.g. `metadata/phase6_manifest.csv`).
- `--output-dir PATH`: Directory where validation reports are written (defaults to current module folder).
- `--skip-mask-values`: Skip pixel-by-pixel mask validation for faster execution.
- `--max-samples N`: Limit validation to the first $N$ samples.

### Example Console Output
```text
============================================================
CBL-04 DATASET VALIDATION SUMMARY
============================================================
Manifest:       phase4_manifest.csv
Status:         PASSED
Manifest Rows:  1000
Unique Images:  1000
Clean Samples:  500
Edited Samples: 500
Unlabeled:      0
Verified Pairs: 500
Broken Pairs:   0
Missing Images: 0
Missing Masks:  0
Errors:         0
Warnings:       0
============================================================
```

---

## 5. Automated Test Suite

A comprehensive test suite is included in `test_dataset_loader.py`. It creates small isolated temporary test fixtures outside the original dataset to verify 8 negative and edge cases, plus 2 full integration tests:

1. `test_1_valid_clean_edited_pair`: Complete clean/edited pair with valid masks.
2. `test_2_missing_image`: Missing screenshot file flagged as fatal error.
3. `test_3_missing_required_mask`: Missing required mask for edited image flagged as fatal error.
4. `test_4_missing_pair_id`: Missing `pair_id` / `source_id` flags warning and skips pairing.
5. `test_5_duplicate_manifest_row`: Duplicate manifest rows detected and reported.
6. `test_6_unlabelled_sample`: Unlabelled real screenshot handled explicitly with `label=None`.
7. `test_7_corrupt_image`: Truncated/garbage image file rejected with decoding error.
8. `test_8_dimension_mismatch`: Mismatch between screenshot and mask dimensions flagged as fatal error.
9. `test_phase4_bulk_manifest`: End-to-end integration test on actual Phase 4 bulk manifest (1,000 rows, 500 pairs).
10. `test_phase6_combined_manifest`: End-to-end integration test on Phase 6 manifest (1,387 rows, 500 pairs, 387 real holdout).

### Running the Tests
```bash
python -m unittest member2/phase1_dataset_loader/test_dataset_loader.py -v
```

All 10 tests execute and pass in $\approx 5.8$ seconds.

---

## 6. Verification Status for Phase 2 Handover

- **Manifest Schema:** Verified. `phase4_manifest.csv` columns (`image_id`, `source_id`, `image_path`, `label`, `mask_path`, `bounding_box`, etc.) are fully mapped and parsed.
- **Pair Relationships:** Verified. 500/500 pairs are complete with zero broken pairs.
- **Image Integrity:** Verified. All 1,000 Phase 4 images and 1,000 masks decode cleanly.
- **Mask Alignment:** Verified. Clean masks are all-black; edited masks are binary and aligned with image differences.
- **Phase 2 Status:** **READY TO PROCEED** with high-resolution sliding-window tile extraction ($256 \times 256$, stride 128).

