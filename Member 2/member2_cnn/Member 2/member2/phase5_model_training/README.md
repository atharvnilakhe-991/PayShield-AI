# Phase 5: Baseline Reproduction & Tile Forensics CNN Training

**Project:** CBL-04 — Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Role:** Member 2 — Tile Forensics CNN & Heatmap Lead  
**Module:** `member2/phase5_model_training/`

---

## 1. Overview & Objective

Phase 5 implements a native-resolution tile-based convolutional neural network (CNN) for detecting manipulated regions in Google Pay dark-theme transaction screenshots. 

The previous global-downsampling baseline developed by Member 1 achieved approximately **50.5% accuracy** (effectively random guessing) because resizing full mobile screenshots (e.g. 1080×2400) to 160×320 destroyed fine text anti-aliasing, compression grid structures, and splice micro-artifacts. Phase 5 evaluates whether operating on **256×256 native-resolution RGB tiles** restores discriminative signal and provides the foundation for pixel-level heatmap reconstruction in Phase 6.

---

## 2. Key Architecture & Design Decisions

### A. Leakage-Safe Group-Aware Dataset Splitting
- **Unit of Partition:** Splits are performed strictly at the transaction-pair level using `split_group_id` (`pair_0001` through `pair_0500`).
- **Proportions:** 70% Training (350 pairs / 700 images), 15% Validation (75 pairs / 150 images), 15% Testing (75 pairs / 150 images).
- **Zero Leakage:** Both the clean and edited screenshots of any transaction pair are guaranteed to reside in the exact same split. Individual tiles from the same screenshot or pair are never split across train/val/test sets.
- **Stratification:** Split assignments are stratified by layout family (`compact`, `enriched`, `bank_progress`, `recharge`) to preserve balanced distribution.
- **Holdout Segregation:** 387 unlabelled real Google Pay screenshots are strictly segregated into a holdout partition and completely excluded from supervised training.

### B. Forensic Filtering & Class Imbalance Mitigation
- **Exclusion of Ambiguous Tiles:** Tiles containing 1–15 foreground tampered pixels (`tile_label == -1`) are strictly excluded from training and validation to prevent label noise.
- **Class Imbalance:** Clean-to-tampered ratio is approximately 11.69 : 1 (92.1% clean, 7.9% tampered).
- **Inverse Frequency Loss Weighting:** Training uses class-weighted `CrossEntropyLoss` with weights computed strictly on the training split:
  $$w_c = \frac{N}{2 \cdot N_c}$$
- **Optional Weighted Sampler:** `WeightedRandomSampler` is also supported for balanced mini-batch extraction.

### C. CNN Architectures
1. **`TileResNet18`:** Industry-standard ResNet-18 backbone modified with a 2-class classification head (`Linear(512, 2)`) regularized by dropout ($p=0.2$).
2. **`SimpleTileCNN`:** Custom 4-stage convolutional neural network (~800k parameters) optimized for high-throughput CPU training and benchmarking.

### D. LRU In-Memory Image Caching
- To avoid redundant disk reads when extracting multiple tiles from the same source screenshot, `TileForensicsDataset` incorporates a Least-Recently-Used (LRU) memory cache, achieving sub-millisecond tile loading during training.

---

## 3. Directory Layout

```
member2/phase5_model_training/
├── __init__.py                   # Package exports
├── training_config.json          # Hyperparameters and pipeline configuration
├── split_dataset.py              # Leakage-safe group splitter & manifest exporter
├── model.py                      # TileResNet18 & SimpleTileCNN architectures
├── training_dataset.py           # PyTorch Dataset, LRU caching, and class weights
├── train.py                      # Training loop with AdamW, CosineAnnealing, & early stopping
├── evaluate.py                   # Comprehensive evaluation, ROC-AUC, PR-AUC, per-family breakdown
├── validate_training.py          # End-to-end orchestration runner
├── test_training_pipeline.py     # 10 unit & integration test cases
├── TRAINING_REPORT.md            # Comprehensive experimental report & baseline comparison
├── README.md                     # This documentation file
└── artifacts/                    # Saved weights, logs, manifests, and diagnostic plots
    ├── best_model.pt             # Best checkpoint weights & metadata
    ├── last_model.pt             # Final epoch checkpoint
    ├── training_history.json     # Per-epoch loss, accuracy, and F1 curves
    ├── loss_f1_curve.png         # Publication-grade loss & F1 visualization
    ├── split_manifest.json       # Complete group & sample split allocations
    ├── split_manifest.csv        # Tabular split assignments
    ├── metrics_test.json         # Held-out test split evaluation metrics
    ├── confusion_matrix.json     # Raw 2x2 confusion matrix counts
    ├── confusion_matrix.png      # Confusion matrix heatmap
    └── baseline_vs_tile_comparison.json # Comparative analysis against Phase 7 baseline
```

---

## 4. Usage Instructions

### Run Comprehensive Unit & Integration Tests
```powershell
python -m unittest member2/phase5_model_training/test_training_pipeline.py -v
```

### Run Full Training & Evaluation Pipeline
```powershell
python member2/phase5_model_training/validate_training.py
```

### Run Fast Verification Mode (Reduced Epochs & Subset)
```powershell
python member2/phase5_model_training/validate_training.py --fast
```

---

## 5. Phase 6 Handover Notes
- The saved model checkpoint `artifacts/best_model.pt` outputs calibrated tile-level binary logits `(B, 2)`.
- In Phase 6 (Heatmap Reconstruction), these logits will be processed via softmax to produce continuous tampering confidence scores $P(\text{tampered})$.
- Using the Phase 3 stride-128 grid metadata `(tile_x, tile_y)`, the overlapping tile probabilities will be blended using a 2D Gaussian or linear distance-weighted averaging kernel to reconstruct continuous, full-resolution pixel tampering heatmaps.

