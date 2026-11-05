# Phase 6 Performance Improvement: Tampered Pixel Detection & Spatial Localization

**Project:** CBL-04 — Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Role:** Member 2 — Tile Forensics CNN & Heatmap Lead  
**Module:** `member2/performance_improvement`  

---

## 1. Executive Summary & Root Cause Analysis

### Baseline Bottleneck (Phases 5 & 6)
In the Phase 5/6 baseline:
1. **Scalar Tile Classification:** The Phase 5 CNN (ResNet-18) was trained as a binary classifier outputting a single scalar probability $p \in [0, 1]$ per 256×256 tile.
2. **False-Positive Pixel Smearing:** When a tile is predicted as tampered ($p \ge \tau$), marking the whole tile or applying Gaussian blending lights up all **65,536 pixels** of the tile. However, in typical Google Pay dark-theme transaction tampering (e.g. editing an amount digit, timestamp, or name), only **50 to 200 pixels** are actually manipulated.
3. **Severe Precision Floor:** This mismatch inherently capped global pixel precision at **0.33%** and global pixel Dice at **0.0054**, generating **587,033 false-positive pixels** across the 150 test screenshots.
4. **Metric Audit Finding:** The reported Phase 6 baseline `mean_image_level_dice` of 0.3818 was mathematically inflated because clean images with zero false alarms receive an empty-set agreement score of $\text{Dice} = 1.0$, masking the near-zero localization performance on tampered images.

### Solution: Direct Spatial Tile Segmentation (`TileSpatialUNet`)
The performance improvement pipeline replaces scalar tile classification with direct pixel-level spatial segmentation:
- **Architecture:** Lightweight 4-stage encoder-decoder `TileSpatialUNet` (~483k parameters) with skip connections predicting an exact $(1, 256, 256)$ pixel probability mask for each tile.
- **Joint Loss Function:** `CombinedBCEDiceLoss` combining weighted Binary Cross-Entropy with differentiable Soft-Dice loss ($0.4 \times \mathcal{L}_{\text{BCE}} + 0.6 \times \mathcal{L}_{\text{Dice}}$) to directly optimize foreground region overlap.
- **Spatial Overlap Blending:** `SpatialHeatmapReconstructor` accumulates the 2D spatial probability maps using a 2D Gaussian window and de-biasing weight normalization to reconstruct seamless native-resolution heatmaps.
- **Connected-Component Noise Filtering:** `HeatmapPostProcessor` applies decision thresholding followed by connected-component area filtering ($\ge A_{\min}$ pixels) to remove isolated noise artifacts while preserving compact text and digit strokes.
- **Strict Validation Tuning:** Optimal decision threshold $\tau^*$ and minimum area $A_{\min}^*$ are tuned jointly and strictly on the validation partition, guaranteeing zero test data contamination.

---

## 2. Directory Structure

```
member2/performance_improvement/
├── artifacts/                                # Output models, metrics, and figures
│   ├── best_improved_model.pt                # Checkpoint of trained TileSpatialUNet
│   ├── validation_tuning_results.json        # Joint (tau, A_min) parameter sweep results
│   ├── improved_pixel_metrics_test.json      # Test evaluation metrics
│   ├── improved_per_image_results.csv        # Per-image forensic results
│   ├── baseline_vs_improved_comparison.json  # Side-by-side metric comparison
│   ├── training_history.json                 # Loss and Dice progression per epoch
│   └── visualizations/                       # 7-panel comparative diagnostic figures
├── config.json                               # Hyperparameter and execution configuration
├── models.py                                 # TileSpatialUNet & CombinedBCEDiceLoss
├── segmentation_dataset.py                   # Paired tile-mask PyTorch dataset
├── spatial_reconstructor.py                  # 2D Gaussian spatial heatmap blender
├── postprocessor.py                          # Connected-component filtering postprocessor
├── train_improved.py                         # Supervised training loop
├── validator_tuning.py                       # Strict validation grid tuner
├── evaluator.py                              # Full-image inferencer & evaluator
├── visualizer.py                             # 7-panel comparative visualizer
├── run_improvement.py                        # Master pipeline execution script
├── test_improvement_pipeline.py              # Comprehensive unit tests
├── PERFORMANCE_IMPROVEMENT_REPORT.md         # Final comprehensive technical report
└── README.md                                 # This documentation file
```

---

## 3. Architecture Specification

### `TileSpatialUNet`
- **Encoder:**
  - Stage 1: Conv2d(3 -> 16, 3x3) -> BatchNorm -> LeakyReLU -> Conv2d(16 -> 16, 3x3) -> BatchNorm -> LeakyReLU
  - MaxPool2d(2, 2): 256×256 -> 128×128
  - Stage 2: ConvBlock(16 -> 32)
  - MaxPool2d(2, 2): 128×128 -> 64×64
  - Stage 3: ConvBlock(32 -> 64)
  - MaxPool2d(2, 2): 64×64 -> 32×32
- **Bottleneck:**
  - ConvBlock(64 -> 128)
- **Decoder:**
  - UpStage 3: ConvTranspose2d(128 -> 64, 2x2) + Skip(Stage 3: 64) -> ConvBlock(128 -> 64)
  - UpStage 2: ConvTranspose2d(64 -> 32, 2x2) + Skip(Stage 2: 32) -> ConvBlock(64 -> 32)
  - UpStage 1: ConvTranspose2d(32 -> 16, 2x2) + Skip(Stage 1: 16) -> ConvBlock(32 -> 16)
- **Projection Head:**
  - Conv2d(16 -> 1, 1x1) producing raw spatial logits map $(1, 256, 256)$.
- **Parameter Count:** ~483,000 parameters.

---

## 4. Execution Commands

### Run Unit Tests
```bash
python -m unittest member2/performance_improvement/test_improvement_pipeline.py -v
```

### Run Full Regression Suite (Phases 1-6 + Improvement)
```bash
python -m unittest discover -s member2 -p "test_*.py" -v
```

### Execute Improvement Pipeline
```bash
python member2/performance_improvement/run_improvement.py --config member2/performance_improvement/config.json
```

