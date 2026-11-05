# Phase 6: Full-Resolution Heatmap Reconstruction & Image-Level Localization

**Project:** CBL-04 — Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Role:** Member 2 — Tile Forensics CNN & Heatmap Lead  
**Module:** `member2/phase6_heatmap_reconstruction/`

---

## 1. Overview & Objective

Phase 6 implements full-resolution tampering probability heatmap reconstruction from Phase 5 tile-level CNN predictions. It bridges tile-level classification and image-level forensic diagnosis by:
1. Mapping 256×256 tile predictions back to their exact native spatial coordinates $(x_k, y_k)$.
2. Blending overlapping predictions (stride 128, $4\times$ spatial coverage) using continuous 2D spatial weighting windows (Gaussian, Hann, or Uniform) to eliminate tile boundary seam artifacts.
3. Optimizing decision thresholds **strictly on the Phase 5 validation set** (with zero test-set peeking).
4. Evaluating pixel-level localization metrics (Pixel Accuracy, Dice coefficient, IoU, Pixel ROC-AUC, PR-AUC) across the held-out test split.
5. Generating multi-panel diagnostic visualizations for synthetic test screenshots and exploratory analysis figures for unlabelled real holdouts.

---

## 2. Directory Layout & Key Modules

```
member2/phase6_heatmap_reconstruction/
├── __init__.py                   # Package exports
├── heatmap_config.json           # Pipeline configuration & threshold parameters
├── reconstructor.py              # Continuous 2D window blending & normalization engine
├── inference.py                  # Batched CNN inference & image-level scoring
├── thresholding.py               # Validation-only threshold optimization engine
├── evaluator.py                  # Pixel-level and image-level test evaluation metrics
├── visualizer.py                 # Multi-panel forensic PNG visualization generator
├── validate_phase6.py            # End-to-end orchestration runner
├── test_heatmap_pipeline.py      # 10 unit & integration test cases
├── HEATMAP_RECONSTRUCTION_REPORT.md # Comprehensive final technical report
├── README.md                     # This documentation file
└── artifacts/                    # Saved metrics, configurations, and visual outputs
    ├── heatmap_config.json       # Active configuration snapshot
    ├── threshold_optimization.json # Validation sweep curves across candidate thresholds
    ├── pixel_metrics_test.json   # Full test split pixel metrics and per-family breakdown
    ├── image_metrics_test.json   # Image-level classification performance
    ├── per_image_results.csv     # Granular per-screenshot forensic evaluation logs
    ├── phase6_summary.json       # Consolidated execution summary
    └── visualizations/           # Diagnostic multi-panel PNGs
        ├── synthetic_*_diagnosis.png    # 6-panel synthetic test diagnostic figures
        └── holdout_*_exploratory.png    # 3-panel unlabelled real holdout figures
```

---

## 3. Mathematical Methodology

### A. 2D Spatial Window Blending
To eliminate artificial boundary seams caused by tiling, each 256×256 patch is weighted by an isotropic 2D Gaussian window:
$$W(u, v) = \exp\left(-\frac{(u - c)^2 + (v - c)^2}{2\sigma^2}\right)$$
where $c = 127.5$ and $\sigma = 64.0$.

For each tile $k$ at canvas position $(x_k, y_k)$ with tampering probability $p_k$:
$$A(y_k + v, x_k + u) \mathrel{+}= p_k \cdot W(u, v)$$
$$\Omega(y_k + v, x_k + u) \mathrel{+}= W(u, v)$$

The continuous full-resolution heatmap is obtained via normalized de-biasing:
$$\text{Heatmap}(y, x) = \frac{A(y, x)}{\max(\Omega(y, x), \epsilon)}$$
Phase 3 boundary anchoring guarantees $\Omega(y, x) > 0$ across 100% of the canvas. Values are strictly constrained to $[0.0, 1.0]$.

### B. Validation-Only Threshold Optimization
- Candidate thresholds $\tau \in [0.10, 0.80]$ were evaluated over 30 validation screenshots (15 clean, 15 edited).
- Optimal threshold $\tau^* = 0.80$ was selected by maximizing validation pixel Dice/F1.
- The test set was evaluated exclusively using this frozen threshold $\tau^* = 0.80$.

---

## 4. Usage Instructions

### Run Phase 6 Unit Tests
```powershell
python -m unittest member2/phase6_heatmap_reconstruction/test_heatmap_pipeline.py -v
```

### Run Full Regression Suite (All Phases 1–6)
```powershell
python -m unittest discover -s member2 -p "test_*.py" -v
```

### Run Phase 6 Heatmap Reconstruction & Evaluation
```powershell
python member2/phase6_heatmap_reconstruction/validate_phase6.py
```

### Run Fast Verification Mode
```powershell
python member2/phase6_heatmap_reconstruction/validate_phase6.py --fast
```

