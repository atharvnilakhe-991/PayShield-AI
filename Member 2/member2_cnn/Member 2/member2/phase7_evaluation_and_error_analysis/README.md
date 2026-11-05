# Phase 7: Evaluation and Error Analysis

**Project:** CBL-04 — Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Role:** Member 2 — Tile Forensics CNN & Heatmap Lead  
**Module:** `member2/phase7_evaluation_and_error_analysis`  

---

## 1. Overview & Objectives

Phase 7 executes a rigorous, publication-grade evaluation and forensic error analysis across the entire project pipeline:
1. **Screenshot-Level Classification:** Evaluates tampering detection on 150 held-out screenshots (75 clean, 75 edited) under explicit, documented decision rules (Area Threshold, Any Foreground Component, Peak Probability).
2. **Pixel-Level Tampering Localization:** Assesses spatial localization fidelity against pixel-accurate ground-truth masks, reporting global metrics and separate clean vs. tampered image metrics.
3. **Confusion Matrices:** Generates and plots separate, publication-quality confusion matrices for screenshot classification and pixel localization.
4. **Forensic Error Analysis:** Identifies and visually overlays representative failure and success cases (False Positives, False Negatives, Excessive Noise, Poor Localization, Successful Localization).
5. **Multi-Condition Robustness Evaluation:** Evaluates the improved model under 14 realistic perturbations (JPEG compression, downsampling/upsampling, Gaussian blur, brightness variations, contrast variations) without retraining.
6. **Tri-Model Baseline Comparison:** Cross-evaluates the Phase 5 ResNet-18 baseline, Phase 6 Heatmap Reconstruction baseline, and Phase 6 Improvement (`TileSpatialUNet`).

---

## 2. Directory Structure

```
member2/phase7_evaluation_and_error_analysis/
├── artifacts/                                # Output evaluation artifacts
│   ├── phase7_summary.json                   # Complete evaluation summary bundle
│   ├── phase7_evaluation_metrics.json        # Screenshot and pixel metrics
│   ├── phase7_per_image_results.csv          # Per-screenshot forensic metrics
│   ├── phase7_robustness_results.json        # Robustness curves across 14 conditions
│   ├── phase7_baseline_comparison.json       # Side-by-side tri-model comparison
│   ├── screenshot_confusion_matrix.png       # Plotted screenshot confusion matrix
│   ├── pixel_confusion_matrix.png            # Plotted pixel confusion matrix
│   └── visualizations/                       # 5-panel forensic overlay figures
├── config.json                               # Evaluation & robustness configuration
├── evaluator_phase7.py                       # Classification & localization evaluator
├── error_analyzer.py                         # 5-panel overlay generator & case selector
├── robustness_evaluator.py                   # Controlled in-memory perturbation suite
├── baseline_comparator.py                    # Tri-model cross-phase comparator
├── run_phase7.py                             # Master pipeline execution script
├── test_phase7_evaluation.py                 # Unit & regression tests for Phase 7
├── PHASE7_EVALUATION_REPORT.md               # Final comprehensive technical report
└── README.md                                 # This documentation file
```

---

## 3. Execution Commands

### Run Phase 7 Unit Tests
```bash
python -m unittest member2/phase7_evaluation_and_error_analysis/test_phase7_evaluation.py -v
```

### Run Full Project Regression Suite (Phases 1-7)
```bash
python -m unittest discover -s member2 -p "test_*.py" -v
```

### Execute Master Phase 7 Evaluation Pipeline
```bash
python member2/phase7_evaluation_and_error_analysis/run_phase7.py --config member2/phase7_evaluation_and_error_analysis/config.json
```

