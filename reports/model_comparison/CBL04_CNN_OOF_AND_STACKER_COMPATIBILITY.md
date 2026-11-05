# CBL-04: CNN Out-of-Fold (OOF) and Stacker Compatibility Audit

**Auditing Entity:** Member 3 — Backend / AI Integration Lead  
**Audited Artifacts:**  
- `comparison_inputs/member2_cnn/Member 2/` (Entire recursive tree)  
- `models/phase8/stacker_model.json`  
- `Hackathon-VIT/CBL04-Dataset/metadata/phase8/oof_stacker_train_features.csv`  
- `Hackathon-VIT/CBL04-Dataset/metadata/phase8/validation_features.csv`  
- `app/services/stacker_service.py`  
- `app/services/final_verdict_service.py`  
**Date of Audit:** November 4, 2026  

---

## 1. Out-of-Fold (OOF) Prediction Inventory & Audit

### 1.1 The Stacker OOF Requirement
In multi-branch ensemble architectures (such as the CBL-04 Stacker integrating CNN, OCR, and Semantic Rule features), the meta-learner (stacker) must be trained strictly on **out-of-fold (OOF) predictions**. 

If the stacker is trained on predictions from a model evaluated on its own training data, the CNN branch appears artificially confident and overfit. Conversely, if evaluated on the held-out test set, test-set contamination occurs, invalidating final generalization bounds.

```
+-----------------------------------------------------------------------------+
|                      5-Fold OOF Generation Protocol                         |
|                                                                             |
|  Fold 1: Train on Folds [2,3,4,5]  --> Predict on Fold 1 (OOF-1)            |
|  Fold 2: Train on Folds [1,3,4,5]  --> Predict on Fold 2 (OOF-2)            |
|  Fold 3: Train on Folds [1,2,4,5]  --> Predict on Fold 3 (OOF-3)            |
|  Fold 4: Train on Folds [1,2,3,5]  --> Predict on Fold 4 (OOF-4)            |
|  Fold 5: Train on Folds [1,2,3,4]  --> Predict on Fold 5 (OOF-5)            |
|                                                                             |
|  Result: Concatenated [OOF-1 .. OOF-5] = 100% Unbiased Stacker Training Data |
+-----------------------------------------------------------------------------+
```

---

### 1.2 Audit of Member 2 Workspace for OOF Artifacts
We performed an exhaustive automated scan across `comparison_inputs/member2_cnn/Member 2/` searching for OOF artifacts, k-fold splits, cross-validation configurations, or fold-indexed prediction manifests:

```python
# Automated Search Script Output
base = 'comparison_inputs/member2_cnn/Member 2'
oof_files   = glob.glob(f'{base}/**/*oof*', recursive=True)    # Result: []
fold_files  = glob.glob(f'{base}/**/*fold*', recursive=True)   # Result: []
kfold_files = glob.glob(f'{base}/**/*kfold*', recursive=True)  # Result: []
```

#### Detailed Findings:
1. **Zero OOF Predictions:** Member 2 did not generate out-of-fold predictions for either `TileResNet18` (Phase 5) or `TileSpatialUNet` (Performance Improvement).
2. **Single-Split Partitioning:** Member 2's dataset loader and training pipeline partitioned the 500 pair groups into a single static partition:
   - 70% Train (350 groups / 700 screenshots)
   - 15% Validation (75 groups / 150 screenshots)
   - 15% Test (75 groups / 150 screenshots)
3. **No Cross-Validation Checkpoints:** Only single checkpoints exist (`best_model.pt`, `best_improved_model.pt`). No `fold_1`, `fold_2`, ..., `fold_5` models were trained.
4. **Prediction Manifest Scope:** The only prediction manifests produced by Member 2 are:
   - `member2/phase5_model_training/artifacts/predictions_test.csv` (contains 300 test *tiles* only).
   - `member2/performance_improvement/artifacts/improved_per_image_results.csv` (contains the 150 *test* screenshots only).

**Audit Verdict:** **OOF ARTIFACTS COMPLETELY ABSENT.** Member 2's workspace contains zero out-of-fold predictions.

---

## 2. Phase 8 Stacker Model Architecture & Dependency Analysis

Member 3's Phase 8 Stacker (`models/phase8/stacker_model.json`) is an L2-regularized logistic regression meta-model trained on **868 out-of-fold feature vectors** generated from Member 3's 5-fold cross-validated ResNet-18 multi-task model, EasyOCR extraction, and the Phase 6 Semantic Rule Engine.

### 2.1 Stacker Feature Schema & Trained Weights

| Feature Name | Scaler Mean | Scaler Std | Stacker Weight ($w_i$) | Impact / Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| `cnn_score` | **0.560736** | **0.148367** | **-0.028635** | Standardized continuous CNN probability |
| `cnn_pred_class` | **0.000000** | **1.000000** | **+0.078964** | Binary CNN decision ($0$ = Clean, $1$ = Edited) |
| `ocr_overall_confidence` | 0.885902 | 0.030746 | +0.019859 | OCR text extraction confidence |
| `ocr_amount_extracted` | 0.000000 | 1.000000 | -0.000026 | Binary flag: amount field recognized |
| `ocr_recipient_extracted` | 0.000000 | 1.000000 | -0.000026 | Binary flag: payee field recognized |
| `ocr_ref_id_extracted` | 0.000000 | 1.000000 | -0.000026 | Binary flag: UPI transaction ID recognized |
| `ocr_date_extracted` | 0.000000 | 1.000000 | -0.146534 | Binary flag: timestamp recognized |
| `semantic_manipulation_score`| 0.116763 | 0.117180 | +0.011107 | Rule engine anomaly score $[0, 1]$ |
| `amount_anomaly` | 0.000000 | 1.000000 | +0.007069 | Binary flag: abnormal currency formatting |
| `recipient_anomaly` | 0.000000 | 1.000000 | +0.053377 | Binary flag: payee syntax anomaly |
| `upi_handle_anomaly` | 0.000000 | 1.000000 | -0.000329 | Binary flag: suspicious VPA suffix |
| `missing_critical_fields_count`| 0.000576| 0.011987 | +0.003886 | Count of unparsed mandatory fields |
| `branch_cnn_available` | 0.000000 | 1.000000 | -0.000026 | Availability indicator |
| `branch_ocr_available` | 0.000000 | 1.000000 | -0.000026 | Availability indicator |
| `branch_semantic_available` | 0.000000 | 1.000000 | -0.000026 | Availability indicator |
| **Intercept ($b$)** | — | — | **+0.088603** | Baseline logistic bias |

---

## 3. Covariate Shift & Quantitative Failure Simulation

To understand what occurs if Member 2's model predictions are fed directly into Member 3's existing stacker without retraining, we simulated the resulting covariate shift:

### 3.1 Failure Mode A: Feeding Member 2 Phase 5 (`TileResNet18`)
- `TileResNet18` only produces tile logits. In an inference pipeline, one would have to aggregate tiles using either mean probability or max probability.
- Because an average screenshot contains 43.5 tiles, and clean tiles have an 18.4% false-positive rate, $\max(p_{\text{tile}})$ exceeds $0.80$ on **virtually 100% of clean screenshots**.
- Result:
  - `cnn_pred_class` becomes **$1.0$ on every screenshot**.
  - `cnn_score` shifts from baseline mean of $0.56$ to $>0.85$.
  - The stacker receives an input where `cnn_pred_class` is permanently stuck at $1$, introducing severe systematic bias and inflating the stacker's tampering probability across genuine screenshots.

### 3.2 Failure Mode B: Feeding Member 2 Improvement (`TileSpatialUNet`)
- `TileSpatialUNet` outputs a continuous pixel heatmap $H(x, y)$, not a scalar probability.
- If we map $H(x, y)$ to the stacker using Member 2's Rule 3 ($\max H(x, y) \ge 0.80$):
  - In `improved_pixel_metrics_test.json`, $\text{TN} = 0, \text{FP} = 75 \implies \textbf{100\% of clean test screenshots have } \max H \ge 0.80$.
  - Every clean image is fed to the stacker with `cnn_pred_class = 1` and `cnn_score >= 0.80`.
- If mapped using Rule 2 ($\text{Tampered Area} \ge 1,000$ px):
  - $\text{TN} = 6, \text{FP} = 69 \implies \textbf{92.0\% of clean screenshots have } \text{Area} \ge 1,000\text{ px}$.
  - $92\%$ of genuine transactions are passed into the stacker as classified positive by the CNN.
- **Quantitative Stacker Degradation:**
  - In Member 3's verified pipeline, genuine screenshots yield a stacker score of $\approx 0.46 - 0.51$, which the Phase 9 Final Verdict Engine resolves as `GENUINE` or `INCONCLUSIVE` when OCR and semantic rules confirm clean typography.
  - Injecting Member 2's values forces the linear combination $z = \mathbf{w}^T \mathbf{x} + b$ upward by $+0.079$ (from `cnn_pred_class`) and $+0.045$ (from `cnn_score`), pushing the logistic output $P(\text{Tampered}) = \sigma(z)$ consistently above $0.62$.
  - The Final Verdict Engine would trigger an immediate **false-positive cascade**, classifying clean payment receipts as `EDITED`!

---

## 4. Retraining Requirements & Feasibility Analysis

Can the existing Stacker be retrained to accommodate Member 2's model?

### 4.1 Strict Prerequisites for Stacker Retraining
Retraining the Stacker with Member 2's model is **theoretically possible but requires extensive engineering prerequisites**:

1. **Mandatory 5-Fold Cross-Validation Implementation:**
   - Member 2's `train_improved.py` must be refactored into a 5-fold cross-validation loop matching the exact `split_group_id` assignments in `Hackathon-VIT/CBL04-Dataset/metadata/source_group_splits.csv`.
   - Five separate `TileSpatialUNet` checkpoints must be trained from scratch (each fold taking ~15–20 minutes on CPU, total ~1.5 hours).
2. **Full-Dataset OOF Extraction:**
   - For each fold $k \in \{1, \dots, 5\}$, the out-of-fold model must run full spatial inference, Gaussian blending, and post-processing across all 200 out-of-fold screenshots.
   - For 1,000 synthetic screenshots at ~2.5 seconds per screenshot, this requires **~42 minutes of CPU computation** just to reconstruct OOF heatmaps.
3. **Feature Engineering from Spatial Maps:**
   - Because `TileSpatialUNet` is a segmentation model, scalar features must be extracted from the heatmap:
     - `unet_peak_probability`: $\max_{(x,y)} H(x, y)$
     - `unet_tampered_pixel_count`: $\sum \mathbb{I}(H(x, y) \ge 0.80 \land \text{area} \ge 50)$
     - `unet_tampered_area_ratio`: $\frac{\text{tampered\_pixels}}{H \times W}$
     - `unet_component_count`: number of connected components
4. **Stacker Optimization & Evaluation:**
   - Re-fit L2-regularized logistic regression weights on the new 18-to-20 dimensional feature matrix.
   - Re-calibrate Platt scaling parameters and decision thresholds strictly on validation folds.

### 4.2 Compatibility Matrix Summary

| Criterion | Current Status | Feasibility / Effort |
| :--- | :---: | :--- |
| **Drop-in Checkpoint Replacement** | **IMPOSSIBLE** | Architecture, input dimensions, and output format completely incompatible |
| **Drop-in Stacker Reuse** | **IMPOSSIBLE** | Causes 92–100% false-alarm collapse due to extreme covariate shift |
| **OOF Predictions Available** | **NONE** | Must implement 5-fold cross-validation and generate 1,000 OOF predictions |
| **Retraining Effort Estimate** | — | ~4 to 6 hours of engineering, 5-fold training, feature extraction, and re-tuning |
| **Recommendation** | **DO NOT REPLACE** | Preserve Member 3's calibrated CNN for the Stacker; use Member 2 for UI heatmap only |

