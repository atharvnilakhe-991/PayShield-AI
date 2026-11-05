# CBL-04: Corrected Phase 7 Evaluation & Forensic Localization Report (Post-Audit Revision)

**Project:** CBL-04 — Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Role:** Member 2 — Tile Forensics CNN & Heatmap Lead  
**Audit Status:** AUDITED & CORRECTED UNDER PHASE 7.1  
**Date:** November 2026  

---

## 1. Executive Summary

This corrected report presents the audited technical evaluation of Member 2's components for CBL-04:
1. **Phase 5 ResNet-18 Tile Classifier Baseline** (operating on isolated 256×256 tiles).
2. **Phase 6 Heatmap Reconstruction Baseline** (continuous 2D Gaussian window blending of tile classifier scores).
3. **Improved `TileSpatialUNet` Model** (direct pixel-level spatial segmentation with connected-component noise filtering).

### Core Verified Findings
- **Substantial Pixel Localization Improvement:** Compared to the Phase 6 reconstruction baseline, the improved `TileSpatialUNet` increased Global Pixel Dice by **32.1×** (from $0.005438$ to $0.174612$) and Global Pixel IoU by **35.1×** (from $0.002726$ to $0.095657$).
- **Reduced False Positive Footprint:** Global pixel precision rose from $0.334\%$ to **$12.082\%$**, eliminating $277,368$ false-positive pixels across the $150$ test screenshots (a $47.25\%$ noise reduction). Simultaneously, true-positive tampered pixel detection increased from $1,969$ to $42,554$ pixels.
- **Mean Tampered Image Dice:** On the $75$ tampered screenshots, the model achieves a mean Dice of **$21.16\%$** ($0.211559$), establishing true spatial localization of modified transaction text.
- **Screenshot-Level Screening Limitations:** Due to false-alarm accumulation across large ~$1$-megapixel canvases, the model cannot function as a standalone binary screenshot classifier. Under the exploratory $\ge 1,000$-pixel cutoff rule, screenshot specificity is only **$8.00\%$** (6/75 clean images correctly classified), with overall classification accuracy of **$50.67\%$**. Under binary existence rules, specificity drops to **$0.00\%$**.
- **Robustness Boundary:** Controlled evaluations show stability under mild JPEG compression ($Q \ge 85$) and minor lighting variations, but significant recall collapse under heavy downsampling ($50\%$ scale: recall drops to $15.36\%$) and strong blur ($\sigma = 1.5$: recall drops to $9.50\%$).
- **Not Production-Ready:** The model serves as an experimental forensic localization tool for highlighting candidate regions, but requires human-in-the-loop inspection or secondary verification.

---

## 2. Dataset & Frozen Split Verification

All evaluations were verified against the **frozen Phase 5 test split**:
- **Screenshots:** $150$ total synthetic dark-theme Google Pay screenshots ($75$ clean, $75$ tampered).
- **Pixel Count:** $126,270,016$ pixels evaluated.
- **Leakage Audit:** $0$ group leakage across train, validation, and test splits. Clean and edited pairs share identical `split_group_id` and remain strictly co-located.
- **Real Holdouts:** $387$ unlabelled real screenshots remain segregated and unlabelled.

---

## 3. Audited Screenshot-Level Classification Results

Evaluated across the 150 test screenshots (75 clean, 75 tampered):

| Decision Rule Strategy | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | PR-AUC | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule 1: Any Component ($>0\text{ px}$)** | 50.00% | 50.00% | 100.00% | **0.00%** | 0.6667 | 0.5588 | 75 | 0 | 75 | 0 |
| **Rule 2: Area Threshold ($\ge 1,000\text{ px}$)** | **50.67%** | **50.36%** | **93.33%** | **8.00%** | **0.6542** | 0.5588 | 70 | 6 | 69 | 5 |
| **Rule 3: Peak Prob ($\ge 0.80$)** | 50.00% | 50.00% | 100.00% | **0.00%** | 0.6667 | 0.5211 | 75 | 0 | 75 | 0 |

### Critical Technical Discussion
Rule 1 and Rule 3 fail completely as screening tools because virtually every dark-theme screenshot (clean or edited) exhibits small typographic edge clusters crossing the $0.80$ probability or $50$-pixel threshold, resulting in $0.00\%$ specificity. Rule 2, an exploratory heuristic threshold of $1,000$ pixels, marginally raises specificity to $8.00\%$, but is still inadequate for autonomous classification. The system must be operated as a localized heatmap overlay for human review.

---

## 4. Audited Pixel-Level Localization Results

Evaluated across all $126.3$ million test pixels:

| Metric | Phase 6 Baseline | Phase 7 Improved Model | Absolute Change | Factor Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Global Pixel Accuracy** | 99.43% | **99.68%** | +0.25% | Consistent high background specificity |
| **Global Pixel Precision** | 0.334% | **12.082%** | **+11.748%** | **36.1× Improvement** |
| **Global Pixel Recall** | 1.456% | **31.476%** | **+30.020%** | **21.6× Improvement** |
| **Global Pixel F1-Score** | 0.0054 | **0.1746** | **+0.1692** | **32.1× Improvement** |
| **Global Pixel Dice** | 0.005438 | **0.174612** | **+0.169174** | **32.1× Improvement** |
| **Global Pixel IoU** | 0.002726 | **0.095657** | **+0.092931** | **35.1× Improvement** |
| **Pixel ROC-AUC** | 0.8010 | **0.9808** | **+0.1798** | High pixel separation |
| **Pixel PR-AUC** | 0.0034 | **0.0894** | **+0.0860** | **26.5× Improvement** |
| **True Positive Pixels (TP)** | 1,969 px | **42,554 px** | **+40,585 px** | **21.6× Increase in Detection** |
| **False Positive Pixels (FP)** | 587,033 px | **309,665 px** | **-277,368 px** | **47.25% Noise Cut** |
| **False Negative Pixels (FN)**| 133,225 px | **92,640 px** | **-40,585 px** | **30.46% Miss Cut** |
| **Mean Tampered Image Dice** | Near 0.00 | **0.211559 (21.16%)**| **+0.211559** | True Edit Overlap |
| **Mean Tampered Image IoU** | Near 0.00 | **0.133788 (13.38%)**| **+0.133788** | True Edit IoU |
| **Clean Total FP Pixels** | 314,200 px | **166,467 px** | **-147,733 px** | 47.0% Clean Noise Cut |
| **Clean Mean FP Pixels / Image** | 4,189.3 px | **2,219.6 px** | **-1,969.7 px** | 47.0% Per-Image Cut |

---

## 5. Audited Confusion Matrices

### 5.1 Screenshot Classification Matrix (Rule 2: $\ge 1,000\text{ px}$)
$$\begin{pmatrix} \text{TN} = 6 & \text{FP} = 69 \\ \text{FN} = 5 & \text{TP} = 70 \end{pmatrix}$$
- Sensitivity (Recall): $93.33\%$
- Specificity: $8.00\%$
- Precision: $50.36\%$

### 5.2 Pixel Localization Matrix
$$\begin{pmatrix} \text{TN} = 125,825,157 & \text{FP} = 309,665 \\ \text{FN} = 92,640 & \text{TP} = 42,554 \end{pmatrix}$$
- Pixel Specificity: $99.754\%$
- Pixel Recall: $31.476\%$
- Pixel Precision: $12.082\%$

---

## 6. Robustness Evaluation & Reconciliation of Dice Discrepancy

### 6.1 Reconciliation of the Dice Discrepancy
- **Main Evaluation Test Set Dice:** **$0.174612$** ($150$ screenshots, $126.3$ million pixels).
- **Robustness Baseline Table Dice:** **$0.175582$** (reported as $0.1756$).
- **Reconciliation Cause:** The difference of $0.00097$ is strictly due to sample size. To maintain tractable CPU runtime across 14 conditions, the robustness suite evaluated a balanced **10-screenshot subset (5 clean, 5 edited)**, where $\text{TP}=1,948, \text{FP}=14,949, \text{FN}=3,344 \implies \text{Dice} = 0.175582$. The exact same weights, thresholds, and code were used.

### 6.2 14-Condition Robustness Summary (10-Screenshot Subset)
- **JPEG Compression:** Resilient at $Q=95$ (Dice $0.1754$, $99.9\%$ retention), $Q=85$ (Dice $0.1740$, $99.1\%$ retention), and $Q=70$ (Dice $0.1717$, $97.8\%$ retention).
- **Downsampling / Upsampling:** Moderate drop at $75\%$ scale (Dice $0.1546$, recall drops to $26.5\%$). Sharp recall collapse at $50\%$ scale (recall drops to $15.36\%$, screenshot recall drops to $0.00\%$ because detected pixels fall below $1,000$).
- **Gaussian Blur:** $\sigma=0.5$ is stable (Dice $0.1664$). $\sigma=1.0$ reduces background noise by $55\%$, raising Dice to $0.1832$ while reducing recall to $22.9\%$. $\sigma=1.5$ obliterates fine character strokes, dropping recall to $9.50\%$.
- **Photometric Jitter:** Stable under $\pm 10\%$ brightness (Dice $0.1735 - 0.1785$) and contrast (Dice $0.1724 - 0.1734$).

---

## 7. Tri-Model Baseline Comparison Matrix

| Metric | Phase 5 ResNet-18 Baseline | Phase 6 Heatmap Reconstruction | Phase 7 Improved Model (`TileSpatialUNet`) |
| :--- | :---: | :---: | :---: |
| **Model Type** | ResNet-18 Classifier | ResNet-18 + Gaussian Blending | Direct `TileSpatialUNet` (483k params) |
| **Parameters** | 11.2M | 11.2M | **483k (23× smaller)** |
| **Loss** | Weighted Cross-Entropy | Weighted Cross-Entropy | **Combined BCE + Soft-Dice** |
| **Global Pixel Accuracy** | N/A (Tile-level only) | 99.43% | **99.68%** |
| **Global Pixel Precision** | 12.07% (Tile-level) | 0.334% (Pixel-level) | **12.082% (Pixel-level)** |
| **Global Pixel Recall** | 30.43% (Tile-level) | 1.456% (Pixel-level) | **31.476% (Pixel-level)** |
| **Global Pixel Dice** | N/A (Undefined on tiles) | 0.005438 | **0.174612 (+32.1×)** |
| **Global Pixel IoU** | N/A (Undefined on tiles) | 0.002726 | **0.095657 (+35.1×)** |
| **False-Positive Pixels** | N/A | 587,033 px | **309,665 px (-47.25%)** |
| **True-Positive Pixels** | N/A | 1,969 px | **42,554 px (+21.6×)** |
| **Mean Tampered Image Dice** | N/A | Near 0.00 | **0.211559 (21.16%)** |

---

## 8. Limitations & Unresolved Issues

1. **Typographic False Alarms on Clean Screens:** Residual false alarms (~2,200 pixels/image) persist along high-contrast text and dividers on clean dark-theme screens.
2. **Sub-50px Single Digit Misses:** The $A_{\min} = 50\text{ px}$ threshold filters out isolated single-digit timestamp alterations smaller than 50 pixels.
3. **Low Screenshot-Level Specificity:** Standalone binary classification of full screenshots cannot be reliably performed with current area thresholds ($8\%$ specificity).

