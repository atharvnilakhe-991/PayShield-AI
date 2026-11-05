# CBL-04: Member 2 vs Member 3 Complete CNN Comparative Audit

**Auditing Entity:** Member 3 — Backend / AI Integration Lead  
**Audited Artifacts:** Member 2's imported project (`comparison_inputs/member2_cnn/Member 2/`)  
**Baseline Artifacts:** Member 3's existing ResNet-18 multi-task pipeline & Phase 9 candidates  
**Workspace:** `/Users/heemanshupjain/Downloads/Member 3`  
**Date of Audit:** November 4, 2026  
**Audit Protocol:** Strictly Read-Only, Cryptographically Verified, Empirical Cross-Examination  

---

## 1. Executive Summary & Synthesis

This report synthesizes the technical, empirical, and architectural audit comparing Member 2's CNN forensic models against Member 3's existing ResNet-18 multi-task baseline and Phase 9 models.

The primary objective was to determine:
1. Whether Member 2's model genuinely improves classification performance, particularly false-positive reduction on clean Google Pay dark-theme transaction screenshots.
2. Whether data leakage, split contamination, or evaluation artifacts occurred in Member 2's reported results.
3. Whether Member 2's models can be plugged into Member 3's Phase 8 OOF Stacker and Phase 9 Final Verdict Engine.
4. Whether Member 2's model is suitable for production integration into the live FastAPI application (`app/`).

### Core Audit Verdicts

| Dimension | Member 3 Baseline (ResNet-18 Multi-Task) | Member 2 Phase 5 (`TileResNet18`) | Member 2 Improvement (`TileSpatialUNet`) | Comparative Determination |
| :--- | :---: | :---: | :---: | :---: |
| **Model Scope** | Global Screenshot Classifier + Coarse Mask | 256×256 Native Tile Binary Classifier | Native Tile Spatial Segmentation U-Net | Completely different modeling objectives |
| **Clean Screenshot FPR** | **56.0%** (Baseline) / **42.0%** (Phase 9 Exp2) | **Not Evaluated** on whole screenshots | **92.0% – 100.0%** (Audited) | **Member 2 WORSE on clean FPR** (near-total false alarm) |
| **Screenshot Accuracy** | **54.2%** (Baseline) / **66.2%** (Phase 9 Exp2) | **77.67%** (on *tiles* only, 92.3% clean) | **50.0% – 50.67%** (Screenshots) | Member 2 Phase 5 77.7% is a **tile metric**, not image accuracy |
| **Pixel Localization IoU** | **0.059** (Truthfully Suppressed in UI) | **0.0027** (Phase 6 Scalar Blending) | **0.0957** (Audited Global Pixel IoU) | **Member 2 TileSpatialUNet SUPERIOR (+35.1×)** |
| **Pixel Localization Dice** | **~0.11** | **0.0054** | **0.1746** (Audited Global Pixel Dice) | **Member 2 TileSpatialUNet SUPERIOR (+32.1×)** |
| **Image Inference Latency** | **~15 ms** (1 forward pass, 320×160) | **~850 ms** (40–60 tiles, 256×256) | **~1,800 – 3,200 ms** (40–60 tiles + U-Net + blending) | Member 3 is **120× faster** on CPU |
| **OOF Stacker Compatibility** | **100% Native** (5-Fold OOF Generated) | **0% Compatible** (No OOF files exist) | **0% Compatible** (No OOF files exist) | Cannot be used in existing Stacker without retraining |
| **Production Classification** | **USABLE IN ENSEMBLE** | **NOT VIABLE** (Tile-only) | **NOT SUITABLE** (FPR 92–100%) | **Member 3 CNN retained for classification** |
| **Production Localization** | **SUPPRESSED** (IoU 0.059 below 0.15 threshold) | **NOT VIABLE** (Dice 0.0054) | **CONDITIONALLY READY** (Assistive overlay) | **Member 2 TileSpatialUNet viable for heatmap only** |

---

## 2. High-Level Findings & Crucial Nuances

### 2.1 The "77.67% Accuracy" Misconception
Member 2's Phase 5 report (`member2/phase5_model_training/TRAINING_REPORT.md`) claimed:
> *"Achieved 77.67% overall accuracy and an ROC-AUC of 0.7288 on the held-out test split, outperforming the downsampled baseline by +27.2 percentage points."*

Our audit of `member2/phase5_model_training/artifacts/metrics_test.json` and `predictions_test.csv` revealed a fundamental nuance:
- The 300 test samples evaluated in Phase 5 were **256×256 TILES**, **NOT FULL SCREENSHOTS**.
- Because transaction tampering only affects small localized areas (e.g. an amount digit or date), **92.3% of tiles across transaction screenshots are completely clean** (277 clean tiles vs. 23 tampered tiles in the test batch).
- On these 300 tiles, the model achieved 226 true negatives and only 7 true positives, while producing 51 false positives (Clean Tile FPR = 18.4%) and 16 false negatives (Tampered Tile Miss Rate = 69.6%).
- Tile-level precision was only **12.07%** and Tile-level F1 was only **0.1728** (compared to Member 1/3's baseline F1 of 0.5243).
- **Conclusion:** Member 2's Phase 5 accuracy of 77.67% was heavily driven by majority-class agreement on clean background tiles, not superior whole-screenshot classification.

### 2.2 The Image-Level Specificity Collapse in Member 2's Model
When Member 2 attempted to reconstruct whole-screenshot decisions from tiles (Phase 6 and Performance Improvement), the spatial accumulation of false-positive tile noise led to an **almost complete collapse of image-level specificity**:
- In `corrected_phase7_report.md` (audited across 150 test screenshots: 75 clean, 75 tampered):
  - **Rule 1 (Any positive pixel $>0$ px):** $\text{TN} = 0, \text{FP} = 75 \implies \textbf{FPR = 100.0\%, Specificity = 0.0\%}$.
  - **Rule 2 (Tampered area $\ge 1,000$ px):** $\text{TN} = 6, \text{FP} = 69, \text{TP} = 70, \text{FN} = 5 \implies \textbf{FPR = 92.0\%, Specificity = 8.0\%}$.
  - **Rule 3 (Peak pixel probability $\ge 0.80$):** $\text{TN} = 0, \text{FP} = 75 \implies \textbf{FPR = 100.0\%, Specificity = 0.0\%}$.
- Member 2's own Phase 7.1 internal audit explicitly confirmed:
  > *"Rule 1 and Rule 3 fail completely as screening tools because virtually every dark-theme screenshot (clean or edited) exhibits small typographic edge clusters crossing the 0.80 probability or 50-pixel threshold, resulting in 0.00% specificity... The system must be operated as a localized heatmap overlay for human review."*

### 2.3 The True Breakthrough: Fine-Grained Micro-Artifact Localization
While Member 2's model fails as a whole-image classifier, its spatial localization capability is a dramatic improvement over Member 3's baseline mask head:
- Member 3's baseline multi-task ResNet-18 downsamples screenshots to $320 \times 160$, shrinking a $1080 \times 2400$ screenshot by **50.6×** in pixel area. A 20-pixel tamper patch shrinks to less than half a pixel, yielding a test mask IoU of **0.059** (properly suppressed in Member 3's UI).
- Member 2's `TileSpatialUNet` operates at native resolution on 256×256 crops. Operating with a 4-stage convolutional encoder-decoder, 2D Gaussian window blending ($\sigma = 64.0$), and connected-component filtering ($A_{\min} = 50$ px), it achieved:
  - **Global Pixel IoU:** $0.0957$ vs. Phase 6 baseline $0.0027$ (**35.1× improvement**).
  - **Global Pixel Dice:** $0.1746$ vs. Phase 6 baseline $0.0054$ (**32.1× improvement**).
  - **Global Pixel Precision:** $12.08\%$ vs. Phase 6 baseline $0.334\%$ (**36.1× improvement**).
  - **Clean False-Positive Pixels:** Reduced from 587,033 px down to 309,665 px (**47.25% noise reduction**).

---

## 3. Comprehensive Feature & Model Comparison Matrix

| Technical Aspect | Member 3 Baseline ResNet-18 | Member 2 Phase 5 `TileResNet18` | Member 2 Improved `TileSpatialUNet` |
| :--- | :--- | :--- | :--- |
| **Checkpoint Path** | `models/phase7/fold_1_resnet18_multitask.pt` | `member2/phase5_model_training/artifacts/best_model.pt` | `member2/performance_improvement/artifacts/best_improved_model.pt` |
| **File Size** | 57.73 MB | 134.26 MB (includes optimizer state) | 5.87 MB |
| **SHA-256 Hash** | `44b654aa846fddfde7be6dab44ceada6687d8dfd35e2c83edbcb60fed909f62a` | `2d47a5eb65ff24f78b2d0d2cbe0ce4fd06dff7ccda7ba2b9d754e9861f28da94` | `b6e3234e734c2801a32b98cb58b66328b907bade26645b03469ecd32bdf5765c` |
| **Total Parameters** | 14,417,024 | 11,187,158 | 484,159 |
| **Backbone** | ResNet-18 (Torchvision standard) | ResNet-18 (Torchvision standard) | Custom 4-stage Encoder-Decoder U-Net |
| **Pretrained Weights** | Yes (ImageNet-1k pretrained) | **No** (`pretrained=False`, trained from scratch) | **No** (Trained from scratch) |
| **Output Heads** | 1. 2-class Classification Head<br>2. ConvTranspose2d Mask Head | 2-class Classification Head (Dropout 0.2 + Linear) | Single $1 \times 1$ Conv Spatial Logit Map $(B, 1, 256, 256)$ |
| **Input Geometry** | Fixed $320 \times 160$ (Canvas H×W) | Tiled $256 \times 256$ native crops | Tiled $256 \times 256$ native crops |
| **Tiling Stride** | None (Global Resize) | 128 pixels (50% horizontal & vertical overlap) | 128 pixels (50% horizontal & vertical overlap) |
| **Tiles / Image ($1080 \times 2400$)** | 1 global image | ~43.5 tiles average (min 18, max 60) | ~43.5 tiles average (min 18, max 60) |
| **Padding & Background** | RGB `(18, 18, 18)` top-left placement | Native resolution edge-anchored | Native resolution edge-anchored |
| **Normalization** | ImageNet Mean & Std | ImageNet Mean & Std | ImageNet Mean & Std |
| **Loss Function** | Weighted Cross-Entropy + BCE Mask Loss | Balanced Cross-Entropy ($w_0=0.54, w_1=6.41$) | $0.4 \times \text{BCE}(\text{pos\_weight}=5) + 0.6 \times \text{SoftDice}$ |
| **Optimizer & Schedule** | AdamW, Cosine Annealing | Adam, lr=1e-4, StepLR | AdamW, lr=5e-4, Cosine Annealing |
| **Training Exposure** | Full dataset cross-validation | 3 epochs, capped at 1,500 tiles/epoch | 6 epochs, balanced 50/50 tile sampling |
| **Cross-Validation** | 5-Fold Stratified Group K-Fold | None (Single 70/15/15 Split) | None (Single 70/15/15 Split) |
| **OOF Predictions** | **Generated across all 5 folds** | **None** | **None** |
| **Stacker Compatibility** | **100% Native** | **Incompatible** (No OOF) | **Incompatible** (No OOF) |
| **Hardware Throughput** | ~65 images/sec (CPU) | ~1.2 images/sec (CPU) | ~0.5 images/sec (CPU) |

---

## 4. Synthesis of Accompanying Technical Reports

The comprehensive audit is organized into four companion reports located in `reports/model_comparison/`:

1. [`CBL04_CNN_PERFORMANCE_AND_LEAKAGE_AUDIT.md`](file:///Users/heemanshupjain/Downloads/Member%203/reports/model_comparison/CBL04_CNN_PERFORMANCE_AND_LEAKAGE_AUDIT.md):
   - Detailed breakdown of all empirical metrics across splits, decision rules, and UI layout families.
   - Comprehensive data leakage audit covering group partitioning, real duplicate images, and threshold selection provenance.
   - Mathematical proof demonstrating why Member 2's Phase 5 accuracy cannot be compared to Member 1/3's whole-screenshot accuracy.

2. [`CBL04_CNN_ARCHITECTURE_AND_PREPROCESSING_COMPARISON.md`](file:///Users/heemanshupjain/Downloads/Member%203/reports/model_comparison/CBL04_CNN_ARCHITECTURE_AND_PREPROCESSING_COMPARISON.md):
   - Layer-by-layer architectural comparison, tensor dimensionality tracing, and parameter distribution analysis.
   - Deep dive into input pipelines: global downsampling vs. sliding-window tile extraction with Gaussian reconstruction.
   - Runtime, latency, and memory profiling across CPU and MPS devices.

3. [`CBL04_CNN_OOF_AND_STACKER_COMPATIBILITY.md`](file:///Users/heemanshupjain/Downloads/Member%203/reports/model_comparison/CBL04_CNN_OOF_AND_STACKER_COMPATIBILITY.md):
   - Comprehensive out-of-fold (OOF) prediction audit.
   - Quantitative evaluation of covariate shift and feature collapse when feeding Member 2 predictions into Member 3's Phase 8 Logistic Stacker.
   - Complete specification of retraining prerequisites.

4. [`CBL04_CNN_FINAL_INTEGRATION_RECOMMENDATION.md`](file:///Users/heemanshupjain/Downloads/Member%203/reports/model_comparison/CBL04_CNN_FINAL_INTEGRATION_RECOMMENDATION.md):
   - Formal integration determination (`NOT SUITABLE` for classification; `CONDITIONALLY READY` as an assistive heatmap overlay).
   - Inventory of affected files in the Member 3 workspace.
   - Prioritized, phased technical roadmap for future decoupling and asynchronous localization service implementation.

---

## 5. Summary Conclusion & Path Forward

Member 2's contribution provides a valuable advancement in **fine-grained micro-artifact localization**, solving the primary limitation of Member 1 and Member 3's global downsampling approach where tiny typographic edits were compressed into sub-pixel space. 

However, because Member 2's models exhibit a **92% to 100% false-positive rate on clean screenshots at the image level**, they cannot replace Member 3's CNN in the screening and classification pipeline. Replacing Member 3's CNN with Member 2's model would collapse the entire multi-branch decision architecture, forcing the stacker and verdict engine to classify virtually all genuine Google Pay screenshots as fraudulent.

The optimal, truthful engineering solution for CBL-04 is:
1. **Retain Member 3's CNN + EasyOCR + Semantic Rule Engine + OOF Stacker** as the primary classification pipeline for the `/api/v1/analyze` endpoint.
2. **Adopt Member 2's `TileSpatialUNet` as an assistive localization module**, deployed either as an optional forensic heatmap generator or an asynchronous inspection worker, providing visual proof to human investigators without poisoning autonomous classification decisions.

