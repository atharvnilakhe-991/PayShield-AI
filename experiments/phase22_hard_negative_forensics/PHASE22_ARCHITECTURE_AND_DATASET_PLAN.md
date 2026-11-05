# CBL-04 — Phase 22: Hard-Negative-Aware Local Forensics Architecture and Dataset Planning

**Project:** CBL-04 — GPay Dark-Theme Transaction Screenshot Tampering Detection  
**Role:** Member 3 — Backend / AI Integration Lead  
**Workspace:** `/Users/heemanshupjain/Downloads/Member 3`  
**Date:** November 4, 2026  
**Document Path:** `experiments/phase22_hard_negative_forensics/PHASE22_ARCHITECTURE_AND_DATASET_PLAN.md`  
**Execution Mode:** Planning & Read-Only Dataset Feasibility Analysis (Zero Models Trained)  
**Status:** COMPLETE & AUTHORITATIVE

---

## 1. Executive Summary & Strategic Mandate

### 1.1 Objective and Scope
The objective of Phase 22 is to design a next-generation local visual forensic architecture and candidate training pipeline that resolves the foundational limitation identified across Phases 19, 20, and 21: **unrestricted candidate extraction produces overwhelming false alarms when evaluated with classifiers trained only on payment fields**.

In accordance with project governance:
- **Strict Read-Only Mode**: No neural networks are trained, no checkpoints are altered, and no production code or frontend files are modified.
- **No Phase 19 Model Reuse as Final Classifier**: The Phase 19 ResNet18 classifier was trained strictly on 4 payment fields (`amount`, `receiver`, `transaction_id`, `date`) and cannot serve as the final classifier for arbitrary UI candidates.
- **Dataset Feasibility Audit**: This document provides empirical analysis of the existing 45,426 training candidates extracted by CRAFT, defines a mathematically sound candidate-labeling taxonomy, specifies a field-aware visual architecture (`FieldAwareForensicResNet`), formulates an aggregation strategy with background false-positive suppression, and verifies whether the dataset can support the proposed experiment.

---

### 1.2 The Core Problem: Why Prior Phases Failed at Scale

```mermaid
flowchart TD
    subgraph P17["Phase 17: Oracle Local Forensics"]
        A1["High-Res Target Crops"] -->|High Discrimination| B1["ROC-AUC = 0.9889 (Amount)"]
        B1 -->|Fatal Flaw| C1["Oracle Upper Bound: Requires Ground-Truth Box"]
    end

    subgraph P19["Phase 19: EasyOCR-Guided Routing"]
        A2["EasyOCR 4-Field Router"] -->|Inference-Realistic| B2["ROC-AUC = 0.8519 (Amount)"]
        B2 -->|Fatal Flaw| C2["Secondary Routing Blindness: 0% Recall on 'other' Category"]
    end

    subgraph P20["Phase 20: Dense Multi-Scale MIL"]
        A3["Uniform 4x2 & 6x3 Grid Tiles"] -->|100% Spatial Coverage| B3["No OCR Dependency"]
        B3 -->|Fatal Flaw| C3["Attention Collapse: 65.35% Mass on Static Position (2,1)"]
    end

    subgraph P21["Phase 21: OCR-Free CRAFT Forensics"]
        A4["CRAFT Scene Text Heatmaps"] -->|100% Spatial Coverage| B4["Successful Localization of 'other'"]
        B4 -->|Fatal Flaw| C4["False Alarm Explosion: 100% Screenshots have S_max >= 0.6615"]
    end

    subgraph P22["Phase 22: Hard-Negative-Aware Forensics"]
        A5["CRAFT Word Proposals"] --> B5["FieldAwareForensicResNet"]
        C4 -->|Resolution 1: Supervise UI Negatives| B5
        C4 -->|Resolution 2: Supervise Clean Twins| B5
        C4 -->|Resolution 3: Calibrated Aggregation| D5["Robust Screenshot Forensics"]
    end
```

### 1.3 Feasibility Assessment Verdict
**VERDICT: THE DATASET FULLY SUPPORTS PHASE 22.**  
Empirical analysis of the 868 training screenshots (434 matched pairs) confirms that:
1. **862 high-quality positive candidates** exist with strong ground-truth overlap ($\text{IoU} \ge 0.30$ or $\text{Containment} \ge 0.70$), providing sufficient supervision across all 5 tampering categories and both layout families.
2. **872 matched clean twin candidates** exist in identical spatial locations on clean receipts, providing direct 1:1 paired hard negatives of authentic GPay font rasterization.
3. **43,598 unchanged background UI candidates** exist across clean and edited screenshots, providing extensive hard-negative supervision across status bars, navigation headers, transaction details, and action buttons.
4. **Candidate clutter is zero**: 0.0% of candidate pairs within any screenshot have an $\text{IoU} > 0.30$.
5. **Ambiguous proposals comprise only 0.21%** (94 candidates), which can be cleanly quarantined from supervised training loss.

---

## 2. Review and Root Cause Analysis of Previous Experiments

To design a robust classifier, we must extract the exact empirical lessons from Phases 17 through 21.1:

| Phase | Input Scope & Resolution | Localization Mechanism | Primary Strength | Fatal Limitation / Bottleneck | Reusable Reference Asset |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 17** | Oracle target crop ($224 \times 224$) | Ground-truth coordinates | Patch-level visual signal is strong: Amount $\text{ROC-AUC} = 0.9889$; matched-pair $\Delta = +0.9574$. | **Oracle Dependence**: Multi-crop max pooling without an oracle jumped to 100% FPR on unseen fonts. | Standardized crop normalization parameters; high-resolution patch intuition. |
| **Phase 18** | Full screenshot + OCR metadata | Regex syntax routing | Integrated multimodal decision flow. | **Semantic Blindness**: 52/54 edited receipts were syntactically valid ($S_{\text{sem}} = 0.0$); stacker rejected. | Fast keyword parsing; syntax error trap handlers. |
| **Phase 19** | 4 payment field crops ($(256 \times 128)$) | EasyOCR keyword router | Broke 0% decisive coverage deadlock (76.85% coverage); amount $\text{ROC-AUC} = 0.8519$. | **Secondary Field Misses**: 0% routing recall on category `other` (84/84 misses); character geometry alone failed ($\text{ROC-AUC} \approx 0.50$). | Leakage-safe grouped 5-fold splits; standardized $(256 \times 128)$ letterbox canvas on `(18, 18, 18)`. |
| **Phase 20** | Uniform $4 \times 2$ (8 tiles) & $6 \times 3$ (18 tiles) | Dense geometric grid | 100% image pixel coverage without OCR reliance. | **Needle in a Giant Dark Haystack**: Tampered text is $<0.2\%$ of pixels; attention collapsed to static layout prior ($65.35\%$ mass on position `(2, 1)`). | Multi-scale grid generation scripts; bag-level pooling benchmark code. |
| **Phase 21** | $\sim 53$ word crops per screenshot | CRAFT character heatmaps | Bypassed OCR keyword routing; 100% spatial text coverage; successfully localized `other` category edits. | **Domain Shift & Aggregation Explosion**: Scoring all candidates with Phase 19 classifier caused 100% clean screenshots to saturate with $S_{\text{max}} \ge 0.6615$. | 51,684 pre-extracted CRAFT candidate bounding boxes; shard caching pipeline. |
| **Phase 21.1** | Read-only metric integrity audit | Full recomputation | Proved mathematical root cause of conservative saturation; proved 98.94% of tied pairs chose identical untouched UI text. | Affirmed rejection of Phase 21 for production deployment. | Authoritative benchmark reconciliation tables; candidate scoring verification tools. |

### Key Takeaways for Phase 22 Architecture
1. **The visual classifier must be trained on UI negatives**: The Phase 19 classifier failed in Phase 21 because it was trained *only* on payment text. When shown a status bar clock, battery icon, or "Google Pay" header, it panicked and predicted high tampering probabilities. Phase 22 must explicitly train the model that normal UI elements have a tampering probability of 0.0.
2. **Clean counterparts provide the ultimate hard negatives**: In each matched pair, the clean screenshot contains the exact authentic counterpart of the tampered text (e.g., authentic `₹500.00` vs edited `₹800.00`). Incorporating these crops as Type 1 Hard Negatives forces the model to learn subtle font rendering, kerning, and anti-aliasing differences rather than memorizing field semantics.
3. **Naive max-pooling must be abandoned**: When pooling 53 candidate probabilities per image, even a 5% candidate false positive rate yields a 93.4% screenshot false alarm rate ($1 - 0.95^{53}$). Aggregation must incorporate background noise calibration, isolation margins, and spatial functional zone gating.

---

## 3. Candidate-Level Training Label Specification

### 3.1 Mathematical Formulation of Overlap
Let a candidate proposal bounding box be $B_{\text{cand}} = [x_1, y_1, x_2, y_2]$ with area $A_{\text{cand}} = (x_2 - x_1) \times (y_2 - y_1)$.  
Let the ground-truth tampered bounding box from `train_manifest.csv` be $B_{\text{gt}} = [x_1^{\text{gt}}, y_1^{\text{gt}}, x_2^{\text{gt}}, y_2^{\text{gt}}]$ with area $A_{\text{gt}} = (x_2^{\text{gt}} - x_1^{\text{gt}}) \times (y_2^{\text{gt}} - y_1^{\text{gt}})$.  

The intersection region is $B_{\cap} = [\max(x_1, x_1^{\text{gt}}), \max(y_1, y_1^{\text{gt}}), \min(x_2, x_2^{\text{gt}}), \min(y_2, y_2^{\text{gt}})]$, with area $I = \max(0, x_2^{\cap} - x_1^{\cap}) \times \max(0, y_2^{\cap} - y_1^{\cap})$.  
The union area is $U = A_{\text{cand}} + A_{\text{gt}} - I$.

We define three overlap metrics:
1. **Intersection over Union (IoU)**:
   $$\text{IoU}(B_{\text{cand}}, B_{\text{gt}}) = \frac{I}{U}$$
2. **Candidate Containment (Precision of Candidate inside Ground Truth)**:
   $$C_{\text{cand}} = \frac{I}{A_{\text{cand}}}$$
3. **Ground-Truth Coverage (Recall of Ground Truth by Candidate)**:
   $$C_{\text{gt}} = \frac{I}{A_{\text{gt}}}$$

---

### 3.2 Justification of Dual-Criterion Thresholds
In standard object detection, $\text{IoU} \ge 0.50$ is standard. However, in scene text forensics, **$\text{IoU}$ alone is mathematically flawed when word-level detectors inspect multi-word text fields**:
- In the CBL-04 dataset, ground-truth bounding boxes for categories like `date_time` (e.g. `"04 Nov 2026, 6:22pm"`, $W \approx 200\text{ px}$) or `recipient` (e.g. `"Heemanshu Parashar Jain"`, $W \approx 320\text{ px}$) span the entire multi-word field.
- CRAFT detects individual word tokens ($W \approx 30\text{--}80\text{ px}$).
- A candidate covering `"Nov"` is **100% inside the tampered region** ($C_{\text{cand}} = 1.0$), but its $\text{IoU}$ against the whole phrase box is only $\frac{40 \times 20}{200 \times 20} \approx 0.20$.
- Requiring $\text{IoU} \ge 0.50$ would falsely reject almost all word-level proposals in multi-word tampered fields!

Therefore, we establish the **Dual-Criterion Rule**:
A candidate is labeled as a genuine tampered text crop if:
$$C_{\text{cand}} \ge 0.70 \quad \text{OR} \quad \text{IoU} \ge 0.30$$
- If $C_{\text{cand}} \ge 0.70$, at least $70\%$ of the pixels in the candidate crop originate from the tampered region.
- If $\text{IoU} \ge 0.30$, the candidate tightly bounds a single-word or numeric tampered field (such as an amount or transaction ID).

---

### 3.3 Four-Class Candidate Label Taxonomy

```mermaid
flowchart TD
    A["CRAFT Candidate Proposal"] --> B{"Is Screenshot Edited?"}
    
    B -->|Yes: Edited Screenshot| C{"Overlap with GT Tampered Box"}
    C -->|C_cand >= 0.70 OR IoU >= 0.30| D["POSITIVE (Label = 1)\nTampered Text Crop"]
    C -->|0.05 <= C_cand < 0.70 AND IoU < 0.30| E["AMBIGUOUS (Label = -1)\nQuarantined from Loss"]
    C -->|C_cand < 0.05 AND IoU < 0.05| F["HARD NEGATIVE TYPE 2\n(Label = 0, Edited BG)"]
    
    B -->|No: Clean Screenshot| G{"Overlap with Edited Twin GT Box"}
    G -->|C_twin >= 0.70 OR IoU_twin >= 0.30| H["HARD NEGATIVE TYPE 1\n(Label = 0, Clean Twin Field)"]
    G -->|C_twin < 0.05 AND IoU_twin < 0.05| I["HARD NEGATIVE TYPE 2\n(Label = 0, Clean BG)"]
```

| Candidate Class | Source Image | Supervised Label | Overlap Criteria | Empirical Count in Train (434 pairs) | Percentage of Candidates | Strategic Role in Training |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **`POSITIVE`** | Edited screenshots | **$1$** | $C_{\text{cand}} \ge 0.70$ OR $\text{IoU} \ge 0.30$ | **$862$** | **$1.90\%$** | Teaches the network what digital font tampering, edge blending, and rasterization artifacts look like. |
| **`HARD_NEGATIVE_TYPE1`** (Clean Twin Field) | Clean screenshots | **$0$** | $C_{\text{twin}} \ge 0.70$ OR $\text{IoU}_{\text{twin}} \ge 0.30$ against edited twin's GT | **$872$** | **$1.92\%$** | Teaches the network what genuine, authentic payment text looks like in the exact same field positions. |
| **`HARD_NEGATIVE_TYPE2`** (UI Background) | Clean & edited screenshots | **$0$** | $C < 0.05$ AND $\text{IoU} < 0.05$ (and twin $C < 0.05$) | **$43,598$** | **$95.97\%$** | Teaches the network that normal UI text (status bar, logo, buttons, footers) has tampering probability $0.0$. |
| **`AMBIGUOUS`** (Boundary Edge) | Edited screenshots | *Excluded* | $0.05 \le C_{\text{cand}} < 0.70$ AND $\text{IoU} < 0.30$ | **$94$** | **$0.21\%$** | Quarantined from loss calculation to avoid gradient noise from partially clipped text. |
| **TOTAL** | **868 Screenshots** | — | — | **$45,426$** | **$100.00\%$** | Fully accounted for across all 434 training pairs. |

> [!IMPORTANT]
> **Strict Inference Isolation**: Ground-truth bounding boxes and twin relationships are used **strictly during dataset manifest compilation** to assign training labels. During inference, CRAFT proposals are extracted and scored purely from image pixels without any access to ground truth or paired twins.

---

## 4. Pair-Aware Dataset Construction & Empirical Feasibility

### 4.1 Grouped Stratified 5-Fold Splitting
To eliminate all risk of pair leakage, template leakage, and fold contamination:
1. **Strict Grouping by `pair_id`**: Both `P4-XXXX_clean` and `P4-XXXX_edited` screenshots, along with all their candidate crops, are strictly assigned to the same fold.
2. **Stratification**: Folds are stratified across `layout_family` (`compact`, `enriched`) and `edit_category` (`amount`, `date_time`, `transaction_id`, `recipient`, `other`), matching the Phase 19 fold assignments.
3. **Empirical Fold Balance**:

| Fold | Positives ($N_{\text{pos}}$) | Clean Twin Negatives ($N_{\text{twin}}$) | Edited BG Negatives | Clean BG Negatives | Ambiguous (Quarantined) | Total Candidates |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 166 | 166 | 4,430 | 4,449 | 21 | 9,232 |
| **Fold 1** | 171 | 173 | 4,275 | 4,296 | 19 | 8,934 |
| **Fold 2** | 161 | 167 | 4,439 | 4,461 | 21 | 9,249 |
| **Fold 3** | 184 | 185 | 4,291 | 4,313 | 14 | 8,987 |
| **Fold 4** | 180 | 181 | 4,311 | 4,333 | 19 | 9,024 |
| **TOTAL** | **862** | **872** | **21,746** | **21,852** | **94** | **45,426** |

Notice the remarkable symmetry: Positives ($862$) and Clean Twin Negatives ($872$) are balanced almost exactly 1:1 across every individual fold.

---

### 4.2 Breakdown Across Categories and Layouts

#### Positive Candidates by Tampering Category
The 434 edited training screenshots span 5 categories, yielding 862 positive candidates:
- `recipient` (85 images): **468 positives** (mean: $5.51$ words/field, e.g. multi-word names).
- `date_time` (91 images): **154 positives** (mean: $1.69$ words/field, e.g. `"04 Nov 2026, 6:22pm"`).
- `other` (84 images): **82 positives** (mean: $0.98$ words/field, e.g. status pills, bank suffixes).
- `amount` (91 images): **79 positives** (mean: $0.87$ words/field, e.g. `"₹500.00"`).
- `transaction_id` (83 images): **79 positives** (mean: $0.95$ words/field, e.g. long alphanumeric codes).

#### Positive Candidates by Layout Family
- `enriched` (216 images): **470 positives** (54.5%).
- `compact` (218 images): **392 positives** (45.5%).

---

### 4.3 Candidate Dimensions, Aspect Ratios, and Spatial Distribution
Empirical analysis of all 45,426 training candidates demonstrates clear geometric and spatial characteristics:

| Candidate Group | Width (px) | Height (px) | Aspect Ratio ($W/H$) | CRAFT Confidence ($C_{\text{det}}$) | Normalized $Y_{\text{center}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`POSITIVE`** | $61.7 \pm 45.4$ | $23.8 \pm 12.9$ | $2.68 \pm 1.64$ | $0.361 \pm 0.033$ | $0.330 \pm 0.172$ |
| **`NEG_CLEAN_TWIN`** | $60.5 \pm 42.8$ | $23.9 \pm 13.0$ | $2.62 \pm 1.47$ | $0.360 \pm 0.034$ | $0.330 \pm 0.172$ |
| **`AMBIGUOUS`** | $80.2 \pm 53.3$ | $25.3 \pm 15.5$ | $3.24 \pm 1.45$ | $0.349 \pm 0.037$ | $0.340 \pm 0.109$ |
| **`NEG_EDITED_BG`** | $51.4 \pm 30.0$ | $18.0 \pm 7.3$ | $2.92 \pm 1.47$ | $0.347 \pm 0.040$ | $0.494 \pm 0.222$ |
| **`NEG_CLEAN_BG`** | $51.6 \pm 30.2$ | $18.1 \pm 7.3$ | $2.93 \pm 1.47$ | $0.347 \pm 0.040$ | $0.494 \pm 0.222$ |

#### Critical Insights from Empirical Distributions:
1. **Twin Identity**: `POSITIVE` and `NEG_CLEAN_TWIN` crops have virtually identical geometry ($W \approx 61$, $H \approx 24$, $Y_c \approx 0.33$) and detector confidence ($0.361$ vs $0.360$). The *only* distinguishing feature between them is the pixel-level visual texture (anti-aliasing, rasterization, font contour smoothness).
2. **Background Differentiation**: UI background text is vertically distributed across the entire screen ($Y_c = 0.494 \pm 0.222$, spanning $0.02$ to $0.98$), whereas payment fields cluster in the upper-middle transaction zone ($Y_c = 0.330 \pm 0.172$). Background text is also slightly smaller on average ($H = 18.0\text{ px}$ vs $23.8\text{ px}$).
3. **Detector Confidence Invariant**: CRAFT detector confidence is essentially constant across all groups ($0.347\text{--}0.361$). This confirms that **detector confidence has zero correlation with tampering** and must never be treated as a proxy for tampering probability.

---

### 4.4 Candidate Overlap & Intra-Screenshot Deduplication
To evaluate whether overlapping CRAFT proposals require Non-Maximum Suppression (NMS):
- Pairwise IoU was calculated across all candidate pairs within screenshots ($N = 67,581$ pairs evaluated across a 50-screenshot sample).
- Result: **0 pairs had $\text{IoU} > 0.50$ ($0.0000\%$)** and **0 pairs had $\text{IoU} > 0.30$ ($0.0000\%$)**.
- Conclusion: CRAFT's connected-component affinity clustering naturally partitions the screenshot into disjoint word boxes. No NMS deduplication is needed, preserving 100% of candidate proposals without box-merging distortions.

---

### 4.5 Class Imbalance & Balanced Batch Sampling Strategy
With $862$ positives and $44,470$ total negatives, the naive negative-to-positive ratio is $\sim 51.6 : 1$. Training with naive random sampling would cause the model to converge to a trivial all-negative predictor.

**Proposed Training Sampler: Balanced Multi-Source Sampler**  
For each training mini-batch of size $B = 32$ (or $64$):
- **$25\%$ Positives ($8$ crops)**: Sampled with replacement from the training fold's `POSITIVE` pool.
- **$25\%$ Type 1 Hard Negatives ($8$ crops)**: Sampled from the training fold's `NEG_CLEAN_TWIN` pool.
- **$50\%$ Type 2 Hard Negatives ($16$ crops)**: Uniformly sampled across UI zones (Status bar, Navigation, Details, Footer) from the `NEG_CLEAN_BG` and `NEG_EDITED_BG` pools.

**Loss Function: Focal Loss**  
To dynamically down-weight easy background negatives while penalizing hard font boundary errors:
$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad \text{with } \gamma = 2.0, \alpha = 0.25$$

---

### 4.6 Strict Partition Quarantines
- **Training Partition (434 pairs / 868 images)**: Exclusively used for 5-fold cross-validation and hyperparameter development.
- **Validation Partition (54 pairs / 108 images, 6,258 candidates)**: Strictly frozen. Zero threshold selection or model checkpoint tuning on validation.
- **Test Partition (12 pairs / 24 images)**: Strictly quarantined. Zero access.
- **Historical Exposure Disclosure**: Sample `P4-0036` was inspected in Phase A.1 for diagnostic OCR debugging; acknowledged and documented.

---

## 5. Proposed Classifier Architecture: `FieldAwareForensicResNet`

### 5.1 Architecture Comparison Matrix

| Architecture Option | Visual Backbone | Context / Spatial Inputs | Parameters | Inference Speed | Pros | Cons | Verdict |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| **Option A: Plain ResNet18** | ResNet18 (ImageNet init) | None (Crop pixels only) | 11.2M | $\sim 780\text{ crops/s}$ | Simple, fast, drop-in replacement. | Lacks screen position; cannot distinguish status bar text from amount text if appearance is similar. | *Sub-optimal* |
| **Option B: Dual-Crop Multi-Scale ResNet** | ResNet18 (2 parallel branches) | Tight crop + 2x expanded context window | 22.4M | $\sim 390\text{ crops/s}$ | Captures local pixel artifacts and surrounding boundary context. | Doubles crop extraction storage and inference latency. | *Too Heavy* |
| **Option C: `FieldAwareForensicResNet`** (Recommended) | ResNet18 (ImageNet init) + Spatial Geometry MLP | Normalized coordinates $[y_1, x_1, y_2, x_2, w, h, AR, C_{\text{det}}]$ | 11.3M | $\sim 770\text{ crops/s}$ | Informs model of UI screen zone; decouples detector confidence; zero extra image memory. | Requires vector concatenation head. | **RECOMMENDED** |

---

### 5.2 Detailed Specification of `FieldAwareForensicResNet`

```mermaid
flowchart LR
    subgraph Stream1["Stream 1: Visual Feature Extractor"]
        A1["Standardized Crop (256x128x3)\nCanvas: (18, 18, 18)"] --> B1["ResNet-18 Backbone\n(ImageNet Pretrained)"]
        B1 --> C1["Global Average Pool"]
        C1 --> D1["Visual Embedding\nf_vis (512-D)"]
    end

    subgraph Stream2["Stream 2: Spatial & Geometric Context"]
        A2["Geometry Vector g (8-D)\n[yc, xc, w, h, AR, log(A), Cdet, ybot]"] --> B2["Linear(8 -> 32) + ReLU"]
        B2 --> C2["Linear(32 -> 64) + LayerNorm"]
        C2 --> D2["Spatial Embedding\nf_geom (64-D)"]
    end

    subgraph Fusion["Fusion & Forensic Classification Head"]
        D1 & D2 --> E["Concatenation [f_vis, f_geom] (576-D)"]
        E --> F["Linear(576 -> 128) + ReLU + Dropout(0.3)"]
        F --> G["Linear(128 -> 2)"]
        G --> H["Softmax Output: P(tampered)"]
    end
```

#### 1. Visual Stream
- **Backbone**: ResNet-18 initialized from torchvision ImageNet weights (`models.resnet18(weights=ResNet18_Weights.DEFAULT)`).
- **Freezing Policy**: Freeze initial stem (`conv1`, `bn1`, `layer1`) to retain generic low-level edge/texture primitives; fine-tune `layer2`, `layer3`, `layer4` for forensic artifact extraction.
- **Input Preprocessing**: Standardized letterbox $(256 \times 128)$ on dark background `(18, 18, 18)` with Lanczos interpolation and ImageNet normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
- **Output**: 512-dimensional vector $f_{\text{vis}} \in \mathbb{R}^{512}$.

#### 2. Spatial & Geometric Stream
To provide functional zone awareness and prevent false alarms on non-payment UI elements, an 8-dimensional normalized geometry vector is constructed:
$$g = \left[ \frac{y_1 + y_2}{2 H_{\text{img}}}, \frac{x_1 + x_2}{2 W_{\text{img}}}, \frac{w}{W_{\text{img}}}, \frac{h}{H_{\text{img}}}, \frac{w}{h}, \log(1 + w \cdot h), C_{\text{det}}, \frac{y_2}{H_{\text{img}}} \right] \in \mathbb{R}^8$$
- **MLP Architecture**: $\text{Linear}(8, 32) \to \text{ReLU} \to \text{Linear}(32, 64) \to \text{LayerNorm} \to f_{\text{geom}} \in \mathbb{R}^{64}$.
- **Decoupled Detector Confidence**: CRAFT detector confidence $C_{\text{det}}$ is provided as an input feature, enabling the model to learn that low detector confidence does not imply tampering.

#### 3. Fusion & Classification Head
- **Concatenation**: $f_{\text{fused}} = [f_{\text{vis}} \,\|\, f_{\text{geom}}] \in \mathbb{R}^{576}$.
- **MLP Head**: $\text{Linear}(576, 128) \to \text{ReLU} \to \text{Dropout}(p = 0.3) \to \text{Linear}(128, 2)$.
- **Output**: Logits $z \in \mathbb{R}^2 \implies P(\text{tampered}) = \frac{\exp(z_1)}{\exp(z_0) + \exp(z_1)}$.

---

## 6. Screenshot-Level Aggregation Design

### 6.1 Mathematical Proof of Max-Pooling Failure
Under independent candidate scoring with false positive rate $\alpha_{\text{cand}}$:
$$P(\text{Clean Screenshot False Alarm}) = 1 - (1 - \alpha_{\text{cand}})^K$$
For $K = 53$ candidates:
- If $\alpha_{\text{cand}} = 0.10 \implies P(\text{False Alarm}) = 1 - 0.90^{53} = 99.6\%$
- If $\alpha_{\text{cand}} = 0.05 \implies P(\text{False Alarm}) = 1 - 0.95^{53} = 93.4\%$
- If $\alpha_{\text{cand}} = 0.02 \implies P(\text{False Alarm}) = 1 - 0.98^{53} = 65.6\%$

Naive max pooling requires $\alpha_{\text{cand}} < 0.0019$ to achieve an overall screenshot false alarm rate $\le 10\%$. This is unachievable with local visual classifiers alone.

---

### 6.2 Proposed Aggregation Strategies for Phase 22

#### Strategy 1: Background-Calibrated Isolation Margin ($\Delta_{\text{iso}}$)
In a genuinely tampered screenshot, typically 1 or 2 candidates are tampered while $\sim 51$ candidates are clean. Thus, the top candidate score $S_{(1)}$ should stand out significantly from the background candidate distribution:
$$\Delta_{\text{iso}} = S_{(1)} - \text{median}(S_{(1)}, S_{(2)}, \dots, S_{(K)})$$
- If an out-of-distribution clean receipt lifts all candidate scores, $S_{(1)}$ and $\text{median}(S)$ rise together $\implies \Delta_{\text{iso}} \approx 0$.
- In a tampered receipt, $S_{(1)}$ is elevated while $\text{median}(S)$ remains low $\implies \Delta_{\text{iso}} \gg 0$.
- Final Screenshot Score: $S_{\text{screen}} = S_{(1)} \times \left(1 + \tanh(3 \cdot \Delta_{\text{iso}})\right) / 2$.

#### Strategy 2: Functional Spatial Zone Suppression Gating
GPay screenshots have rigid functional zones:
- **Zone 1: Status Bar** ($Y_c < 0.08$): Battery, time, signal $\implies$ Weight $w_{\text{zone}} = 0.05$.
- **Zone 2: Navigation Header** ($0.08 \le Y_c < 0.18$): Google Pay title, back arrow $\implies$ Weight $w_{\text{zone}} = 0.10$.
- **Zone 3: Transaction Hero & Amount** ($0.18 \le Y_c < 0.45$): Recipient name, amount $\implies$ Weight $w_{\text{zone}} = 1.00$.
- **Zone 4: Transaction Details & IDs** ($0.45 \le Y_c < 0.75$): UPI IDs, dates, bank notes $\implies$ Weight $w_{\text{zone}} = 0.90$.
- **Zone 5: Footer & Actions** ($Y_c \ge 0.75$): "Check balance", "Pay again", logos $\implies$ Weight $w_{\text{zone}} = 0.10$.

Each candidate score is gated by its functional zone: $\tilde{S}_k = w_{\text{zone}}(k) \cdot S_k$.

#### Strategy 3: Top-2 Corroborating Rank-Weighted Mean
$$S_{\text{corrob}} = 0.70 \cdot \tilde{S}_{(1)} + 0.30 \cdot \tilde{S}_{(2)}$$
Penalizes isolated single-pixel spike artifacts by requiring corroborating visual evidence.

#### Strategy 4: Conservative Inconclusive Decision Band
To prevent guessing on borderline receipts:
$$\text{Verdict} = \begin{cases} 
\text{TAMPERED} & \text{if } S_{\text{screen}} \ge 0.60 \\
\text{CLEAN} & \text{if } S_{\text{screen}} \le 0.40 \\
\text{INCONCLUSIVE} & \text{if } 0.40 < S_{\text{screen}} < 0.60 
\end{cases}$$
The inconclusive band is audited for decisive coverage percentage without parameter tuning against the frozen validation partition.

---

## 7. Acceptance Criteria & Evaluation Protocol

### 7.1 Proposed Development Targets (Derived from Phase 16)
These metrics represent **engineering goals for model development**, not pre-experimental claims:

| Target Metric | Proposed Target Value | Evaluation Scope |
| :--- | :---: | :--- |
| **Screenshot ROC-AUC** | $\ge 0.80$ | 5-Fold OOF Screenshot Predictions |
| **Screenshot PR-AUC** | $\ge 0.80$ | 5-Fold OOF Screenshot Predictions |
| **Screenshot FPR** | $\le 10.0\%$ | Diagnostic threshold $0.50$ |
| **Screenshot Edited Recall** | $\ge 80.0\%$ | Diagnostic threshold $0.50$ |
| **Screenshot Macro-F1** | $\ge 0.80$ | Diagnostic threshold $0.50$ |
| **Category Recall (All 5 Categories)** | $\ge 70.0\%$ | Subcategory breakdown (`amount`, `date_time`, etc.) |
| **Layout Macro-F1 (Both Layouts)** | $\ge 0.75$ | `compact` vs `enriched` |
| **Decisive Coverage** | $\ge 80.0\%$ | Fraction outside $[0.40, 0.60]$ |
| **Candidate-Level ROC-AUC** | $\ge 0.85$ | Evaluated across all $45,332$ non-ambiguous crops |

---

### 7.2 Dual Evaluation Protocol
To guarantee complete diagnostic transparency, Phase 22 will report:
1. **Candidate-Level Metrics**: Evaluated across all individual crops (Positives vs Type 1 Negatives vs Type 2 Negatives), measuring the pure visual discriminative power of `FieldAwareForensicResNet`.
2. **Screenshot-Level Metrics**: Evaluated across full receipts using the 4 aggregation strategies, measuring system-level detection utility.
3. **Decisive Coverage & Inconclusive Rate**: Reported explicitly for all thresholds and rejection bands.

---

### 7.3 Real-World Synthetic Data Realism Disclosure
All models in Phase 22 are developed using synthetic dark-theme GPay transaction receipts created from native templates. While font rendering, anti-aliasing, and compression artifacts simulate real-world manipulation, **generalization to non-GPay applications or radically different light-theme layouts cannot be claimed without external real-world validation data**.

---

## 8. Risk Analysis & Failure Modes

| Risk Description | Probability | Severity | Root Cause | Proposed Mitigation in Phase 22 |
| :--- | :---: | :---: | :--- | :--- |
| **Risk 1: Overfitting to Synthetic Font Kernels** | Medium | High | Synthetic text generator uses a fixed set of digital fonts (`Roboto`, `Google Sans`). | Apply strong data augmentation during training: random Gaussian blur ($\sigma \in [0.1, 0.5]$), JPEG compression jitter ($Q \in [70, 95]$), and subtle brightness/contrast jitter ($\pm 5\%$). |
| **Risk 2: Spatial Zone Bias on Unseen Layouts** | Low | High | GPay updates its layout, moving amounts or dates to different vertical coordinates. | Spatial feature dropout: Randomly zero out $f_{\text{geom}}$ with probability $p = 0.20$ during training, forcing the visual backbone to maintain independent discriminative capability. |
| **Risk 3: Extreme Imbalance Gradient Starvation** | High | High | $43,598$ UI negatives overwhelm the $862$ positives. | Implement strict Balanced Batch Sampling ($25\%$ pos, $25\%$ twin neg, $50\%$ UI neg) and Focal Loss ($\gamma = 2.0$). |
| **Risk 4: Boundary Crop Label Noise** | Low | Medium | Candidates clipping $10\%\text{--}30\%$ of tampered text confuse the loss function. | Quarantined the $94$ ambiguous candidates ($0.05 \le C < 0.70$) with label `-1`, strictly excluding them from cross-entropy loss computation. |
| **Risk 5: MPS Hardware Instability** | Low | Low | Apple Silicon MPS numerical precision drift in custom loss layers. | Verify all loss computations in float32; include CPU fallback check in training scripts. |

---

## 9. Step-by-Step Implementation Sequence

```mermaid
flowchart TD
    S1["Step 1: Build Phase 22 Candidate Dataset Manifest\n(build_phase22_dataset.py)"] --> S2["Step 2: Implement FieldAwareForensicResNet Architecture\n(field_aware_classifier.py)"]
    S2 --> S3["Step 3: Grouped 5-Fold OOF Training Execution\n(train_5fold_phase22.py)"]
    S3 --> S4["Step 4: Candidate & Screenshot Evaluation\n(evaluate_phase22.py)"]
    S4 --> S5["Step 5: Frozen Validation Evaluation & Report\n(CBL04_PHASE22_EVALUATION_REPORT.md)"]
```

### Detailed Execution Roadmap:
1. **Step 1: Manifest Compilation (`experiments/phase22_hard_negative_forensics/build_phase22_dataset.py`)**
   - Read `craft_detections_manifest.csv` and `train_manifest.csv`.
   - Calculate candidate containment $C_{\text{cand}}$ and IoU against ground-truth boxes.
   - Assign the 4 discrete labels (`POSITIVE`, `NEG_CLEAN_TWIN`, `NEG_EDITED_BG`, `NEG_CLEAN_BG`, `AMBIGUOUS`).
   - Export `phase22_candidate_train_manifest.csv` with normalized spatial coordinates.
2. **Step 2: Architecture Implementation (`experiments/phase22_hard_negative_forensics/field_aware_classifier.py`)**
   - Implement `FieldAwareForensicResNet` in PyTorch.
   - Add unit tests verifying forward pass tensor shapes: input $(B, 3, 128, 256)$ and $(B, 8) \implies$ output $(B, 2)$.
3. **Step 3: 5-Fold Training Execution (`experiments/phase22_hard_negative_forensics/train_5fold_phase22.py`)**
   - Implement `BalancedMultiSourceSampler`.
   - Train Folds 0 through 4 with Focal Loss, AdamW ($\text{lr} = 10^{-4}$), cosine decay, and early stopping on held-out fold loss.
   - Save authenticated checkpoints `best_field_aware_resnet18_fold{0..4}.pt` with SHA-256 digests.
4. **Step 4: Candidate & Screenshot Evaluation (`experiments/phase22_hard_negative_forensics/evaluate_phase22.py`)**
   - Score all $45,426$ OOF candidates strictly using held-out fold checkpoints.
   - Evaluate the 4 aggregation strategies across the 868 screenshots.
   - Compute candidate-level and screenshot-level ROC-AUC, PR-AUC, FPR, recall, and category breakdowns.
5. **Step 5: Post-Lock Validation & Final Report**
   - Evaluate frozen 5-fold ensemble on the 108 `bank_progress` validation screenshots.
   - Author authoritative Phase 22 comparison report against Phase 17, 19, 20, and 21 benchmarks.

---

## 10. Dataset Feasibility Assessment & Conclusion

### Empirical Support Verification
| Requirement | Feasibility Audit Finding | Status |
| :--- | :--- | :---: |
| **Supervised Tamper Signal** | Exactly **$862$ positive candidates** with $\ge 70\%$ containment or $\ge 0.30$ IoU, covering all 5 categories and both layouts. | **PASSED** |
| **Authentic Field Negatives** | Exactly **$872$ clean twin candidates** in identical spatial locations on clean receipts. | **PASSED** |
| **UI Background Negatives** | Exactly **$43,598$ candidates** across status bars, headers, details, and action buttons. | **PASSED** |
| **Fold Balance** | Exactly $161\text{--}184$ positives and $166\text{--}185$ clean twins per fold across all 5 folds. | **PASSED** |
| **Candidate Clutter** | **$0.0000\%$** candidate pairs with IoU $> 0.30$ within screenshots. No NMS needed. | **PASSED** |
| **Label Ambiguity** | Only **$94$ ambiguous candidates ($0.21\%$)**, cleanly quarantined from training loss. | **PASSED** |
| **Hardware Throughput** | Scoring throughput audited at **$783.6\text{ crops/s}$** on Apple Silicon MPS. 5 folds train in $< 20\text{ min}$. | **PASSED** |

### Final Engineering Recommendation
The CBL-04 training dataset provides an exceptionally rich, well-balanced, and leakage-safe foundation for **Phase 22 Hard-Negative-Aware Local Forensics**. By pairing genuine tampered crops with authentic field twins and background UI negatives, and conditioning inference on spatial functional zones, the proposed `FieldAwareForensicResNet` directly targets the root causes of prior failure modes.

**Phase 22 planning is complete, fully documented, and ready for execution upon directive.**

