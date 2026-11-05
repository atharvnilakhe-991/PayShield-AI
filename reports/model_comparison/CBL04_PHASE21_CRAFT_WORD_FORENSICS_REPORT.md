# CBL-04 — Phase 21: OCR-Free Word-Level Visual Forensics Report

**Project:** CBL-04 — GPay Dark-Theme Transaction Screenshot Tampering Detection  
**Role:** Member 3 — Backend / AI Integration Lead  
**Workspace:** `/Users/heemanshupjain/Downloads/Member 3`  
**Date:** November 4, 2026  
**Artifact Directory:** `experiments/phase21_craft_word_forensics/`  
**Report Path:** `reports/model_comparison/CBL04_PHASE21_CRAFT_WORD_FORENSICS_REPORT.md`  
**Audited & Reconciled:** Phase 21.1 Audit ([`CBL04_PHASE21_1_METRIC_INTEGRITY_AUDIT.md`](file:///Users/heemanshupjain/Downloads/Member%203/reports/model_comparison/CBL04_PHASE21_1_METRIC_INTEGRITY_AUDIT.md))

---

## 1. Executive Summary

Phase 21 investigated an inference-realistic visual forensic pipeline designed to eliminate the secondary field routing bottleneck discovered in Phase 19. In Phase 19, an EasyOCR keyword-based router extracted four structured transaction fields (`amount`, `receiver`, `transaction_id`, `date`), achieving high detection performance on those fields (e.g., $0.8519$ validation ROC-AUC on `amount`), but completely missed tampering located in unrouted regions (specifically the `other` category, which suffered a $0.0\%$ routing recall).

To achieve universal spatial text coverage without keyword reliance or ground-truth annotations, Phase 21 developed **OCR-Free Word-Level Visual Forensics**:
1. **Scene Text Detection via CRAFT**: The CRAFT (Character Region Awareness for Text Detection) deep neural network extracted all word-level text regions across every screenshot based purely on pixel-level character heatmaps (`score_text`) and affinity linkages (`score_link`). EasyOCR text recognition was strictly disabled ($0\%$ recognition text used).
2. **Standardized Glyph Cropping**: Candidate regions were expanded with conservative glyph padding ($m = \max(4, \text{int}(0.15 \times \min(w, h)))$), clipped to image boundaries, letterboxed to $(256, 128)$ on dark background `(18, 18, 18)`, and normalized to match Phase 19 training preprocessing.
3. **Strict Leakage-Safe Out-of-Fold (OOF) Scoring**: All $51,684$ candidate word crops across the 868 training screenshots (434 pairs) were scored strictly by their corresponding held-out Phase 19 ResNet18 fold checkpoints. Validation screenshots (108 `bank_progress` images) were scored by the frozen 5-fold ensemble.
4. **Predefined Screenshot Aggregations**: Five aggregation strategies (Maximum, Top-3 Mean, Top-5 Mean, Confidence-Filtered Mean, and Conservative Inconclusive) were evaluated without parameter tuning against validation labels.

---

### Key Empirical Findings (Reconciled Benchmarks)

| Experimental Branch | Candidate Selection Mechanism | Input Scope & Resolution | OOF Screenshot ROC-AUC | OOF Screenshot PR-AUC | Val Screenshot ROC-AUC | Val Screenshot PR-AUC | Decisive Coverage ($\notin [0.45, 0.55]$) | Mean Separation Delta ($\bar{S}_{\text{edited}} - \bar{S}_{\text{clean}}$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 15 Global CNN** | Whole screenshot | Squashed $320 \times 160$ | 0.5240 | 0.5180 | 0.4938 | 0.5028 | 0.00% (0 / 108) | -0.0005 |
| **Phase 17 Oracle Crops** | Oracle GT target box | Oracle $224 \times 224$ crop | N/A | N/A | **0.6677** | **0.6551** | **62.04% (67 / 108)** | **+0.1938** |
| — *Phase 17 Amount Subcategory*| *Oracle Amount box* | *Oracle $224 \times 224$ crop* | N/A | N/A | *0.9889* | *0.9890* | *90.00% (9 / 10)* | *+0.6214* |
| **Phase 18 Semantic Fusion** | Regex-guided routing | Fused screenshot | 0.5180 | 0.5120 | 0.5185 | 0.5120 | 1.85% (2 / 108) | +0.0056 |
| **Phase 19 Local Visual Branch** | EasyOCR 4-field routing | 4 field crops $(256 \times 128)$ | **0.5904** | **0.6086** | **0.5574** | **0.6011** | **76.85% (83 / 108)** | **+0.0478** |
| — *Phase 19 Amount Subcategory* | *OCR Amount field* | *Single crop $(256 \times 128)$* | *0.9925* | *0.9764* | *0.8519* | *0.6751* | *80.00% (8 / 10)* | *+0.3066* |
| **Phase 20 Dense MIL** | Uniform $4 \times 2$ grid | 8 dense tiles $(224 \times 224)$ | 0.5007 | 0.5006 | 0.5000 | 0.4993 | 0.00% (0 / 108) | 0.0000 |
| **Phase 21 CRAFT Word (Max)** | Pure CRAFT word CCs | ~53 word crops $(256 \times 128)$ | **0.4960** | **0.5008** | **0.4954** | **0.5173** | **100.00% (108 / 108)** | **-0.0005** |
| **Phase 21 CRAFT Word (Top-3)** | Pure CRAFT word CCs | Mean of top 3 crops | 0.5000 | 0.5067 | 0.4839 | 0.5012 | 100.00% (108 / 108) | -0.0010 |
| **Phase 21 CRAFT Word (Top-5)** | Pure CRAFT word CCs | Mean of top 5 crops | 0.5000 | 0.5012 | 0.4868 | 0.5141 | 100.00% (108 / 108) | -0.0008 |
| **Phase 21 CRAFT Word (Conf-Filt)** | CRAFT words ($C_{\text{det}} \ge 0.50$) | Filtered mean | 0.4999 | 0.5046 | 0.4938 | 0.5090 | 53.70% (58 / 108) | -0.0001 |
| **Phase 21 CRAFT Word (Conservative)**| Margin $S_{\text{max}} \notin (0.35, 0.65)$ | Decisive subset | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 100.00% (0 inc) | -0.0005 |

*Note on Historical Reconciliations*: Phase 17 is restored to its authoritative validation benchmark of **$0.6677$** (overall native target crop) and **$0.9889$** (amount crop), labeled strictly as an **Oracle Upper Bound**. Phase 19 is restored to its authoritative audited benchmark of **$0.5574$ validation ROC-AUC** with **$76.85\%$ decisive coverage** ($0.8519$ on amount).

---

### Scientific Verdict & Production Gate

1. **Successful Localization vs Aggregation Breakdown**:
   - CRAFT text detection **successfully resolved the Phase 19 localization limitation**. For screenshots in the unrouted `other` category (e.g. `P4-0281_edited`), CRAFT placed word bounding boxes with high overlap ($\text{IoU} = 0.5707$, $100\%$ containment) over the manipulated text.
   - However, scoring these candidates through the frozen Phase 19 classifier resulted in **complete screenshot-level performance collapse** ($\text{ROC-AUC} \approx 0.4954\text{--}0.4960$).
2. **Root Cause 1: Catastrophic False Positive Accumulation**:
   - In Phase 19, EasyOCR routed at most $3 \text{ to } 4$ structured crops per screenshot.
   - In Phase 21, CRAFT extracted an average of **$52.95 \pm 3.12$ word crops per screenshot**.
   - If an individual clean crop has a false positive rate of even $\alpha = 0.05$ ($95\%$ specificity), the screenshot-level false alarm rate under maximum pooling scales exponentially:
     $$P(\text{False Alarm}) = 1 - (1 - \alpha)^K = 1 - (0.95)^{53} = 1 - 0.066 = 0.934 \quad (93.4\%)$$
   - Empirically, **100% of clean screenshots** contained at least one non-tampered UI word (e.g. "Google Pay", status banners, buttons, small footer links) that triggered a visual score $P \ge 0.6615$ (clean screenshot mean: $0.9347$, median: $0.9666$).
3. **Root Cause 2: Domain Shift of the Local Classifier**:
   - The Phase 19 ResNet18 classifier was trained exclusively on 3,291 crops extracted from 4 structured transaction fields (`amount`, `receiver`, `transaction_id`, `date`).
   - When exposed to arbitrary UI words, punctuation, and navigation buttons, the out-of-distribution feature activations saturated the classifier's positive logit, generating pervasive high-scoring false alarms across clean screenshots.
4. **Gate Recommendation**: **REJECT Phase 21 for production deployment.**
   - Do **NOT** integrate unconstrained CRAFT word extraction into the FastAPI backend or frontend.
   - Maintain the **Phase 19 OCR-guided local branch** as the active visual baseline.

---

## 2. Pre-Scoring Implementation Audit

Before executing scoring, a comprehensive code audit of [`merge_and_score_crops.py`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase21_craft_word_forensics/merge_and_score_crops.py) was performed to guarantee leakage safety and mathematical fidelity.

| Audit Parameter | Verification Requirement | Implementation Details | Finding |
| :--- | :--- | :--- | :---: |
| **Crop Extraction Logic** | Extract bounding box directly from RGB image | `img_rgb.crop(exp_box)` using `expanded_bbox` | **PASS** |
| **Padding & Boundary Clamping** | Safe margin expansion within $[0, W] \times [0, H]$ | $m = \max(4, \text{int}(0.15 \times \min(w, h)))$, strictly clipped to image dimensions | **PASS (0 breaches)** |
| **Coordinate Retention** | Retain original unpadded coordinates & dimensions | Recorded `orig_x1..y2`, `exp_x1..y2`, $W, H$, aspect ratio in all manifests | **PASS** |
| **Letterbox & Normalization** | Exact match to Phase 19 training preprocessing | $(256, 128)$ canvas, $(18, 18, 18)$ dark background, LANCZOS resize, ImageNet $\mu/\sigma$ | **PASS ($\Delta_{\max} = 0.0$)** |
| **Fold Checkpoint Assignment** | Score training samples via held-out fold checkpoint | Screenshot `pair_id` mapped via `fold_assignments.csv`; Fold $k$ scored by `fold{k}.pt` | **PASS (Strict OOF)** |
| **Training Pair Exclusion** | Ensure training pair is absent from scoring checkpoint | Checkpoint `fold{k}.pt` was trained strictly on Folds $\{0,1,2,3,4\} \setminus \{k\}$ | **PASS (Zero Leakage)** |
| **Pair Consistency** | Same checkpoint for clean and edited pair members | Both members share `pair_id` and are scored by the identical fold checkpoint | **PASS (100% paired)** |
| **Validation Scoring** | Independent evaluation on frozen 108 validation images | Quarantined partition scored via 5-fold ensemble average ($\frac{1}{5}\sum_{f=0}^4 P_f$) | **PASS** |
| **Zero Annotation Leakage** | Independence from labels, GT boxes, or categories | Candidate extraction and scoring executed blind to all ground-truth metadata | **PASS (100% Blind)** |

---

## 3. Checkpoint Provenance and Immutability Verification

All five Phase 19 ResNet18 fold checkpoints were loaded with `weights=None` and strict state dictionary matching. Checkpoint binaries were verified byte-identical against their authoritative hashes:

| Fold Index | Checkpoint File | Checkpoint SHA-256 Digest | Phase 19 Best Epoch | Training Crop Count | Held-Out Val AUC |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **Fold 0** | `best_crop_resnet18_fold0.pt` | `17d2fafe8c9fd4075726d11e5c7128c68b5e3195d659d578e3859aa87d42bbc9` | 8 | 2,632 | 0.7100 |
| **Fold 1** | `best_crop_resnet18_fold1.pt` | `cb0326ee515459298ec99f00c473f1d4fe6ea06d779f9e7de7e3745647d1045c` | 3 | 2,633 | 0.7213 |
| **Fold 2** | `best_crop_resnet18_fold2.pt` | `d0a4a504b6242eceba2d95a7b95ab484e9d49a866522992f861369d40f8b5468` | 5 | 2,633 | 0.7417 |
| **Fold 3** | `best_crop_resnet18_fold3.pt` | `8f1fe61d6d0fba89d19a01e9766b9c613eab40ca70049d04102ea40d9c2ae3b0` | 6 | 2,633 | 0.7607 |
| **Fold 4** | `best_crop_resnet18_fold4.pt` | `6ad10f181c6b0bb137d57459ff99ab0564f6de141181ec5bb07ae1d2b370a20f` | 3 | 2,633 | 0.7104 |

---

## 4. CRAFT Candidate Extraction, Throughput & Geometry Audit

Four parallel CPU worker processes extracted word-level bounding boxes from the 976 screenshots.

### Shard Processing Summary

| Shard | Worker Task | Index Range | Processed Records | JSON Output Path | Size | SHA-256 Digest | Runtime | Throughput |
| :---: | :---: | :---: | :---: | :--- | :---: | :--- | :---: | :---: |
| **Shard 0** | `task-4363` | `[0:244]` | 244 / 244 | [`craft_shard_0.json`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase21_craft_word_forensics/data/craft_shard_0.json) | 4.76 MB | `b5ea66eeacdcf506d9f146835c69c801a77b335e4c5f6ab7779ba795215f51aa` | 784.01s | 0.31 screenshots/s |
| **Shard 1** | `task-4366` | `[244:488]` | 244 / 244 | [`craft_shard_1.json`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase21_craft_word_forensics/data/craft_shard_1.json) | 4.77 MB | `6228377d793b5aeb476a9c4684c14e2b235f95fd14c5a5d15f40dcaec3320b66` | 1164.19s | 0.21 screenshots/s |
| **Shard 2** | `task-4368` | `[488:732]` | 244 / 244 | [`craft_shard_2.json`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase21_craft_word_forensics/data/craft_shard_2.json) | 4.72 MB | `8b9c6acfe960e439ef5ac6b7bb67a061a6de1dcbad986afabd56381b2b9af637` | 1159.11s | 0.21 screenshots/s |
| **Shard 3** | `task-4370` | `[732:976]` | 244 / 244 | [`craft_shard_3.json`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase21_craft_word_forensics/data/craft_shard_3.json) | 4.95 MB | `3802196db723ef865dd6fd71d335b76aba94101ed512570034738979c5b9935e` | 1056.30s | 0.23 screenshots/s |

### Candidate Accounting and Scoring Throughput
- **Total Candidates Detected**: Exactly **$51,684$ candidate word regions**.
- **OOF Scored Crops**: Exactly **$45,426$ crops** in [`oof_crop_predictions.csv`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase21_craft_word_forensics/results/oof_crop_predictions.csv).
- **Validation Scored Crops**: Exactly **$6,258$ crops** in [`val_crop_predictions.csv`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase21_craft_word_forensics/results/val_crop_predictions.csv).
- **Accounting Verification**: $45,426 + 6,258 = 51,684$ ($100.0\%$, exactly 0 dropped or failed crops).
- **Scoring Throughput**:
  - Training OOF: $57.97\text{ s}$ for 868 screenshots $\implies$ **$15.0$ screenshots/s** or **$783.6$ crops/s**.
  - Validation: $27.48\text{ s}$ for 108 screenshots $\implies$ **$3.9$ screenshots/s** or **$227.7$ crops/s** (5-fold ensemble = $1,138.6$ model evaluations/s).

### Candidate Box Morphology & Density Breakdown
- **Standard Word Tokens** ($w \ge 10, h \ge 8$): **51,055 candidates (98.78%)**
- **Tiny Characters / Punctuation** ($w < 10$ or $h < 8$): **44 candidates (0.09%)** (retained, not pruned)
- **Elongated Line Tokens** ($w / h > 8.0$): **585 candidates (1.13%)** (retained, not pruned)
- **Excessively Large Monolithic Patches** ($w > 0.85W$ or $h > 0.35H$): **0 candidates (0.00%)**
- **Intra-Screenshot Spatial Overlaps**: $1,358,900$ intra-image candidate pairs evaluated. Exactly **0 pairs** exhibited $\text{IoU} \ge 0.50$, confirming discrete, non-redundant word boundary extraction.

---

## 5. Predefined Screenshot-Level Aggregation Strategies

To prevent post-hoc metric hacking, five aggregation strategies were defined prior to validation scoring:

1. **Maximum Candidate Score ($S_{\text{max}}$)**:
   $$S_{\text{max}}(I) = \max_{c \in C_i} s_c$$
   Reflects the standard forensic assumption that a screenshot is tampered if any single text region exhibits tampering evidence.
2. **Mean of Top 3 Candidate Scores ($S_{\text{top3}}$)**:
   $$S_{\text{top3}}(I) = \frac{1}{\min(|C_i|, 3)} \sum_{c \in \text{Top-3}} s_c$$
   Dampens single-crop outlier spikes by requiring multi-token agreement.
3. **Mean of Top 5 Candidate Scores ($S_{\text{top5}}$)**:
   $$S_{\text{top5}}(I) = \frac{1}{\min(|C_i|, 5)} \sum_{c \in \text{Top-5}} s_c$$
   Further reduces sensitivity to isolated noise artifacts.
4. **Detector-Confidence Filtered Mean ($S_{\text{conf\_filt}}$)**:
   $$S_{\text{conf\_filt}}(I) = \frac{1}{|C_i^{(\ge 0.50)}|} \sum_{c \in C_i^{(\ge 0.50)}} s_c \quad \text{where } C_i^{(\ge 0.50)} = \{c \in C_i : C_{\text{detector}}(c) \ge 0.50\}$$
   Scores only high-confidence scene text regions ($C_{\text{detector}} \ge 0.50$), falling back to screenshot mean if none qualify.
5. **Conservative Strategy with Inconclusive Margin**:
   $$\hat{y}(I) = \begin{cases} \text{Tampered (1)} & \text{if } S_{\text{max}}(I) \ge 0.65 \\ \text{Clean (0)} & \text{if } S_{\text{max}}(I) \le 0.35 \\ \text{INCONCLUSIVE} & \text{if } 0.35 < S_{\text{max}}(I) < 0.65 \end{cases}$$
   Rejects ambiguous scores within margin $(0.35, 0.65)$ around the decision threshold.

---

## 6. Comprehensive Screenshot-Level Evaluation Results

### 1. Training Set Out-Of-Fold (OOF) Evaluation (868 Screenshots)

| Strategy | ROC-AUC | PR-AUC | Clean Mean ($\mu \pm \sigma$) | Edited Mean ($\mu \pm \sigma$) | Separation Delta ($\Delta$) | Accuracy (at 0.5) | Recall (at 0.5) | Specificity (at 0.5) | FPR (at 0.5) | Macro-F1 (at 0.5) | Decisive Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Max Score ($S_{\text{max}}$)** | **0.4960** | **0.5008** | $0.9347 \pm 0.0769$ | $0.9340 \pm 0.0765$ | **-0.0007** | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 100.00% (868 / 868) |
| **Top-3 Mean ($S_{\text{top3}}$)** | 0.5000 | 0.5067 | $0.9022 \pm 0.0910$ | $0.9020 \pm 0.0906$ | -0.0003 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 100.00% (868 / 868) |
| **Top-5 Mean ($S_{\text{top5}}$)** | 0.5000 | 0.5012 | $0.8788 \pm 0.1015$ | $0.8786 \pm 0.1010$ | -0.0002 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 100.00% (868 / 868) |
| **Conf-Filt Mean** | 0.4999 | 0.5046 | $0.5595 \pm 0.1356$ | $0.5595 \pm 0.1356$ | 0.0000 | 0.5046 | 0.5806 | 0.4286 | 0.5714 | 0.5017 | 59.68% (518 / 868) |
| **Conservative Strategy** | 0.5000 | 0.5000 | — | — | -0.0007 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | **100.00% (0 inc)** |

> [!NOTE]
> **Conservative Inconclusive Audit Finding**: The Conservative Strategy exhibited $100\%$ decisive coverage and $0$ inconclusive cases because across all 868 training screenshots, $\min(S_{\text{max}}) = 0.6615 > 0.65$. With $100\%$ of screenshots scoring above $0.65$, all samples were classified as tampered, collapsing the strategy to an all-positive decision.

#### Confusion Matrices on Training OOF (868 Screenshots)

| Strategy | Diagnostic Threshold | True Positive (TP) | False Positive (FP) | True Negative (TN) | False Negative (FN) | Precision | Recall | Specificity | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Max Score** | $\tau = 0.30$ | 434 | 434 | 0 | 0 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| **Max Score** | $\tau = 0.50$ | 434 | 434 | 0 | 0 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| **Max Score** | $\tau = 0.70$ | 429 | 430 | 4 | 5 | 0.4994 | 0.9885 | 0.0092 | 0.9908 | 0.0115 |
| **Top-3 Mean** | $\tau = 0.50$ | 434 | 434 | 0 | 0 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| **Top-5 Mean** | $\tau = 0.50$ | 434 | 434 | 0 | 0 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| **Conf-Filt Mean**| $\tau = 0.50$ | 252 | 248 | 186 | 182 | 0.5040 | 0.5806 | 0.4286 | 0.5714 | 0.4194 |

---

### 2. Frozen Validation Set Evaluation (108 Screenshots)

| Strategy | ROC-AUC | PR-AUC | Clean Mean ($\mu \pm \sigma$) | Edited Mean ($\mu \pm \sigma$) | Separation Delta ($\Delta$) | Accuracy (at 0.5) | Recall (at 0.5) | Specificity (at 0.5) | FPR (at 0.5) | Macro-F1 (at 0.5) | Decisive Coverage ($\notin [0.45, 0.55]$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Max Score ($S_{\text{max}}$)** | **0.4954** | **0.5173** | $0.8228 \pm 0.0378$ | $0.8223 \pm 0.0379$ | **-0.0005** | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 100.00% (108 / 108) |
| **Top-3 Mean ($S_{\text{top3}}$)** | 0.4839 | 0.5012 | $0.7958 \pm 0.0274$ | $0.7948 \pm 0.0269$ | -0.0010 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 100.00% (108 / 108) |
| **Top-5 Mean ($S_{\text{top5}}$)** | 0.4868 | 0.5141 | $0.7767 \pm 0.0255$ | $0.7759 \pm 0.0255$ | -0.0008 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 100.00% (108 / 108) |
| **Conf-Filt Mean** | 0.4938 | 0.5090 | $0.5537 \pm 0.0186$ | $0.5536 \pm 0.0184$ | -0.0001 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 53.70% (58 / 108) |
| **Conservative Strategy** | 0.5000 | 0.5000 | — | — | -0.0005 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | **100.00% (0 inc)** |

#### Confusion Matrices on Frozen Validation (108 Screenshots)

| Strategy | Diagnostic Threshold | TP | FP | TN | FN | Precision | Recall | Specificity | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Max Score** | $\tau = 0.30$ | 54 | 54 | 0 | 0 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| **Max Score** | $\tau = 0.50$ | 54 | 54 | 0 | 0 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| **Max Score** | $\tau = 0.70$ | 54 | 54 | 0 | 0 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| **Top-3 Mean** | $\tau = 0.50$ | 54 | 54 | 0 | 0 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| **Top-5 Mean** | $\tau = 0.50$ | 54 | 54 | 0 | 0 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| **Conf-Filt Mean**| $\tau = 0.50$ | 54 | 54 | 0 | 0 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

---

## 7. Matched-Pair Score Analysis & Spatial UI Verification

Evaluating pairwise differences ($\Delta_{\text{pair}} = S_{\text{edited}} - S_{\text{clean}}$) isolates tampering signals from background layout variations:

| Partition | Total Pairs | Mean Pair $\Delta$ | Median Pair $\Delta$ | Std Pair $\Delta$ | Pairs Edited $>$ Clean | Pairs Clean $>$ Edited | Pairs Tied ($|\Delta| \le 10^{-4}$) | Separation Success Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Set OOF** | 434 | **-0.0007** | 0.0000 | 0.0199 | 24 (5.53%) | 33 (7.60%) | **377 (86.87%)** | **5.53%** |
| **Frozen Validation Set**| 54 | **-0.0005** | 0.0000 | 0.0076 | 3 (5.56%) | 4 (7.41%) | **47 (87.04%)** | **5.56%** |

### Candidate-Level Verification of Tied Pairs
A spatial coordinate audit across all 377 tied training pairs demonstrated:
- In **$373 / 377$ tied pairs ($98.94\%$)**, the winning candidate bounding box on the clean screenshot and the winning candidate bounding box on the edited screenshot shared an **$\text{IoU} \ge 0.80$** ($\text{mean IoU} = 0.9903$).
- Qualitative inspection confirmed that these winning boxes bounded identical background UI tokens (e.g. status bar time, battery percentage, Google Pay header banner) that exist identically on both images of the transaction pair. The identical feature activations on these pristine UI tokens produced identical scores ($P \approx 0.95\text{--}0.99$), explaining the $86.87\%$ tie rate.

---

## 8. Breakdown by Tampering Category and Layout Family

### Protocol A: Matched-Pair Within-Category Evaluation (1:1 Balanced Design)

| Split | Category | Exact Pairs ($N_{\text{pos}} = N_{\text{neg}}$) | Total Images | ROC-AUC | PR-AUC | Mean Pair $\Delta$ | Median Pair $\Delta$ | Std Pair $\Delta$ | Edited $>$ Clean | Clean $>$ Edited | Tied Pairs ($|\Delta| \le 10^{-4}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OOF** | `amount` | 91 | 182 | 0.4763 | 0.4963 | -0.0043 | +0.0000 | 0.0358 | 10.99% (10) | 20.88% (19) | **68.13% (62)** |
| **OOF** | `date_time` | 91 | 182 | 0.5018 | 0.5114 | +0.0017 | +0.0000 | 0.0107 | 5.49% (5) | 2.20% (2) | **92.31% (84)** |
| **OOF** | `other` | 84 | 168 | 0.5036 | 0.5072 | +0.0012 | +0.0000 | 0.0070 | 4.76% (4) | 1.19% (1) | **94.05% (79)** |
| **OOF** | `recipient` | 85 | 170 | 0.4955 | 0.5112 | -0.0019 | +0.0000 | 0.0216 | 4.71% (4) | 12.94% (11) | **82.35% (70)** |
| **OOF** | `transaction_id`| 83 | 166 | 0.5001 | 0.5101 | +0.0000 | +0.0000 | 0.0000 | 1.20% (1) | 0.00% (0) | **98.80% (82)** |
| **Val** | `amount` | 5 | 10 | 0.4400 | 0.4800 | -0.0054 | +0.0000 | 0.0109 | 0.00% (0) | 20.00% (1) | **80.00% (4)** |
| **Val** | `date_time` | 6 | 12 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 0.0000 | 0.00% (0) | 0.00% (0) | **100.00% (6)** |
| **Val** | `other` | 16 | 32 | 0.4961 | 0.5463 | +0.0003 | +0.0000 | 0.0120 | 6.25% (1) | 6.25% (1) | **87.50% (14)** |
| **Val** | `recipient` | 14 | 28 | 0.5000 | 0.5682 | -0.0001 | +0.0000 | 0.0019 | 14.29% (2) | 14.29% (2) | **71.43% (10)** |
| **Val** | `transaction_id`| 13 | 26 | 0.5000 | 0.5184 | +0.0000 | +0.0000 | 0.0000 | 0.00% (0) | 0.00% (0) | **100.00% (13)** |

### Protocol B: Unmatched Category Edited vs Global Clean Population

| Split | Category | Edited ($N$) | Global Clean ($N$) | Prevalence | ROC-AUC | PR-AUC | Edited Mean | Clean Mean | Delta ($\Delta$) | Recall (at 0.5) | Precision (at 0.5) | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OOF** | `amount` | 91 | 434 | 17.33% | 0.5027 | 0.1732 | 0.9381 | 0.9347 | +0.0034 | 1.0000 | 0.1733 | 0.2955 |
| **OOF** | `date_time` | 91 | 434 | 17.33% | 0.5050 | 0.1812 | 0.9376 | 0.9347 | +0.0029 | 1.0000 | 0.1733 | 0.2955 |
| **OOF** | `other` | 84 | 434 | 16.22% | 0.4868 | 0.1606 | 0.9342 | 0.9347 | -0.0005 | 1.0000 | 0.1622 | 0.2791 |
| **OOF** | `recipient` | 85 | 434 | 16.38% | 0.4998 | 0.1864 | 0.9287 | 0.9347 | -0.0060 | 1.0000 | 0.1638 | 0.2815 |
| **OOF** | `transaction_id`| 83 | 434 | 16.05% | 0.4841 | 0.1599 | 0.9308 | 0.9347 | -0.0039 | 1.0000 | 0.1605 | 0.2767 |
| **Val** | `amount` | 5 | 54 | 8.47% | 0.4963 | 0.1468 | 0.8244 | 0.8228 | +0.0016 | 1.0000 | 0.0847 | 0.1562 |
| **Val** | `date_time` | 6 | 54 | 10.00% | 0.3719 | 0.0981 | 0.8076 | 0.8228 | -0.0152 | 1.0000 | 0.1000 | 0.1818 |
| **Val** | `other` | 16 | 54 | 22.86% | 0.5052 | 0.2854 | 0.8233 | 0.8228 | +0.0005 | 1.0000 | 0.2286 | 0.3721 |
| **Val** | `recipient` | 14 | 54 | 20.59% | 0.4974 | 0.2308 | 0.8230 | 0.8228 | +0.0002 | 1.0000 | 0.2059 | 0.3415 |
| **Val** | `transaction_id`| 13 | 54 | 19.40% | 0.5377 | 0.2285 | 0.8265 | 0.8228 | +0.0037 | 1.0000 | 0.1940 | 0.3250 |

---

## 9. Crop-Level vs. Screenshot-Level Metrics

To isolate whether the failure originated in the crop classifier itself or in the aggregation step, unambiguous candidate crops were evaluated independently:

- Unambiguous Positive Crop: Overlaps GT tamper box with $\text{IoU} \ge 0.30$ or $\text{coverage} \ge 0.50$ (Label 1).
- Unambiguous Negative Crop: On clean image OR non-overlapping on edited image ($\text{IoU} < 0.05$ and $\text{cov} < 0.05$) (Label 0).
- Ambiguous crops ($0.05 \le \text{overlap} < 0.30$) were excluded from crop-level metrics.

| Evaluation Metric | Training OOF Crops ($N = 45,065$) | Frozen Validation Crops ($N = 6,224$) |
| :--- | :---: | :---: |
| **Positive Tampered Crops** | 572 (1.27% prevalence) | 86 (1.38% prevalence) |
| **Negative Clean Crops** | 44,493 (98.73%) | 6,138 (98.62%) |
| **Crop-Level ROC-AUC** | **0.5283** | **0.5785** |
| **Crop-Level PR-AUC** | **0.0157** | **0.0200** |
| **Crop-Level Accuracy (at 0.5)** | 0.4457 | 0.3292 |
| **Crop-Level Precision (at 0.5)** | 0.0136 | 0.0154 |
| **Crop-Level Recall (at 0.5)** | 0.5944 | 0.7558 |
| **Crop-Level Specificity (at 0.5)** | 0.4438 | 0.3232 |
| **Crop-Level False Positive Rate (FPR)** | **55.62%** | **67.68%** |
| **Crop-Level False Negative Rate (FNR)** | 40.56% | 24.42% |

### Critical Finding on Candidate Discrimination
The local ResNet18 classifier achieved an FPR of **$55.62\%$ on negative CRAFT crops**. Because the negative crops included thousands of non-field UI elements (icons, badges, small buttons, URLs), the model—trained solely on clean transaction text fields—falsely flagged over half of all clean UI elements as tampered.

---

## 10. Candidate Geometry and Score Correlations

Pearson correlation coefficients ($\rho$) between candidate tampering scores and geometric/detection attributes:

| Paired Variables | Pearson Correlation ($\rho$) | Forensic Interpretation |
| :--- | :---: | :--- |
| **Score vs Width ($w$)** | **-0.0581** | Very weak negative correlation with box width. |
| **Score vs Height ($h$)** | **+0.0729** | Taller glyphs show marginally higher tampering scores. |
| **Score vs Area ($w \times h$)** | **+0.0381** | Bounding box area does not meaningfully dictate tampering score. |
| **Score vs Aspect Ratio ($w/h$)** | **-0.1384** | Elongated text lines exhibit slightly lower scores than compact words. |
| **Score vs CRAFT Confidence ($C_{\text{det}}$)** | **+0.0324** | Detector confidence is completely independent of tampering probability. |

These correlations prove that the classifier did not latch onto trivial box size shortcuts. The failure was strictly conceptual: out-of-domain UI text elements triggered false-positive feature activations.

---

## 11. Comparison Across Historical CBL-04 Phases

| Phase | Pipeline Architecture | Candidate Selection | Supervision Scope | Evaluation Unit | Validation ROC-AUC | Decisive Coverage | Matched-Pair Delta ($\Delta$) | Production Gate Verdict |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Phase 15** | Global Multi-Task ResNet18 ($320 \times 160$) | Whole image | Screenshot-level | Screenshot ($n=108$) | **0.4938** | 0.00% | -0.0005 | **REJECTED** (spatial dilution) |
| **Phase 17** | Oracle Target Field Crop ($224 \times 224$) | **Oracle Ground Truth** | Crop-level | **Crop ($n=108$)** | **0.6677** | **62.04%** | **+0.1938** | **ORACLE UPPER BOUND ONLY** |
| — *Phase 17 Amount*| *Oracle Amount Crop* | *Oracle Ground Truth* | *Crop-level* | *Crop ($n=10$)* | *0.9889* | *90.00%* | *+0.6214* | *Oracle upper bound on amount* |
| **Phase 18** | Global CNN + Regex Routing | Keyword regex | Screenshot fusion | Screenshot ($n=108$) | **0.5185** | 1.85% | +0.0056 | **REJECTED** (brittle heuristics) |
| **Phase 19** | EasyOCR Keyword Routing + ResNet18 | 4 structured fields | Crop-level / Max | Screenshot ($n=108$) | **0.5574** | **76.85%** | **+0.0478** | **ACCEPTED (Active Visual Baseline)** |
| — *Phase 19 Amount*| *OCR Amount Field* | *Inference EasyOCR* | *Crop-level / Max* | *Screenshot ($n=10$)* | *0.8519* | *80.00%* | *+0.3066* | *Primary high-confidence detector* |
| **Phase 20** | Uniform $4 \times 2$ Grid + Attention MIL | 8 uniform tiles | Bag-level MIL | Screenshot ($n=108$) | **0.5002** | 0.00% | 0.0000 | **REJECTED** (layout prior collapse) |
| **Phase 21** | Pure CRAFT Word Detections + ResNet18 | Pure scene text CCs | Zero OCR text / Max | Screenshot ($n=108$) | **0.4954** | **100.00%** | **-0.0005** | **REJECTED (FP accumulation collapse)** |

---

## 12. Test Partition Quarantine and Data Governance

1. **Test Set Quarantine**: The 24 test screenshots in `metadata/phase6/test_manifest.csv` were **never loaded, accessed, or scored** during Phase 21. Checksum `3f14f6a7...` remained unmodified throughout execution.
2. **Historical Exposure Disclosure**: Sample `P4-0036` was inspected read-only during Phase A.1 preflight. In accordance with strict data integrity standards, the test partition is **not described as pristine**; it is designated as **quarantined with historical read-only exposure of sample P4-0036**.
3. **Repository Immutability**: All Phase 19 checkpoints, production service files (`app/services/`), configuration settings, and manifest files remained strictly untouched.
4. **Regression Safety**: Backend test suite passed **159 / 159 tests (100%)** via `pytest tests/`, and the frontend built cleanly in **127ms**.

---

## 13. Production Recommendation and Next Steps

### Production Recommendation
**DO NOT DEPLOY Phase 21 to production.**
- The Phase 21 OCR-free word-level pipeline achieved chance-level screenshot discrimination ($\text{ROC-AUC} = 0.4954$) due to false positive accumulation across ~53 unconstrained word crops per image.
- **Phase 19 remains the superior, production-ready local visual branch** ($\text{ROC-AUC} = 0.5574$ overall on validation, $0.8519$ on the critical `amount` field, $+0.0478$ matched-pair separation).

### Technical Lessons for Future Research
1. **Unconstrained Text Candidate Explosion**: Extracting every scene text token creates an intolerable false positive multiplier ($K \approx 53$) for any local visual classifier with non-zero FPR.
2. **Field Pruning is Essential**: For local visual classifiers to succeed at screenshot level, candidate count must be restricted to $K \le 5$ high-value transaction elements.
3. **Requirement for Hard Negative Retraining**: If scene-text word detection is to be used in future research, the classifier must be explicitly trained with hard negative mining on hundreds of arbitrary clean UI tokens to drive crop-level FPR down from $55.6\%$ to $< 0.1\%$.
