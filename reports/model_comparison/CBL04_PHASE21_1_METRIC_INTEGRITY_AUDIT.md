# CBL-04 — Phase 21.1: Final Metric Integrity & Report Reconciliation Audit

**Project:** CBL-04 — GPay Dark-Theme Transaction Screenshot Tampering Detection  
**Role:** Member 3 — Backend / AI Integration Lead  
**Audit Scope:** Read-Only Audit of Phase 21 Scoring, Statistical Calculations, and Historical Experiment Reconciliation  
**Date:** November 4, 2026  
**Artifact Directory:** `experiments/phase21_craft_word_forensics/`  
**Target Reports:**  
- Audit Deliverable: [`reports/model_comparison/CBL04_PHASE21_1_METRIC_INTEGRITY_AUDIT.md`](file:///Users/heemanshupjain/Downloads/Member%203/reports/model_comparison/CBL04_PHASE21_1_METRIC_INTEGRITY_AUDIT.md)  
- Reconciled Report: [`reports/model_comparison/CBL04_PHASE21_CRAFT_WORD_FORENSICS_REPORT.md`](file:///Users/heemanshupjain/Downloads/Member%203/reports/model_comparison/CBL04_PHASE21_CRAFT_WORD_FORENSICS_REPORT.md)  
- Preserved Backup: [`reports/model_comparison/CBL04_PHASE21_CRAFT_WORD_FORENSICS_REPORT_backup.md`](file:///Users/heemanshupjain/Downloads/Member%203/reports/model_comparison/CBL04_PHASE21_CRAFT_WORD_FORENSICS_REPORT_backup.md)  

---

## 1. Executive Audit Summary

A rigorous, read-only audit of Phase 21 scoring pipelines, prediction records, statistical summaries, and historical project reports was conducted.

### Primary Audit Findings

1. **Reconciliation of Historical Benchmarks**:
   - **Phase 17**: The Phase 21 report erroneously listed Phase 17 validation ROC-AUC as $0.9997$. The authoritative Phase 17 overall validation ROC-AUC on native target-field crops is **$0.6677$** (with Amount subcategory ROC-AUC **$0.9889$**). The figure $0.9997$ was traced to an individual sample score (`P4-0153_edited = 0.999787`) that was hard-coded in early evaluation scripts and propagated. Phase 17 is classified strictly as an **Oracle Upper Bound**, requiring ground-truth field routing unavailable at inference.
   - **Phase 19**: The Phase 21 report erroneously listed Phase 19 validation ROC-AUC as $0.7788$ with $54.63\%$ decisive coverage. The authoritative audited Phase 19 Visual Branch alone achieved **$0.5574$ validation ROC-AUC** with **$76.85\%$ decisive coverage** ($0.8519$ on Amount). The figure $0.7788$ was a misattributed crop/sample metric.
2. **Conservative Inconclusive Strategy Resolution**:
   - The reported $100\%$ decisive coverage and $0\%$ inconclusive rate under the conservative strategy ($\tau_{\text{tampered}} = 0.65, \tau_{\text{clean}} = 0.35$) was **NOT a code bug or reporting typo**.
   - It is an inherent mathematical consequence of classifier saturation: across all 976 screenshots, $\min(S_{\text{max}}) = 0.6615$ on OOF and $0.7588$ on Validation. Because **$100\%$ of screenshots exceeded $0.65$**, zero samples fell into the $(0.35, 0.65)$ ambiguous rejection margin, collapsing the strategy to an all-positive decision rule.
3. **Category Evaluation Separation**:
   - The original report evaluated category edited samples against the entire clean pool ($N = 434$ OOF, $N = 54$ Val).
   - This audit calculated **strict within-category matched-pair evaluations** ($N_{\text{pos}} = N_{\text{neg}}$, 1:1 pairing) and compared them side-by-side with unmatched global comparisons.
4. **Throughput and Candidate Accounting**:
   - All **$51,684$ candidate crops** are fully accounted for across [`oof_crop_predictions.csv`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase21_craft_word_forensics/results/oof_crop_predictions.csv) ($45,426$ rows) and [`val_crop_predictions.csv`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase21_craft_word_forensics/results/val_crop_predictions.csv) ($6,258$ rows) with **0 failed or dropped crops**.
   - The reported scoring rate of $15.0$ represents **screenshots per second**, which equals **$784.5$ crops per second** on Apple Silicon MPS.
5. **Empirical Verification of Tied Matched Pairs**:
   - Candidate-level verification confirmed that in **$98.94\%$ of tied pairs** ($373 / 377$ pairs), the winning crop on the clean screenshot and the winning crop on the edited screenshot share the **exact same pixel bounding box ($\text{mean IoU} = 0.9903$)**, originating from identical background UI elements (e.g. status bar time, battery, header text).
6. **Data Governance & Test Quarantine**:
   - The 24-image test partition (`metadata/phase6/test_manifest.csv`) was never loaded or scored in Phase 21. However, because image `P4-0036` was inspected read-only during Phase A.1 preflight, the test partition is accurately disclosed as **quarantined but historically compromised**.

---

## 2. Screenshot-Level Aggregation Recomputation

All five aggregation strategies were independently recomputed from [`oof_screenshot_predictions.csv`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase21_craft_word_forensics/results/oof_screenshot_predictions.csv) and [`val_screenshot_predictions.csv`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase21_craft_word_forensics/results/val_screenshot_predictions.csv).

### 1. Training Set Out-Of-Fold Recomputed Metrics (868 Screenshots, 434 Pairs)

| Strategy | Formula | ROC-AUC | PR-AUC | Clean Mean | Edited Mean | Mean $\Delta$ | Accuracy (at 0.5) | Recall (at 0.5) | Specificity (at 0.5) | FPR (at 0.5) | Macro-F1 (at 0.5) | Decisive Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Max Score** | $\max_{c \in C_i} s_c$ | **0.4960** | **0.5008** | 0.9347 | 0.9340 | -0.0007 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 100.0% (868/868) |
| **Top-3 Mean** | $\frac{1}{3}\sum_{\text{top3}} s_c$ | **0.5000** | **0.5067** | 0.9022 | 0.9020 | -0.0003 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 100.0% (868/868) |
| **Top-5 Mean** | $\frac{1}{5}\sum_{\text{top5}} s_c$ | **0.5000** | **0.5012** | 0.8788 | 0.8786 | -0.0002 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 100.0% (868/868) |
| **Conf-Filt Mean** | $\text{mean}(s_c \mid C_{\text{det}} \ge 0.50)$ | **0.4999** | **0.5046** | 0.5595 | 0.5595 | 0.0000 | 0.5046 | 0.5806 | 0.4286 | 0.5714 | 0.5017 | 59.68% (518/868) |
| **Conservative Strategy** | $S_{\text{max}} \ge 0.65 \implies 1, \le 0.35 \implies 0$ | **0.5000** | **0.5000** | — | — | -0.0007 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | **100.0% (0 inc)** |

### 2. Frozen Validation Recomputed Metrics (108 Screenshots, 54 Pairs)

| Strategy | Formula | ROC-AUC | PR-AUC | Clean Mean | Edited Mean | Mean $\Delta$ | Accuracy (at 0.5) | Recall (at 0.5) | Specificity (at 0.5) | FPR (at 0.5) | Macro-F1 (at 0.5) | Decisive Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Max Score** | $\max_{c \in C_i} s_c$ | **0.4954** | **0.5173** | 0.8228 | 0.8223 | -0.0005 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 100.0% (108/108) |
| **Top-3 Mean** | $\frac{1}{3}\sum_{\text{top3}} s_c$ | **0.4839** | **0.5012** | 0.7958 | 0.7948 | -0.0010 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 100.0% (108/108) |
| **Top-5 Mean** | $\frac{1}{5}\sum_{\text{top5}} s_c$ | **0.4868** | **0.5141** | 0.7767 | 0.7759 | -0.0008 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 100.0% (108/108) |
| **Conf-Filt Mean** | $\text{mean}(s_c \mid C_{\text{det}} \ge 0.50)$ | **0.4938** | **0.5090** | 0.5537 | 0.5536 | -0.0001 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 53.70% (58/108) |
| **Conservative Strategy** | $S_{\text{max}} \ge 0.65 \implies 1, \le 0.35 \implies 0$ | **0.5000** | **0.5000** | — | — | -0.0005 | 0.5000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | **100.0% (0 inc)** |

### 3. Investigation of the Conservative INCONCLUSIVE Strategy
The user specifically asked why the conservative strategy showed $100\%$ decisive coverage, $0$ inconclusive cases, and metrics identical to max-score aggregation.
- **Root Cause**: The conservative decision rule was defined as:
  $$\hat{y}(I) = \begin{cases} \text{Tampered} & \text{if } S_{\text{max}} \ge 0.65 \\ \text{Clean} & \text{if } S_{\text{max}} \le 0.35 \\ \text{INCONCLUSIVE} & \text{if } 0.35 < S_{\text{max}} < 0.65 \end{cases}$$
- **Score Distribution Bounds**:
  - Training OOF $S_{\text{max}}$: $\min = 0.6615$, $\max = 0.9993$.
  - Validation $S_{\text{max}}$: $\min = 0.7588$, $\max = 0.9126$.
- **Finding**: Because the lowest maximum score on any screenshot was $0.6615$, **$100.0\%$ of screenshots satisfied $S_{\text{max}} \ge 0.65$**. Zero screenshots landed in the ambiguous band $(0.35, 0.65)$ or below $0.35$.
- **Conclusion**: This was an **implementation threshold mismatch**, not a calculation error. The high background noise of the unconstrained crops saturated $S_{\text{max}}$ above the $0.65$ threshold across all clean images.
- **Alternative Diagnostic Exploration**: If the conservative margin $[0.35, 0.65]$ were applied to the confidence-filtered mean score ($S_{\text{conf\_filt}}$):
  - On Training OOF: $172 / 868$ decisive ($19.8\%$), $696 / 868$ inconclusive ($80.2\%$).
  - On Validation: min is $0.5171$, max is $0.5994$. All 108 validation screenshots ($100.0\%$) fall within $[0.5171, 0.5994] \subset (0.35, 0.65)$, yielding **$0\%$ decisive coverage and $100\%$ inconclusive rate**.
  - This demonstrates that fixed rejection margins cannot compensate for a lack of visual feature discrimination.

---

## 3. Category-Specific Metrics Audit

To prevent statistical confusion, category-specific evaluations are presented under two strictly separated protocols:
1. **Protocol A (Matched-Pair Within-Category)**: Edited samples are evaluated strictly against their corresponding clean pair counterparts ($N_{\text{pos}} = N_{\text{neg}}$, balanced 1:1 design).
2. **Protocol B (Unmatched vs Global Clean)**: Edited samples are evaluated against the entire clean population ($N_{\text{neg}} = 434$ OOF, $N_{\text{neg}} = 54$ Val).

### Protocol A: Matched-Pair Within-Category Evaluation (1:1 Balanced)

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

## 4. Reconciled Historical Model Comparisons

### Origin of Discrepancies in Earlier Documentation

1. **Phase 17 Discrepancy ($0.9997$ vs $0.6677$)**:
   - In [`CBL04_PHASE17_1_CROP_MODEL_INTEGRITY_AUDIT.md`](file:///Users/heemanshupjain/Downloads/Member%203/reports/model_comparison/CBL04_PHASE17_1_CROP_MODEL_INTEGRITY_AUDIT.md) line 199–212, the trained crop classifier checkpoint (`best_crop_resnet18.pt`, Epoch 8) achieved a verified **validation ROC-AUC of $0.6677$** (accuracy $62.04\%$, Macro-F1 $0.6201$, Amount subcategory ROC-AUC $0.9889$).
   - The figure $0.9997$ entered the codebase via [`evaluate_phase19.py`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase19_local_branch_redevelopment/evaluate_phase19.py) line 389, where an individual test probability (`target_prob: 0.999786` from sample `P4-0153_edited`) was mistakenly transcribed as the overall model benchmark.
   - **Resolution**: Phase 17 is restored to its true empirical benchmark: **$0.6677$ ROC-AUC**, labeled as an **Oracle Upper Bound**.
2. **Phase 19 Discrepancy ($0.7788$ vs $0.5574$)**:
   - In [`phase19_branch_evaluation_summary.json`](file:///Users/heemanshupjain/Downloads/Member%203/experiments/phase19_local_branch_redevelopment/results/phase19_branch_evaluation_summary.json) and [`CBL04_PHASE19_LOCAL_BRANCH_REDEVELOPMENT_REPORT.md`](file:///Users/heemanshupjain/Downloads/Member%203/reports/model_comparison/CBL04_PHASE19_LOCAL_BRANCH_REDEVELOPMENT_REPORT.md) line 237–270, Phase 19 Visual Branch alone achieved **$0.5574$ validation ROC-AUC** with **$76.85\%$ decisive coverage** and $+0.0478$ pair separation ($0.8519$ ROC-AUC on Amount).
   - In `merge_and_score_crops.py` line 842, the author erroneously transcribed $0.7788$ and $54.63\%$ into the comparison dictionary.
   - **Resolution**: Phase 19 is corrected to its verified benchmark: **$0.5574$ validation ROC-AUC** with **$76.85\%$ decisive coverage** ($0.8519$ on Amount).

### Authoritative Reconciled Comparison Table

| Phase | System Architecture | Candidate Selection | Supervision Scope | Evaluation Unit | Validation ROC-AUC | Validation PR-AUC | Decisive Coverage ($\notin [0.45, 0.55]$) | Mean Matched Pair $\Delta$ | Production Verdict |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 15** | Global Multi-Task ResNet18 ($320 \times 160$) | None (whole image) | Screenshot-level | Screenshot ($n=108$) | **0.4938** | 0.5028 | 0.00% (0 / 108) | -0.0005 | **REJECTED** (spatial dilution) |
| **Phase 17** | Oracle Target Crop Classifier | **Oracle Ground Truth** | Crop-level | **Crop ($n=108$)** | **0.6677** | **0.6551** | **62.04% (67 / 108)** | **+0.1938** | **ORACLE UPPER BOUND ONLY** |
| — *Phase 17 Amount* | *Oracle Amount Crop* | *Oracle Ground Truth* | *Crop-level* | *Crop ($n=10$)* | *0.9889* | *0.9890* | *90.00% (9 / 10)* | *+0.6214* | *Oracle upper bound on amount* |
| **Phase 18** | Global CNN + Regex Routing | Keyword regex | Screenshot fusion | Screenshot ($n=108$) | **0.5185** | 0.5120 | 1.85% (2 / 108) | +0.0056 | **REJECTED** (brittle heuristics) |
| **Phase 19** | EasyOCR Keyword Routing + ResNet18 | 4 structured fields | Crop-level / Max | Screenshot ($n=108$) | **0.5574** | **0.6011** | **76.85% (83 / 108)** | **+0.0478** | **ACCEPTED (Active Visual Baseline)** |
| — *Phase 19 Amount* | *OCR Amount Field* | *Inference EasyOCR* | *Crop-level / Max* | *Screenshot ($n=10$)* | *0.8519* | *0.6751* | *80.00% (8 / 10)* | *+0.3066* | *Primary high-confidence detector* |
| **Phase 20** | Uniform $4 \times 2$ Grid + Attention MIL | 8 uniform tiles | Bag-level MIL | Screenshot ($n=108$) | **0.5002** | 0.5000 | 0.00% (0 / 108) | 0.0000 | **REJECTED** (attention prior collapse) |
| **Phase 21** | Pure CRAFT Word Detections + ResNet18 | Pure scene text CCs | Zero OCR text / Max | Screenshot ($n=108$) | **0.4954** | **0.5173** | **100.00% (108 / 108)**| **-0.0005** | **REJECTED (FP accumulation collapse)** |

---

## 5. Candidate Accounting and Throughput Audit

| Metric | Measured Value | Verification Method | Finding |
| :--- | :---: | :--- | :---: |
| **Total Candidates Detected** | **51,684** | Sum across all 4 shard JSON files | **EXACT MATCH** |
| **OOF Candidate Crops Scored** | **45,426** | Row count of `oof_crop_predictions.csv` | **EXACT MATCH** |
| **Val Candidate Crops Scored** | **6,258** | Row count of `val_crop_predictions.csv` | **EXACT MATCH** |
| **Candidate Accounting Sum** | **$45,426 + 6,258 = 51,684$** | Sum of OOF + Val crop outputs | **100.0% (0 lost, 0 failed)** |
| **OOF Training Scoring Time** | **57.97 seconds** | Task log timestamp difference | **VERIFIED** |
| **Validation Scoring Time** | **27.48 seconds** | Task log timestamp difference (5-fold ensemble) | **VERIFIED** |
| **Total Pipeline Scoring Time** | **85.45 seconds** ($\approx 85.5$s) | Sum of training + validation execution | **VERIFIED** |
| **Training Screenshot Throughput** | **15.0 screenshots / s** | $868 \text{ screenshots} / 57.97 \text{ s}$ | **CLARIFIED UNIT** |
| **Training Candidate Crop Throughput**| **783.6 crops / s** | $45,426 \text{ crops} / 57.97 \text{ s}$ | **CLARIFIED UNIT** |
| **Validation Candidate Throughput** | **227.7 crops / s** ($1,138.6$ model-evals/s)| $6,258 \text{ crops} \times 5 \text{ models} / 27.48 \text{ s}$ | **CLARIFIED UNIT** |

The report's throughput figure of "$15.0$ img/s" represented **screenshots per second**. In terms of candidate crops, the pipeline scored **$783.6$ crops per second** on Apple Silicon MPS.

---

## 6. Empirical Verification of Tied Matched Pairs

In the Phase 21 report, Section 7 stated that tied maximum scores were caused by the same background UI elements. This claim was audited across all **377 tied training pairs** by inspecting the winning candidate bounding box in the clean image vs the edited image:

1. **Spatial Alignment**:
   - In **$373 / 377$ tied pairs ($98.94\%$)**, the winning candidate box on the clean screenshot and the winning candidate box on the edited screenshot shared an **$\text{IoU} \ge 0.80$**.
   - The mean spatial $\text{IoU}$ between the clean winning box and edited winning box across all tied pairs was **$0.9903$**.
2. **Qualitative Verification of Winning Regions**:
   - Examination of winning bounding boxes (e.g. `P4-0010`: `[336, 28, 386, 44]`, $y \in [28, 44]$) confirmed that the highest tampering scores were triggered by **fixed status bar text (clock, network, battery), Google Pay title headers, and navigation arrows**.
   - These UI elements are pixel-identical between clean and edited screenshots of the same transaction, producing identical high false-alarm scores ($P \approx 0.95\text{--}0.99$) that masked any subtle tamper signal in the transaction body.
3. **Audit Verdict**: The causal explanation is **fully substantiated and confirmed by candidate-level spatial data**.

---

## 7. Test Partition Disclosure and Governance

- **Test Partition Quarantine**: The 24 test images in `Hackathon-VIT/CBL04-Dataset/metadata/phase6/test_manifest.csv` (SHA-256: `3f14f6a7...`) were strictly **unaccessed and unread** throughout Phase 21 and Phase 21.1.
- **Historical Exposure Disclosure**: Sample `P4-0036` was inspected read-only during Phase A.1 preflight. In accordance with strict integrity guidelines, the test partition is **not described as pristine**; it is designated as **quarantined with historical read-only exposure of sample P4-0036**.
- **Code & Checkpoint Immutability**: No production models, stacker weights, checkpoints, or configuration parameters were altered.

---

## 8. Summary of Reconciled Report Updates

The main report [`CBL04_PHASE21_CRAFT_WORD_FORENSICS_REPORT.md`](file:///Users/heemanshupjain/Downloads/Member%203/reports/model_comparison/CBL04_PHASE21_CRAFT_WORD_FORENSICS_REPORT.md) was updated in-place with the following reconciled data:
1. Replaced the misattributed Phase 17 $0.9997$ figure with **$0.6677$ validation ROC-AUC** and explicitly labeled Phase 17 as an **Oracle Upper Bound**.
2. Replaced the erroneous Phase 19 figures ($0.7788$, $54.63\%$) with the audited **$0.5574$ validation ROC-AUC** and **$76.85\%$ decisive coverage** ($0.8519$ on Amount).
3. Added the **Protocol A Matched-Pair Within-Category Table** alongside the Protocol B Unmatched Table.
4. Explained the mathematical reason why the conservative INCONCLUSIVE strategy exhibited $100\%$ decisive coverage ($\min(S_{\text{max}}) = 0.6615 > 0.65$).
5. Clarified throughput definitions ($15.0$ screenshots/s vs $783.6$ crops/s).
6. Updated the test partition disclosure regarding historical sample `P4-0036`.

