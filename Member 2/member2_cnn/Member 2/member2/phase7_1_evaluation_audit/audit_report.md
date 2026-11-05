# CBL-04: Phase 7.1 Independent Technical Audit Report

**Project:** CBL-04 — Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Role:** Member 2 — Tile Forensics CNN & Heatmap Lead  
**Audit Scope:** Verification of Phase 7 Evaluation, Metrics, Test Counts, and Robustness Baselines  
**Audit Verdict:** **PROVISIONALLY APPROVED WITH RECONCILED DOCUMENTATION**  
**Date:** November 2026  

---

## 1. Executive Summary

This independent audit (Phase 7.1) was performed to rigorously evaluate all reported metrics, code implementations, test split integrity, and threshold selections in Phase 7.

### Primary Audit Findings
1. **Dice Discrepancy Reconciled:** The difference between the main test set Dice ($0.174612$) and the robustness baseline Dice ($0.175582$, reported as $0.1756$) was investigated. It is **$100\%$ accounted for by evaluation sample size**: the main evaluation ran across all **150 test screenshots**, whereas the multi-condition robustness suite evaluated a balanced **10-screenshot subset (5 clean, 5 edited)** to preserve tractable CPU execution. The identical model checkpoint, weights, thresholds ($\tau = 0.80, A_{\min} = 50$), and preprocessing code were used.
2. **Mathematical Verification of Screenshot Classification:** All three screenshot-level decision rules were mathematically recomputed from the raw per-image results:
   - **Rule 1 (Any Component $>0\text{ px}$):** TP=75, TN=0, FP=75, FN=0. Accuracy: 50.00%, Specificity: **0.00%**.
   - **Rule 2 (Area $\ge 1,000\text{ px}$):** TP=70, TN=6, FP=69, FN=5. Accuracy: 50.67%, Sensitivity: 93.33%, Specificity: **8.00%**, Precision: 50.36%.
   - **Rule 3 (Peak Prob $\ge 0.80$):** TP=75, TN=0, FP=75, FN=0. Accuracy: 50.00%, Specificity: **0.00%**.
   - **Conclusion:** The model is a **spatial pixel-level localization model**, not a standalone global binary screenshot classifier.
3. **Strict Test Split Integrity:** Verified that all 150 test screenshots (75 clean, 75 tampered) match the Phase 5 split manifest bitwise. Zero group leakage exists across train, validation, and test splits. Zero test data was used for model training or parameter tuning.
4. **Threshold Provenance & Leakage Audit:**
   - Heatmap threshold $\tau = 0.80$ and minimum component area $A_{\min} = 50\text{ px}$ were tuned strictly on the **Phase 5 validation split** (`validation_tuning_results.json`) with zero test set peeking.
   - The $1,000$-pixel screenshot cutoff was an exploratory heuristic introduced in `config.json`, which achieves only $8.00\%$ specificity. It is now explicitly labeled as an uncalibrated heuristic.
5. **Test Discovery Verification:**
   - Standard `unittest discover -s member2` discovers and passes **86 tests** (including Phase 1-7 and Phase 7.1).
   - `member2/performance_improvement/test_improvement_pipeline.py` contains **8 tests** that pass when executed directly.
   - Total verified tests in codebase: **94 tests** (all passing 100%).
6. **Report Language Overhaul:** Overly promotional phrasing ("outstanding", "near-perfect", "high practical utility") has been stripped and replaced with sober, technically precise engineering conclusions in [`corrected_phase7_report.md`](file:///c:/Users/athar/OneDrive/Desktop/Member%202/member2/phase7_1_evaluation_audit/corrected_phase7_report.md).

---

## 2. Screenshot-Level Classification Audit

The per-image results across all 150 held-out test screenshots were audited. Mathematical verification of the confusion matrices and metrics yields:

| Rule Name | Decision Condition | TP | TN | FP | FN | Accuracy | Precision | Recall | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule 1** | $\sum M_{\text{pred}} > 0$ | 75 | 0 | 75 | 0 | 50.00% | 50.00% | 100.00% | **0.00%** | 0.6667 |
| **Rule 2** | $\sum M_{\text{pred}} \ge 1,000\text{ px}$ | 70 | 6 | 69 | 5 | **50.67%** | **50.36%** | **93.33%** | **8.00%** | **0.6542** |
| **Rule 3** | $\max H(x, y) \ge 0.80$ | 75 | 0 | 75 | 0 | 50.00% | 50.00% | 100.00% | **0.00%** | 0.6667 |

### Mathematical Validation:
- Rule 2 Accuracy: $\frac{\text{TP} + \text{TN}}{150} = \frac{70 + 6}{150} = \frac{76}{150} = 50.667\%$.
- Rule 2 Precision: $\frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{70}{70 + 69} = \frac{70}{139} = 50.360\%$.
- Rule 2 Recall: $\frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{70}{70 + 5} = \frac{70}{75} = 93.333\%$.
- Rule 2 Specificity: $\frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{6}{6 + 69} = \frac{6}{75} = 8.000\%$.
- Rule 2 F1-Score: $\frac{2 \times 0.503597 \times 0.933333}{0.503597 + 0.933333} = 0.654206$.

**Audit Verdict:** Confirmed accurate. The rules are not interchangeable; Rules 1 and 3 suffer complete failure of specificity, while Rule 2 provides minimal specificity ($8.00\%$). The model must be documented strictly as an assistive spatial localization overlay.

---

## 3. Robustness Baseline Metric Reconciliation

### Discrepancy Investigated:
- Main Test Evaluation Dice: **$0.174612$**
- Robustness Baseline Table Dice: **$0.175582$** (reported as $0.1756$)
- Difference: $+0.000970$

### Root Cause Analysis:
1. **Sample Size:** The main evaluation evaluated all **150 test screenshots** ($126,270,016$ total pixels). The robustness suite evaluated a balanced subset of **10 screenshots (5 clean, 5 edited)** to allow 14 conditions ($140$ forward passes) to complete within acceptable CPU time bounds.
2. **Pixel Sums on 150-sample set:**
   - $\text{TP} = 42,554$, $\text{FP} = 309,665$, $\text{FN} = 92,640$
   - $\text{Dice} = \frac{2 \times 42,554}{2 \times 42,554 + 309,665 + 92,640} = \frac{85,108}{487,413} = 0.174612$
3. **Pixel Sums on 10-sample subset:**
   - $\text{TP} = 1,948$, $\text{FP} = 14,949$, $\text{FN} = 3,344$
   - $\text{Dice} = \frac{2 \times 1,948}{2 \times 1,948 + 14,949 + 3,344} = \frac{3,896}{22,189} = 0.175582 \approx 0.1756$
4. **Code & Weight Verification:** The model weights (`best_improved_model.pt`), inference batch size, threshold ($\tau = 0.80$), min area ($A_{\min} = 50$), Gaussian sigma ($64.0$), and ImageNet normalization were bitwise identical.

**Audit Verdict:** RECONCILED. The discrepancy is purely sample-size dependent ($10$ vs $150$ images) and mathematically consistent.

---

## 4. Test Split Consistency Audit

We verified the sample IDs across all phases using `test_split_consistency.json`:
- **Phase 5 Split Manifest Test IDs:** $150$ samples ($75$ clean, $75$ tampered).
- **Phase 6 CSV Test IDs:** $150$ samples ($0$ mismatches).
- **Performance Improvement Test IDs:** $150$ samples ($0$ mismatches).
- **Phase 7 CSV Test IDs:** $150$ samples ($0$ mismatches).
- **Group Isolation:** Zero transaction-pair overlap across train, validation, and test splits.
- **Holdout Segregation:** All $387$ real holdout screenshots were verified untouched.

**Audit Verdict:** CONFIRMED ZERO DATA LEAKAGE.

---

## 5. Threshold Provenance & Leakage Audit

Audit of all operational parameters:
1. **Decision Threshold $\tau = 0.80$:** Verified selected from 2D grid sweep over $[0.15, \dots, 0.80]$ strictly on the **Phase 5 validation split** (`validation_tuning_results.json`). Zero test peeking.
2. **Connected-Component Min Area $A_{\min} = 50\text{ px}$:** Verified selected jointly with $\tau$ on the **Phase 5 validation split**. Zero test peeking.
3. **Screenshot Area Threshold $\theta_{\text{pixel}} = 1,000\text{ px}$:** Audited as an **exploratory heuristic** introduced in Phase 7 config. It was not derived from validation tuning. Since it achieves only $8.00\%$ specificity, it cannot be promoted as an optimal classifier threshold.

**Audit Verdict:** Localization thresholds ($\tau, A_{\min}$) are completely clean of leakage. The screenshot threshold is properly reclassified as an uncalibrated heuristic.

---

## 6. Test Discovery & Count Verification

Verification of actual test discovery:
- `unittest discover -s member2 -p "test_*.py"`:
  - Phase 1: 10 tests
  - Phase 2: 16 tests
  - Phase 3: 15 tests
  - Phase 4: 14 tests
  - Phase 5: 10 tests
  - Phase 6: 10 tests
  - Phase 7: 5 tests
  - Phase 7.1: 6 tests
  - **Subtotal Discovered:** **86 tests** (all passing).
- `member2/performance_improvement/test_improvement_pipeline.py`:
  - 8 tests (all passing).
- **Total Existing Tests:** **94 tests** across codebase.
- **Audit Explanation:** The earlier claim of "88/88" was a manual sum of 80 tests from phases 1-7 plus 8 tests from performance improvement. However, because `performance_improvement` lacked an `__init__.py`, `unittest discover` ran 80, not 88. With Phase 7.1 tests added, `discover` runs 86, and direct runs bring the total to 94.

---

## 7. Robustness Condition Audits

We verified the 4 specific empirical phenomena observed across the 14 robustness conditions:
1. **50% Downsampling Recall Collapse (0.00%):** At 50% scale, total detected pixels across all 5 tampered images fell to $813$ pixels (~$162$ px/image), all below the $1,000$-pixel cutoff. Hence, all were predicted as 0, yielding $0.00\%$ recall.
2. **Gaussian Blur $\sigma = 1.0$ Dice Increase ($0.1756 \to 0.1832$):** Mild blur smoothed out background sub-pixel typography, cutting false-positive pixels by $55\%$ ($14,949 \to 6,740$), which mathematically raised the Dice ratio despite a drop in recall ($36.8\% \to 22.9\%$).
3. **Gaussian Blur $\sigma = 1.5$ Recall Collapse ($36.8\% \to 9.50\%$):** Severe blur diffused 1-pixel font strokes into the dark canvas, pulling signal below the $\tau = 0.80$ cutoff.
4. **Brightness +10% Minor Gain ($0.1756 \to 0.1785$):** Slightly enhanced stroke contrast against dark cards without increasing background noise (-$555$ FP pixels).
5. **Geometric Alignment:** In `apply_resize_resample`, images are downscaled and immediately upscaled back to $(w, h)$, maintaining exact spatial correspondence with ground-truth masks.

---

## 8. Final Audit Conclusions & Status

- **Phase 7.1 Status:** **PROVISIONALLY APPROVED WITH RECONCILED DOCUMENTATION**.
- **Corrected Report:** Saved at [`member2/phase7_1_evaluation_audit/corrected_phase7_report.md`](file:///c:/Users/athar/OneDrive/Desktop/Member%202/member2/phase7_1_evaluation_audit/corrected_phase7_report.md).
- **Phase 8 Readiness:** As instructed, Phase 8 has **NOT** been started. Member 2's components are frozen and audited.

