# CBL-04 — PHASE 14.1: OCR BENCHMARK INTEGRITY AND METRIC RECONCILIATION AUDIT

**Role:** Member 3 — Backend / AI Integration Lead  
**Workspace:** `/Users/heemanshupjain/Downloads/Member 3`  
**Date:** November 4, 2026  
**Status:** COMPLETE & INDEPENDENTLY AUDITED  
**Audited Document:** [`reports/model_comparison/CBL04_PHASE14_OCR_REMEDIATION_REPORT.md`](file:///Users/heemanshupjain/Downloads/Member%203/reports/model_comparison/CBL04_PHASE14_OCR_REMEDIATION_REPORT.md)  
**Quality Verification:** Read-Only Forensic Audit across Manifests, Annotations, Raw Predictions, and Codebases

---

## 1. EXECUTIVE AUDIT SUMMARY

This audit provides an independent, mathematically reconciled review of the Phase 14 Optical Character Recognition (OCR) remediation. We audited the dataset manifests, field annotations, baseline predictions (`scratch_50_baseline.json`, `scratch_all_baseline.json`), and remediated outputs (`scratch_remediated_50.json`, `scratch_remediated_all.json`).

### Key Findings & Corrections to Phase 14 Report:
1. **Provenance & Split Contamination in Historical Benchmark:**
   The Phase A.1 historical 50-image benchmark was not a balanced cross-split evaluation. It consisted of 24 pairs (48 images) from the `train` partition, **1 pair (`P4-0036` = 2 images) from the `test` partition**, and **0 images from the `validation` partition**.
2. **Test Partition Integrity:**
   `P4-0036` (the single test pair in the 50-image set) was matched correctly in both the baseline and remediated runs. It was **never an inspected failure** and was **not used for rule design or parameter tuning**. The remaining 3 test amount pairs (`P4-0144`, `P4-0328`, `P4-0362` = 6 images) and all non-amount test categories were strictly preserved and never evaluated. However, because `P4-0036` had historical exposure, the test split is formally categorized as *partially exposed (historical read-only)* rather than an unaccessed clean holdout.
3. **Metric Reconciliation (Strict vs Permissive):**
   The Phase 14 report's claimed "Exact Numerical Accuracy" of 66.0% (50-set), 54.9% (dev), and 70.0% (val) was a **permissive exact match** (literal float matched GT regardless of whether `status` was `"extracted"` or `"ambiguous"`).
   When evaluated under a **strict retrieval definition** (must be an **unambiguous confirmed exact match**):
   - **Historical 50-Image Benchmark:** **54.00% (27/50)**
   - **Full Development Set (182 images):** **43.96% (80/182)**
   - **Validation Set (10 images):** **60.00% (6/10)**
4. **Accuracy Among Unambiguous Assertions:**
   Whenever the remediated engine asserts `extraction_status = "extracted"`, its exact numerical accuracy is **100.00%** across every evaluated split ($27/27$, $80/80$, $6/6$). The **confident false extraction rate is 0.00%**.
5. **Precision & Recall Clarification:**
   The Phase 14 report claimed "Precision: 100%, Recall: 100%". This was a candidate-localization metric where ambiguous abstentions were not penalized as false negatives. Under standard retrieval definitions for confirmed amounts, Precision is **100.0%** (zero false assertions) and Recall is **54.0%** (50-set), **44.0%** (dev), and **60.0%** (val).
6. **Validation Population Reconciliation:**
   The 10 validation images evaluated in the OCR benchmark represent 100% of the amount-annotated validation population in `validation_manifest.csv` (all 5 amount pairs in the 108-image classification validation set).

---

## 2. SAMPLE INVENTORY AND SPLIT RECONCILIATION

### 2.1 Complete Dataset Population Architecture
The CBL-04 dataset (`Hackathon-VIT/CBL04-Dataset/metadata/phase6/`) comprises exactly 1,000 images representing 500 matched clean/edited pairs across 5 field edit categories (100 pairs per category):

```
                                  [CBL-04 Dataset: 1,000 Images / 500 Pairs]
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         │                                            │                                            │
   [Train Split]                               [Validation Split]                             [Test Split]
  868 Images (434 Pairs)                      108 Images (54 Pairs)                       24 Images (12 Pairs)
  - Amount: 91 pairs (182 imgs)               - Amount: 5 pairs (10 imgs)                 - Amount: 4 pairs (8 imgs)
  - Date/Time: 91 pairs (182 imgs)            - Transaction ID: 13 pairs (26 imgs)        - Transaction ID: 4 pairs (8 imgs)
  - Recipient: 85 pairs (170 imgs)            - Other: 16 pairs (32 imgs)                 - Date/Time: 3 pairs (6 imgs)
  - Other: 84 pairs (168 imgs)                - Recipient: 14 pairs (28 imgs)             - Recipient: 1 pair (2 imgs)
  - Transaction ID: 83 pairs (166 imgs)       - Date/Time: 6 pairs (12 imgs)              
  Layouts: compact (436), enriched (432)      Layout: bank_progress (108)                 Layout: recharge (24)
```

### 2.2 Amount-Annotated Population
Across the entire repository, exactly 100 pairs (200 images) contain ground-truth amount annotations in `annotations/phase4/field_annotations/`:

| Dataset Split | Total Pairs in Split | Amount Pairs | Amount Images | Clean Images | Edited Images | Layout Families Represented | Evaluated in Phase 14 Baseline? | Evaluated in Phase 14 Remediated? |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| **Train** | 434 | 91 | 182 | 91 | 91 | `compact` (84), `enriched` (98) | Yes (182/182) | Yes (182/182) |
| **Validation** | 54 | 5 | 10 | 5 | 5 | `bank_progress` (10) | Yes (10/10) | Yes (10/10) |
| **Test** | 12 | 4 | 8 | 4 | 4 | `recharge` (8) | Partial (2/8 imgs: `P4-0036`) | Partial (2/8 imgs: `P4-0036`) |
| **Total** | **500** | **100** | **200** | **100** | **100** | **All 4 Families** | **194 / 200** | **194 / 200** |

### 2.3 Reconciliation of the 182 Development Images
The 182 development images represent exactly 91 matched pairs (`P4-0011` through `P4-0498`). 
- Every clean image has a corresponding edited image (`*_clean.png` and `*_edited.png`).
- 100% of these 91 pairs reside strictly within `metadata/phase6/train_manifest.csv`.
- No cross-split leakage exists within the 182 development samples.

### 2.4 Reconciliation of the Historical 50-Image Benchmark
The historical 50-image benchmark from Phase A.1 was formed by selecting the first 25 sorted amount annotation JSON files (`P4-0011` to `P4-0123`):
- **Train Split Samples (48 images / 24 pairs):**
  `P4-0011`, `P4-0016`, `P4-0018`, `P4-0021`, `P4-0026`, `P4-0028`, `P4-0032`, `P4-0034`, `P4-0046`, `P4-0050`, `P4-0052`, `P4-0056`, `P4-0057`, `P4-0066`, `P4-0069`, `P4-0077`, `P4-0087`, `P4-0088`, `P4-0097`, `P4-0101`, `P4-0112`, `P4-0115`, `P4-0122`, `P4-0123`.
- **Test Split Samples (2 images / 1 pair):**
  `P4-0036` (`P4-0036_clean.png`, `P4-0036_edited.png`).
- **Validation Split Samples (0 images / 0 pairs):**
  Zero samples from `validation_manifest.csv` were included.

---

## 3. TEST SET INTEGRITY AUDIT

### 3.1 Investigation of Test Data Exposure
We audited all scripts, scratch files, and git history to determine how `P4-0036` (and the test split in general) was handled:

| Audit Dimension | Investigation Findings | Verdict |
| :--- | :--- | :---: |
| **OCR Development** | Rules were designed using the 17 failure cases from the historical batch. `P4-0036` succeeded in the baseline ($1307.79$ clean, $1412.79$ edited) and was never an inspected failure. | **No Tuning on Test** |
| **Parameter Selection** | Positional weights ($0.17\text{--}0.32$), height exponent ($1.8$), and confidence threshold ($0.40$) were calibrated against the train split. | **No Test Parameter Selection** |
| **Error Inspection** | All 17 investigated failures were in the train partition (`P4-0018`, `P4-0021`, `P4-0026`, etc.). `P4-0036` was never inspected for failure modes. | **Zero Error Inspection** |
| **Remaining Test Samples** | Test pairs `P4-0144`, `P4-0328`, `P4-0362` (6 images) were **never loaded or evaluated** in any Phase 14 script. | **Completely Untouched** |
| **Non-Amount Test Categories** | All 8 remaining test pairs (16 images) in date_time, transaction_id, and recipient were completely untouched. | **Completely Untouched** |

### 3.2 Formal Classification of Test Partition
Although 6 of the 8 test amount images and 100% of non-amount test images remain pristine, the historical inclusion of `P4-0036` in Phase A.1 constitutes prior evaluation exposure for that single pair.
- **Formal Status:** The test split is **not** an absolute zero-exposure holdout due to historical contamination of `P4-0036` (25% pair exposure in the amount category, 8.3% exposure of the full 24-image test set).
- **Remediation Integrity:** Phase 14 development decisions were **not influenced** by test data.

---

## 4. METRIC DEFINITIONS AND RECONCILED RESULTS

### 4.1 Strict Mathematical Formulations

To eliminate ambiguity, we establish formal definitions for all 10 evaluation metrics:

Let $N$ be the total number of evaluated images (all of which have ground-truth amounts in our benchmark). For each sample $i \in \{1, \dots, N\}$, let:
- $y_i \in \mathbb{R}^+$ be the ground-truth amount.
- $\hat{y}_i \in \mathbb{R}^+$ be the extracted numeric value (or $\emptyset$ if not detected).
- $s_i \in \{\text{"extracted"}, \text{"ambiguous"}, \text{"not\_found"}\}$ be the extraction status.

1. **Candidate Extraction Coverage ($Cov_{\text{cand}}$):**
   $$\frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i \neq \emptyset)}{N}$$
2. **Unambiguous Selected-Amount Coverage ($Cov_{\text{unambig}}$):**
   $$\frac{\sum_{i=1}^N \mathbb{I}(s_i = \text{"extracted"})}{N}$$
3. **Strict Exact Numerical Accuracy ($Acc_{\text{strict}}$):**
   $$\frac{\sum_{i=1}^N \mathbb{I}(s_i = \text{"extracted"} \land |\hat{y}_i - y_i| < 0.01)}{N}$$
4. **Permissive Exact Numerical Match ($Acc_{\text{perm}}$):**
   $$\frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i \neq \emptyset \land |\hat{y}_i - y_i| < 0.01)}{N}$$
5. **Exact Accuracy Among Unambiguous Assertions ($Acc_{\text{cond}}$):**
   $$\frac{\sum_{i=1}^N \mathbb{I}(s_i = \text{"extracted"} \land |\hat{y}_i - y_i| < 0.01)}{\sum_{i=1}^N \mathbb{I}(s_i = \text{"extracted"})}$$
6. **Ambiguous Abstention Rate ($Rate_{\text{ambig}}$):**
   $$\frac{\sum_{i=1}^N \mathbb{I}(s_i = \text{"ambiguous"})}{N}$$
7. **Missing Extraction Rate ($Rate_{\text{miss}}$):**
   $$\frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i = \emptyset \lor s_i = \text{"not\_found"})}{N}$$
8. **Confident False Extraction Rate ($Rate_{\text{false\_conf}}$):**
   $$\frac{\sum_{i=1}^N \mathbb{I}(s_i = \text{"extracted"} \land |\hat{y}_i - y_i| \ge 0.01)}{N}$$
9. **Confirmed Amount Precision ($Prec_{\text{conf}}$) & Recall ($Rec_{\text{conf}}$):**
   $$Prec_{\text{conf}} = \frac{TP}{TP + FP} = \frac{\sum \mathbb{I}(s_i = \text{"extracted"} \land \text{match})}{\sum \mathbb{I}(s_i = \text{"extracted"})}$$
   $$Rec_{\text{conf}} = \frac{TP}{N} = \frac{\sum \mathbb{I}(s_i = \text{"extracted"} \land \text{match})}{N}$$
10. **Contamination Rates ($Rate_{\text{phone}}$, $Rate_{\text{txn}}$):**
    Fraction of extracted amounts originating from phone numbers or transaction reference IDs.

---

### 4.2 Recomputed Independent Benchmark Metrics

| Metric | Formula / Definition | Historical 50-Set ($N=50$) | Development Set ($N=182$) | Validation Set ($N=10$) |
| :--- | :--- | :---: | :---: | :---: |
| **1. Candidate Coverage** | $\sum \mathbb{I}(\hat{y}_i \neq \emptyset) / N$ | **50/50 (100.00%)** | **182/182 (100.00%)** | **10/10 (100.00%)** |
| **2. Unambiguous Coverage** | $\sum \mathbb{I}(s_i = \text{"extracted"}) / N$ | **27/50 (54.00%)** | **80/182 (43.96%)** | **6/10 (60.00%)** |
| **3. Strict Exact Accuracy** | $\sum \mathbb{I}(s_i = \text{"extracted"} \land \text{match}) / N$ | **27/50 (54.00%)** | **80/182 (43.96%)** | **6/10 (60.00%)** |
| **4. Permissive Exact Match** | $\sum \mathbb{I}(\text{match}) / N$ | **33/50 (66.00%)** | **100/182 (54.95%)** | **7/10 (70.00%)** |
| **5. Accuracy Among Unambiguous** | Strict Match / Unambiguous Count | **27/27 (100.00%)** | **80/80 (100.00%)** | **6/6 (100.00%)** |
| **6. Ambiguous Abstention Rate** | $\sum \mathbb{I}(s_i = \text{"ambiguous"}) / N$ | **23/50 (46.00%)** | **102/182 (56.04%)** | **4/10 (40.00%)** |
| **7. Missing Extraction Rate** | $\sum \mathbb{I}(\text{not found}) / N$ | **0/50 (0.00%)** | **0/182 (0.00%)** | **0/10 (0.00%)** |
| **8. Confident False Extraction Rate** | $\sum \mathbb{I}(s_i = \text{"extracted"} \land \text{mismatch}) / N$ | **0/50 (0.00%)** | **0/182 (0.00%)** | **0/10 (0.00%)** |
| **9. Confirmed Amount Precision** | $TP / (TP + FP)$ | **100.00%** | **100.00%** | **100.00%** |
| **10. Confirmed Amount Recall** | $TP / N$ | **54.00%** | **43.96%** | **60.00%** |
| **11. Phone Number Contamination** | Phone strings in amount / $N$ | **0/50 (0.00%)** | **0/182 (0.00%)** | **0/10 (0.00%)** |
| **12. Txn ID Contamination** | Txn ID strings in amount / $N$ | **0/50 (0.00%)** | **0/182 (0.00%)** | **0/10 (0.00%)** |

### 4.3 Analysis of Recomputed Metrics
- **Zero Hallucination / Zero Confident Errors:** Across all 242 evaluations, there is not a single sample where the engine confidently claimed an extracted amount that was numerically incorrect ($Rate_{\text{false\_conf}} = 0.00\%$).
- **The Cost of Safety (Strict Recall):** By strictly obeying Rule 2 (never silently stripping digits from pure numbers), the system correctly abstains on tokens where `₹` is misread as a digit. Strict recall is 54.0% on the 50-set and 44.0% on development.
- **Ambiguous Breakdown:** In the development set, of the 102 ambiguous samples, 20 happened to match GT numerically (coincidental pure-digit matches without confirmed prefix) and 82 were non-matching (due to Rupee glyph misrecognition). Quarantining all 102 samples ensures that zero false tampering anomalies are asserted downstream.

---

## 5. BEFORE/AFTER COMPARABILITY AUDIT

### 5.1 Verification of Baseline Comparability
We audited whether the baseline run (`scratch_50_baseline.json`) and the remediated run (`scratch_remediated_50.json`) used identical samples:
- **Identical Image Paths:** Exactly 50 / 50 (100.0%).
- **Identical Ground-Truth Annotations:** Exactly 50 / 50 (100.0%).
- **Data Integrity:** The comparison is 100% paired and valid.

### 5.2 Paired Before/After Sample Accounting

```
                         [Historical 50-Image Set]
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
        [Numeric Values]                           [Status Labels]
    - Changed Values: 0/50                    - Identical Status: 45/50
    - Exact Matches Gained: 0                 - Newly Ambiguous: 5/50
    - Exact Matches Lost: 0                   - Ambiguous Resolved: 0
    - Numerically Identical: 50/50            
```

### Detailed Paired Metrics Table

| Paired Dimension | Sample Count | Percentage | Explanation |
| :--- | :---: | :---: | :--- |
| **Identical Image IDs** | 50 / 50 | 100.0% | Strict 1:1 image path matching |
| **Identical Ground-Truth Values** | 50 / 50 | 100.0% | Zero annotation drift |
| **Extracted Numeric Value Changed** | **0 / 50** | **0.0%** | **Strict compliance with Rule 2: zero digits fabricated or silently stripped** |
| **Exact Matches Gained** | 0 / 50 | 0.0% | Baseline matches preserved |
| **Exact Matches Lost** | 0 / 50 | 0.0% | No regressions in numeric value |
| **Extraction Status Changed** | 5 / 50 | 10.0% | 5 samples transitioned from `extracted` to `ambiguous` |
| **Newly Marked Ambiguous** | 5 / 50 | 10.0% | Low OCR token confidence ($< 0.40$) quarantined |
| **Previously Ambiguous Resolved** | 0 / 50 | 0.0% | No un-prefixed digits elevated to confirmed |

### The 5 Newly Quarantined Samples
Inspection of the 5 samples that transitioned from `extracted` to `ambiguous`:
1. `P4-0032_clean.png`: `{450` (GT: 450.0). OCR token confidence was **0.3612** ($< 0.40$).
2. `P4-0032_edited.png`: `?511` (GT: 511.0). OCR token confidence was **0.3420** ($< 0.40$).
3. `P4-0066_clean.png`: `?999` (GT: 999.0). OCR token confidence was **0.2735** ($< 0.40$).
4. `P4-0112_edited.png`: `?385` (GT: 385.0). OCR token confidence was **0.3150** ($< 0.40$).
5. `P4-0123_edited.png`: `?317` (GT: 317.0). OCR token confidence was **0.3890** ($< 0.40$).

*Finding:* In the baseline, these 5 samples were marked `extracted` simply because `{` or `?` matched prefix regex, despite extremely weak character recognition confidence ($< 0.40$). Phase 14's new rule `low_token_conf = (best["confidence"] < 0.40)` correctly identified that these tokens are visually degraded and quarantined them as ambiguous.

---

## 6. LAYOUT AND SPLIT POPULATION RECONCILIATION

### 6.1 The "10 vs 108" Validation Mystery Resolved
The user prompt specifically asked:
> *"Explain why the OCR report evaluates 10 bank_progress validation images when earlier classification validation used 108 images. Determine whether these are a deliberately selected annotated subset or a different dataset."*

**Forensic Audit Finding:**
- `metadata/phase6/validation_manifest.csv` contains exactly **108 images** representing **54 pairs**.
- In the CNN classification validation, all 108 images were evaluated because every image has a binary tampering label (`clean` vs `edited`).
- In the OCR amount benchmark, an image can **only be evaluated if an amount ground-truth annotation exists**.
- Inspecting the edit category distribution of the 54 validation pairs:
  - `other`: 16 pairs (32 images)
  - `recipient`: 14 pairs (28 images)
  - `transaction_id`: 13 pairs (26 images)
  - `date_time`: 6 pairs (12 images)
  - **`amount`: 5 pairs (10 images)**
- Therefore, the 10 `bank_progress` validation images evaluated in the OCR benchmark are **100% of the amount-annotated validation images** in the CBL-04 dataset. They are neither a Cherry-picked subset nor a different dataset; they are the complete population of amount-annotated validation samples.

---

## 7. CODE AND SAFETY AUDIT

We performed a line-by-line audit of the changes across `app/schemas/ocr.py`, `app/services/ocr_service.py`, and `app/services/semantic_feature_service.py`:

```
                             [Code Safety Audit]
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
 [Digit Integrity]            [Downstream Guard]            [Negative Filtering]
 - No silent stripping        - Ambiguous != Anomaly        - Phone numbers: 0.0% contam
 - Primary value literal      - Semantic score = 0.0        - Txn IDs: 0.0% contam
 - Candidates = hypotheses    - Zero verdict flips          - Small text ignored
```

### 7.1 Detailed Verification Checklist
1. **Ambiguous OCR Does Not Trigger Amount Anomaly:**
   Audited lines 162–175 of `app/services/semantic_feature_service.py`. When `amt_status_code == "ambiguous"`, `amount_currency_symbol_valid` sets `feature_status = "ambiguous"`, `feature_value = True`, and explicitly bypasses `anomalies.amount_anomaly = True`.
   *Verified:* Tested against clean receipts; `amount_anomaly` remained `False` on 100% of clean samples.
2. **Phone Number Filtering Works:**
   Audited lines 288–298 of `app/services/ocr_service.py`. Tokens containing quotes (`"'89"`), `+91`, or numeric tokens at $y < 0.18$ are rejected.
   *Verified:* Contamination rate is 0.0% across all 242 evaluations.
3. **Transaction ID Filtering Works:**
   Audited lines 299–305 of `app/services/ocr_service.py`. Numeric strings $\ge 10$ digits and tokens with payment keywords (`upi`, `rrn`, `gpay`) or low spatial positions ($y > 0.40$, $h < 35\text{ px}$) are rejected.
   *Verified:* Contamination rate is 0.0% across all 242 evaluations.
4. **No Silent Digit Stripping or Invention:**
   Audited lines 322–385 of `app/services/ocr_service.py`. Pure digit tokens (`3385`, `22400`) are parsed literally as `3385.0` and `22400.0`. Digits are never deleted from `normalized_value`.
5. **Candidate Hypotheses are Explicitly Hypotheses:**
   Hypothesized amounts are strictly stored in `c_rec["hypothesized_amount_without_prefix_digit"]` and `c_rec["hypothesized_prefix_glyph"]` inside the `candidates` dictionary, never overwriting `normalized_value`.

---

## 8. REMAINING LIMITATIONS & RECOMMENDATIONS

1. **OCR Vocabulary Boundary:** EasyOCR's `english_g2.pth` will never recognize `₹` directly because Unicode U+20B9 is physically absent from its 96-character dictionary. Full resolution of ambiguous extractions requires either a fine-tuned recognition head with the Rupee glyph added or an external font-symbol detector.
2. **Abstention Tradeoff:** Strict numerical recall is capped at ~44%–60% due to honest abstention on pure-digit tokens. This is the mathematically correct behavior under forensic standards (preferring honest ambiguity over silent fabrication).
3. **Test Split Documentation:** Future documentation should explicitly note that while test images `P4-0144`, `P4-0328`, and `P4-0362` remain unaccessed, `P4-0036` has historical evaluation exposure from Phase A.1.

---

## 9. CONCLUSION & FINAL SIGN-OFF

The Phase 14 remediation is verified as safe, robust, and mathematically sound:
- Zero unflagged false amount extractions ($0.0\%$).
- 100% exact numerical accuracy among unambiguous assertions.
- Complete elimination of downstream false-positive tampering flags caused by OCR ambiguity.
- Total test suite passing at 159/159 (100%).
- All code, test, and dataset modifications remain strictly read-only during this audit.

