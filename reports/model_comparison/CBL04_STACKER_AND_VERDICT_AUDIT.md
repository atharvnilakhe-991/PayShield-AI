# CBL-04 — OUT-OF-FOLD STACKER & FINAL VERDICT ENGINE AUDIT

**Project:** CBL-04 — Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Workspace:** `/Users/heemanshupjain/Downloads/Member 3`  
**Role:** Member 3 — Backend / AI Integration Lead  
**Scope:** Read-Only Forensic Audit of `app/services/stacker_service.py`, `models/phase8/stacker_model.json`, `app/services/final_verdict_service.py`, and Frontend Presentation Components.  
**Date:** 2026-11-04  
**Status:** COMPLETE AUDIT DELIVERABLE (Deliverable 3 of 3)

---

## 1. Executive Summary

This report documents the forensic investigation into why unmanipulated, authentic Google Pay dark-theme transaction screenshots are repeatedly classified as **`EDITED`** with a displayed confidence score of **`60%`** across the live CBL-04 application.

### Key Audit Conclusions:
1. **The Displayed "60%" is an Artificial Heuristic Floor, Not a Model Output:**
   In `app/services/final_verdict_service.py` (lines 221 and 239), the verdict confidence is computed via:
   $$\text{confidence} = \min\left(0.95, \max\left(0.60, 0.50 + |\text{stacker\_prob} - 0.50|\right)\right)$$
   Whenever the ensemble stacker outputs a probability near the decision boundary ($\approx 0.50$), the term $0.50 + |\text{stacker\_prob} - 0.50|$ evaluates to $\approx 0.50$. The hardcoded `max(0.60, ...)` clamp forces the confidence to **0.60**. The React frontend (`ConfidenceScore.jsx`) multiplies this by 100 and renders **`60%`** accompanied by a red badge proclaiming **`"High Tamper Confidence"`**. This creates the deceptive appearance of high certainty when the underlying model was actually in a state of maximum entropy (a coin flip).

2. **The Guardrail Ambiguity Band is Systematically Bypassed:**
   The final verdict engine contains an ambiguity guardrail designed to catch uncertain predictions in the range $[0.45, 0.55]$ and output `INCONCLUSIVE` (Rule 5). However, Rule 5 is guarded by `and not has_semantic_anom`. Because OCR extraction failures trigger semantic rule violations on clean receipts, `has_semantic_anom` is evaluated as `True`, **completely bypassing the ambiguity band** and forcing the execution of Rule 6 (`EDITED`).

3. **Stacker Positive Intercept Bias ($+0.088603$):**
   The L2-regularized logistic stacker (`models/phase8/stacker_model.json`) was fitted with an uncentered positive intercept ($b = +0.088603$) and a large positive weight on `cnn_pred_class` ($+0.078964$). When the baseline CNN suffers layout mode collapse on compact receipts and predicts class 1, the stacker logit is pushed above 0, shifting the baseline probability above the $0.50$ threshold ($0.5008$–$0.5090$) even when visual manipulation is zero.

4. **Zero Effective Weight on Critical Forensic Features:**
   Out of 18 features in the stacker schema, 8 features have weights essentially equal to zero ($0.0$ or $\approx -2.6 \times 10^{-5}$), including transaction reference ID anomaly, status anomaly, and branch availability flags. The stacker effectively operates as a 3-variable model driven almost exclusively by `ocr_date_extracted` ($-0.1465$), `cnn_pred_class` ($+0.0789$), and `recipient_anomaly` ($+0.0534$).

---

## 2. Stacker Architecture & Mathematical Specification

The CBL-04 Stacker Service (`app/services/stacker_service.py`) implements a meta-learner designed to combine the continuous probabilities and structured discrete flags from the three upstream perception branches (CNN, OCR, and Semantic Consistency).

### 2.1 Complete 18-Feature Schema & Scaling Parameters

The feature vector $\mathbf{x} \in \mathbb{R}^{18}$ is standardized using pre-computed sample means $\mu_i$ and standard deviations $\sigma_i$ stored in `models/phase8/stacker_model.json`:

$$\tilde{x}_i = \frac{x_i - \mu_i}{\sigma_i}$$

| Index | Feature Identifier | Scaler Mean ($\mu$) | Scaler Std ($\sigma$) | Stacker Weight ($w_i$) | Absolute Rank | Influence Direction |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `ocr_date_extracted` | 0.000000 | 1.000000 | **-0.146534** | 1 | **Protective** (Reduces Tamper Prob) |
| 2 | `cnn_pred_class` | 0.000000 | 1.000000 | **+0.078964** | 2 | **Aggravating** (Increases Tamper Prob) |
| 3 | `recipient_anomaly` | 0.000000 | 1.000000 | **+0.053377** | 3 | **Aggravating** (Increases Tamper Prob) |
| 4 | `cnn_score` | 0.560736 | 0.148367 | **-0.028635** | 4 | Inverted (Negative on raw prob!) |
| 5 | `ocr_overall_confidence` | 0.885902 | 0.030746 | **+0.019859** | 5 | Inverted (Higher conf $\to$ higher tamper) |
| 6 | `semantic_manipulation_score` | 0.116763 | 0.117180 | **+0.011107** | 6 | Aggravating |
| 7 | `amount_anomaly` | 0.000000 | 1.000000 | **+0.007069** | 7 | Aggravating |
| 8 | `missing_critical_fields_count` | 0.000576 | 0.011987 | **+0.003886** | 8 | Aggravating |
| 9 | `upi_handle_anomaly` | 0.000000 | 1.000000 | **-0.000329** | 9 | Negligible |
| 10 | `ocr_amount_extracted` | 0.000000 | 1.000000 | **-0.000026** | 10 | Inactive ($w \approx 0$) |
| 11 | `ocr_recipient_extracted` | 0.000000 | 1.000000 | **-0.000026** | 11 | Inactive ($w \approx 0$) |
| 12 | `ocr_ref_id_extracted` | 0.000000 | 1.000000 | **-0.000026** | 12 | Inactive ($w \approx 0$) |
| 13 | `branch_cnn_available` | 0.000000 | 1.000000 | **-0.000026** | 13 | Inactive ($w \approx 0$) |
| 14 | `branch_ocr_available` | 0.000000 | 1.000000 | **-0.000026** | 14 | Inactive ($w \approx 0$) |
| 15 | `branch_semantic_available` | 0.000000 | 1.000000 | **-0.000026** | 15 | Inactive ($w \approx 0$) |
| 16 | `date_anomaly` | 0.000000 | 1.000000 | **0.000000** | 16 | Completely Inactive ($w = 0$) |
| 17 | `ref_id_anomaly` | 0.000000 | 1.000000 | **0.000000** | 17 | Completely Inactive ($w = 0$) |
| 18 | `status_anomaly` | 0.000000 | 1.000000 | **0.000000** | 18 | Completely Inactive ($w = 0$) |

### 2.2 Mathematical Forward Pass

The stacker computes logit $z$ and probability $P(\text{edited})$ via:

$$z = b + \sum_{i=1}^{18} w_i \cdot \tilde{x}_i = 0.088603 + \sum_{i=1}^{18} w_i \cdot \frac{x_i - \mu_i}{\sigma_i}$$

$$P(\text{manipulation}) = \sigma(z) = \frac{1}{1 + e^{-\text{clamp}(z, -30.0, 30.0)}}$$

### 2.3 Mathematical Anomalies in Trained Weights
1. **Contradictory CNN Weights:**
   - `cnn_pred_class` has a positive weight: $+0.078964$.
   - `cnn_score` has a negative weight: $-0.028635$.
   When the CNN score increases from $0.51$ to $0.90$ with `cnn_pred_class = 1`, the continuous score actually *decreases* the logit by $-0.028635 \times \frac{0.90 - 0.5607}{0.1484} = -0.0655$, partially cancelling the discrete class prediction. This is an artifact of multicollinearity during L2 optimization on synthetic data.
2. **Inverted OCR Confidence Influence:**
   `ocr_overall_confidence` carries a *positive* weight ($+0.019859$). High OCR text legibility increases the tampering logit, which is counter-intuitive for authentic receipts.
3. **Severe Inactivity:**
   Features 10 through 18 have near-zero or exact zero weights because in the synthetic training folds (Folds 1 and 2), reference ID anomalies and status anomalies never varied independently, resulting in zero gradient updates under L2 regularization.

---

## 3. Training vs. Live Feature Distribution Divergence

The stacker model was trained on 868 synthetic receipts derived exclusively from Folds 1 and 2. 

### 3.1 Training Distribution vs. Live Inference Distribution

| Feature | Training Set Mean ($\mu$) | Live Clean Compact Receipt | Divergence Impact on Stacker Logit |
| :--- | :---: | :---: | :--- |
| `cnn_score` | 0.5607 | 0.7527 | Continuous score standardized to $+1.29\sigma$, subtracting $-0.037$ from logit. |
| `cnn_pred_class` | 0.0000 | 1.0000 | Discrete class standardized to $+1.00\sigma$, adding **$+0.0789$** to logit. |
| `recipient_anomaly` | 0.0000 | 1.0000 (when OCR misreads) | Binary flag standardized to $+1.00\sigma$, adding **$+0.0534$** to logit. |
| `amount_anomaly` | 0.0000 | 1.0000 (when OCR drops amount) | Binary flag standardized to $+1.00\sigma$, adding **$+0.0071$** to logit. |
| `ocr_date_extracted`| 0.0000 | 1.0000 (when date extracted) | Binary flag standardized to $+1.00\sigma$, subtracting **$-0.1465$** from logit. |

### 3.2 The Logit Equilibrium on Clean Receipts

When a clean receipt is processed:
- Base Intercept: $b = +0.0886$
- CNN class 1: $+0.0790$
- CNN score continuous ($0.75$): $-0.0370$
- OCR Date Extracted: $-0.1465$
- Base Logit (No Anomalies):
  $$z = 0.0886 + 0.0790 - 0.0370 - 0.1465 = -0.0159 \implies P = \frac{1}{1 + e^{0.0159}} \approx 0.4960$$
  This places the clean receipt directly in the center of the $[0.45, 0.55]$ ambiguity margin.

However, if **any** semantic anomaly trips (e.g. `recipient_anomaly` $= 1$ due to OCR receiver parse collision):
$$z = -0.0159 + 0.0534 = +0.0375 \implies P = \frac{1}{1 + e^{-0.0375}} \approx 0.5094$$
The logit crosses 0, pushing $P(\text{manipulation}) \ge 0.5000$.

---

## 4. Final Verdict Engine Forensic Breakdown (`app/services/final_verdict_service.py`)

The Final Verdict Service evaluates a deterministic 7-rule sequence to classify screenshots as `GENUINE`, `EDITED`, or `INCONCLUSIVE`.

```
                  ┌────────────────────────────────────────┐
                  │ 1. Branch Availability Check           │
                  │ Both CNN & OCR down?                   │
                  └──────────────────┬─────────────────────┘
                                     │ No
                  ┌──────────────────▼─────────────────────┐
                  │ 2. Signal Extraction & Anomaly Audit   │
                  │ stacker_prob, cnn_score, ocr_conf      │
                  └──────────────────┬─────────────────────┘
                                     │
                  ┌──────────────────▼─────────────────────┐
                  │ 3. Low OCR Legibility Guardrail        │
                  │ ocr_conf < 0.45 AND missing >= 3?      │
                  └──────────────────┬─────────────────────┘
                                     │ No
                  ┌──────────────────▼─────────────────────┐
                  │ 4. Severe Branch Conflict Guardrail    │
                  │ CNN >= 0.80 AND Sem == 0.0 AND         │
                  │ OCR >= 0.85 AND NOT has_semantic_anom? │
                  └──────────────────┬─────────────────────┘
                                     │ No
                  ┌──────────────────▼─────────────────────┐
                  │ 5. Ambiguity Band Guardrail            │
                  │ 0.45 <= stacker_prob <= 0.55 AND       │
                  │ NOT has_semantic_anom?                 │
                  └──────┬──────────────────────────┬──────┘
                   Yes   │                          │ No
        ┌────────────────▼─────────┐   ┌────────────▼──────────────┐
        │ VERDICT: INCONCLUSIVE    │   │ 6. Decisive EDITED Rule   │
        │ rule_ambiguity_band      │   │ stacker_prob >= 0.50 OR   │
        └──────────────────────────┘   │ has_semantic_anom OR      │
                                       │ (CNN >= 0.65 & Prob>=0.45)│
                                       └────────────┬──────────────┘
                                        Yes         │ No
                            ┌───────────────────────▼──────────────┐
                            │ VERDICT: EDITED (rule_edited_decision)│
                            │ confidence = max(0.60, ...) -> 60%   │
                            └───────────────────────┬──────────────┘
                                                    │
                                       ┌────────────▼──────────────┐
                                       │ 7. Decisive GENUINE Rule  │
                                       │ rule_genuine_decision     │
                                       └───────────────────────────┘
```

### 4.1 Detailed Analysis of Decision Rules

#### Rule 1: Perception Branch Availability
```python
if not cnn_available and not ocr_available:
    return FinalVerdictResult(verdict=VerdictEnum.INCONCLUSIVE, decision_rule="rule_all_branches_unavailable")
```
- **Evaluation:** Sound. Correctly marks inputs as `INCONCLUSIVE` when primary sensory inputs fail.

#### Rule 3: Low OCR Confidence / Severe Occlusion Guardrail
```python
if ocr_available and ocr_conf < self.min_ocr_confidence and missing_fields >= 3:
    return FinalVerdictResult(verdict=VerdictEnum.INCONCLUSIVE, decision_rule="rule_low_ocr_legibility")
```
- **Evaluation:** Ineffective against subtle OCR failures. OCR confidence on degraded or merged tokens is often high ($0.85$–$0.90$) even when the transcription is completely incorrect (e.g. `"{2,400"` has $0.879$ confidence). The guardrail never fires for standard OCR symbol substitutions.

#### Rule 4: Severe Inter-Branch Conflict Guardrail
```python
if cnn_available and sem_available and cnn_score >= 0.80 and sem_score == 0.0 and ocr_conf >= 0.85 and not has_semantic_anom:
    return FinalVerdictResult(verdict=VerdictEnum.INCONCLUSIVE, decision_rule="rule_severe_branch_conflict")
```
- **Evaluation:** Narrow condition. Requires `cnn_score >= 0.80`. In our clean receipt test suite, CNN scores on clean receipts hovered between $0.59$ and $0.78$. Thus, this conflict guardrail did not fire on any of the 10 clean test receipts.

#### Rule 5: Ambiguity Band Guardrail (`rule_ambiguity_band`)
```python
lower_bound = self.operating_threshold - self.ambiguity_margin  # 0.45
upper_bound = self.operating_threshold + self.ambiguity_margin  # 0.55
if lower_bound <= stacker_prob <= upper_bound and not has_semantic_anom:
    return FinalVerdictResult(
        verdict=VerdictEnum.INCONCLUSIVE,
        confidence=round(1.0 - abs(stacker_prob - 0.5) * 2.0, 2),
        decision_rule="rule_ambiguity_band",
    )
```
- **The Design Intent:** Inconclusive margin $[0.45, 0.55]$ prevents coin-flip model outputs from generating false accusations of fraud.
- **The Fatal Flaw:** The condition `and not has_semantic_anom`.
  When EasyOCR drops a 4-digit amount (due to line 274 of `ocr_service.py`) and extracts `"24"` from the date or phone suffix, Rule `R_AMT_004` flags `amount_anomaly`. This sets `has_semantic_anom = True`.
  **The ambiguity band is bypassed completely**, even though `stacker_prob` is $0.5008$ (essentially $50\%$).

#### Rule 6: Decisive EDITED Rule (`rule_edited_decision`)
```python
if stacker_prob >= self.operating_threshold or has_semantic_anom or (cnn_score >= 0.65 and stacker_prob >= 0.45):
    verdict = VerdictEnum.EDITED
    ...
    calc_conf = round(min(0.95, max(0.60, 0.50 + abs(stacker_prob - 0.50))), 2)
    return FinalVerdictResult(verdict=verdict, confidence=calc_conf, decision_rule="rule_edited_decision")
```
- **The Compound Trigger:** Three separate disjunctive OR conditions trigger `EDITED`:
  1. `stacker_prob >= 0.50` (Trivially satisfied when CNN class 1 pushes probability to $0.5008$).
  2. `has_semantic_anom` (Trivially satisfied when OCR misreads `₹` or drops 4-digit amount).
  3. `cnn_score >= 0.65 and stacker_prob >= 0.45` (Satisfied on nearly all compact receipts).
- **Result:** A clean receipt has virtually **zero mathematical probability of escaping Rule 6**!

---

## 5. Forensic Dissection of the Displayed "60%" Confidence Score

Why does the UI specifically display **`60%`**?

### 5.1 The Mathematical Clamp in Backend Code

In `app/services/final_verdict_service.py`, lines 221 and 239:
```python
calc_conf = round(min(0.95, max(0.60, 0.50 + abs(stacker_prob - 0.50))), 2)
```

Let us evaluate this function across different values of `stacker_prob`:

| Stacker Probability ($P$) | Deviation from Boundary $|P - 0.50|$ | Linear Confidence $0.50 + |P - 0.50|$ | `max(0.60, ...)` Clamp | Final Returned Confidence |
| :---: | :---: | :---: | :---: | :---: |
| **0.5000** (Pure coin flip) | 0.0000 | 0.5000 | **0.6000** | **0.60 (60%)** |
| **0.5008** (Sample P4-0007) | 0.0008 | 0.5008 | **0.6000** | **0.60 (60%)** |
| **0.5015** (Sample P4-0010) | 0.0015 | 0.5015 | **0.6000** | **0.60 (60%)** |
| **0.5090** (Sample P4-0008) | 0.0090 | 0.5090 | **0.6000** | **0.60 (60%)** |
| **0.5500** | 0.0500 | 0.5500 | **0.6000** | **0.60 (60%)** |
| **0.5900** | 0.0900 | 0.5900 | **0.6000** | **0.60 (60%)** |
| **0.6100** | 0.1100 | 0.6100 | 0.6100 | **0.61 (61%)** |
| **0.7500** | 0.2500 | 0.7500 | 0.7500 | **0.75 (75%)** |
| **0.9600** | 0.4600 | 0.9600 | 0.9600 (min 0.95) | **0.95 (95%)** |

### 5.2 The Frontend Rendering Chain

1. **Backend Response:** `AnalyzeResponse.confidence` is set to `0.6`.
2. **Frontend Ingestion (`ResultsDashboard.jsx`):**
   ```javascript
   const confidencePercent = Math.round((confidence || 0) * 100); // 60
   ```
3. **Frontend Component (`ConfidenceScore.jsx`):**
   ```javascript
   const normalizedConfidence = Math.min(Math.max(confidence ?? 0, 0), 1); // 0.6
   const confidencePercent = Math.round(normalizedConfidence * 100);       // 60
   let themeClass = 'confidence-fake';
   let statusText = 'High Tamper Confidence';
   ```
4. **Visual Display:**
   - Big bold text: **`60%`**
   - Meter progress bar: Filled to **`60%`** with red gradient (`#ef4444`).
   - Badge: **`High Tamper Confidence`**.
   - Explanation: `"Potential tampering detected: semantic consistency rules detected formatting or structural inconsistencies."`

### 5.3 The Deception of the Clamp
The underlying model had a manipulation probability of **`0.5008`**—it was $99.8\%$ uncertain. However, the artificial `max(0.60, ...)` floor converts maximum model ignorance into an affirmative, confident declaration of digital forgery.

---

## 6. Reason Code Generation & Explainability Failure

When a false positive occurs under Rule 6, the system attempts to provide localized explainability:

### 6.1 Bounding Box Hallucination on Clean Layouts
In `final_verdict_service.py` (lines 201–209):
```python
if sem_available and semantic_result:
    for feat_name, ev in semantic_result.feature_evidence.items():
        if ev.feature_status in ["invalid_format", "ambiguous"] and ev.source_bounding_box:
            suspicious_regions.append(
                SuspiciousRegion(
                    label=f"Semantic: {feat_name}",
                    score=round(ev.ocr_confidence or 0.8, 2),
                    box_normalized=ev.source_bounding_box,
                )
            )
```
- Because EasyOCR picked up the date `"24"` as the amount in `P4-0007`, the semantic engine flagged the amount field as `invalid_format` (missing Rupee prefix).
- The system then highlighted the date bounding box at the top of the receipt and labeled it:
  `"Semantic: amount"` with `score: 0.89`.
- To an end-user or auditor, the application appears to highlight the date and claim it is a tampered currency amount!

### 6.2 Reason Code Incoherence

| Sample | Actual Image Content | Extracted Value | Reason Code Generated | Plain-English False Accusation |
| :--- | :--- | :--- | :--- | :--- |
| **P4-0007** | Clean receipt, amount ₹450 | `"24"` | `AMOUNT_CURRENCY_SYMBOL_MISMATCH` | Blames receipt for lacking Indian Rupee formatting. |
| **P4-0008** | Clean receipt, amount ₹280 | `"{280"` | `SENDER_RECEIVER_IDENTICAL` | OCR mapped same line to sender & receiver, accusing user of circular transfer fraud. |
| **P4-0010** | Clean receipt, amount ₹999 | `"'17"` | `AMOUNT_FORMAT_ANOMALY` + `AMOUNT_CURRENCY_SYMBOL_MISMATCH` | Highlights phone number suffix and accuses transaction of corrupted monetary layout. |

---

## 7. Comparative Assessment: Member 2 CNN vs. Member 3 Stacker

A central question in the Member 2 vs Member 3 audit is whether integrating Member 2's CNN solves this false-positive crisis:

1. **Member 2's CNN Alone Does NOT Fix the 60% EDITED Bug:**
   Even if Member 2's CNN provides a clean score of $0.05$ on clean screenshots:
   - If EasyOCR drops the 4-digit amount, `has_semantic_anom` will still be `True`.
   - In Rule 6: `if stacker_prob >= 0.50 or has_semantic_anom or ...`
   - Because `has_semantic_anom` is `True`, the verdict will **still be `EDITED`**!
   - And the confidence will **still be clamped to `60%`**!
2. **True Root Cause is Multi-Tiered:**
   The false-positive crisis cannot be resolved simply by swapping the CNN checkpoint. It requires synchronizing the OCR tokenization, semantic anomaly thresholds, ambiguity margin logic, and confidence formula.

---

## 8. Prioritized Remediation & Redesign Plan

To permanently eliminate false positives on genuine receipts and remove the deceptive 60% confidence floor, the following changes are specified:

### Remediation Step 1: Remove the Artificial 0.60 Confidence Floor
**File:** `app/services/final_verdict_service.py` (lines 221 and 239)
**Action:** Replace heuristic clamp with linear or calibrated probability distance from threshold:
```python
# Before (FLAWED):
calc_conf = round(min(0.95, max(0.60, 0.50 + abs(stacker_prob - 0.50))), 2)

# After (TRUE UNCERTAINTY):
# When stacker_prob is 0.5015, distance is 0.0015 -> Confidence is 0.50 (low certainty)
calc_conf = round(min(0.98, max(0.50, 0.50 + abs(stacker_prob - self.operating_threshold))), 4)
```

### Remediation Step 2: Protect the Ambiguity Band from Noisy Semantic Overrides
**File:** `app/services/final_verdict_service.py` (lines 181 and 197)
**Action:** Do not allow single-rule semantic anomalies to override the ambiguity band when stacker probability is near 50%:
```python
# Rule 5 Ambiguity Band: Protect receipts in [0.45, 0.55] unless corroborated by BOTH CNN and Semantic
if lower_bound <= stacker_prob <= upper_bound:
    # Only override ambiguity if there is strong multi-branch agreement
    strong_corroboration = (cnn_score >= 0.70 and sem_score >= 0.50)
    if not strong_corroboration:
        return FinalVerdictResult(
            verdict=VerdictEnum.INCONCLUSIVE,
            confidence=round(1.0 - abs(stacker_prob - 0.5) * 2.0, 2),
            decision_rule="rule_ambiguity_band",
        )
```

### Remediation Step 3: Fix OCR Amount Extraction & 4-Digit Filter
**File:** `app/services/ocr_service.py` (lines 142, 274)
**Action:**
1. Update `AMOUNT_PREFIX_REGEX` to accept `{`, `[`, `?`, `3`, `7`, `2` when adjacent to numeric digits.
2. Remove the flawed `len(num_str) == 4 and "." not in num_str` filter that discards amounts between ₹1,000 and ₹9,999.
3. Reject candidates from known phone number vertical regions ($y < 0.20$ when phone label is present).

### Remediation Step 4: Re-Train / Re-Scale the Stacker Meta-Learner
**File:** `models/phase8/stacker_model.json`
**Action:**
1. Re-fit L2 logistic regression weights on a balanced dataset that includes all 4 layout families (`compact`, `bank_progress`, `enriched`, `standard`).
2. Force `cnn_pred_class` to have zero weight or eliminate it to avoid collinear cancellation with `cnn_score`.
3. Center the intercept to $0.0$ on a balanced 50/50 prior.

---

## 9. Deliverable Verification Matrix

| Audit Requirement | Verification Status | Artifact Location / Code Reference |
| :--- | :---: | :--- |
| **Complete 18-Feature Schema** | **VERIFIED** | `models/phase8/stacker_model.json` lines 6–25 |
| **Standardization Scaler ($\mu, \sigma$)** | **VERIFIED** | `models/phase8/stacker_model.json` lines 49–122 |
| **Weight Vector & Intercept ($b = 0.0886$)** | **VERIFIED** | `models/phase8/stacker_model.json` lines 28–48 |
| **Rules 1–7 Decision Logic** | **VERIFIED** | `app/services/final_verdict_service.py` lines 91–252 |
| **Ambiguity Band Bypass Mechanism** | **VERIFIED** | `app/services/final_verdict_service.py` line 181 (`and not has_semantic_anom`) |
| **Origin of Displayed "60%"** | **VERIFIED** | `app/services/final_verdict_service.py` line 221 (`max(0.60, ...)`) |
| **Frontend Metric Multiplier** | **VERIFIED** | `ConfidenceScore.jsx` line 10 & `ResultsDashboard.jsx` line 59 |
| **Explainability Hallucination** | **VERIFIED** | `app/services/final_verdict_service.py` lines 201–209 |

---
**Report Author:** Member 3 — Backend / AI Integration Lead  
**Audit Status:** Final Deliverable Completed. System ready for remediation planning upon user authorization.

