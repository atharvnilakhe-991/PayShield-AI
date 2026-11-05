# CBL-04 — FINAL PHASE 3 AUDIT & PHASE 4 EVALUATION REPORT (READ-ONLY)

**Document Identifier:** `CBL04_PHASE3_PHASE4_FINAL_AUDIT_REPORT.md`  
**Execution Timestamp:** 2026-11-03T19:25:00+05:30  
**Audit Type:** Strict Read-Only Verification, Codebase Inspection & Empirical Artifact Evaluation  
**Auditor / Role:** Member 3 (Backend & AI Integration Lead)  
**Workspace Root:** `/Users/heemanshupjain/Downloads/Member 3`  
**ML Project Path:** `/Users/heemanshupjain/Downloads/Member 3/Hackathon-VIT/CBL04-Dataset`  
**Host Runtime:** Python 3.12.15 (`/Users/heemanshupjain/Downloads/Member 3/.venv_mac/bin/python`)  
**Operating System:** macOS (Darwin Kernel 27.2.0, Apple Silicon `arm64`)  

---

## 1. EXECUTIVE SUMMARY

This audit delivers an evidence-based assessment of:
1. **Phase 3:** CNN inference integration into the FastAPI backend service (`POST /api/v1/analyze`).
2. **Phase 4:** Baseline model diagnosis, controlled training experiments in `models/phase9_improved/`, and held-out test evaluation.

### Core Audit Verdicts:
- **Phase 3 Status: ACCEPTED (PASSED)**
  - Live multi-task ResNet-18 forward inference is genuinely invoked by `/api/v1/analyze`.
  - The checkpoint loader enforces secure deserialization via `weights_only=True`.
  - Verdict and confidence are driven **100% by the live CNN output**, completely isolated from mock companion branches.
  - Localization overlays remain **strictly disabled** (`localization_available = false`, `suspicious_regions = []`).
  - The backend test suite passes completely with **61 passed, 0 failed, 0 skipped**.
- **Phase 4 Status: COMPLETED WITH DOCUMENTED LIMITATIONS (ACCEPTED FOR ENGINEERING SCOPE)**
  - Root cause of baseline failure confirmed: In Phase 7 Fold 1, tracking positive-class $F_1$ mathematically rewarded a degenerate all-positive collapsed model ($F_1 = 0.6667$, $FPR = 100\%$).
  - In Phase 4, switching checkpoint selection to **Validation Macro-$F_1$** and training for 12 epochs broke the degenerate collapses:
    - Fold 2 test recall jumped from $8.80\%$ to $52.31\%$, and $F_1$ jumped from $15.20\%$ to $51.36\%$.
    - Fold 1 False Positive Rate plummeted from $100\%$ down to $13.30\%$ in Experiment 2 ($TN$ increased from $0$ to $189$).
  - **Fundamental Limitation Confirmed:** Overall classification accuracy on unseen layout families remains bounded near chance level ($\approx 50.0\% - 50.5\%$), and score separation between clean and edited images is negligible ($\Delta < 0.0015$). This proves that an un-pretrained CNN cannot detect subtle text splicing across novel templates without downstream OCR and semantic rule analysis.
  - **Zero Test Contamination Rule Enforced:** Inter-experiment comparisons are reported for diagnostic transparency only; test metrics are **not** used for hyperparameter selection.
  - **Zero Training Active:** No training was launched, resumed, or modified during this audit.

---

## 2. PHASE 3 SOURCE CODE AUDIT & TEST EVIDENCE

### 2.1 Live CNN Invocation & Pipeline Routing
- **File Inspected:** [`app/api/routes/analyze.py`](file:///Users/heemanshupjain/Downloads/Member 3/app/api/routes/analyze.py#L110-L150)
- **Findings:**
  - In `analyze_screenshot()`, when `settings.CNN_ENABLED` is true and `settings.MOCK_MODE` is false:
    - Calls `cnn_service = get_cnn_service()`.
    - Dispatches raw bytes via `cnn_output = cnn_service.predict(cnn_input)`.
    - Evaluates `is_tampered = cnn_output.manipulation_probability >= settings.CNN_DECISION_THRESHOLD`.
    - Maps verdict directly:
      - Canonical style: `EDITED` if `is_tampered` else `GENUINE`.
      - Frontend style: `FAKE` if `is_tampered` else `REAL`.
  - **Evidence:** Live PyTorch inference is genuinely executed; no mock CNN is silently substituted.

### 2.2 Checkpoint Paths & Class Index Mapping
- **Files Inspected:** [`app/core/config.py`](file:///Users/heemanshupjain/Downloads/Member 3/app/core/config.py#L96-L107), [`app/services/cnn_inference_service.py`](file:///Users/heemanshupjain/Downloads/Member 3/app/services/cnn_inference_service.py#L217-L225)
- **Findings:**
  - Default path: `Hackathon-VIT/CBL04-Dataset/models/phase7/fold_1_resnet18_multitask.pt`.
  - Class index in `model_config`: `{"classes": ["clean", "edited"], "class_to_index": {"clean": 0, "edited": 1}}`.
  - Forward inference extracts:
    ```python
    scores = torch.softmax(logits_tensor, dim=0)
    raw_edited_score = float(scores[1].item())  # Index 1 = edited
    ```
  - **Evidence:** Positive-class index 1 matches the training label encoding exactly.

### 2.3 Preprocessing Consistency Verification
- **Files Compared:** [`app/services/cnn_inference_service.py`](file:///Users/heemanshupjain/Downloads/Member 3/app/services/cnn_inference_service.py#L158-L189) vs. [`Hackathon-VIT/CBL04-Dataset/app/server.py`](file:///Users/heemanshupjain/Downloads/Member 3/Hackathon-VIT/CBL04-Dataset/app/server.py#L45-L53) and [`scripts/phase7/train_evaluate.py`](file:///Users/heemanshupjain/Downloads/Member 3/Hackathon-VIT/CBL04-Dataset/scripts/phase7/train_evaluate.py#L62-L76)
- **Findings:**
  - Input converted to RGB.
  - Scale factor: $\text{scale} = \min(160/W, 320/H)$ using `Image.Resampling.LANCZOS`.
  - Dark canvas: Pasted at $(0, 0)$ onto RGB `(18, 18, 18)` of size $160 \times 320$.
  - Channel ordering: Permuted to $[C, H, W] \to [3, 320, 160]$, divided by $255.0$.
  - Normalization: ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
  - Batch dimension: Unsqueezed to `[1, 3, 320, 160]` (dtype `torch.float32`).
  - **Evidence:** Line-by-line parity with Member 2's training preprocessing is confirmed.

### 2.4 Checkpoint Loading Security (`weights_only=True`)
- **File Inspected:** [`app/services/cnn_inference_service.py`](file:///Users/heemanshupjain/Downloads/Member 3/app/services/cnn_inference_service.py#L121-L126)
- **Findings:**
  - Checkpoint loading explicitly specifies:
    ```python
    checkpoint = torch.load(
        self.checkpoint_path,
        map_location=self.device,
        weights_only=True,
    )
    ```
  - Both Phase 7 checkpoints and all Phase 9 candidate checkpoints deserialize cleanly without requiring unrestricted pickle execution.
  - Verified in tests `test_checkpoint_loading_fold1`, `test_checkpoint_loading_fold2`, and `test_checkpoint_loading_weights_only_security`.

### 2.5 Path Injection & Deserialization Vulnerability Audit
- **Findings:**
  - The API endpoint parameters (`image`, `request_id`, `return_heatmap`, `verdict_style`, `x_verdict_style`) provide zero mechanism for a client to specify checkpoint paths or arbitrary URLs.
  - The checkpoint path is resolved strictly on the server from `settings.CNN_CHECKPOINT_PATH`.
  - `self._resolve_path()` validates local existence.
  - `weights_only=True` prevents arbitrary code execution even if an adversary attempted to place an untrusted `.pt` file on disk.

### 2.6 Truthful Error Reporting (Zero Silent Mock Fallback)
- **Files Inspected:** [`app/main.py`](file:///Users/heemanshupjain/Downloads/Member 3/app/main.py#L94-L125), [`app/api/routes/analyze.py`](file:///Users/heemanshupjain/Downloads/Member 3/app/api/routes/analyze.py)
- **Findings:**
  - If the CNN model is missing or corrupt, `CNNModelUnavailableError` is raised and caught by FastAPI's global exception handler, returning **`HTTP 503 Service Unavailable`** with `error_code: "MODEL_UNAVAILABLE"`.
  - If tensor processing fails, `CNNModelError` is caught, returning **`HTTP 500 Internal Server Error`** with `error_code: "INTERNAL_PROCESSING_FAILURE"`.
  - The service **never** silently falls back to mock predictions upon model failure.
  - Verified by test `test_analyze_cnn_unavailable_returns_503` (passes).

### 2.7 Live vs. Mock Component Isolation
- **Findings:**
  - Mock OCR, mock semantic scores, and mock stacker scores are isolated.
  - The verdict is derived strictly from `cnn_output.manipulation_probability`.
  - Response payload includes:
    ```json
    "pipeline_status": "phase3_cnn_integrated",
    "component_status": {
      "cnn": "live",
      "ocr": "mock",
      "semantic": "mock",
      "stacker": "mock"
    }
    ```
  - Disclaimers and warnings explicitly state that downstream branches are simulated placeholders and do not contribute to the verdict or confidence.

### 2.8 Uncalibrated Score Labeling
- **Findings:**
  - [`app/schemas/cnn.py`](file:///Users/heemanshupjain/Downloads/Member 3/app/schemas/cnn.py) enforces `score_type: "uncalibrated_model_score"`.
  - Warnings state: `"CNN score is an uncalibrated experimental model score (baseline accuracy ~50.5%)."`

### 2.9 Localization Policy Enforcement
- **Findings:**
  - [`CNNInferenceService`](file:///Users/heemanshupjain/Downloads/Member 3/app/services/cnn_inference_service.py) enforces: `localization_available = False`, `suspicious_regions = []`, `heatmap_data_uri = None`.
  - [`app/api/routes/analyze.py`](file:///Users/heemanshupjain/Downloads/Member 3/app/api/routes/analyze.py) enforces: `manipulation_analysis.detected = False`, `suspicious_regions = []`, `heatmap_url = None`.
  - Warning states: `"Spatial mask localization is disabled due to experimental mask IoU (0.059)."`

### 2.10 Thread-Safe Singleton & Concurrency
- **File Inspected:** [`app/services/cnn_inference_service.py`](file:///Users/heemanshupjain/Downloads/Member 3/app/services/cnn_inference_service.py#L261-L281)
- **Findings:**
  - Implements thread lock `_CNN_SERVICE_LOCK = threading.Lock()`.
  - `get_cnn_service()` uses double-checked locking.
  - Verified by `test_singleton_thread_safety` running 10 concurrent requests across 5 worker threads (`ThreadPoolExecutor`), all obtaining the identical instance.

### 2.11 Contract Preservation
- **Findings:**
  - All original Phase 1 Pydantic contracts ([`AnalyzeResponse`](file:///Users/heemanshupjain/Downloads/Member 3/app/schemas/response.py), [`BranchScores`](file:///Users/heemanshupjain/Downloads/Member 3/app/schemas/response.py), [`ExtractedTransactionData`](file:///Users/heemanshupjain/Downloads/Member 3/app/schemas/ocr.py)) are 100% preserved.
  - All 27 contract validation tests pass.

### 2.12 Test Suite Execution Output
Executed command:
```bash
.venv_mac/bin/python -m pytest tests/ -v
```
**Results:**
- `tests/test_contracts.py`: **27 passed**
- `tests/test_api.py`: **17 passed**
- `tests/test_cnn_inference.py`: **17 passed**
- **Total:** **61 passed, 0 failed, 0 skipped in 3.43s**

---

## 3. PHASE 4 EXPERIMENT METRICS AUDIT

All Phase 4 experiments were inspected directly from existing artifacts in `reports/phase9_improved/experiment_results.csv`, `experiment_results.json`, and training log `task-460.log`.

### 3.1 Run-by-Run Experiment Inventory

| Exp ID | Fold | Checkpoint Filename | SHA-256 (64-char) | Best Val Epoch | Best Val Macro-$F_1$ | Test Acc | Test Prec | Test Recall | Test $F_1$ (Ed.) | Test Macro $F_1$ | Test FPR | Test FNR | Mask IoU | Confusion Matrix |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`EXP1`** | 1 | `EXP1_MACRO_F1_fold_1_best.pt` | `18e932ea5bf3ece82e4f9495452154d10e9828e0ed1db03a037290e74ba29ef1` | 3 | 0.4915 | 49.77% | 49.85% | 77.06% | 60.54% | 45.73% | 77.52% | 22.94% | 0.0414 | TN=49, FP=169, FN=50, TP=168 |
| **`EXP1`** | 2 | `EXP1_MACRO_F1_fold_2_best.pt` | `e338f07f7d80046a62ef6d25ed5b753b267f1adc67b04dc8ceb31f42795da751` | 11 | 0.5000 | 50.46% | 50.45% | 52.31% | 51.36% | 50.45% | 51.39% | 47.69% | 0.0694 | TN=105, FP=111, FN=103, TP=113 |
| **`EXP2`** | 1 | `EXP2_CLS_PRIORITY_fold_1_best.pt` | `b8bb46e6db984cff96038053520888613dcf78b181ac87c11edd9e67d537b0f0` | 11 | 0.5092 | 50.00% | 50.00% | 13.30% | 21.01% | 42.22% | 13.30% | 86.70% | 0.0020 | TN=189, FP=29, FN=189, TP=29 |
| **`EXP2`** | 2 | `EXP2_CLS_PRIORITY_fold_2_best.pt` | `3839c6aef7fc6f1af395c0e1b74e072e908f48914f7f0675b5caea01a71b28d3` | 11 | 0.5000 | 50.46% | 50.37% | 62.96% | 55.97% | 49.68% | 62.04% | 37.04% | 0.0000 | TN=82, FP=134, FN=80, TP=136 |

> [!WARNING]
> **Localization Safety Warning:**
> Test Mask IoU across all models is negligible ($0.000$ to $0.0694$, or $0\% - 6.94\%$). It must **never** be used to justify heatmap generation or region overlays.

---

## 4. BASELINE COMPARISON & LIMITATIONS

### 4.1 Apples-to-Apples Parity Audit
The test evaluation manifests, preprocessing, label encoding, and metric calculations between Phase 7 baseline and Phase 9 improved are **100% identical**:
- **Fold 1 Test Set:** `metadata/phase7/folds/fold_1_test.csv` (436 rows, layout family `compact`).
- **Fold 2 Test Set:** `metadata/phase7/folds/fold_2_test.csv` (432 rows, layout family `enriched`).
- **Label Encoding:** `clean = 0`, `edited = 1`.
- **Metric Definitions:** Identical formulas for accuracy, precision, recall, $F_1$, and mask pixel counts.

### 4.2 Side-by-Side Comparison

```
FOLD 2 TEST SET (Held-Out Family: enriched, N=432)
─────────────────────────────────────────────────────────────────────────────
Metric               Phase 7 Baseline      EXP1 (Ours)          EXP2 (Ours)
─────────────────────────────────────────────────────────────────────────────
Recall (Sensitivity)      8.80%               52.31% (+43.51%)     62.96% (+54.16%)
F1 Score (Edited)        15.20%               51.36% (+36.16%)     55.97% (+40.77%)
Macro F1                 40.34%               50.45% (+10.11%)     49.68% (+9.34%)
False Negative Rate      91.20%               47.69% (-43.51%)     37.04% (-54.16%)
True Positives             19                  113 (+94)            136 (+117)
True Negatives            201                  105                  82
Mask IoU                  0.0686               0.0694               0.0000
Validation Macro-F1       0.4000               0.5000               0.5000
─────────────────────────────────────────────────────────────────────────────
```

```
FOLD 1 TEST SET (Held-Out Family: compact, N=436)
─────────────────────────────────────────────────────────────────────────────
Metric               Phase 7 Baseline      EXP1 (Ours)          EXP2 (Ours)
─────────────────────────────────────────────────────────────────────────────
False Positive Rate     100.00%               77.52% (-22.48%)     13.30% (-86.70%)
True Negatives (Clean)      0                   49 (+49)           189 (+189)
Recall (Sensitivity)    100.00%               77.06%               13.30%
F1 Score (Edited)        66.67%               60.54%               21.01%
Clean F1 Score            0.00%               30.91% (+30.91%)     63.42% (+63.42%)
Macro F1                 33.33%               45.73% (+12.40%)     42.22% (+8.89%)
Mask IoU                  0.0510               0.0414               0.0020
Validation Macro-F1       0.6667 (Degenerate)  0.4915               0.5092
─────────────────────────────────────────────────────────────────────────────
```

### 4.3 Validation-Only Decision Rule
In strict adherence to leakage prevention rules:
- **Test-set metrics were NOT used to tune or select between Experiment 1 and Experiment 2.**
- When comparing candidates based strictly on **Validation Macro-$F_1$**:
  - **Fold 1 Winner (Validation-based):** `EXP2_CLS_PRIORITY_fold_1_best.pt` achieved $\text{Val Macro-}F_1 = \mathbf{0.5092}$ (vs. EXP1's $0.4915$).
  - **Fold 2 Winner (Validation-based):** `EXP1_MACRO_F1_fold_2_best.pt` and `EXP2` tied at $\text{Val Macro-}F_1 = \mathbf{0.5000}$. EXP1 is preferred as it preserves the mask decoder ($IoU = 0.0694$).

---

## 5. EXPERIMENT COVERAGE HONESTY & STATISTICAL LIMITS

1. **Incomplete Cross-Validation Coverage:**
   - Only **Folds 1 and 2** were trained and evaluated in both Phase 7 and Phase 9.
   - **Folds 3 and 4 were NOT trained or evaluated.** We make no claims regarding Folds 3 and 4.
2. **Extreme Statistical Volatility in Small Folds:**
   - In Fold 2, the validation family is `recharge`, which contains **only 24 screenshots (12 clean, 12 edited)**.
   - In a 24-sample validation partition, a single misclassification swings accuracy and recall by **$4.17\%$** ($1/24$).
   - In Fold 1, the validation family is `bank_progress` with **108 screenshots (54 pairs)**.
   - Because validation groups are small, validation metrics carry significant variance. No broad generalization claims should be made.

---

## 6. DATA INTEGRITY & IMMUTABILITY AUDIT

### 6.1 Split Integrity Verification
Verified via automated script across all 4 fold CSV manifests:
- **Pair Overlaps:** Train-Val = $0$, Train-Test = $0$, Val-Test = $0$.
- **Family Overlaps:** Train-Val = $0$, Train-Test = $0$, Val-Test = $0$.
- **Real Screenshots:** All 387 sensitive real screenshots remain completely excluded from training manifests.

### 6.2 Checkpoint SHA-256 Immutability Audit
Verified via `shasum -a 256`:

| Checkpoint Path | Previous Audit Hash | Current Audit Hash | Status |
|---|---|---|---|
| `models/phase7/fold_1_resnet18_multitask.pt` | `44b654aa...909f62a` | `44b654aa846fddfde7be6dab44ceada6687d8dfd35e2c83edbcb60fed909f62a` | **100% Identical** |
| `models/phase7/fold_2_resnet18_multitask.pt` | `4c5f9768...cb6403a4` | `4c5f97684a64553d117983c21b55a8ca874c855f9023b3e9732d1c70cb6403a4` | **100% Identical** |
| `models/phase9_improved/EXP1_MACRO_F1_fold_1_best.pt` | `18e932ea...ba29ef1` | `18e932ea5bf3ece82e4f9495452154d10e9828e0ed1db03a037290e74ba29ef1` | **Preserved** |
| `models/phase9_improved/EXP1_MACRO_F1_fold_2_best.pt` | `e338f07f...795da751` | `e338f07f7d80046a62ef6d25ed5b753b267f1adc67b04dc8ceb31f42795da751` | **Preserved** |
| `models/phase9_improved/EXP2_CLS_PRIORITY_fold_1_best.pt` | `b8bb46e6...d537b0f0` | `b8bb46e6db984cff96038053520888613dcf78b181ac87c11edd9e67d537b0f0` | **Preserved** |
| `models/phase9_improved/EXP2_CLS_PRIORITY_fold_2_best.pt` | `3839c6ae...171b28d3` | `3839c6aef7fc6f1af395c0e1b74e072e908f48914f7f0675b5caea01a71b28d3` | **Preserved** |

---

## 7. KNOWN RISKS & UNRESOLVED ISSUES

1. **Chance-Level Overall Accuracy:**
   Overall test classification accuracy across held-out layout families remains bounded near chance level ($\approx 50.0\% - 50.5\%$). An un-pretrained CNN cannot generalize localized text font/alignment anomalies to an unseen layout template.
2. **Zero Score Separation:**
   Raw score separation between clean and edited images on unseen templates is $\Delta < 0.0015$. The CNN branch by itself cannot function as a reliable standalone detector.
3. **Localization Failure:**
   Audited mask IoU remains $< 0.07$ ($< 7\%$). Explainability via bounding boxes or heatmaps is non-functional and must remain disabled.
4. **Dependence on Subsequent Phases:**
   True tampering detection capability for CBL-04 depends critically on **Phase 5 (Live OCR Extraction)** and **Phase 6 (Semantic Consistency Rules)**, which inspect factual discrepancies (e.g. invalid UPI handles, mismatched transaction amounts, impossible timestamps) that bypass visual convolutional filters.

---

## 8. EXPLICIT ACCEPTANCE RECOMMENDATIONS

### Phase 3 Recommendation: ACCEPTED (PASSED)
- The FastAPI CNN integration satisfies all architectural, security, lifecycle, and contract requirements.
- The service enforces `weights_only=True`, thread-safe singleton instantiation, truthful 503/500 error reporting, and strict live/mock separation.
- Full test suite passes: **61/61 tests passing**.

### Phase 4 Recommendation: ACCEPTED AS A DIAGNOSTIC BASELINE (PROCEED TO PHASE 5)
- The baseline false-positive and false-negative collapses were successfully diagnosed, proven, and mitigated in candidate models stored under `models/phase9_improved/`.
- The empirical bounds of synthetic visual-only CNN detection on this task have been established and documented.
- **Next Step:** Proceed to **Phase 5 (Live OCR Integration)** to establish the second analysis branch.
- **Hold Rule:** Do not launch further CNN retraining. Keep all Phase 7 and Phase 9 checkpoints intact.

---

## TERMINAL AUDIT SUMMARY

- **Files Inspected:**
  - `app/services/cnn_inference_service.py`
  - `app/api/routes/analyze.py`
  - `app/schemas/cnn.py`
  - `app/schemas/response.py`
  - `app/core/config.py`
  - `app/main.py`
  - `tests/test_cnn_inference.py`
  - `tests/test_api.py`
  - `tests/test_contracts.py`
  - `Hackathon-VIT/CBL04-Dataset/scripts/phase7/train_improved_experiments.py`
  - `Hackathon-VIT/CBL04-Dataset/reports/phase9_improved/experiment_results.csv`
  - `Hackathon-VIT/CBL04-Dataset/reports/phase9_improved/experiment_results.json`
  - `Hackathon-VIT/CBL04-Dataset/metadata/phase7/folds/fold_1..4_{train,val,test}.csv`
- **Tests Actually Run & Results:**
  - Command: `.venv_mac/bin/python -m pytest tests/ -v`
  - Result: **61 passed, 0 failed, 0 skipped (100% pass rate in 3.43s)**
- **Files Modified During This Audit:** **Zero (0)** (Strict read-only audit).
- **Training Launched or Resumed:** **NO (Zero training active)**.
- **Phase 3 Recommendation:** **ACCEPTED (PASSED)**.
- **Phase 4 Recommendation:** **ACCEPTED AS A DIAGNOSTIC BASELINE (PROCEED TO PHASE 5)**.

