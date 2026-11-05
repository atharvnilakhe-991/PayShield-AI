# CBL-04 — Phase 8: Out-Of-Fold (OOF) Stacker Documentation

## 1. Executive Summary

Phase 8 implements the Out-Of-Fold (OOF) Stacker Meta-Learner for project **CBL-04: Google Pay Dark-Theme Transaction Screenshot Tampering Detection**. The stacker synthesizes first-stage predictions from three specialized branches:
1. **CNN Visual Perception Branch:** ResNet-18 multi-task manipulation detector.
2. **OCR Extraction Branch:** EasyOCR 1.7.2 text and spatial bounding box extractor.
3. **Semantic Consistency Branch:** 22 deterministic heuristics evaluating transaction metadata, UPI rules, font ratios, and spatial alignment.

To eliminate target leakage, the stacker trains strictly on genuine **Out-Of-Fold (OOF)** predictions (868 samples across `compact` and `enriched` layout families) where the predicting CNN model never observed the layout family during training. Development decisions and hyperparameter regularization selection utilize strictly the independent **validation partition** (108 samples, `bank_progress` layout family). The 24-sample held-out test partition (`recharge` layout family) remains completely untouched until final Phase 9 evaluation.

---

## 2. Input Schema Contract

The stacker consumes an 18-dimensional feature vector. Target leakage and shortcut features are strictly prohibited:
- **Excluded Features:** `sample_id`, `pair_id`, `layout_family`, `split`, `true_label`, file paths, and filenames.

### Feature Specification

| Index | Feature Name | Type | Range | Description / Provenance |
| :---: | :--- | :---: | :---: | :--- |
| 0 | `cnn_score` | Float | $[0.0, 1.0]$ | Raw uncalibrated manipulation probability from CNN branch. Neutral prior $0.5$ if CNN unavailable. |
| 1 | `cnn_pred_class` | Binary | $\{0, 1\}$ | Binary class flag ($1$ if `cnn_score` $\ge 0.50$, else $0$). |
| 2 | `ocr_overall_confidence` | Float | $[0.0, 1.0]$ | Mean confidence across all extracted OCR text tokens. $0.0$ if OCR failed. |
| 3 | `ocr_amount_extracted` | Binary | $\{0, 1\}$ | $1$ if amount or amount_numeric extracted; $0$ otherwise. |
| 4 | `ocr_recipient_extracted` | Binary | $\{0, 1\}$ | $1$ if payee name or payee UPI ID extracted; $0$ otherwise. |
| 5 | `ocr_ref_id_extracted` | Binary | $\{0, 1\}$ | $1$ if 12-digit UPI RRN or Google Transaction ID extracted. |
| 6 | `ocr_date_extracted` | Binary | $\{0, 1\}$ | $1$ if transaction date or timestamp extracted; $0$ otherwise. |
| 7 | `semantic_manipulation_score` | Float | $[0.0, 1.0]$ | Heuristic manipulation score from SemanticFeatureService. |
| 8 | `amount_anomaly` | Binary | $\{0, 1\}$ | Flagged format, currency, or decimal anomaly in amount. |
| 9 | `recipient_anomaly` | Binary | $\{0, 1\}$ | Flagged anomaly in payee name or UPI address. |
| 10 | `date_anomaly` | Binary | $\{0, 1\}$ | Flagged anomaly in date or time chronology. |
| 11 | `ref_id_anomaly` | Binary | $\{0, 1\}$ | Flagged anomaly in UPI 12-digit RRN pattern. |
| 12 | `status_anomaly` | Binary | $\{0, 1\}$ | Flagged anomaly in payment status text or casing. |
| 13 | `upi_handle_anomaly` | Binary | $\{0, 1\}$ | Flagged invalid bank handle suffix (e.g. missing `@ok...`). |
| 14 | `missing_critical_fields_count` | Float | $[0.0, 1.0]$ | Normalized count of missing critical fields ($k / 4.0$). |
| 15 | `branch_cnn_available` | Binary | $\{0, 1\}$ | $1$ if live CNN executed successfully; $0$ otherwise. |
| 16 | `branch_ocr_available` | Binary | $\{0, 1\}$ | $1$ if live EasyOCR executed successfully; $0$ otherwise. |
| 17 | `branch_semantic_available` | Binary | $\{0, 1\}$ | $1$ if Semantic Consistency engine executed successfully. |

---

## 3. Mathematical Formulation

To ensure complete interpretability, zero arbitrary code execution risks (avoiding Python `pickle`), and strict compatibility with sandbox environments where `scikit-learn` is absent, the stacker is implemented as an L2-regularized logistic regression model optimized via `scipy.optimize.minimize`:

$$\min_{w, b} J(w, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \sigma(w^T x_i + b) + (1 - y_i) \ln (1 - \sigma(w^T x_i + b)) \right] + \frac{\lambda}{2} \|w\|_2^2$$

where:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Continuous features ($x_0, x_2, x_7, x_{14}$) are standardized using training set mean $\mu$ and standard deviation $\sigma$: $\tilde{x} = \frac{x - \mu}{\sigma}$.
- Indicator features retain binary $0/1$ representation.
- $\lambda$ is tuned via grid search $\lambda \in [10^{-4}, 10^1]$ selecting the candidate maximizing validation Macro-F1.

### Model Serialization
Model parameters are stored in human-readable, auditable JSON:
`models/phase8/stacker_model.json`
containing:
- `stacker_version`: "v1.0.0-phase8"
- `l2_regularization`: selected $\lambda$
- `intercept`: scalar bias $b$
- `weights`: dictionary mapping each feature name to its learned regression coefficient $w_j$
- `scaler`: dictionary mapping each feature name to its `mean` and `std`
- `operating_threshold`: $0.50$
- `calibration_status`: `"uncalibrated"`

---

## 4. Missing Branch Handling

The stacker explicitly accounts for branch failures without throwing unhandled exceptions or inventing fake telemetry:
1. **CNN Failure:** If CNN inference fails or weights are unavailable:
   - `branch_cnn_available = 0.0`
   - `cnn_score = 0.50` (uninformative neutral prior)
   - `cnn_pred_class = 0.0`
2. **OCR Failure:** If EasyOCR crashes, fails image decode, or times out:
   - `branch_ocr_available = 0.0`
   - `ocr_overall_confidence = 0.0`
   - All OCR presence flags ($x_3, x_4, x_5, x_6$) set to $0.0$.
3. **Semantic Failure:** If OCR is missing or semantic rules encounter an unhandled exception:
   - `branch_semantic_available = 0.0`
   - `semantic_manipulation_score = 0.0`
   - All anomaly indicators set to $0.0$.
   - `missing_critical_fields_count = 1.0` ($4 / 4$ missing).
4. **All Branches Unavailable:**
   - Evaluates baseline neutral prediction with `decision_metadata` documenting total absence of input signals.

---

## 5. Verification & Testing

Unit tests in `tests/test_stacker_service.py` establish:
- **Thread Safety:** Thread-safe singleton access via `get_stacker_service()`.
- **Target Leakage Prohibition:** Automatic assertion verifying no forbidden tokens (`label`, `split`, `pair`, `family`, `path`) exist in feature schemas.
- **Safe Fallback:** Correct initialization with fallback baseline weights when model file is not yet persisted.
- **Deterministic Inference:** Idempotent predictions across repeated forward passes on identical inputs.
- **Graceful Degradation:** Robust scoring when CNN, OCR, or Semantic branches are individually or simultaneously suppressed.

