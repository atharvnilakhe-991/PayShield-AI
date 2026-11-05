# CBL-04 — PHASE 13: CONFIDENCE SEMANTICS AND INCONCLUSIVE UI CORRECTION REPORT

**Project:** CBL-04 — Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Workspace:** `/Users/heemanshupjain/Downloads/Member 3`  
**Role:** Member 3 — Backend / AI Integration Lead  
**Date:** 2026-11-04  
**Status:** IMPLEMENTATION & VERIFICATION COMPLETE  

---

## 1. Executive Summary

During Phase A.1 and the Phase B Readiness Gate, forensic inspection revealed a critical semantic distortion in the application pipeline: receipts with highly ambiguous multi-branch signals (stacker manipulation score near the 0.50 decision boundary, e.g. $P = 0.5008$) received an `INCONCLUSIVE` verdict but displayed a **"100% Model Confidence"** meter in the frontend.

Phase 13 resolves this issue end-to-end by implementing rigorous semantic boundaries across the backend API contract and frontend presentation:
1. **Uncertainty is no longer disguised as Confidence:** Boundary proximity ambiguity ($1.0 - 2|P - 0.50|$) is strictly decoupled into a dedicated `uncertainty_score` field.
2. **`INCONCLUSIVE` Verdicts Expose `confidence: null`:** When the multi-branch ensemble is indeterminate or branches conflict, `confidence` is returned as `null` (`None`).
3. **Primary Uncalibrated `decision_score` Introduced:** The continuous ensemble score evaluated by decision logic is explicitly exposed as `decision_score` with clear documentation that it is an uncalibrated experimental indicator, not a statistical probability.
4. **UI Progress Meter & Percentage Suppressed on `INCONCLUSIVE`:** The frontend now renders a dedicated indeterminate status badge and informative notice box for ambiguous screenshots, completely suppressing progress bars and percentage numbers.
5. **No Claims of Bank-Side Settlement:** Decisive verdicts (`GENUINE` / `EDITED`) strictly label scores as "Decision Score (Uncalibrated)" and carry explicit disclaimers that digital screenshot analysis does not prove bank-side settlement.
6. **Zero Model Overwrites or Threshold Drift:** CNN checkpoints, stacker model weights, training/validation/test manifests, and operating thresholds remain strictly frozen and bitwise verified.

---

## 2. PART 1 — Pre-Remediation Code & Data Flow Trace

### 2.1 The Defective Path
Prior to Phase 13, the pipeline traced as follows:
```
[Perception Branches]
  ├── CNN Visual Branch (ResNet-18): 0.5200
  ├── EasyOCR Extraction: conf=0.90, 0 anomalies
  └── Semantic Rule Engine: 0.0000
             │
             ▼
[OOF Stacker Meta-Learner]
  LogisticRegression: prob = 0.5008
             │
             ▼
[Final Verdict Engine (app/services/final_verdict_service.py)]
  Inside ambiguity margin [0.45, 0.55] -> Triggers rule_ambiguity_band -> INCONCLUSIVE
  DEFECTIVE FORMULA: confidence = round(1.0 - abs(0.5008 - 0.50) * 2.0, 4) = 0.9984
             │
             ▼
[API Route (app/api/routes/analyze.py)]
  AnalyzeResponse(verdict=INCONCLUSIVE, confidence=0.9984, manipulation_probability=0.5008)
             │
             ▼
[Frontend (ConfidenceScore.jsx & ResultsDashboard.jsx)]
  confidencePercent = Math.round(0.9984 * 100) = 100%
  UI renders: "100%" with a fully filled horizontal progress bar!
```

### 2.2 Mathematical Cause of the Distortion
The expression $U(P) = 1.0 - 2|P - 0.50|$ is a normalized triangular **Uncertainty Index / Boundary Entropy metric**:
- At $P = 0.5000$ (maximum ambiguity at threshold boundary), $U(0.50) = 1.0$ ($100\%$ uncertainty).
- At $P = 0.0000$ or $P = 1.0000$ (maximum boundary separation), $U = 0.0$ ($0\%$ uncertainty).

Because the backend returned this uncertainty value inside the field named `confidence`, the frontend legitimately interpreted $0.9984$ as maximum confidence, turning maximum model ignorance into maximum apparent certainty.

---

## 3. PART 2 — Backend API Semantic Corrections

### 3.1 Strict Separation of Concepts
| Semantic Concept | Field Name | Type | Value Domain | Meaning & Guardrail |
| :--- | :--- | :--- | :--- | :--- |
| **Decision Score** | `decision_score` | `Optional[float]` | $[0.0, 1.0]$ | Uncalibrated continuous score evaluated by decision logic. Never described as a calibrated probability. |
| **Uncertainty Score** | `uncertainty_score` | `Optional[float]` | $[0.0, 1.0]$ | Boundary-based ambiguity index ($1.0$ = maximum threshold ambiguity). Populated only for indeterminate states. |
| **Decision Margin** | `confidence` | `Optional[float]` | $[0.0, 1.0]$ or `null` | Heuristic decision margin for decisive verdicts ($0.50 + \|P - 0.50\|$). **Strictly `null` for `INCONCLUSIVE`.** |
| **Manipulation Score** | `manipulation_probability` | `float` | $[0.0, 1.0]$ | Raw ensemble output retained for backward compatibility; docstring clarifies it is uncalibrated. |

### 3.2 Schema Definition Changes (`app/schemas/response.py`)
```python
# Before
confidence: float = Field(
    ...,
    ge=0.0,
    le=1.0,
    description="Overall decision margin score in the assigned verdict [0.0 to 1.0]; uncalibrated heuristic metric"
)

# After
confidence: Optional[float] = Field(
    default=None,
    ge=0.0,
    le=1.0,
    description=(
        "Optional heuristic decision margin score in assigned verdict [0.0 to 1.0] for decisive verdicts, "
        "or null for INCONCLUSIVE. Uncalibrated heuristic metric; not a statistical probability."
    ),
)
decision_score: Optional[float] = Field(
    default=None,
    ge=0.0,
    le=1.0,
    description=(
        "Primary uncalibrated decision score from the multi-branch ensemble pipeline [0.0 to 1.0]. "
        "Represents continuous decision logic score; not a calibrated probability."
    ),
)
uncertainty_score: Optional[float] = Field(
    default=None,
    ge=0.0,
    le=1.0,
    description=(
        "Boundary-based normalized ambiguity index [0.0 to 1.0] measuring proximity to decision boundary; "
        "1.0 indicates maximum ambiguity. Populated for indeterminate states."
    ),
)
```

Automatic synchronization was added to `sync_branch_scores` so that `decision_score` is guaranteed to populate from `stacker_score` or `manipulation_probability` if omitted, ensuring backward compatibility with external API consumers.

### 3.3 Decision Engine Corrections (`app/services/final_verdict_service.py`)
In `evaluate_verdict()`, all 5 `INCONCLUSIVE` branch rules were updated to guarantee `confidence=None` and route ambiguity metrics to `uncertainty_score`:
1. **`rule_ambiguity_band`** ($0.45 \le P \le 0.55$):
   - `confidence = None`
   - `decision_score = round(stacker_prob, 4)`
   - `uncertainty_score = round(1.0 - abs(stacker_prob - 0.50) * 2.0, 4)`
2. **`rule_low_ocr_legibility`** ($conf < 0.45$, missing $\ge 3$ fields):
   - `confidence = None`
   - `decision_score = round(stacker_prob, 4)`
   - `uncertainty_score = round(1.0 - ocr_conf, 4)`
3. **`rule_severe_branch_conflict`** (CNN $\ge 0.75$, Sem $== 0.0$):
   - `confidence = None`
   - `decision_score = round(stacker_prob, 4)`
   - `uncertainty_score = 1.0`
4. **`rule_branch_disagreement`** (Sem flag, CNN $\le 0.35$):
   - `confidence = None`
   - `decision_score = round(stacker_prob, 4)`
   - `uncertainty_score = round(abs(cnn_score - sem_score), 4)`
5. **`rule_all_branches_unavailable`**:
   - `confidence = None`
   - `decision_score = 0.50`
   - `uncertainty_score = 1.0`

For decisive verdicts (`rule_edited_decision` and `rule_genuine_decision`):
- `confidence = round(min(0.98, max(0.50, 0.50 + decision_margin)), 4)`
- `decision_score = round(stacker_prob, 4)`
- `uncertainty_score = None`

---

## 4. PART 3 — Frontend UI Presentation Corrections

### 4.1 Before vs. After UI Behavior

| Verdict State | Previous UI Presentation | Corrected Phase 13 UI Presentation |
| :--- | :--- | :--- |
| **`INCONCLUSIVE` (near boundary)** | "100%" numeric stat, 100% full progress meter, "Elevated Indeterminate Margin" | **Suppressed percentage meter**, **Suppressed progress bar**, displays "Indeterminate", status "Insufficient or ambiguous evidence", displays Boundary Ambiguity Index `0.9994` |
| **`INCONCLUSIVE` (branch conflict)** | "50%" progress bar, labeled "Model Confidence (Simulated)" | **Suppressed progress bar**, "Indeterminate" badge, explicit explanation of branch conflict requiring visual review |
| **`GENUINE`** | "Model Confidence: 85%", implied statistical probability | Labeled **"Decision Score (Uncalibrated): 85%"**, scale `0.0 (Clean Pattern)` to `1.0 (Tamper Pattern)`, explicit settlement disclaimer |
| **`EDITED`** | "Model Confidence: 90%", "Probability of Tampering" | Labeled **"Decision Score (Uncalibrated): 90%"**, visual anomaly score breakdown, explicit settlement disclaimer |

### 4.2 Component Architecture Updates
1. **`ConfidenceScore.jsx`**:
   - Props: `confidence`, `decisionScore`, `uncertaintyScore`, `verdict`, `scoreType`, `_pipelineStatus`.
   - When `isInconclusive = true`:
     - Entire `<div className="confidence-meter-track" ...>` progress bar is removed from the DOM.
     - Percentage number `{confidencePercent}%` is removed from the DOM.
     - Renders an informative amber alert box explaining that perception signals fell into the indeterminate zone and manual inspection is required.
     - Displays `Boundary Ambiguity Index: <value>` and `Uncalibrated Decision Score: <value>`.
   - When decisive (`GENUINE` / `EDITED`):
     - Renders uncalibrated decision score percentage and raw value.
     - Progress bar is labeled `aria-label="Model uncalibrated decision score meter"`.
     - Scale markers updated to `0.0 (Clean Pattern)`, `0.50 Threshold`, `1.0 (Tamper Pattern)`.
2. **`ResultsDashboard.jsx`**:
   - Top header stat tile renders `Assessment Status: Indeterminate` for ambiguous screenshots, with footnote displaying `Ambiguity Index: <val>`.
   - Multi-branch AI breakdown labels updated from `Raw manipulation prob` to `Visual anomaly score`, and `Ensemble probability` to `Ensemble decision score`.
   - Forwards `decision_score` and `uncertainty_score` to `ConfidenceScore`.

---

## 5. PART 4 — Verification, Test Results & Quality Gates

### 5.1 New Dedicated Unit & Integration Tests
Created [`tests/test_phase13_confidence_semantics.py`](file:///Users/heemanshupjain/Downloads/Member%203/tests/test_phase13_confidence_semantics.py) covering 10 distinct edge cases:
1. `test_inconclusive_ambiguity_band_has_null_confidence_and_high_uncertainty`: Verifies that $P = 0.4997$ returns `confidence is None`, `uncertainty_score >= 0.99`, and `decision_score == 0.4997`.
2. `test_inconclusive_branch_conflict_has_null_confidence`: Verifies severe branch conflict returns `confidence is None` and `uncertainty_score == 1.0`.
3. `test_inconclusive_low_ocr_legibility_has_null_confidence`: Verifies low OCR quality returns `confidence is None` and `uncertainty_score == 0.75`.
4. `test_inconclusive_all_branches_failed_has_null_confidence`: Verifies all-branch failure returns `confidence is None`.
5. `test_genuine_decisive_has_uncalibrated_decision_score_and_null_uncertainty`: Verifies decisive GENUINE returns decision score and null uncertainty.
6. `test_edited_decisive_has_uncalibrated_decision_score_and_null_uncertainty`: Verifies decisive EDITED returns decision score and null uncertainty.
7. `test_response_allows_null_confidence_for_inconclusive`: Verifies Pydantic model validation and JSON serialization with null confidence.
8. `test_response_auto_syncs_decision_score`: Verifies automatic fallback synchronization of `decision_score`.
9. `test_response_handles_missing_optional_scores_gracefully`: Verifies backwards compatibility when optional fields are omitted.
10. `test_analyze_endpoint_blank_image_returns_inconclusive_with_null_confidence`: Live HTTP API end-to-end integration test validating FastAPI serialization.

### 5.2 Full Test Suite Execution Summary
- **Backend Test Suite:**
  - Command: `./.venv_mac/bin/pytest tests/ -q`
  - Total tests executed: **150 passed** (0 failed, 66 framework warnings)
  - Execution latency: **38.25s**
- **Frontend Linter:**
  - Command: `npm run lint` (`oxlint`)
  - Result: **0 warnings, 0 errors** across 19 files (19ms)
- **Frontend Production Build:**
  - Command: `npm run build` (`vite build`)
  - Result: **Successfully built in 125ms** (bundle: `dist/assets/index-6qY1Njwu.js` 335.66 kB, gzip: 106.91 kB)

---

## 6. PART 5 — Strict Constraints & Asset Invariance Audit

To confirm that no unauthorized modifications occurred, cryptographic SHA-256 hashes were recorded before and after Phase 13:

| Asset Name | Expected Path | SHA-256 Checksum | Invariance Status |
| :--- | :--- | :--- | :--- |
| **Member 3 Multi-Task CNN Checkpoint** | `Hackathon-VIT/CBL04-Dataset/models/phase7/fold_1_resnet18_multitask.pt` | `44b654aa846fddfde7be6dab44ceada6687d8dfd35e2c83edbcb60fed909f62a` | **UNTOUCHED / IDENTICAL** |
| **Phase 8 Stacker Meta-Learner Weights** | `models/phase8/stacker_model.json` | `230df8204929411087eba9e184cd31ae203d4f605474833ccbf7063d973e7438` | **UNTOUCHED / IDENTICAL** |
| **Training Partition Manifest** | `Hackathon-VIT/CBL04-Dataset/metadata/phase6/train_manifest.csv` | `4e1822efa1aa5a8078468e34e132389711a1af34dab8eb7ba5ad0b5fe467d2d7` | **UNTOUCHED / IDENTICAL** |
| **Validation Partition Manifest** | `Hackathon-VIT/CBL04-Dataset/metadata/phase6/validation_manifest.csv` | `d66bfcb6e9056473ea97266c64ee0a09c927d2f4cb5cb91c4dee83af3d7773fa` | **UNTOUCHED / IDENTICAL** |
| **Frozen Test Partition Manifest** | `Hackathon-VIT/CBL04-Dataset/metadata/phase6/test_manifest.csv` | `3f14f6a773069fe6f539f194299d0c14dd5cfc0952dd84ed099b8f391d624856` | **UNTOUCHED / IDENTICAL** |
| **Held-Out Real Partition Manifest** | `Hackathon-VIT/CBL04-Dataset/metadata/phase6/restricted_real_holdout_manifest.csv` | `1e3fcd574df2cae0abd027158d17b2c08a8d4fab2d20b19105dcd46ace6cf75e` | **UNTOUCHED / IDENTICAL** |
| **Operating Decision Threshold** | `app/services/final_verdict_service.py:68` | `operating_threshold = 0.50` | **UNTOUCHED / IDENTICAL** |
| **Ambiguity Margin** | `app/services/final_verdict_service.py:69` | `ambiguity_margin = 0.05` | **UNTOUCHED / IDENTICAL** |

---

## 7. Modified Files & Backward Compatibility

### 7.1 Modified Files Summary
1. `app/schemas/response.py`: Updated `AnalyzeResponse` with `confidence: Optional[float] = None`, added `decision_score` and `uncertainty_score`, updated validator.
2. `app/services/final_verdict_service.py`: Updated `FinalVerdictResult` to carry `decision_score` and `uncertainty_score`; set `confidence=None` on all 5 `INCONCLUSIVE` branch rules.
3. `app/api/routes/analyze.py`: Forwarded `decision_score` and `uncertainty_score` to `AnalyzeResponse`.
4. `app/services/mock_analysis_service.py`: Updated mock generator to conform to new semantic fields.
5. `cbl04-frontend/src/components/ConfidenceScore.jsx`: Implemented suppression of progress meters and percentages on `INCONCLUSIVE`, added explicit uncalibrated labeling.
6. `cbl04-frontend/src/components/ResultsDashboard.jsx`: Updated header stat tile for indeterminate status, forwarded decision and uncertainty scores.
7. `tests/test_api.py`, `tests/test_phase10_frontend_backend_integration.py`, `tests/test_final_verdict_service.py`, `tests/test_ocr_service.py`, `tests/test_phase_a_remediation.py`: Updated assertion semantics to accommodate `confidence: None`.
8. `tests/test_phase13_confidence_semantics.py`: Added 10 dedicated regression tests.

### 7.2 Backward Compatibility Assessment
- **External API Clients:** Clients reading `response["confidence"]` will receive `null` instead of a misleading `0.9994` on inconclusive verdicts. For decisive verdicts, `confidence` continues to provide the decision margin.
- **New API Clients:** Can consume `response["decision_score"]` directly for uncalibrated multi-branch ranking and `response["uncertainty_score"]` for ambiguity threshold monitoring.
- **Frontend Clients:** The teammate-developed React frontend cleanly consumes both legacy and updated fields without runtime errors or crashes.

---
**Report Author:** Member 3 — Backend / AI Integration Lead  
**Handover Status:** Complete. Confidence semantics and indeterminate UI presentation verified.

