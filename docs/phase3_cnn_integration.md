# CBL-04 — Phase 3: CNN Inference Integration Architecture & Specification

## 1. Executive Summary

This document specifies the technical architecture and implementation details for **Phase 3: CNN Inference Integration into FastAPI** for project **CBL-04: Google Pay Dark-Theme Transaction Screenshot Tampering Detection**.

Phase 3 achieves the primary milestone of bridging Member 2's trained multi-task PyTorch ResNet-18 model with Member 3's FastAPI backend service, replacing the simulated CNN branch with live model inference while keeping downstream OCR, semantic analysis, and ensemble stacking explicitly identified as simulated components.

---

## 2. Architecture & Design Principles

```
                                  [ Client Request ]
                                          │
                                          ▼
                            [ POST /api/v1/analyze ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [ Binary Validation ]                           [ Streaming Guard ]
         (PNG/JPG/WEBP, Size)                            (Max 10MB Ceiling)
                  │
                  ▼
        [ CNNInferenceService ] (Singleton)
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
 [ Preprocessing ]      [ PyTorch Model ]
 - Aspect Lanczos       - ResNet-18 Multitask
 - Dark Pad (18,18,18)  - [1, 3, 320, 160]
 - ImageNet Norm        - CPU / Apple Silicon MPS
      │                       │
      └───────────┬───────────┘
                  │
                  ▼
         [ CNNModelOutput ]
         - Real manipulation_probability
         - score_type: uncalibrated_model_score
         - localization_available: False
                  │
                  ├───────────────────────────────────┐
                  ▼                                   ▼
      [ Mock OCR & Semantic ]             [ AnalyzeResponse Synthesis ]
      - ExtractedTransactionData          - verdict (GENUINE/EDITED/REAL/FAKE)
      - SemanticReasonCodes               - live cnn_score
      - Stacker calibration               - mock branch scores
                                          - pipeline_status: phase3_cnn_integrated
                                          - component_status: cnn=live, others=mock
                                          - transparent warnings & legal disclaimer
```

### 2.1 Core Architectural Principles
1. **Contract Integrity:** The integration strictly preserves the Pydantic v2 schemas defined in Phase 1 (`CNNModelInput`, `CNNModelOutput`, `AnalyzeResponse`, `BranchScores`).
2. **State Isolation & Singleton Lifecycle:** The PyTorch model is loaded once into memory via `get_cnn_service()` and reused across HTTP requests, minimizing latency and eliminating memory churn.
3. **Transparent Operational Labeling:** Every response includes `pipeline_status: "phase3_cnn_integrated"` and granular `component_status: {"cnn": "live", "ocr": "mock", "semantic": "mock", "stacker": "mock"}` to clearly distinguish live AI inference from simulated pipeline branches.
4. **Honest Model Representation:** Because the baseline checkpoint achieves $\approx 50.46\%$ accuracy and $0.059$ mask IoU, the output explicitly flags `score_type: "uncalibrated_model_score"`, sets `localization_available: False`, and omits misleading false-positive bounding box overlays.
5. **No Disruption of Upstream Artifacts:** Original checkpoint files (`fold_1_resnet18_multitask.pt`, `fold_2_resnet18_multitask.pt`) and the legacy Windows virtual environment are preserved without alteration.

---

## 3. CNN Inference Service Implementation

### 3.1 Class Architecture (`app/services/cnn_inference_service.py`)
`CNNInferenceService` implements the `CNNInferenceInterface` abstract base class defined in `app/schemas/cnn.py`:

```python
class CNNInferenceService(CNNInferenceInterface):
    TARGET_HEIGHT = 320
    TARGET_WIDTH = 160
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    BACKGROUND_RGB = (18, 18, 18)
```

### 3.2 Preprocessing Pipeline
Input screenshots undergo exact aspect-preserving preprocessing to match the training pipeline:
1. **RGB Conversion:** Converts any image format (RGBA, Greyscale) to 3-channel RGB.
2. **Aspect-Preserving Resize:**
   $$\text{scale} = \min\left(\frac{160}{W}, \frac{320}{H}\right)$$
   Scaled using `Image.Resampling.LANCZOS` filter.
3. **Dark Canvas Padding:** The scaled image is pasted at top-left $(0, 0)$ on an RGB $(18, 18, 18)$ dark theme canvas of size $160 \times 320$.
4. **Tensor Conversion & Normalization:**
   - Permuted from $[H, W, C]$ to $[C, H, W]$ $\to$ `[3, 320, 160]`.
   - Scaled to $[0.0, 1.0]$.
   - Normalized using ImageNet mean $(0.485, 0.456, 0.406)$ and standard deviation $(0.229, 0.224, 0.225)$.
   - Unsqueezed to batch dimension: `[1, 3, 320, 160]`.

### 3.3 Forward Inference Execution
Forward inference runs inside a `torch.inference_mode()` context:
```python
with torch.inference_mode():
    output = self.model(x)
    logits_tensor = output["logits"][0]  # Shape: [2]
    scores = torch.softmax(logits_tensor, dim=0)
    raw_edited_score = float(scores[1].item())
```

### 3.4 Spatial Localization Disablement
Audit of the Phase 7 training artifacts revealed that the mask decoder achieves a validation IoU of only $0.059$ ($5.9\%$), which generates high false-positive spatial noise. In accordance with strict explainability guidelines:
- `suspicious_regions = []`
- `heatmap_url = None`
- `localization_available = False`
- `score_type = "uncalibrated_model_score"`

---

## 4. API Endpoint Integration (`app/api/routes/analyze.py`)

### 4.1 Route Operation
`POST /api/v1/analyze` and its compatibility alias `POST /analyze` process incoming requests:
1. **Streaming Input Inspection:** Reads image in 1 MB chunks up to the 10 MB limit (`MAX_FILE_SIZE_BYTES`), raising `413 Content Too Large` if exceeded.
2. **Magic-Byte Format Validation:** Rejects non-image and unsupported formats (e.g. GIF, PDF) with `415 Unsupported Media Type`.
3. **Dimension Bounds:** Validates width and height are between $100 \times 100$ and $4096 \times 4096$ pixels, raising `422 Unprocessable Entity` if violated.
4. **Live CNN Prediction:** Dispatches to `CNNInferenceService.predict()` when `settings.CNN_ENABLED` is true and `settings.MOCK_MODE` is false.
5. **Simulated Companion Branches:** Generates mock OCR data and semantic consistency flags, fused into the response structure.
6. **Verdict Styling:** Maps verdicts based on client preference (`canonical` $\to$ `GENUINE`/`EDITED`; `frontend` $\to$ `REAL`/`FAKE`).

---

## 5. Error Handling & Fault Isolation

| Exception | Root Cause | HTTP Status | ErrorCodeEnum | Behavior |
|---|---|---:|---|---|
| `ImageValidationError` | Corrupt image, invalid dimensions, unsupported MIME | 400, 413, 415, 422 | `CORRUPT_IMAGE`, `FILE_TOO_LARGE`, etc. | Handled with clean sanitized error JSON |
| `CNNModelUnavailableError` | Missing checkpoint file or unparseable weights | 503 | `MODEL_UNAVAILABLE` | Returns service unavailable; zero stack traces exposed |
| `CNNModelError` | Tensor permutation failure, unidentifiable image bytes | 500 | `INTERNAL_PROCESSING_FAILURE` | Internal error logged on server; client receives safe message |
| `RequestValidationError` | Missing `image` multipart field | 400 | `MISSING_REQUIRED_INPUT` | Explicit field notice returned |

---

## 6. Test Suite & Verification Matrix

The test suite contains **57 automated tests** across three test modules:

```text
tests/test_contracts.py       27 Passed (100%)
tests/test_api.py             17 Passed (100%)
tests/test_cnn_inference.py   13 Passed (100%)
------------------------------------------------
Total Suite                   57 Passed (100%)
Execution Time                2.50s
```

### Coverage Highlights:
- **Singleton Lifecycle:** Verified `get_cnn_service()` returns identical instance; `reset_cnn_service()` cleanly resets.
- **Checkpoint Loading:** Verified loading of both `fold_1_resnet18_multitask.pt` and `fold_2_resnet18_multitask.pt`.
- **Preprocess Shapes:** Verified exact `[1, 3, 320, 160]` float32 tensor output on CPU/MPS.
- **Forward Pass:** Verified inference on synthetic PNG, producing valid probabilities between 0.0 and 1.0.
- **Missing Checkpoint Safety:** Verified 503 response and `MODEL_UNAVAILABLE` code when checkpoint is absent.
- **Mock Mode Toggle:** Verified system cleanly falls back to pure mock pipeline when `MOCK_MODE=True`.
- **Security Sanitization:** Verified zero leakage of stack traces, local paths, or internal variables across all error paths.

---

## 7. Model Performance & Operational Limitations Disclosure

> [!WARNING]
> **Baseline Model Status Notice:**
> 1. **Classification Reliability:** The integrated Phase 7 ResNet-18 model is an **experimental baseline** with a reported validation accuracy of approximately **50.46%** (indistinguishable from random guessing on balanced data). It exhibits severe false-positive bias on synthetic clean receipts.
> 2. **Uncalibrated Model Scores:** The score returned in `cnn_score` is a raw softmax probability from uncalibrated logits. It must not be treated as a verified fraud probability.
> 3. **Spatial Localization Inactive:** Heatmap generation and bounding box localization are explicitly disabled due to an audited validation IoU of $0.059$ ($5.9\%$).
> 4. **No Banking Settlement Verification:** The API only checks visual and layout features. It does not interface with banking networks, UPI NPCI switches, or bank ledgers.

