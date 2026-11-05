# CBL-04: Google Pay Screenshot Manipulation Detection
## Phase 2 FastAPI Foundation — Setup and Integration Guide

**Role:** Member 3 (FastAPI Backend, OCR, Semantic Analysis, OOF Stacker)  
**Collaborators:** Member 2 (CNN Image Manipulation Model), Member 4 (Frontend Integration)  
**Status:** ✅ Phase 2 Complete (FastAPI Backend, Image Validation, Mock Pipeline, Automated Tests)  
**Framework:** FastAPI 0.115+ / Python 3.10+  
**Documentation Version:** `v1.0.0`  
**Date:** 2026-11-03  

---

## 1. Backend Architecture Overview

Phase 2 establishes the production FastAPI foundation for the CBL-04 detection backend. The application is structured in clean modular layers separating schemas, HTTP routing, image validation, configuration, and analysis orchestration.

```
Member 3/
├── app/
│   ├── __init__.py                  # Package entry point
│   ├── main.py                      # FastAPI app initialization, CORS, global error handlers
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                # Pydantic Settings (env overrides, upload limits, CORS)
│   ├── schemas/                     # Phase 1 Pydantic v2 data & model contracts
│   │   ├── __init__.py
│   │   ├── common.py                # Bounding boxes, coordinate conversions, model info
│   │   ├── enums.py                 # VerdictEnum, ReasonCodeEnum, ErrorCodeEnum
│   │   ├── request.py               # AnalyzeRequest, ImageUploadConfig
│   │   ├── ocr.py                   # ExtractedTransactionData, OCRResult
│   │   ├── cnn.py                   # CNNModelInput, CNNModelOutput, CNNInferenceInterface
│   │   ├── semantic.py              # SemanticBranchOutput, SemanticFeatureVector
│   │   ├── stacker.py               # StackerInput, StackerOutput, StackerInferenceInterface
│   │   ├── response.py              # AnalyzeResponse, BranchScores, SuspiciousRegion
│   │   └── errors.py                # APIErrorResponse, error sanitizer
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_validation_service.py # Binary decode, dimension, format & size validator
│   │   └── mock_analysis_service.py # Deterministic mock pipeline simulating Phases 3-6
│   └── api/
│       ├── __init__.py
│       └── routes/
│           ├── __init__.py          # Consolidated API router
│           ├── health.py            # GET /health and GET /api/v1/health
│           └── analyze.py           # POST /api/v1/analyze and POST /analyze
├── tests/
│   ├── __init__.py
│   ├── test_contracts.py            # 27 Phase 1 contract unit tests
│   └── test_api.py                  # 17 Phase 2 FastAPI integration & validation tests
├── docs/
│   ├── api_contract.md              # Phase 1 contract documentation
│   └── phase2_setup.md              # Phase 2 setup, execution, and integration guide
└── requirements.txt                 # Pinned project dependencies
```

---

## 2. Installation and Environment Setup

### Prerequisites
- **Python:** 3.10 to 3.13 (Verified on Python 3.13.7)
- **Pip:** 24+

### Step 1: Install Dependencies
From the `Member 3/` project directory:
```bash
pip install -r requirements.txt
```

Core dependencies in `requirements.txt`:
- `fastapi`: High-performance asynchronous API framework
- `uvicorn[standard]`: Lightning-fast ASGI production web server
- `python-multipart`: Streaming parser for multipart form image uploads
- `pillow`: Safe image format decoding and dimension verification
- `pydantic`: Schema validation and serialization
- `pydantic-settings`: Type-safe `.env` configuration management
- `httpx`: HTTP client backing FastAPI `TestClient`
- `pytest`: Automated test runner

---

## 3. Environment Variables Configuration

Create an optional `.env` file in the project root to override defaults:

```env
# Application Metadata
ENVIRONMENT=development
API_VERSION=0.1.0

# CORS Allowed Origins (Comma-separated or JSON list)
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000

# Image Upload Limits
MAX_FILE_SIZE_BYTES=10485760   # 10 MB in bytes
MIN_FILE_SIZE_BYTES=1          # Rejects empty files

# Mock Mode Configuration
MOCK_MODE=true
DEFAULT_VERDICT_STYLE=canonical # 'canonical' (GENUINE/EDITED) or 'frontend' (REAL/FAKE)
```

> **Security Note:** By default, CORS is restricted to local development origins (`http://localhost:5173`, `http://localhost:3000`). It does not use wildcard `*` by default.

---

## 4. Starting the Server

Run the Uvicorn ASGI server with hot-reload enabled:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Once running:
- **API Base URL:** `http://localhost:8000`
- **Interactive Swagger UI:** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`
- **OpenAPI JSON Specification:** `http://localhost:8000/openapi.json`

---

## 5. Endpoints Reference

### A. Health Endpoint

Non-blocking liveness check. Does not require image input or model availability.

- **Route:** `GET /health` (Alias: `GET /api/v1/health`)
- **HTTP Method:** `GET`
- **Response Model:** `HealthResponse`

#### Example Response (`200 OK`):
```json
{
  "status": "healthy",
  "service": "cbl04-gpay-detection-api",
  "version": "0.1.0"
}
```

---

### B. Analyze Endpoint

Primary screenshot verification endpoint.

- **Route:** `POST /api/v1/analyze` (Alias: `POST /analyze`)
- **HTTP Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Form Parameters:**
  - `image` *(Required, Binary)*: The Google Pay receipt screenshot (`.jpg`, `.jpeg`, `.png`, `.webp`, max 10MB).
  - `request_id` *(Optional, String)*: Trace identifier for logging/correlation.
  - `return_heatmap` *(Optional, Boolean, default: true)*: Request Grad-CAM spatial heatmap.
  - `verdict_style` *(Optional, String)*: Override verdict formatting (`"canonical"` or `"frontend"`).

#### Image Content & Security Validation
Every incoming upload undergoes strict multi-tier verification before any pipeline execution:
1. **Size-Capped Streaming:** Stream read in 1MB chunks; immediately raises `413 Payload Too Large` if exceeding 10MB without loading unlimited memory.
2. **Empty File Rejection:** Rejects 0-byte uploads with `400 Bad Request` (`CORRUPT_IMAGE`).
3. **True Binary Decoding:** Uses Pillow (`PIL.Image.open`) to verify decodable image headers. Does not trust client-supplied MIME or extension alone.
4. **Format Restriction:** Strictly allows only `JPEG`, `PNG`, and `WebP`. Unallowed formats (GIF, TIFF, BMP, PDF) are rejected with `415 Unsupported Media Type` (`INVALID_IMAGE_FORMAT`).
5. **Dimension Bounds:** Enforces minimum ($100 \times 100\text{px}$) and maximum ($4096 \times 4096\text{px}$) bounds. Out-of-range files return `422 Unprocessable Entity` (`IMAGE_DIMENSIONS_OUT_OF_RANGE`).
6. **Ephemeral Lifetime:** Buffers are read in-memory and closed immediately via `await image.close()`. No user screenshots are persisted permanently to disk.

---

## 6. Example Request & Response

### Example `curl` Upload:
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Accept: application/json" \
  -F "image=@receipt_screenshot.png" \
  -F "request_id=trace-demo-101" \
  -F "return_heatmap=true"
```

### Example `AnalyzeResponse` (`200 OK`):
```json
{
  "request_id": "trace-demo-101",
  "verdict": "EDITED",
  "confidence": 0.95,
  "manipulation_probability": 0.94,
  "message": "Analysis completed. Image exhibits localized tampering around amount and font inconsistencies.",
  "ocr_data": {
    "transaction_id": "UPI/234809283402",
    "amount": "₹5,000.00",
    "amount_numeric": 5000.0,
    "currency": "INR",
    "date": "2026-11-03",
    "time": "14:30:00",
    "sender": "Rahul Sharma",
    "sender_upi": "rahul@okaxis",
    "receiver": "Merchant Store",
    "receiver_upi": "store@okhdfcbank",
    "status": "Paid",
    "google_txn_id": null
  },
  "manipulation_analysis": {
    "detected": true,
    "suspicious_regions": [
      {
        "label": "Amount field",
        "score": 0.94,
        "x": 48.0,
        "y": 132.0,
        "width": 104.0,
        "height": 36.0,
        "box_normalized": {
          "xmin": 0.12,
          "ymin": 0.22,
          "xmax": 0.38,
          "ymax": 0.28
        }
      },
      {
        "label": "Transaction ID alignment",
        "score": 0.88,
        "x": 40.0,
        "y": 228.0,
        "width": 180.0,
        "height": 30.0,
        "box_normalized": {
          "xmin": 0.1,
          "ymin": 0.38,
          "xmax": 0.55,
          "ymax": 0.43
        }
      }
    ],
    "heatmap_url": null
  },
  "branch_scores": {
    "cnn_score": 0.95,
    "semantic_score": 0.92,
    "stacker_score": 0.94,
    "additional_scores": null
  },
  "cnn_score": 0.95,
  "semantic_score": 0.92,
  "stacker_score": 0.94,
  "reason_codes": [
    "AMOUNT_FORMAT_ANOMALY",
    "TEXT_LAYOUT_INCONSISTENCY"
  ],
  "model_versions": {
    "pipeline_version": "v0.1.0-phase2-mock",
    "cnn_model_version": "v0.1-mock",
    "ocr_engine_version": "v0.1-mock",
    "semantic_engine_version": "v0.1-mock",
    "stacker_model_version": "MockStacker_v0.1"
  },
  "processing_time_ms": 2.15,
  "warnings": [
    "MOCK PIPELINE: Analysis results generated by Phase 2 mock service. Actual OCR, CNN, and stacking models will be integrated in Phases 3-6."
  ],
  "disclaimer": "Notice: Screenshot analysis performs digital artifact and layout integrity checks only. It does not verify bank-side settlement, banking ledger entries, or prove that a financial payment occurred."
}
```

---

## 7. Error Handling and Sanitization

All errors conform to the Phase 1 `APIErrorResponse` schema. Internal stack traces, Windows/Linux filesystem paths, and server line numbers are strictly redacted.

### Example Error Response:
```json
{
  "error_code": "INVALID_IMAGE_FORMAT",
  "message": "Unsupported file extension in 'receipt.pdf'. Accepted extensions: .jpg, .jpeg, .png, .webp",
  "detail": null,
  "request_id": "req-c39f10a8",
  "status_code": 415,
  "timestamp": "2026-11-03T15:50:00.000000+00:00"
}
```

### HTTP Status Code & Error Code Mapping
| Status Code | Error Code | Trigger Condition |
|---|---|---|
| `400 Bad Request` | `MISSING_REQUIRED_INPUT` | The required `image` form field was not provided in the upload request. |
| `400 Bad Request` | `CORRUPT_IMAGE` | The uploaded file was 0 bytes or contained corrupt, undecodable image bytes. |
| `400 Bad Request` | `IMAGE_UNREADABLE` | PIL failed to parse the image data stream. |
| `413 Payload Too Large`| `FILE_TOO_LARGE` | Uploaded image size exceeds configured limit (default: 10 MB). |
| `415 Unsupported Media`| `INVALID_IMAGE_FORMAT` | Unaccepted extension or image format (e.g. PDF, GIF, TIFF, BMP). |
| `422 Unprocessable` | `IMAGE_DIMENSIONS_OUT_OF_RANGE` | Image dimensions are below $100\times100\text{px}$ or exceed $4096\times4096\text{px}$. |
| `422 Unprocessable` | `VALIDATION_ERROR` | Request form parameter validation failure. |
| `500 Internal Error` | `INTERNAL_PROCESSING_FAILURE` | Unhandled server exception (stack trace logged server-side only). |

---

## 8. Frontend Integration & Compatibility

The FastAPI backend is built to connect seamlessly with Member 4's `cbl04-frontend` without requiring modifications to frontend source code:

1. **Endpoint Alignment:** Route `/api/v1/analyze` matches frontend `PROPOSED_ENDPOINTS.ANALYZE`.
2. **Form Parameter:** Frontend sends `FormData.append('image', imageFile)`, which maps directly to FastAPI's `image: UploadFile = File(...)`.
3. **Verdict Mapping:**
   - Frontend checks: `verdict === 'REAL'` or `verdict === 'FAKE'` or `verdict === 'SUSPICIOUS'`.
   - Backend supports canonical (`GENUINE`/`EDITED`) and frontend styles (`REAL`/`FAKE`).
   - Frontend can request `REAL`/`FAKE` via header `X-Verdict-Style: frontend` or query param `?verdict_style=frontend`, or configure `DEFAULT_VERDICT_STYLE=frontend` in `.env`.
4. **Card Field Alignment:**
   - `ocr_data`: `transaction_id`, `amount`, `date`, `sender`, `receiver` populate `OCRDataCard.jsx` without key transformation.
   - `manipulation_analysis`: `detected`, `suspicious_regions` (`x`, `y`, `width`, `height`, `score`, `label`), and `heatmap_url` feed `ResultsDashboard.jsx` and `HeatmapView.jsx`.
5. **Timeout Grace:** The frontend's 45-second timeout in `api.js` provides ample runway for future heavy CNN and OCR inferences.

---

## 9. How the Mock Pipeline Works and Future Phase Integration

In Phase 2, `MockAnalysisService` fulfills the exact architectural contracts designed in Phase 1:

1. **Deterministic Profiles:** Automatically selects a test profile based on filename keywords:
   - Contains `"real"`, `"genuine"`, or `"authentic"` $\to$ Genuine profile (`GENUINE` / `REAL`, prob 0.02, clean reason codes).
   - Contains `"suspicious"` or `"uncertain"` $\to$ Suspicious profile (`SUSPICIOUS`, prob 0.62, compression reason codes).
   - Default / other $\to$ Edited profile (`EDITED` / `FAKE`, prob 0.94, amount/layout anomaly reason codes).
2. **Future Phase Pluggability:**
   - **Phase 3 (OCR Extraction):** Replace `MockAnalysisService.generate_mock_ocr_result` with the EasyOCR service wrapping receipt bounding box extraction and field association.
   - **Phase 4 (Semantic Feature Engine):** Replace `generate_mock_semantic_result` with the semantic rule engine computing banking syntax, font aspect ratios, and heuristic anomaly flags.
   - **Phase 5 (OCR/Semantic Baseline):** Connect the live OCR and Semantic branches into an initial baseline verdict.
   - **Phase 6 (CNN + OOF Stacker Integration):** Integrate Member 2's `CNNInferenceInterface` and the calibrated ensemble stacker into `AnalyzeResponse`.

---

## 10. Running the Test Suite

Execute pytest from the `Member 3/` directory:

```bash
# Run all tests (Contracts + API)
pytest -v

# Run only Phase 2 API tests
pytest -v tests/test_api.py

# Run only Phase 1 contract tests
pytest -v tests/test_contracts.py
```

