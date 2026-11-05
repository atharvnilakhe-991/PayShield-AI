# CBL-04 — Phase 5: GPay Dark-Theme OCR Integration Architecture & Specification

## 1. Executive Summary

This document specifies the technical architecture and implementation details for **Phase 5: GPay Dark-Theme OCR Implementation** for project **CBL-04: Google Pay Dark-Theme Transaction Screenshot Tampering Detection**.

Phase 5 transitions the Optical Character Recognition branch from mock simulation to **live, production-grade text extraction and spatial parsing** using EasyOCR (PyTorch CRAFT detection + CRNN recognition). The OCR branch extracts visible receipt text tokens, bounding box coordinates, and token confidences, while an conservative parsing layer extracts key transaction fields (`amount`, `transaction_id`, `receiver`, `receiver_upi`, `sender`, `sender_upi`, `date`, `time`, `status`, `google_txn_id`).

### Scope Boundaries:
- **Scope Target:** Strictly Google Pay dark-theme transaction receipts.
- **Independence:** The CNN branch and its verdict remain 100% independent and unaffected by OCR output during Phase 5.
- **Truthful Status:** `component_status: {"cnn": "live", "ocr": "live", "semantic": "mock", "stacker": "mock"}` and `pipeline_status: "phase5_ocr_integrated"`.
- **Zero Mock Fallback:** Engine or inference failures return HTTP 503 or 500 errors; mock data is never silently substituted.
- **Banking Settlement Disclaimer:** OCR extracts visible text for auditing; it does NOT verify banking settlement or prove that a financial transaction occurred.

---

## 2. System Architecture & Component Flow

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
                  ├───────────────────────────────────────────────┐
                  ▼                                               ▼
      [ CNNInferenceService ] (Live)                     [ EasyOCRService ] (Live)
      - ResNet-18 Multitask                               - CRAFT Text Detector
      - Manipulation Probability                          - CRNN Recognizer
                  │                                               │
                  │                                               ▼
                  │                                    [ Raw OCR Token Stream ]
                  │                                    - text, confidence
                  │                                    - pixel & normalized bboxes
                  │                                               │
                  │                                               ▼
                  │                                 [ TransactionFieldExtractor ]
                  │                                 - Amount / Currency
                  │                                 - Status (Completed/Paid)
                  │                                 - Receiver & VPA
                  │                                 - Sender & VPA
                  │                                 - Date & Time
                  │                                 - UPI Ref ID / Google Txn ID
                  │                                               │
                  │                                               ▼
                  │                                      [ ExtractedTransactionData ]
                  │                                               │
                  ├───────────────────────────────────────────────┘
                  ▼
      [ Mock Semantic & Stacker ]
      - Simulated companion branches for contract adherence (Phases 6-7)
                  │
                  ▼
      [ AnalyzeResponse Synthesis ]
      - verdict: derived strictly from live CNN score
      - confidence: uncalibrated CNN confidence
      - ocr_data: live ExtractedTransactionData
      - pipeline_status: "phase5_ocr_integrated"
      - component_status: {"cnn": "live", "ocr": "live", "semantic": "mock", "stacker": "mock"}
      - model_versions: CNN v1.0.0, EasyOCR 1.7.2, Mock Semantic/Stacker
      - transparent warnings & legal settlement disclaimer
```

---

## 3. OCR Engine Selection & Operational Characteristics

### 3.1 Why EasyOCR?
1. **Mandated in Project Specifications:** `app/schemas/ocr.py`, `docs/api_contract.md`, and `docs/phase2_setup.md` explicitly designated EasyOCR as the canonical OCR extraction engine for CBL-04.
2. **PyTorch Integration:** EasyOCR builds natively on PyTorch (`torch`), which was already installed, optimized, and verified in `.venv_mac` during Phase 2.
3. **High Spatial Precision:** EasyOCR's CRAFT (Character Region Awareness for Text Detection) outputs polygon vertices $[[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]]$, enabling accurate calculation of pixel bounding boxes and resolution-invariant normalized bounding boxes $[0.0, 1.0]$.
4. **Token-Level Confidences:** Returns true floating-point probabilities $[0.0, 1.0]$ without artificial rounding or fabrication.
5. **Offline Local Model Storage:** Model weights (`craft_mlt_25k.pth` 79MB and `english_g2.pth` 14MB) are stored locally in `models/easyocr/`. In production, `download_enabled=False` ensures offline execution inside the security sandbox.

### 3.2 Operational Telemetry
- **Device:** CPU (default, deterministic) / Apple Silicon MPS.
- **Languages:** `["en"]` (English alphanumeric).
- **Latency:** $\approx 1.2\text{s} - 1.8\text{s}$ per dark-theme receipt on Apple Silicon.
- **Lifecycle:** Thread-safe singleton (`get_ocr_service()`) with double-checked locking (`_OCR_SERVICE_LOCK`).

---

## 4. Structured Field Extraction Logic

The extraction layer (`TransactionFieldExtractor`) parses the raw token stream into `ExtractedTransactionData` and `detailed_fields: Dict[str, OCRExtractedField]`:

| Field | Extraction Pattern / Heuristic | Normalization | Extraction Status |
|---|---|---|---|
| **`amount`** | Prominent numeric token with currency prefix (`₹`, `{`, `Rs`, `INR`) in top $15\%-45\%$ of receipt | Parsed float into `amount_numeric`, formatted string `amount` (e.g. `₹1,250.00`) | `extracted`, `ambiguous`, `not_found` |
| **`status`** | Keywords: `"Completed"`, `"Payment completed"`, `"Paid"`, `"Processing"`, `"Transferred"`, `"Success"` | Canonical title string (e.g. `"Completed"`) | `extracted`, `not_found` |
| **`receiver`** | Prefixes `"To:"`, `"To "`, `"Paid to "` or token following `"To"` | Stripped payee / merchant display name | `extracted`, `not_found` |
| **`receiver_upi`** | VPA regex `[a-zA-Z0-9.\-_]+@[a-zA-Z0-9]+` appearing before sender section | Canonical UPI handle (e.g. `store@okhdfcbank`) | `extracted`, `not_found` |
| **`sender`** | Prefixes `"From:"`, `"From "`, `"Banking Name:"` | Stripped payer display name | `extracted`, `not_found` |
| **`sender_upi`** | VPA regex appearing at or after `"From:"` section | Canonical sender UPI handle | `extracted`, `not_found` |
| **`date`** | Date regex: `\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}` or `\d{4}-\d{2}-\d{2}` | Normalized date string (e.g. `"17 Apr 2025"`) | `extracted`, `not_found` |
| **`time`** | Time regex: `\d{1,2}[:.]\d{2}\s*(?:am|pm|AM|PM)?` | Normalized time string (e.g. `"1.52am"`) | `extracted`, `not_found` |
| **`transaction_id`** | Token near `"UPI transaction ID"` or `"UPI Ref ID"`, or 12-digit RRN | Alphanumeric transaction reference string | `extracted`, `not_found` |
| **`google_txn_id`** | Token near `"Google transaction ID"` or `DEMO-GPAY-...` / `CIC...` | Google payment identifier | `extracted`, `not_found` |

### Conservative Rules:
1. **No Value Invention:** If a field is not detected in the OCR stream, it is set to `None`, with `is_detected = False` and `extraction_status = "not_found"`. Missing fields never cause crashes.
2. **No Tampering Decision:** The OCR layer is purely an extractor. It does not decide whether an amount or date is genuine or fraudulent (that is deferred to Phase 6 semantic rules and Phase 7 stacking).
3. **Provenance Tracking:** Every extracted field in `detailed_fields` maintains direct references to its source `raw_text`, `confidence`, `pixel_box`, and `bounding_box`.

---

## 5. Error Handling & Security

- **`OCRInitializationError`**: Raised if model weights are missing or corrupt. FastAPI exception handler catches this and returns **`HTTP 503 Service Unavailable`** with `error_code: "MODEL_UNAVAILABLE"`.
- **`OCRImageError`**: Raised if image bytes cannot be decoded or have invalid dimensions. Returns **`HTTP 400 Bad Request`** with `error_code: "IMAGE_CORRUPTED"`.
- **`OCRInferenceError`**: Raised if PyTorch/EasyOCR execution crashes. Returns **`HTTP 500 Internal Server Error`** with `error_code: "INTERNAL_PROCESSING_FAILURE"`.
- **Zero Silent Mock Fallback:** Under no circumstances does an OCR failure silently revert to mock predictions.
- **Privacy & Redaction:** Logger outputs report only token counts and processing latency. Full transaction IDs, bank accounts, and customer names are never logged.

---

## 6. Verification & Test Evidence

The dedicated test suite `tests/test_ocr_service.py` provides 17 unit, extraction, and API integration tests:
- `test_singleton_behavior` & `test_singleton_thread_safety`
- `test_ocr_extraction_valid_png` & `test_ocr_extraction_valid_jpeg`
- `test_bounding_box_coordinates_validity` & `test_confidence_range_validity`
- `test_blank_image_truthful_empty_result` & `test_low_resolution_image_does_not_crash`
- `test_unsupported_image_bytes_raises_error`
- `test_missing_model_weights_raises_initialization_error`
- `test_inference_failure_raises_ocr_inference_error`
- `test_extract_fields_from_synthetic_receipt` & `test_dataset_clean_screenshot_extraction`
- `test_no_fabricated_values_on_missing_fields`
- `test_api_returns_live_ocr_component_status`
- `test_cnn_verdict_remains_unaffected_by_ocr`
- `test_ocr_unavailable_returns_503_error`

Total workspace test coverage: **78 passed, 0 failed, 0 skipped**.

