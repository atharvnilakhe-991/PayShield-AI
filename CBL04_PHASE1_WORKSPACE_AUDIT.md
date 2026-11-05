# CBL-04 — PHASE 1: WORKSPACE VERIFICATION & PROJECT FREEZE AUDIT REPORT

**Document Identifier:** `CBL04_PHASE1_WORKSPACE_AUDIT.md`  
**Audit Timestamp:** 2026-11-03T18:14:04+05:30  
**Target Application:** Google Pay (Dark Theme) UPI Transaction Screenshot Tampering Detection  
**Auditor Role:** Member 3 (Backend & AI Integration Lead)  
**Audit Mode:** Strictly Read-Only Audit & Project Freeze Verification  
**Workspace Root:** `/Users/heemanshupjain/Downloads/Member 3`  
**Host Environment:** macOS (Darwin Kernel Version 27.2.0, arm64), Host Python: 3.14.3  

---

## CLAIM CLASSIFICATION TAXONOMY
Every technical finding and architectural claim in this report is explicitly classified under one of the following five standardized labels:
- `[VERIFIED IN CURRENT WORKSPACE]`: Confirmed directly by reading files, inspecting source code, or examining filesystem artifacts within `/Users/heemanshupjain/Downloads/Member 3`.
- `[HISTORICAL DOCUMENTATION ONLY]`: Documented in historical Phase 1–8 reports or handover files within the workspace, but represents historical logs rather than a verified live service.
- `[NOT VERIFIED]`: Referenced in documentation or code comments, but unverified due to missing dependencies, environment constraints, or unexecuted test suites.
- `[NOT PRESENT]`: Searched for systematically across the workspace directory tree and confirmed to be completely absent.
- `[INFERENCE]`: Deduced from structural evidence, file paths, configurations, or codebase conventions without explicit narrative documentation.

---

## 1. EXECUTIVE SUMMARY

`[VERIFIED IN CURRENT WORKSPACE]`  
A complete, non-destructive, read-only audit of the CBL-04 project workspace was conducted at `/Users/heemanshupjain/Downloads/Member 3`. No source code, configuration files, dataset files, metadata manifests, or binary model checkpoints were modified, moved, renamed, or deleted.

The primary findings of the Phase 1 audit are:
1. **Dual-Tree Codebase Structure:** The workspace contains two distinct, unlinked development trees:
   - **Member 3 FastAPI Backend Tree** (`app/`, `schemas/`, `services/`, `tests/`, `docs/`): Fully structured FastAPI 0.115+ foundation with comprehensive Pydantic v2 data contracts, strict 10MB streaming image upload validation, sanitized global error handling, and 44 unit/integration tests. However, the analysis route currently invokes a deterministic mock service (`MockAnalysisService`).
   - **Hackathon ML Pipeline, Dataset & Local Demo Tree** (`Hackathon-VIT/CBL04-Dataset/`): Complete synthetic data generation pipeline, 387 unique real dark-theme screenshots, 500 matched synthetic pairs across 4 layout families, two trained PyTorch ResNet-18 multi-task checkpoints (`models/phase7/`), and a standalone Python `ThreadingHTTPServer` browser demo.
2. **Disconnected Inference:** The FastAPI backend is **completely disconnected** from the PyTorch model checkpoints. The FastAPI backend serves mock results, while the standalone demo runs on an independent standard-library server (`app/server.py` on port 8080).
3. **Preserved Baseline Checkpoint Reality:** Checkpoint `models/phase7/fold_1_resnet18_multitask.pt` (57.7 MB) is preserved as the official baseline. However, historical evaluation reveals a pooled two-fold accuracy of **50.46% (effectively coin-flip random guessing)**, a severe positive bias (Fold 1 classified 100% of compact clean images as edited), and a mask localization IoU of only **0.059 (5.9%)**. Folds 3 and 4 were never trained.
4. **Missing Official Frontend:** Member 4's official React/Vite frontend (`cbl04-frontend`) is **[NOT PRESENT]** in the workspace. A fallback vanilla HTML/JS demo exists under `Hackathon-VIT/CBL04-Dataset/app/`, but its API contract diverges from the FastAPI schema.
5. **Environment Divergence:** The host system is macOS arm64 (Python 3.14.3), while the existing virtual environment (`.venv_phase7`) is a Windows x64 virtualenv containing `.exe` binaries created under Python 3.12.10 on Windows. PyTorch and FastAPI dependencies are not installed in the host environment.
6. **Project Scope:** Strictly confined to Google Pay dark-theme UPI transaction receipts. The system performs digital image manipulation and layout consistency checks only; it cannot and does not verify bank-side ledger settlement.

---

## 2. WORKSPACE AND DIRECTORY STRUCTURE IDENTIFICATION

### 2.1 Workspace Root and Source Lineage
- **Active Workspace Root:** `/Users/heemanshupjain/Downloads/Member 3` `[VERIFIED IN CURRENT WORKSPACE]`
- **Subproject Root:** `/Users/heemanshupjain/Downloads/Member 3/Hackathon-VIT/CBL04-Dataset` `[VERIFIED IN CURRENT WORKSPACE]`
- **Operating System:** macOS (Darwin 27.2.0, arm64 / Apple Silicon) `[VERIFIED IN CURRENT WORKSPACE]`
- **Host Python Version:** Python 3.14.3 (`/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`) `[VERIFIED IN CURRENT WORKSPACE]`
- **Apple System Python:** Python 3.9.6 (`/usr/bin/python3`) `[VERIFIED IN CURRENT WORKSPACE]`
- **Version Control Status:** **Not a Git repository**. Running `git status` or checking for `.git` directories returns:  
  `fatal: not a git repository (or any of the parent directories): .git` `[VERIFIED IN CURRENT WORKSPACE]`
- **Workspace Origin:** `[INFERENCE]` The workspace appears to be an unversioned filesystem export or manual folder download combining two separate working trees transferred from a Windows development environment. Evidence: `.venv_phase7/pyvenv.cfg` explicitly references `command = C:\Users\dell\AppData\Local\Programs\Python\Python312\python.exe -m venv C:\Users\dell\Desktop\Hackathon-VIT\CBL04-Dataset\.venv_phase7`.
- **Other Discovered Copies:** No other member directories (e.g., `Member 1`, `Member 2`, `Member 4`) exist within the accessible workspace boundary `[VERIFIED IN CURRENT WORKSPACE]`.

### 2.2 Verified Complete Project File Tree (Excluding `.venv` and `__pycache__`)

```
/Users/heemanshupjain/Downloads/Member 3/
├── .gitignore
├── requirements.txt                              # Member 3 FastAPI dependencies
├── docs/
│   ├── api_contract.md                           # Phase 1 contract specification (470 lines)
│   └── phase2_setup.md                           # Phase 2 setup and integration guide (342 lines)
├── app/
│   ├── __init__.py                               # Package initialization
│   ├── main.py                                   # FastAPI application, CORS, error handlers (197 lines)
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                             # Pydantic Settings v2 configuration (107 lines)
│   ├── schemas/                                  # Phase 1 Pydantic v2 data contracts
│   │   ├── __init__.py
│   │   ├── common.py                             # Bounding boxes (normalized & pixel), ModelVersionInfo
│   │   ├── enums.py                              # VerdictEnum, ReasonCodeEnum, ErrorCodeEnum
│   │   ├── request.py                            # ImageUploadConfig, AnalyzeRequest
│   │   ├── response.py                           # AnalyzeResponse, BranchScores, SuspiciousRegion
│   │   ├── ocr.py                                # OCRExtractedField, ExtractedTransactionData, OCRResult
│   │   ├── cnn.py                                # CNNModelInput, CNNModelOutput, CNNInferenceInterface
│   │   ├── semantic.py                           # SemanticFeatureVector, SemanticBranchOutput
│   │   ├── stacker.py                            # StackerInput, StackerOutput, StackerInferenceInterface
│   │   └── errors.py                             # APIErrorResponse, regex error sanitizer
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_validation_service.py           # 10MB streaming size check, Pillow decode, bounds validator
│   │   └── mock_analysis_service.py              # Phase 2 deterministic mock pipeline (427 lines)
│   └── api/
│       ├── __init__.py
│       └── routes/
│           ├── __init__.py                       # Router aggregator
│           ├── health.py                         # GET /health & GET /api/v1/health
│           └── analyze.py                        # POST /api/v1/analyze & POST /analyze
├── tests/
│   ├── __init__.py
│   ├── test_contracts.py                         # 27 Phase 1 contract unit tests (450 lines)
│   └── test_api.py                               # 17 Phase 2 FastAPI integration tests (253 lines)
└── Hackathon-VIT/
    └── CBL04-Dataset/
        ├── README.md                             # Dataset, model, and demo documentation (230 lines)
        ├── requirements.txt                      # PIL requirements
        ├── requirements-phase7.txt               # PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, PIL, numpy
        ├── .venv_phase7/                         # Windows-generated Python 3.12 venv (.exe binaries)
        ├── data/
        │   ├── real_gpay/raw/Dataset-Hackathon/  # 394 raw JPEG screenshots (387 unique contents)
        │   ├── synthetic_gpay/                   # Pilot 1 (20 clean, 20 edited, 40 PNG masks)
        │   ├── synthetic_gpay_phase3/            # Phase 3 Visual Pilot (12 clean, 12 edited, 24 PNG masks)
        │   └── synthetic_gpay_phase4/            # Phase 4 Bulk Dataset (500 clean, 500 edited, 1000 PNG masks)
        ├── annotations/
        │   ├── bounding_boxes/bounding_boxes.csv # Pilot 1 bounding boxes
        │   ├── field_annotations/                # Pilot 1 JSON annotations (SYN-20261103-001..020.json)
        │   ├── phase3/                           # Phase 3 JSON annotations & bounding_boxes.csv
        │   └── phase4/                           # Phase 4 JSON annotations (500 files) & bounding_boxes.csv (707 rows)
        ├── metadata/
        │   ├── manifest.csv                      # Pilot 1 manifest
        │   ├── source_group_splits.csv           # Pilot 1 splits
        │   ├── pilot_config.json                 # Pilot 1 configuration
        │   ├── phase3_manifest.csv               # Phase 3 manifest (24 rows)
        │   ├── phase4_manifest.csv               # Phase 4 bulk manifest (1000 rows)
        │   ├── phase4_batches/                   # Batches 1 to 5 manifests
        │   ├── phase6/                           # Phase 6 split manifests (train, val, test, real)
        │   ├── phase6_manifest.csv               # Phase 6 master manifest
        │   └── phase7/                           # Phase 7 4-fold manifests and training_config.json
        ├── models/
        │   └── phase7/
        │       ├── fold_1_resnet18_multitask.pt  # 57,727,691 bytes (Preserved Baseline Checkpoint Fold 1)
        │       └── fold_2_resnet18_multitask.pt  # 57,727,691 bytes (Preserved Checkpoint Fold 2)
        ├── app/
        │   ├── server.py                         # Standalone Python ThreadingHTTPServer (port 8080)
        │   ├── index.html                        # Local demo browser interface
        │   ├── styles.css                        # Local demo stylesheet
        │   └── app.js                            # Local demo vanilla JS frontend
        ├── scripts/
        │   ├── generator.py                      # Pilot generator
        │   ├── generate_pilot.py                 # CLI pilot runner
        │   ├── audit_real_dataset.py             # Hashes, dimensions, format auditor for real archive
        │   ├── second_stage_audit.py             # Scope and visual verification compiler
        │   ├── validate_dataset.py               # Pixel difference and annotation validator
        │   ├── build_layout_family_inventory.py  # Layout family clustering
        │   ├── analyze_layout_families.py        # Layout analysis
        │   ├── create_quality_contact_sheets.py  # Visual review sheet generator
        │   ├── phase3/                           # Phase 3 generator, renderer, validator
        │   ├── phase4/                           # Phase 4 bulk generator (500 pairs) & sheet builder
        │   ├── phase5/                           # Phase 5 full dataset audit
        │   ├── phase6/                           # Phase 6 leakage-safe split generator & validator
        │   ├── phase7/
        │   │   ├── model.py                      # PyTorch GPayTamperModel (ResNet-18 + U-Net decoder)
        │   │   ├── prepare_folds.py              # 4-fold family-held-out fold builder
        │   │   ├── train_evaluate.py             # Multitask training runner (epochs 1-3)
        │   │   ├── evaluate_saved_folds.py       # Standalone checkpoint evaluator
        │   │   └── validation_threshold_eval.py  # Validation-only threshold sensitivity analysis
        │   └── phase8/
        │       └── smoke_test_app.py             # Phase 8 HTTP smoke test runner
        ├── templates/
        │   ├── phase3/layout_families.json       # Layout family geometry templates
        │   └── phase7/model_config.json          # Architecture and training defaults
        ├── backups/                              # Pre-phase backup READMEs and manifests
        └── reports/                              # Detailed markdown and JSON audit/eval reports
            ├── dataset_inventory.csv / .md / .json
            ├── real_image_classification.csv
            ├── validation_report.json
            ├── phase3_validation_report.md / .json
            ├── phase4/                           # Phase 4 validation, duplicate, and distribution reports
            ├── phase5/                           # Phase 5 inventory, annotation, privacy, quality audits
            ├── phase6/                           # Phase 6 split strategy, leakage, balance reports
            ├── phase7/                           # Phase 7 training, evaluation, error analysis reports
            └── phase8/                           # Phase 8 app testing, final project report, handover
```

---

## 3. AUDIT OF THE FASTAPI BACKEND

### 3.1 Core Architecture and Initialization
`[VERIFIED IN CURRENT WORKSPACE]`
- **Application Factory:** `app/main.py` defines `create_application() -> FastAPI` and instantiates the global `app` object.
- **Title and Version:** Configured via `app/core/config.py` (`Settings.API_TITLE = "CBL-04 Google Pay Screenshot Manipulation Detection API"`, `API_VERSION = "0.1.0"`).
- **Interactive Documentation:** Registered at `/docs` (Swagger UI), `/redoc` (ReDoc), and `/openapi.json`.
- **CORS Middleware:** Configured with `allow_credentials=True`, `allow_methods=["GET", "POST", "OPTIONS"]`, `allow_headers=["*"]`. Configured allowed origins default to `["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"]` (parsed from string or list via Pydantic validator).
- **Request Tracing Middleware:** An HTTP middleware intercepts all requests, extracts incoming `X-Request-ID` or generates `req-{uuid4.hex[:12]}`, attaches it to `request.state.trace_id`, and injects `X-Request-ID` into the HTTP response headers.
- **Global Exception Handling:** Implemented for four specific exception tiers, all returning structured JSON conforming to `APIErrorResponse`:
  1. `ImageValidationError`: Maps custom domain error codes (`FILE_TOO_LARGE`, `INVALID_IMAGE_FORMAT`, `CORRUPT_IMAGE`, `IMAGE_DIMENSIONS_OUT_OF_RANGE`) to HTTP 400, 413, 415, 422.
  2. `RequestValidationError`: Distinguishes missing `image` parameter (HTTP 400 `MISSING_REQUIRED_INPUT`) from general payload validation errors (HTTP 422 `VALIDATION_ERROR`). Sanitizes error details.
  3. `HTTPException`: Maps status codes to standard `ErrorCodeEnum` values.
  4. `Exception`: Catches unhandled errors, logs the full stack trace server-side only, and returns HTTP 500 (`INTERNAL_PROCESSING_FAILURE`) without leaking internal filesystem paths or stack traces.

### 3.2 Endpoint Inventory

| Endpoint Route | HTTP Method | Expected Inputs | Response Model | Current Implementation | Inference Type |
|---|---|---|---|---|---|
| `/health` | `GET` | None | `HealthResponse` | `app/api/routes/health.py` | Liveness check (Non-ML) |
| `/api/v1/health` | `GET` | None (Alias) | `HealthResponse` | `app/api/routes/health.py` | Liveness check (Non-ML) |
| `/api/v1/analyze` | `POST` | Multipart Form: `image` (UploadFile, required), `request_id` (str, opt), `return_heatmap` (bool, default True), `verdict_style` (query, opt), `X-Verdict-Style` (header, opt) | `AnalyzeResponse` | `app/api/routes/analyze.py` | **MOCK IMPLEMENTATION** (`MockAnalysisService.run_mock_pipeline`) |
| `/analyze` | `POST` | Multipart Form (Same as above, compatibility alias) | `AnalyzeResponse` | `app/api/routes/analyze.py` | **MOCK IMPLEMENTATION** |

### 3.3 Image Validation Service (`app/services/image_validation_service.py`)
`[VERIFIED IN CURRENT WORKSPACE]`  
Uploaded images undergo strict multi-tier verification before entering the analysis pipeline:
1. **Streaming Size Limit:** Reads chunks of 1MB; raises `ImageValidationError` (`FILE_TOO_LARGE`, HTTP 413) if cumulative bytes exceed `MAX_FILE_SIZE_BYTES` (default: 10 MB / 10,485,760 bytes).
2. **Empty File Check:** Raises `CORRUPT_IMAGE` (HTTP 400) if file size $< 1$ byte.
3. **Pillow Header Verification:** Decodes image bytes using `PIL.Image.open(io.BytesIO(file_bytes))` and verifies image integrity. Does not trust client-supplied MIME or extension headers alone.
4. **Format Restriction:** Strictly allows `{"JPEG", "PNG", "WEBP"}`. Disallowed formats (GIF, TIFF, BMP, PDF) raise `INVALID_IMAGE_FORMAT` (HTTP 415).
5. **Dimension Bounds:** Enforces min dimensions ($100 \times 100\text{ px}$) and max dimensions ($4096 \times 4096\text{ px}$). Out-of-bounds images raise `IMAGE_DIMENSIONS_OUT_OF_RANGE` (HTTP 422).
6. **Ephemeral Lifetime:** Buffers are read in-memory; `await image.close()` is guaranteed in a `finally` block. No temporary files are written to disk.

### 3.4 Analysis Route Inference Status: Mock vs Real
`[VERIFIED IN CURRENT WORKSPACE]`  
The analysis route `POST /api/v1/analyze` **does NOT call a real machine learning model**.
In `app/api/routes/analyze.py` (lines 92–98):
```python
# Execute analysis pipeline (Phase 2 Mock Service)
response = MockAnalysisService.run_mock_pipeline(
    image=validated_image,
    request_id=request_id,
    return_heatmap=return_heatmap,
    verdict_style=style,
)
return response
```
`MockAnalysisService` (`app/services/mock_analysis_service.py`) uses deterministic keyword matching on the uploaded filename:
- Filenames containing `"real"`, `"genuine"`, or `"authentic"` $\to$ Genuine mock profile (`GENUINE` / `REAL`, prob 0.02, clean reason codes).
- Filenames containing `"suspicious"` or `"uncertain"` $\to$ Suspicious mock profile (`SUSPICIOUS`, prob 0.62, compression anomaly codes).
- Default / other filenames $\to$ Edited mock profile (`EDITED` / `FAKE`, prob 0.94, amount/layout anomaly codes).
- Top-level warnings array contains: `"MOCK PIPELINE: Analysis results generated by Phase 2 mock service. Actual OCR, CNN, and stacking models will be integrated in Phases 3-6."`

### 3.5 Schema Contracts (`app/schemas/`)
`[VERIFIED IN CURRENT WORKSPACE]`  
Nine Pydantic v2 schema modules establish strict typing across all pipeline interfaces:
- `app/schemas/common.py`: `NormalizedBoundingBox` ($xmin, ymin, xmax, ymax \in [0.0, 1.0]$), `PixelBoundingBox` ($x, y, width, height$), coordinate conversion methods (`to_pixel()`, `to_normalized()`), `ModelVersionInfo`.
- `app/schemas/enums.py`: `VerdictEnum` (`GENUINE`, `SUSPICIOUS`, `EDITED`, `INCONCLUSIVE`, `REAL`, `FAKE`), `ReasonCodeEnum` (17 structured codes), `ErrorCodeEnum` (10 machine-readable error codes).
- `app/schemas/request.py`: `ImageUploadConfig`, `AnalyzeRequest`.
- `app/schemas/response.py`: `AnalyzeResponse`, `BranchScores` (`cnn_score`, `semantic_score`, `stacker_score`), `SuspiciousRegion`, `ManipulationAnalysisEvidence`, `ModelVersionMetadata`.
- `app/schemas/ocr.py`: `OCRExtractedField`, `ExtractedTransactionData` (`transaction_id`, `amount`, `amount_numeric`, `currency`, `date`, `time`, `sender`, `sender_upi`, `receiver`, `receiver_upi`, `status`, `google_txn_id`), `OCRResult`.
- `app/schemas/cnn.py`: `CNNModelInput`, `CNNModelOutput`, `CNNSuspiciousRegion`, `CNNInferenceInterface` (abstract base class).
- `app/schemas/semantic.py`: `SemanticFeatureVector` (15 structured features), `FieldAnomalyIndicators`, `OCRConfidenceInfo`, `SemanticBranchOutput`.
- `app/schemas/stacker.py`: `StackerInput`, `StackerOutput`, `CalibrationMetadata`, `StackerInferenceInterface` (abstract base class).
- `app/schemas/errors.py`: `APIErrorResponse` (`error_code`, `message`, `detail`, `request_id`, `status_code`, `timestamp`), `_sanitize_string()` path/traceback redactor.

---

## 4. AUDIT OF CNN MODEL ARTIFACTS

### 4.1 Checkpoint Inventory

`[VERIFIED IN CURRENT WORKSPACE]`

| Checkpoint Path | File Size | Extension | Exists & Readable | Architecture | Training Fold & Scope | Best Recorded Metric | Status |
|---|---:|---|---|---|---|---|---|
| `Hackathon-VIT/CBL04-Dataset/models/phase7/fold_1_resnet18_multitask.pt` | 57,727,691 bytes (~55.05 MB) | `.pt` | Yes (Readable) | `GPayTamperModel` (ResNet-18 + U-Net Mask Decoder) | Fold 1: Test=`compact`, Val=`bank_progress`, Train=`enriched`+`recharge` | Best Val F1 = 0.6667 (Epoch 2 of 3) | **Preserved Baseline Checkpoint** |
| `Hackathon-VIT/CBL04-Dataset/models/phase7/fold_2_resnet18_multitask.pt` | 57,727,691 bytes (~55.05 MB) | `.pt` | Yes (Readable) | `GPayTamperModel` (ResNet-18 + U-Net Mask Decoder) | Fold 2: Test=`enriched`, Val=`recharge`, Train=`compact`+`bank_progress` | Best Val F1 = 0.4000 (Epoch 3 of 3) | **Preserved Experimental Checkpoint** |

### 4.2 Model Architecture and Configuration
`[VERIFIED IN CURRENT WORKSPACE]`  
Defined in `Hackathon-VIT/CBL04-Dataset/scripts/phase7/model.py` and configured via `templates/phase7/model_config.json`:
- **Encoder:** Torchvision ResNet-18 backbone (`stem_conv`, `stem_bn`, `stem_relu`, `stem_pool`, `layer1`, `layer2`, `layer3`, `layer4`).
- **Pretrained Weights:** `pretrained=False` (Random initialization; no weights download).
- **Classification Head:** `AdaptiveAvgPool2d((1, 1))` $\to$ `Dropout(p=0.2)` $\to$ `Linear(512, 2)`. Returns unnormalized `logits` `[B, 2]` (Index 0: clean, Index 1: edited).
- **Localization Head (Mask Decoder):** Lightweight U-Net skip decoder with 4 `ConvNormAct` blocks upsampling and concatenating skip features from `layer3`, `layer2`, `layer1`, and `stem`. Terminal `mask_head` has $1 \times 1$ convolution returning `mask_logits` `[B, 1, H, W]`.
- **Input Tensor Dimensions:** Float32 tensor `[B, 3, 320, 160]` (Height: 320, Width: 160).
- **Preprocessing Pipeline:**
  1. Aspect-preserving resize using `PIL.Image.Resampling.LANCZOS` so image fits within $320 \times 160$.
  2. Paste at top-left `(0, 0)` onto an RGB `(18, 18, 18)` dark canvas.
  3. Convert to float tensor $[0.0, 1.0]$.
  4. Normalize with ImageNet channel statistics: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`.

### 4.3 Baseline Model Evaluation Reality (Historical Phase 7 & 8 Results)
`[HISTORICAL DOCUMENTATION ONLY]`  
Source: `reports/phase7/model_evaluation_report.md` and `reports/phase8/final_model_evaluation.md`:

| Scope | Test Samples ($n$) | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Fold 1 (`compact`)** | 436 | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0 | 218 | 0 | 218 |
| **Fold 2 (`enriched`)** | 432 | 0.5093 | 0.5588 | 0.0880 | 0.1520 | 201 | 15 | 197 | 19 |
| **Pooled Partial Baseline** | **868** | **0.5046 (~50.5%)** | **0.5043** | **0.5461** | **0.5243** | **201** | **233** | **197** | **237** |
| **Fold 3 (`bank_progress`)**| 0 | — | — | — | — | — | — | — | — |
| **Fold 4 (`recharge`)** | 0 | — | — | — | — | — | — | — | — |

**Critical Baseline Deficiencies:**
1. **Coin-Flip Accuracy:** Overall accuracy is $50.46\%$, indicating no reliable generalization.
2. **Severe Class Collapse in Fold 1:** Fold 1 predicted **100% of held-out compact screenshots as edited**, yielding 218 false positives out of 218 clean controls.
3. **Severe Miss Rate in Fold 2:** Fold 2 detected only 19 of 216 edited enriched screenshots (recall $8.8\%$).
4. **Localization Failure:** Mask IoU is **0.0593 (5.9%)** pooled on the $320 \times 160$ canvas (compact: 0.051, enriched: 0.069). Region highlighting was intentionally disabled in the demo because masks generate overwhelming false-positive noise.
5. **Incomplete Cross-Validation:** Folds 3 and 4 were never trained or evaluated.
6. **No Real-World Testing:** None of the 387 real screenshots were evaluated.

### 4.4 Inference Code Availability and Execution Capability
`[VERIFIED IN CURRENT WORKSPACE]`
- **Inference Code:** Working standalone PyTorch inference code exists in `Hackathon-VIT/CBL04-Dataset/app/server.py` (`load_model()`, `preprocess()`, `analyze()`).
- **Execution Capability in Host Environment:** **Cannot be loaded in system Python 3.14** (`No module named 'torch'`). Can only be executed inside a dedicated environment where PyTorch is installed.

### 4.5 Safe Checkpoint Isolation Rule
`[VERIFIED IN CURRENT WORKSPACE]`
- `models/phase7/fold_1_resnet18_multitask.pt` and `models/phase7/fold_2_resnet18_multitask.pt` must remain strictly untouched and preserved as immutable baselines.
- `scripts/phase7/train_evaluate.py` line 180 explicitly checks: `if output_dir.exists() and any(output_dir.iterdir()): raise RuntimeError("Output directory already exists and is not empty")`.
- Future training outputs must be isolated in a new directory (e.g., `models/phase9_improved/`).

---

## 5. AUDIT OF DATASET AND TRAINING PIPELINE

### 5.1 Dataset Directory Inventory
`[VERIFIED IN CURRENT WORKSPACE]`

```
Hackathon-VIT/CBL04-Dataset/data/
├── real_gpay/raw/Dataset-Hackathon/  # 394 raw JPEG screenshots
├── synthetic_gpay/                   # Pilot 1: 20 clean, 20 edited, 40 masks (540 × 1600)
├── synthetic_gpay_phase3/            # Phase 3 Visual Pilot: 12 clean, 12 edited, 24 masks
└── synthetic_gpay_phase4/            # Phase 4 Bulk Dataset: 500 clean, 500 edited, 1000 masks
```

### 5.2 Exact Image Counts and Classification

| Partition | Format | Total Physical Files | Unique Content Count | Clean Images | Edited Images | Clean Masks | Edited Masks | Matched Pairs | Ground Truth Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Real Archive** | JPEG | 394 | **387** | — | — | None | None | — | **Unlabeled / Unknown** |
| **Pilot 1** | PNG | 40 | 40 | 20 | 20 | 20 | 20 | 20 | Verified Synthetic |
| **Phase 3 Pilot** | PNG | 24 | 24 | 12 | 12 | 12 | 12 | 12 | Verified Synthetic |
| **Phase 4 Bulk** | PNG | 1,000 | 1,000 | 500 | 500 | 500 | 500 | 500 | Verified Synthetic |
| **Total Physical** | | **1,458** | **1,451** | **532** | **532** | **532** | **532** | **532** | |

`[VERIFIED IN CURRENT WORKSPACE]`
- Real Archive: 394 physical JPEG files. SHA-256 analysis in `reports/dataset_inventory.md` confirmed **7 exact duplicate pairs** (14 files involved, 7 redundant copies), leaving **387 unique image contents**. All 387 are classified as dark-theme Google Pay screenshots in `reports/real_image_classification.csv`.
- Real screenshots contain sensitive payment information (names, UPI handles, account digits) and are **unlabeled for tampering**. They were excluded from all training and validation splits.

### 5.3 Layout Families and Edit Distributions (Phase 4 Bulk Dataset)
`[VERIFIED IN CURRENT WORKSPACE]`  
Phase 4 bulk dataset contains 500 pairs (1,000 screenshots) distributed across 4 layout families:
1. `compact`: 218 pairs (436 screenshots) — Minimalist dark screen with large centered amount.
2. `enriched`: 216 pairs (432 screenshots) — Detailed transaction view with bank cards and promotions.
3. `bank_progress`: 54 pairs (108 screenshots) — Multi-stage bank processing view.
4. `recharge`: 12 pairs (24 screenshots) — Mobile/utility recharge receipt format.

#### Edit Categories (100 pairs each across 5 categories):
- `amount`: 100 pairs (numeric value substitution).
- `recipient_merchant`: 100 pairs (payee name or handle modification).
- `date_time`: 100 pairs (temporal string changes).
- `transaction_id_reference`: 100 pairs (UPI 12-digit RRN or Google Txn ID altered).
- `other_account_suffix`: 100 pairs (4-digit bank account suffix modified, restricted to `compact`, `enriched`, and `bank_progress`).

### 5.4 Annotations and Bounding Boxes
`[VERIFIED IN CURRENT WORKSPACE]`
- `annotations/phase4/field_annotations/`: 500 JSON files (`P4-0001.json` .. `P4-0500.json`) recording source ID, layout family, dimensions, file paths, edit category, original value, replacement value, seed, and bounding boxes.
- `annotations/phase4/bounding_boxes.csv`: 707 bounding box rows ($source\_id, box\_index, xmin, ymin, xmax, ymax, field\_name$).
- Clean masks: 100% black (all zeros).
- Edited masks: Binary PNG masks marking pixels differing between clean and edited RGB renders.

### 5.5 Train / Validation / Test Splits & Leakage Prevention Logic
`[VERIFIED IN CURRENT WORKSPACE]`
- Defined in `metadata/phase6/` and `metadata/phase7/`.
- **Leakage Prevention Rule:** Clean/edited pairs are strictly indivisible. Entire layout families and 64-bit perceptual hash (pHash) connected components (Hamming distance $\le 6$) are kept in the same split.
- **Phase 6 Realized Split:** Train = 434 pairs (86.8%, `compact` + `enriched`), Val = 54 pairs (10.8%, `bank_progress`), Test = 12 pairs (2.4%, `recharge`).
- **Phase 7 Four-Fold Family Rotation (`metadata/phase7/folds/`):**
  - Fold 1: Test = `compact` (218 pairs), Val = `bank_progress` (54 pairs), Train = `enriched` + `recharge` (228 pairs).
  - Fold 2: Test = `enriched` (216 pairs), Val = `recharge` (12 pairs), Train = `compact` + `bank_progress` (272 pairs).
  - Fold 3 (Unstarted): Test = `bank_progress` (54 pairs), Val = `compact` (218 pairs), Train = `enriched` + `recharge` (228 pairs).
  - Fold 4 (Unstarted): Test = `recharge` (12 pairs), Val = `enriched` (216 pairs), Train = `compact` + `bank_progress` (272 pairs).

### 5.6 Dataset Generation and Validation Scripts
`[VERIFIED IN CURRENT WORKSPACE]`
- `scripts/phase4/generate_bulk.py`: Deterministic Pillow-based batch generator.
- `scripts/validate_dataset.py`, `scripts/phase3/validate_phase3.py`, `scripts/phase6/validate_splits.py`: Validation scripts checking image decodability, pixel difference alignment, mask binarity, and group isolation.

---

## 6. AUDIT OF OCR, SEMANTIC AND STACKER IMPLEMENTATIONS

| Component | Classification | File Path & Function / Class | Evidentiary Basis |
|---|---|---|---|
| **OCR Engine (EasyOCR / Tesseract)** | **PARTIALLY IMPLEMENTED (MOCK ONLY)** | `app/services/mock_analysis_service.py` (`MockAnalysisService.generate_mock_ocr_result`) | Returns canned transaction fields based on filename keyword matching. No real OCR library is imported or executed. |
| **OCR Field Extraction** | **PARTIALLY IMPLEMENTED (MOCK ONLY)** | `app/schemas/ocr.py` (`ExtractedTransactionData`, `OCRExtractedField`) | Pydantic v2 schemas are fully specified and tested, but extraction logic returns static mock values. |
| **Transaction ID Extraction** | **PARTIALLY IMPLEMENTED (MOCK ONLY)** | `app/services/mock_analysis_service.py` (lines 68, 84, 102) | Populates static strings (`"CIC492049203"`, `"UNKNOWN"`, `"UPI/234809283402"`). No pattern regex or bounding box association. |
| **Amount / Date / Sender / Receiver Extraction** | **PARTIALLY IMPLEMENTED (MOCK ONLY)** | `app/services/mock_analysis_service.py` | Populates mock amounts (`"₹350.00"`, `"₹5,000.00"`), dates, and party handles. Real extraction not implemented. |
| **Semantic Feature Extraction** | **PARTIALLY IMPLEMENTED (MOCK ONLY)** | `app/services/mock_analysis_service.py` (`generate_mock_semantic_result`), `app/schemas/semantic.py` (`SemanticFeatureVector`) | `SemanticFeatureVector` schema defines 15 numeric/boolean features. In the service, features are hardcoded for 3 profiles. |
| **Semantic Validation Rules** | **NOT IMPLEMENTED** | `app/schemas/enums.py` (`ReasonCodeEnum`) | 17 structured reason codes are defined in enums, but no active rule engine evaluates banking regex, typography aspect ratios, or chronology. |
| **OOF Prediction Generation** | **PARTIALLY IMPLEMENTED (CNN ONLY)** | `Hackathon-VIT/CBL04-Dataset/reports/phase7/available_folds_1_2_predictions.csv` | CNN out-of-fold predictions exist for Folds 1 and 2 only. Zero OOF predictions exist for OCR, semantic features, or Folds 3 and 4. |
| **Stacker Training** | **NOT IMPLEMENTED** | None found | No training script exists for a LightGBM, Logistic Regression, or MLP ensemble stacker. |
| **Probability Calibration** | **PARTIALLY IMPLEMENTED (MOCK ONLY)** | `app/services/mock_analysis_service.py` (`generate_mock_stacker_result`) | Returns mock calibrated confidence (`calibrated_confidence = 0.95`, `calibration_method = "mock_isotonic"`). No calibration model exists. |
| **Verdict Generation** | **VERIFIED IN CURRENT WORKSPACE (MOCK LOGIC)** | `app/schemas/enums.py` (`VerdictEnum.to_canonical`), `app/services/mock_analysis_service.py` | Canonical mapping between `GENUINE`/`EDITED` and `REAL`/`FAKE` is implemented and verified by 27 contract tests. |
| **Explainability / Evidence Generation** | **PARTIALLY IMPLEMENTED (MOCK ONLY)** | `app/services/mock_analysis_service.py` (`generate_mock_cnn_result`), `app/schemas/response.py` | Formats `SuspiciousRegion` and mock heatmap URLs. Live Grad-CAM/ELA generation is not implemented in the PyTorch code. |

---

## 7. AUDIT OF FRONTEND AND API INTEGRATION

### 7.1 Frontend Status
`[VERIFIED IN CURRENT WORKSPACE]`
1. **Official Member 4 React/Vite Frontend:**
   - **Status:** **[NOT PRESENT]**.
   - Components described in contracts (`cbl04-frontend`, `ResultsDashboard.jsx`, `OCRDataCard.jsx`, `HeatmapView.jsx`, `apiContract.js`) were searched for using file system queries and confirmed absent from the workspace.
2. **Standalone Phase 8 Fallback Browser Demo:**
   - **Status:** **[VERIFIED IN CURRENT WORKSPACE]**.
   - Location: `Hackathon-VIT/CBL04-Dataset/app/` (`index.html`, `styles.css`, `app.js`, `server.py`).
   - Served by: Python standard-library `ThreadingHTTPServer` on port 8080.
   - Technology: Vanilla HTML5, Vanilla JavaScript (ES6+), Vanilla CSS.
   - Verified via: `scripts/phase8/smoke_test_app.py` execution report (`reports/phase8/application_smoke_test_results.json`).

### 7.2 Contract Mismatch Matrix (FastAPI Backend vs Phase 8 Demo Frontend)
`[VERIFIED IN CURRENT WORKSPACE]`  
A detailed comparison between Member 3's FastAPI backend (`app/main.py`) and the Phase 8 fallback demo (`Hackathon-VIT/CBL04-Dataset/app/app.js` and `server.py`) reveals severe divergence:

| Contract Dimension | Member 3 FastAPI Backend (`app/main.py`) | Phase 8 Demo Server (`server.py`) & Client (`app.js`) | Discrepancy Analysis |
|---|---|---|---|
| **Endpoint URL** | `POST /api/v1/analyze` (Alias: `POST /analyze`) | `POST /api/analyze` | Client targets `/api/analyze`; FastAPI expects `/api/v1/analyze` or `/analyze`. |
| **Default Port** | `8000` (Uvicorn) | `8080` (`server.py`) | Port mismatch prevents direct communication without reconfiguration. |
| **Primary Verdict Field**| `verdict`: `"GENUINE"`, `"EDITED"`, `"SUSPICIOUS"`, `"REAL"`, `"FAKE"` | `model_guess`: `"clean"`, `"edited"` | Key name mismatch: `app.js` reads `result.model_guess`. |
| **Probability Score Field**| `manipulation_probability` (float $[0.0, 1.0]$) | `edited_score_uncalibrated` (float) | Key name mismatch: `app.js` reads `result.edited_score_uncalibrated`. |
| **Decision Threshold Field**| In `BranchScores` / `CalibrationMetadata` | `decision_threshold_validation_selected` | `app.js` expects threshold at root level to render threshold notice. |
| **OCR Metadata Card** | Full `ocr_data` (`ExtractedTransactionData`) | Completely omitted | `app.js` does not render or inspect OCR data. |
| **Suspicious Regions** | `manipulation_analysis.suspicious_regions` list | `localization_available: false` (Hardcoded) | `app.js` does not render canvas bounding box overlays. |
| **Heatmap Visualization**| `manipulation_analysis.heatmap_url` | Completely omitted | `server.py` does not generate or return heatmap URLs/URIs. |
| **Reason Codes** | `reason_codes: List[ReasonCodeEnum]` | Completely omitted | `app.js` does not display structured reason codes. |
| **Disclaimer / Legal Notice**| `disclaimer: str` (Mandatory banking notice) | `warning: str` | Key name mismatch: `app.js` reads `result.warning`. |

---

## 8. AUDIT OF DEPENDENCIES AND EXECUTION ENVIRONMENT

### 8.1 Dependency Specifications
`[VERIFIED IN CURRENT WORKSPACE]`
1. **Member 3 FastAPI Backend (`requirements.txt`):**
   ```text
   fastapi>=0.115.0
   uvicorn[standard]>=0.30.0
   python-multipart>=0.0.12
   pillow>=10.4.0
   pydantic>=2.9.0
   pydantic-settings>=2.5.0
   httpx>=0.27.0
   pytest>=8.3.0
   ```
2. **Dataset Generation (`Hackathon-VIT/CBL04-Dataset/requirements.txt`):**
   ```text
   Pillow>=10.0,<14
   ```
3. **PyTorch Model Pipeline (`Hackathon-VIT/CBL04-Dataset/requirements-phase7.txt`):**
   ```text
   torch==2.15.0+cpu
   torchvision==0.29.2+cpu
   Pillow==13.0.0
   numpy==2.5.4
   ```

### 8.2 Execution Environment Analysis & Incompatibilities
`[VERIFIED IN CURRENT WORKSPACE]`
- **Host OS & Architecture:** macOS (Darwin 27.2.0, arm64).
- **Host Python:** Python 3.14.3.
  - Installed packages in host Python: `contourpy 1.3.3`, `cycler 0.12.1`, `fonttools 4.63.0`, `kiwisolver 1.5.0`, `matplotlib 3.10.9`, `numpy 2.4.6`, `packaging 26.2`, `pandas 3.0.3`, `pillow 12.2.0`, `pip 26.1.1`, `pyparsing 3.3.2`, `python-dateutil 2.9.0.post0`, `six 1.17.0`.
  - Missing in host Python: `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `httpx`, `pytest`, `torch`, `torchvision`.
- **Existing Virtual Environment (`.venv_phase7`):**
  - Path: `Hackathon-VIT/CBL04-Dataset/.venv_phase7/`
  - Origin: Windows 64-bit virtual environment created under `C:\Users\dell\Desktop\Hackathon-VIT\CBL04-Dataset\.venv_phase7` using Python 3.12.10.
  - Contains: Windows `.exe` binaries (`python.exe`, `pip.exe`, `Scripts/Activate.ps1`, Windows CPU PyTorch wheels).
  - Incompatibility: **Cannot be activated or executed natively on macOS arm64**. Attempting to execute its Python binary from macOS fails.
- **Docker / Containerization:** No Dockerfile or `docker-compose.yml` exists in the workspace.
- **Secrets Audit:** No `.env` file, hardcoded API keys, JWT secrets, database credentials, or private authentication tokens were found in the codebase. All credentials remain secure.

---

## 9. TESTING EVIDENCE AND LIMITATIONS

### 9.1 Static Test Suite Inspection
`[VERIFIED IN CURRENT WORKSPACE]`
- `tests/test_contracts.py`: **27 unit tests** covering:
  - Bounding box normalization invariants ($0.0 \le xmin \le xmax \le 1.0$).
  - Coordinate transformations (normalized to pixel and pixel to normalized).
  - Verdict enum mapping and compatibility aliases (`REAL` $\to$ `GENUINE`, `FAKE` $\to$ `EDITED`).
  - OCR null tolerance and optional field serialization.
  - CNN and Stacker interface input/output validations.
  - API error response sanitization and path redaction.
- `tests/test_api.py`: **17 integration tests** covering:
  - Health check endpoints (`/health` and `/api/v1/health`).
  - Image format acceptance (PNG, JPEG, WebP).
  - Endpoint alias compatibility (`/analyze` vs `/api/v1/analyze`).
  - Mock response structure validation.
  - Frontend verdict style query parameter overrides (`?verdict_style=frontend`).
  - Strict upload validation error triggers (missing file 400, empty file 400, corrupt image 400, invalid extension 415, GIF format 415, dimensions $<100$px 422, dimensions $>4096$px 422, payload $>10$MB 413).
  - Sanitization of error responses (ensuring no `C:\`, `/home/`, or `Traceback` leakage).
- **Total Test Count:** **44 tests**.

### 9.2 Execution Limitations in Current Read-Only Phase
`[VERIFIED IN CURRENT WORKSPACE]`
- Running `pytest -v` from the host macOS terminal failed with exit code 127 (`zsh: command not found: pytest`) because `pytest` is not installed in the host macOS environment and dependencies were not modified during this read-only phase.
- Therefore, **no claim is made that the pytest suite passed in this execution session**. The tests exist, were structurally inspected, and are documented as passing in historical reports (`docs/phase2_setup.md`).
- Historical Phase 8 smoke test evidence (`reports/phase8/application_smoke_test_results.json`) confirms that the standalone demo server (`server.py`) passed all 8 image format and error handling smoke checks when executed on its original Windows environment.

---

## 10. RISKS AND IMPLEMENTATION BLOCKERS

### Priority 0: Critical Blockers (Must Resolve Before Implementation)

1. **Risk P0-1: FastAPI Backend Disconnected from PyTorch Inference**
   - *Description:* `app/api/routes/analyze.py` exclusively calls `MockAnalysisService`. The trained ResNet-18 model (`fold_1_resnet18_multitask.pt`) is completely unreferenced by the FastAPI application.
   - *Impact:* System cannot perform actual model inference through the FastAPI API.
   - *Files Affected:* `app/api/routes/analyze.py`, `app/services/mock_analysis_service.py`, `Hackathon-VIT/CBL04-Dataset/scripts/phase7/model.py`.
   - *Remediation:* Create a concrete `PyTorchCNNService` implementing `CNNInferenceInterface` (`app/schemas/cnn.py`), load the checkpoint, preprocess images to $320 \times 160$, and integrate into `analyze.py`.

2. **Risk P0-2: Execution Environment Incompatibility (Windows Virtualenv on macOS)**
   - *Description:* `.venv_phase7` contains Windows binaries (`.exe`) and cannot execute on macOS. The host macOS Python 3.14 environment lacks PyTorch, torchvision, FastAPI, and pytest.
   - *Impact:* Model inference and backend testing cannot run on macOS without setting up a native virtual environment.
   - *Files Affected:* `Hackathon-VIT/CBL04-Dataset/.venv_phase7/`, `requirements.txt`, `requirements-phase7.txt`.
   - *Remediation:* In Phase 2, build a clean macOS virtual environment (`.venv_mac`) using Python 3.10–3.12 and install native wheels for PyTorch and FastAPI.

3. **Risk P0-3: Contract and Port Divergence Between Demo Frontend and FastAPI Backend**
   - *Description:* The fallback demo (`app.js`) connects to `http://127.0.0.1:8080/api/analyze` and expects flattened keys (`model_guess`, `edited_score_uncalibrated`). The FastAPI backend runs on port 8000 and returns nested `AnalyzeResponse` schemas.
   - *Impact:* The demo UI cannot communicate with the FastAPI backend.
   - *Files Affected:* `Hackathon-VIT/CBL04-Dataset/app/app.js`, `app/api/routes/analyze.py`.
   - *Remediation:* Update `app.js` to target port 8000 (`/api/v1/analyze`) and align response parsing to read `result.verdict` and `result.manipulation_probability`.

4. **Risk P0-4: Missing Member 4 React/Vite Frontend Repository**
   - *Description:* The official React frontend described in contracts is absent from the workspace.
   - *Impact:* A modern interactive dashboard (with canvas bounding boxes and OCR data cards) cannot be demonstrated unless the fallback demo is expanded or the React repository is imported.
   - *Files Affected:* Entire frontend integration.
   - *Remediation:* Adapt the Phase 8 vanilla demo to render full `AnalyzeResponse` data (OCR card, branch scores, reason codes) as a fallback, or import the React codebase from Member 4.

### Priority 1: Required for Working Demonstration & Reliable Evaluation

5. **Risk P1-1: Near-Chance CNN Baseline Accuracy & Severe False Positive Bias**
   - *Description:* Checkpoint Fold 1 achieved only 50.0% accuracy on held-out compact screenshots by predicting 100% of samples as edited (218 false positives out of 218 clean controls). Demo smoke tests confirmed all 8 synthetic pilot samples were flagged as edited.
   - *Impact:* Baseline model is unreliable and behaves as a random/biased classifier.
   - *Remediation:* Preserve `fold_1_resnet18_multitask.pt` as the immutable baseline. Train an improved checkpoint (Phase 4) using pretrained weights (`pretrained=True`), conservative data augmentation, and threshold calibration, isolating outputs in `models/phase9_improved/`.

6. **Risk P1-2: Failed Mask Localization Quality (IoU = 0.059)**
   - *Description:* Test mask IoU is only $5.9\%$. The mask decoder produces overwhelming false-positive pixel noise.
   - *Impact:* Displaying spatial bounding boxes on user screenshots creates misleading, inaccurate highlights.
   - *Remediation:* Keep spatial bounding box highlighting disabled (`localization_available = False`) until an improved model demonstrates $IoU > 0.50$.

7. **Risk P1-3: Mock OCR, Semantic, and Stacker Services**
   - *Description:* Transaction data extraction, banking heuristic rules, and stacking ensemble are currently static mocks.
   - *Impact:* Dual-branch multimodal capability does not yet exist.
   - *Remediation:* Progressively implement live EasyOCR (Phase 5), semantic regex/typography heuristics (Phase 6), and a calibrated meta-stacker (Phase 8).

8. **Risk P1-4: Incomplete 4-Fold Cross-Validation Suite**
   - *Description:* Cross-validation folds for `bank_progress` (Fold 3) and `recharge` (Fold 4) were never executed.
   - *Impact:* Complete cross-family out-of-fold metrics across all 500 pairs do not exist.
   - *Remediation:* Execute Folds 3 and 4 training in a separate directory to establish the complete 4-fold baseline.

### Priority 2: Important Implementation Gaps

9. **Risk P2-1: Unlabeled Real Screenshot Archive**
   - *Description:* All 387 unique real screenshots remain unlabeled for tampering, preventing real-world validation.
   - *Remediation:* Retain real images in `restricted_real_holdout` until a privacy-reviewed, verified ground-truth subset is prepared.

10. **Risk P2-2: Synthetic Template Overfitting**
    - *Description:* All synthetic training data originates from only 4 renderer templates, leading to domain shift when testing out-of-family.
    - *Remediation:* Add slight geometric jitter and background noise variations during model training.

### Priority 3: Refinements & Future Scope

11. **Risk P3-1: Lack of Probability Calibration on Out-of-Distribution Data**
    - *Description:* Raw softmax scores are uncalibrated and cannot be interpreted as posterior probabilities.
    - *Remediation:* Implement Platt scaling or Isotonic Regression on validation fold predictions.

12. **Risk P3-2: Scope Boundaries and Settlement Limitation**
    - *Description:* The system must not be represented as a bank ledger settlement verifier or multi-app detector.
    - *Remediation:* Maintain prominent UI disclaimers stating that the tool performs digital image manipulation checks only and does not verify bank-side ledger entries.

---

## 11. RECOMMENDED 11-PHASE IMPLEMENTATION SEQUENCE

Based on verified code dependencies and discovered blockers, the intended 11-phase roadmap is structurally sound but requires specific execution adjustments:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       RECOMMENDED 11-PHASE IMPLEMENTATION ROADMAP               │
├───────┬───────────────────────────────────────────┬─────────────────────────────┤
│ Phase │ Phase Objective                           │ Critical Alignment Notes    │
├───────┼───────────────────────────────────────────┼─────────────────────────────┤
│ 1     │ Workspace Verification & Freeze (Current) │ Non-destructive audit       │
│ 2     │ Environment Setup & Baseline Verification │ Build macOS native venv     │
│ 3     │ CNN Inference Integration                 │ Bridge FastAPI to PyTorch   │
│ 4     │ CNN Evaluation & Improvement              │ Train isolated new weights  │
│ 5     │ GPay OCR Implementation                   │ Implement live EasyOCR      │
│ 6     │ Semantic Feature Engine                   │ Rule engine & heuristics    │
│ 7     │ Branch Validation & OOF Preparation       │ Generate real branch scores │
│ 8     │ OOF Stacker Implementation                │ Calibrated meta-classifier  │
│ 9     │ Final Verdict, Calibration & Evidence     │ Synthesize AnalyzeResponse  │
│ 10    │ Frontend / Backend Integration            │ Align contracts & demo UI   │
│ 11    │ End-to-End Testing & Final Handover       │ Full pipeline verification  │
└───────┴───────────────────────────────────────────┴─────────────────────────────┘
```

### Detailed Roadmap Adjustments:
- **Phase 2 (Environment Setup):** Must create a native macOS virtual environment (`.venv_mac`) using Python 3.10–3.12, install both `requirements.txt` and native macOS PyTorch wheels, and verify that the 44 existing tests run and pass.
- **Phase 3 (CNN Inference Integration):** Must occur *before* model improvements so the end-to-end pipeline wiring is established and verified using the preserved baseline checkpoint (`fold_1_resnet18_multitask.pt`).
- **Phase 4 (CNN Improvement):** Must strictly isolate all new training runs in a new directory (`models/phase9_improved/`). Checkpoint `fold_1_resnet18_multitask.pt` must never be overwritten.
- **Phase 5 & 6 (OCR & Semantic):** Can proceed sequentially to replace `MockAnalysisService` stubs with real text extraction and heuristic rule verification.
- **Phase 7 & 8 (OOF Data & Stacker):** Train a calibrated meta-classifier (e.g., Logistic Regression or LightGBM) fusing CNN probabilities and semantic features.
- **Phase 10 (Frontend Integration):** Because Member 4's React repository is absent, Phase 10 must upgrade the Phase 8 vanilla demo (`Hackathon-VIT/CBL04-Dataset/app/`) to fully render the `AnalyzeResponse` schema (verdict, calibrated probability, OCR card, branch scores, reason codes).

---

## 12. EXPLICIT LIST OF PRESERVED FILES AND ARTIFACTS

To ensure zero regression and maintain complete reproducibility, the following files and directories are designated as **IMMUTABLE BASELINES**:

### 12.1 Immutable Model Checkpoints
- `Hackathon-VIT/CBL04-Dataset/models/phase7/fold_1_resnet18_multitask.pt` (Baseline Checkpoint, 57.7 MB)
- `Hackathon-VIT/CBL04-Dataset/models/phase7/fold_2_resnet18_multitask.pt` (Fold 2 Checkpoint, 57.7 MB)

### 12.2 Immutable Dataset Files
- `Hackathon-VIT/CBL04-Dataset/data/real_gpay/raw/Dataset-Hackathon/*` (All 394 raw JPEG screenshots)
- `Hackathon-VIT/CBL04-Dataset/data/synthetic_gpay/*` (Pilot 1 images and masks)
- `Hackathon-VIT/CBL04-Dataset/data/synthetic_gpay_phase3/*` (Phase 3 Pilot images and masks)
- `Hackathon-VIT/CBL04-Dataset/data/synthetic_gpay_phase4/*` (Phase 4 Bulk images and masks)

### 12.3 Immutable Manifests and Annotations
- `Hackathon-VIT/CBL04-Dataset/annotations/phase4/field_annotations/*` (500 JSON annotation files)
- `Hackathon-VIT/CBL04-Dataset/annotations/phase4/bounding_boxes.csv` (707 bounding box records)
- `Hackathon-VIT/CBL04-Dataset/metadata/phase4_manifest.csv` (1,000 screenshot records)
- `Hackathon-VIT/CBL04-Dataset/metadata/phase6/*` (Phase 6 split manifests)
- `Hackathon-VIT/CBL04-Dataset/metadata/phase7/folds/*` (Phase 7 4-fold cross-validation manifests)
- `Hackathon-VIT/CBL04-Dataset/metadata/phase7/training_config.json`

### 12.4 Immutable Historical Reports
- `Hackathon-VIT/CBL04-Dataset/reports/phase7/*` (All Phase 7 training and evaluation records)
- `Hackathon-VIT/CBL04-Dataset/reports/phase8/*` (All Phase 8 threshold and testing records)
- `docs/api_contract.md` and `docs/phase2_setup.md`

---

## 13. EXACT LIST OF UNVERIFIED INFORMATION

`[NOT VERIFIED]`
1. **Member 4 React Frontend Codebase:** The entire `cbl04-frontend` React/Vite source code is absent from the workspace. Its actual component implementation, styling, and dependency tree could not be verified.
2. **Real-World Screenshot Tampering Labels:** Ground-truth tampering labels for the 387 unique real screenshots do not exist. Whether any real screenshot has been manipulated cannot be verified.
3. **CNN Cross-Validation Folds 3 and 4 Performance:** Folds 3 (`bank_progress`) and 4 (`recharge`) were never trained. Their cross-validation metrics are unknown.
4. **Live Execution of the 44 PyTorch & FastAPI Tests on macOS Host:** Due to the read-only constraint of Phase 1 and missing dependencies in host Python 3.14, test suites were statically audited but not executed in this session.
5. **Real EasyOCR Accuracy on Dark-Theme Screenshots:** Because EasyOCR is not yet integrated, its character recognition accuracy on Google Pay dark-theme fonts cannot be verified.

---

*Report certified by: Member 3 (Backend & AI Integration Lead)*  
*Audit Completed — Project Frozen — Ready for Human Review Prior to Phase 2*

