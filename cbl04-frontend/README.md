# CBL-04: Fake / Manipulated GPay Receipt Detection System
## Phase 3 (H3–H5): Results Dashboard & Heatmap Visualization

**Team Role:** Member 4 — Frontend Development & Integration  
**Backend Partner:** Member 3 — FastAPI Backend & ML/OCR Pipeline  
**IDE:** Antigravity (Google DeepMind Agentic IDE)  
**Status:** ✅ Phase 1 Complete (Setup & API Contract) • ✅ Phase 2 Complete (Upload Page & Mock API) • ✅ Phase 3 Complete (Results Dashboard & Heatmap)

---

## 1. Project Purpose
Google Pay (GPay) and UPI receipt spoofing is a rapidly rising vector for digital fraud. Fraudsters generate fabricated payment confirmation screenshots using fake receipt generators or image editing software to trick merchants and individuals.

**CBL-04** is an AI-powered detection system designed to analyze payment screenshots, extract receipt metadata via OCR, detect localized pixel/font/metadata tampering, and deliver an authenticity verdict (`REAL`, `FAKE`, `SUSPICIOUS`) with explainable visual evidence (bounding boxes and Grad-CAM/ELA heatmaps).

---

## 2. Frontend Technology Stack
- **Framework:** [React 19](https://react.dev/)
- **Build Tool:** [Vite 8](https://vite.dev/)
- **Language:** JavaScript (ES6+ / JSX) — *pure JS, no TypeScript*
- **Styling:** Modern modular CSS with custom design tokens (Strict Light Theme)
- **HTTP Client:** [Axios](https://axios-http.com/) with request/response interceptors & standardized error formatting
- **Icons:** [Lucide React](https://lucide.dev/)
- **Simulation Layer:** Client-side Mock API service adhering 100% to confirmed contract

---

## 3. Project Directory Structure
```
cbl04-frontend/
├── .env                      # Local environment variables (ignored by Git)
├── .env.example              # Template for environment configuration
├── .gitignore                # Git ignore rules (node_modules, dist, .env)
├── README.md                 # Complete setup, API, UI design & Phase 3 guide
├── package.json              # Project dependencies & scripts
├── vite.config.js            # Vite configuration
├── index.html                # HTML entry point
├── public/                   # Static assets & favicons
└── src/
    ├── assets/               # Local images and branding icons
    ├── components/           # Reusable UI components
    │   ├── Header.jsx        # Navigation header & tab switcher (Phase 1 vs 2/3)
    │   ├── UploadCard.jsx    # Drag-and-drop file upload zone & validation
    │   ├── ImagePreview.jsx  # Screenshot preview with dimensions & actions
    │   ├── LoadingState.jsx  # 2–3s simulated inference spinner & progress
    │   ├── MockResultPreview.jsx # Compact verdict preview (Phase 2 legacy)
    │   ├── ResultsDashboard.jsx # Comprehensive verification dashboard (Phase 3)
    │   ├── ConfidenceScore.jsx  # Clamped semantic confidence meter & thresholds
    │   ├── OCRDataCard.jsx      # Formatted simulated OCR receipt data with fallbacks
    │   ├── SuspiciousRegionsCard.jsx # Bounding box coordinates & localized scores
    │   └── HeatmapView.jsx      # Side-by-side original vs Grad-CAM heatmap view
    ├── pages/                # Page-level views
    │   └── Home.jsx          # Primary GPay verification workspace
    ├── services/             # API services & simulation layer
    │   ├── api.js            # Axios client & analyzeImage() placeholder
    │   ├── apiContract.js    # Living API contract & confirmed schemas
    │   └── mockApi.js        # Client-side 2–3s simulated inference service
    ├── utils/                # Utility helpers & constants
    │   ├── constants.js      # App metadata & phase configurations
    │   └── fileValidation.js # MIME, extension, size & empty file validator
    ├── App.jsx               # Root application coordinator with tab navigation
    ├── App.css               # Light theme component stylesheets
    ├── index.css             # Light theme design tokens, reset & base styles
    └── main.jsx              # React DOM mounting entry point
```

---

## 4. Setup & Running the Frontend

### Prerequisites
- **Node.js**: v18.0.0 or later (Tested on Node v24)
- **npm**: v9.0.0 or later

### Step 1: Install Dependencies
Open your terminal in `cbl04-frontend`:
```bash
npm install
```

### Step 2: Environment Variable Setup
Copy the `.env.example` file to `.env`:
```bash
# On Windows PowerShell:
Copy-Item .env.example .env

# On macOS/Linux:
cp .env.example .env
```

Contents of `.env`:
```env
# Base URL of the FastAPI backend (Member 3)
VITE_API_BASE_URL=http://localhost:8000
```
> **Note:** The backend URL is completely configurable via `VITE_API_BASE_URL`. Do not hardcode production or staging URLs in source code.

### Step 3: Run the Development Server
```bash
npm run dev
```
By default, Vite will start at: `http://localhost:5173/`

### Step 4: Build for Production & Lint Check
To verify production bundle build and code quality:
```bash
npm run build
npm run lint
```

---

## 5. Phase 2: Upload Interface & Mock API Workflow

1. **Header Navigation Switcher:**
   - **Upload & Verify (Phase 2 & 3):** Primary user upload and analysis workspace.
   - **API Contract & Health (Phase 1):** Backend connection tester, endpoint specification, and hackathon roadmap.
2. **Drag-and-Drop & File Selection:**
   - Drag and drop screenshots directly onto the dropzone or click "Browse Files".
   - Generates an instant local object URL for preview without transmitting bytes over the network.
   - Automatically revokes object URLs on replacement or component unmount to prevent browser memory leaks.
3. **Client-Side Validation:**
   - Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`.
   - Supported MIME types: `image/jpeg`, `image/png`, `image/webp`.
   - Max file size limit: `10 MB` (`IMAGE_CONSTRAINTS.MAX_FILE_SIZE_BYTES`).
   - Rejects empty files (0 bytes) and non-image files with inline, accessible error messages.
4. **Mock API Simulation (`analyzeImageMock`):**
   - Simulates realistic 2–3s model inference latency.
   - Updates progressive loading stages (`Preprocessing` -> `Layout Analysis` -> `Synthesis`).
   - Offers an **Auto-Cycle** mode (rotates `FAKE -> REAL -> SUSPICIOUS`) or **Forced Verdicts** (`Force Fake`, `Force Real`, `Force Suspicious`) so evaluators can test all three verdict branches.

---

## 6. Phase 3: Detailed Results Dashboard & Heatmap Visualization

In Phase 3, the minimal preview is transformed into a full-scale, professional verification dashboard driven by the confirmed mock response fixtures:

### A. Verdict Summary Card
- **Verdict Badges:** Displays prominent, semantic badges:
  - `REAL`: Authentic Receipt Verified (Soft green: `#15803d` on `#f0fdf4`)
  - `FAKE`: Manipulated / Fake Receipt Detected (Soft red: `#b91c1c` on `#fef2f2`)
  - `SUSPICIOUS`: Suspicious / Inconclusive Receipt (Soft amber: `#b45309` on `#fffbeb`)
- Displays synthetic trace ID, engine message from response, and simulated-result notice.

### B. Confidence Visualization (`ConfidenceScore.jsx`)
- Reads the decimal `confidence` score (e.g. `0.95`), converts it to `95%`, and clamps it strictly within `0%–100%`.
- Visual horizontal progress track with color-coded gradient fills corresponding to the verdict.
- Threshold context badge (`High Authenticity Confidence`, `High Tamper Confidence`, or `Moderate Uncertainty`).
- Labeled clearly as **"Model Confidence (Simulated)"**.

### C. Side-by-Side Screenshot Comparison & Heatmap (`HeatmapView.jsx`)
- **Desktop Comparison:** Left panel shows the untouched uploaded screenshot (`Input Receipt`), and right panel shows the explainability visualization (`Grad-CAM / ELA Heatmap`). Stacks fluidly on tablet and mobile.
- **Heatmap Placeholder:** When `heatmap_url` is `null` (as in current mock fixtures), renders a professional light-theme empty-state card stating:
  > *"Heatmap visualization will use actual CNN output once Member 2 integration is available."*
  > *Status: Awaiting CNN Model Layer (Phase 4).*
- Explicitly guarantees that no random or fabricated pixels are generated.

### D. Suspicious Region Analysis (`SuspiciousRegionsCard.jsx`)
- Parses `manipulation_analysis.suspicious_regions` dynamically.
- For each flagged region, displays:
  - Label (e.g., `"Amount field"`, `"Transaction ID alignment"`)
  - Bounding box coordinates: `X: 100px, Y: 200px`
  - Dimensions: `140px × 48px`
  - Computed area in `px²`
  - Localized anomaly confidence score (e.g., `94%`)
- **Empty State:** When no regions are flagged (such as for `REAL` receipts), renders a clean integrity verification badge with confirmation text.

### E. Simulated OCR Information Card (`OCRDataCard.jsx`)
- Dynamically maps `ocr_data` fields:
  - **Transaction ID / UPI Ref:** Monospaced display
  - **Transaction Amount:** Highlighted with currency symbol (`₹500`)
  - **Date / Timestamp:** Extracted date string
  - **Sender Name / UPI ID:** Sender handle
  - **Receiver / Merchant:** Payee name
- **Missing Value Handling:** Displays `"Not available"` in muted italics when any field is `null`, empty, or undetected.

### F. Dashboard Interaction & Navigation
- Smooth auto-scrolling into the dashboard upon mock analysis completion.
- **"Change Screenshot"** action allows replacing the current image.
- **"Analyze Another Screenshot"** action resets the workflow for a new test run.
- Preserves the **Mock Simulation Mode** selector bar (`Auto-Cycle`, `Force Fake`, `Force Real`, `Force Suspicious`) so evaluators can re-run and compare different outputs instantly.

---

## 7. Confirmed Backend API Contract (POST /api/v1/analyze)

### Endpoint Details
- **Method:** `POST`
- **Route:** `/api/v1/analyze`
- **Content-Type:** `multipart/form-data`

### Request Parameters
| Field | Type | Required | Description |
|---|---|---|---|
| `image` | `File` (Binary) | **Yes** | The uploaded GPay screenshot (`.jpg`, `.jpeg`, `.png`, `.webp`). Configurable placeholder max size: **10 MB**. |

---

### Response Schema (JSON)
```json
{
  "request_id": "demo-001",
  "verdict": "REAL | FAKE | SUSPICIOUS",
  "confidence": 0.95,
  "message": "Analysis completed",
  "ocr_data": {
    "transaction_id": "example-id",
    "amount": "₹500",
    "date": "2026-11-03",
    "sender": "Demo Sender",
    "receiver": "Demo Receiver"
  },
  "manipulation_analysis": {
    "detected": true,
    "suspicious_regions": [
      {
        "label": "Amount field",
        "x": 100,
        "y": 200,
        "width": 120,
        "height": 50,
        "score": 0.92
      }
    ],
    "heatmap_url": null
  }
}
```

---

### Contract Specifications & Definitions

1. **Verdict Meanings (`verdict`):**
   - `"REAL"`: Image shows genuine GPay fonts, consistent compression, authentic UPI alignment, and no signs of tampering.
   - `"FAKE"`: Localized tampering, copy-move splicing, inconsistent font rendering, or invalid digital signature detected.
   - `"SUSPICIOUS"`: Artifacts or heavy compression detected that prevent high-confidence verification. Manual inspection recommended.

2. **Confidence Score Range (`confidence`):**
   - Floating-point value from `0.00` to `1.00`.
   - Represents the model's overall probability score for the assigned verdict.

3. **OCR Fields (`ocr_data`):**
   - `transaction_id`: Extracted UPI reference or Google transaction ID (`null` if unreadable).
   - `amount`: Transaction amount with currency symbol (`null` if unreadable).
   - `date`: Parsed timestamp (`null` if unreadable).
   - `sender`: Sender's name or UPI handle (`null` if missing).
   - `receiver`: Payee's name or merchant identifier (`null` if missing).

4. **Coordinate Convention for Suspicious Regions (`suspicious_regions`):**
   - `(x, y)`: Pixel coordinate of the top-left corner relative to the original uploaded image's natural dimensions.
   - `width`, `height`: Bounding box dimensions in pixels.
   - `score`: Confidence score (`0.00` to `1.00`) of localized anomaly in this region.
   - `label`: Human-readable identifier (e.g., `"Amount field"`, `"Transaction ID alignment"`).

5. **Heatmap Delivery Method (`heatmap_url`):**
   - **Primary**: URL pointing to statically served Grad-CAM/ELA image (e.g., `http://localhost:8000/static/heatmaps/<id>.png`).
   - **Alternative**: Base64 data URI string (`data:image/png;base64,...`).
   - If no heatmap is generated (e.g. authentic image or awaiting CNN), `heatmap_url` is `null`.

6. **HTTP Status Codes:**
   - `200 OK`: Analysis successfully performed.
   - `400 Bad Request`: Missing `image` form parameter or corrupt file.
   - `413 Payload Too Large`: Image file exceeds maximum allowed size.
   - `415 Unsupported Media Type`: Non-image file uploaded.
   - `422 Unprocessable Entity`: Validation failure.
   - `500 Internal Server Error`: Server/ML pipeline exception.

---

## 8. UI Design Guidelines

To maintain visual excellence, professional polish, and accessibility throughout the hackathon, all frontend components across all phases must adhere to these design rules:

1. **Light Theme Only:**
   - Dark mode or black backgrounds are strictly forbidden across the entire application.
   - The main application canvas uses a very light gray/off-white background (`--color-background: #f8fafc`).

2. **White Cards on Light Backgrounds:**
   - All content containers, inspection cards, and data sections must use pure white backgrounds (`--color-surface: #ffffff`) with subtle light borders (`--color-border: #e2e8f0`).
   - Use soft elevation (`--shadow-sm`, `--shadow-md`) to separate layers gently without heavy black drop shadows.

3. **Consistent Blue Primary Actions:**
   - Main call-to-actions, upload triggers, and interactive controls must use professional blue (`--color-primary: #2563eb`) with a darker hover state (`--color-primary-hover: #1d4ed8`).
   - Soft blue tints (`--color-primary-light: #eff6ff`) should be used for selected tabs, active pills, and icon backgrounds.

4. **Dark Readable Typography:**
   - Primary headings and data values use dark navy/charcoal (`--color-text-primary: #0f172a`) to guarantee high contrast.
   - Body labels and descriptions use muted slate (`--color-text-secondary: #475569`).

5. **Subtle Semantic Indicators:**
   - Success (Authentic receipt): Soft green (`#15803d` text on `#f0fdf4` background with `#bbf7d0` border).
   - Warning (Suspicious / Inconclusive): Soft amber (`#b45309` text on `#fffbeb` background with `#fde68a` border).
   - Danger (Manipulated / Fake receipt): Soft red (`#b91c1c` text on `#fef2f2` background with `#fecaca` border).
   - Avoid neon, overly saturated, or jarring highlight colors.

6. **Consistent Spacing & Radii:**
   - Use standardized border radii: small buttons and tags (`--radius-sm: 6px`), cards (`--radius-lg: 14px`), and pills (`--radius-full: 9999px`).
   - Maintain uniform grid gaps (`1.25rem` to `1.75rem`) and container padding.

7. **Accessible Contrast & Clean Layouts:**
   - All text and iconography must satisfy WCAG AA contrast standards against their container background.
   - Code and JSON blocks must use clean light monospaced surfaces (`--color-surface-muted: #f8fafc`) with dark readable text.

8. **Universal Responsive Design:**
   - All layouts must fluidly adapt from high-resolution desktop monitors down to mobile viewports (`max-width: 992px`, `max-width: 768px`, and `max-width: 540px`).

---

## 9. Hackathon Roadmap
- **Phase 1 (H0–H1):** ✅ Project Setup, Light Theme & API Contract.
- **Phase 2 (H1–H3):** ✅ Upload Screen, drag-and-drop zone, client-side validation & Mock API service.
- **Phase 3 (H3–H5):** ✅ Results Dashboard, confidence score meter, OCR details, suspicious regions & side-by-side heatmap comparison.
- **Phase 4 (H9–H12):** Live FastAPI backend integration, real CNN Grad-CAM overlay & final presentation polish.
