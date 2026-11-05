# CBL-04 — FINAL PROJECT HANDOVER & LAUNCH GUIDE

**Project:** Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Workspace:** `/Users/heemanshupjain/Downloads/Member 3`  
**Backend Port:** `8000` (FastAPI)  
**Frontend Port:** `5173` (React / Vite)  
**Handover Date:** 2026-11-03  

---

## 1. PREREQUISITES & ENVIRONMENT SETUP

### 1.1 System Requirements
- **Operating System:** macOS (Apple Silicon or Intel) or Linux
- **Python Version:** Python 3.12.x (configured in `.venv_mac`)
- **Node.js Version:** Node.js v18+ or v20+ with `npm`
- **Hardware Acceleration:** MPS (Apple Silicon Metal) or CPU (CPU is default for deterministic evaluation)

### 1.2 Directory Structure
```text
/Users/heemanshupjain/Downloads/Member 3
├── .venv_mac/               # Pre-configured Python virtual environment
├── app/                     # FastAPI backend application
│   ├── api/routes/          # Route handlers (/health, /analyze)
│   ├── core/                # Configuration and Pydantic settings
│   ├── schemas/             # Request, response, and component contracts
│   ├── services/            # CNN, OCR, Semantic, Stacker, and Final Verdict services
│   └── main.py              # Application entry point
├── cbl04-frontend/          # React + Vite frontend application
│   ├── src/                 # UI components, pages, services, styling
│   ├── package.json         # Frontend dependencies and scripts
│   └── vite.config.js       # Vite bundler configuration
├── models/                  # Offline model weights (EasyOCR, Phase 8 Stacker)
│   ├── easyocr/             # CRAFT (83MB) and CRNN English (15MB)
│   └── phase8/              # Stacker model JSON
└── Hackathon-VIT/           # Dataset, CNN models, and evaluation scripts
```

---

## 2. LAUNCHING THE APPLICATION

### 2.1 Start the FastAPI Backend
Open a terminal and run:

```bash
cd "/Users/heemanshupjain/Downloads/Member 3"
./.venv_mac/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- **Health Check URL:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) or [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
- **Interactive Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 2.2 Start the React Frontend
Open a second terminal window and run:

```bash
cd "/Users/heemanshupjain/Downloads/Member 3/cbl04-frontend"
npm run dev -- --host 127.0.0.1 --port 5173
```

- **Application URL:** [http://127.0.0.1:5173](http://127.0.0.1:5173)

---

## 3. HOW TO DEMONSTRATE & USE THE SYSTEM

### 3.1 Uploading a Screenshot
1. Open [http://127.0.0.1:5173](http://127.0.0.1:5173) in your browser.
2. Under **GPay Receipt Verification**, drag-and-drop or browse for a Google Pay dark-theme screenshot (`.png`, `.jpg`, `.jpeg`, or `.webp`, up to 10 MB).
   - Sample images are located in:  
     `Hackathon-VIT/CBL04-Dataset/data/synthetic_gpay_phase4/clean/`  
     `Hackathon-VIT/CBL04-Dataset/data/synthetic_gpay_phase4/edited/`
3. Verify that the **API Pipeline Mode** is set to **Live FastAPI Backend** (the default setting).
4. Click **Analyze with Live Backend**.
5. The live pipeline will process the image through CNN, EasyOCR, Semantic Rules, the OOF Stacker, and the Final Verdict Engine, displaying the comprehensive Results Dashboard in ~1.5–1.7 seconds.

### 3.2 Interpreting the Results Dashboard

| Verdict State | Visual Indicator | Meaning & Criteria |
| :--- | :--- | :--- |
| **`GENUINE`** | Green Banner / Shield | Visual layout, typography, and extracted transaction metadata match authentic Google Pay receipt signatures without detected anomalies. |
| **`EDITED`** | Red Banner / Octagon | The ensemble detected strong visual tampering (CNN score $\ge 0.65$ & stacker $\ge 0.45$), explicit semantic rule failures (e.g. invalid rupee symbol, mismatched timestamps), or meta-learner manipulation score $\ge 0.50$. |
| **`INCONCLUSIVE`** | Amber Banner / Question | Analysis is indeterminate. Triggered when the stacker manipulation score falls in the ambiguity margin ($[0.45, 0.55]$) without corroborating anomalies, when OCR confidence is critically low ($< 0.45$), or when branches irreconcilably conflict. |

### 3.3 Inspecting Additional Evidence Tiles
- **Multi-Branch AI Evidence Breakdown:** Displays the raw uncalibrated scores for the CNN Visual Branch (ResNet-18), Semantic Consistency Rules, and OOF Stacker Meta-Learner.
- **Extracted OCR Metadata Card:** Shows key payment parameters extracted by EasyOCR: Transaction ID, Amount, Timestamp/Date, Sender, and Receiver.
- **Suspicious Region Analysis:** Lists specific regions flagged by semantic layout rules with normalized coordinates.
- **Heatmap Comparison:** In accordance with forensic explainability guidelines, pixel-level Grad-CAM heatmaps are suppressed because baseline segmentation IoU ($0.059$) is below operating quality.

---

## 4. STOPPING THE APPLICATION

To stop both services:
1. Press `Ctrl + C` in each terminal window.
2. Alternatively, terminate the processes by port:
   ```bash
   lsof -ti :8000 | xargs kill -TERM
   lsof -ti :5173 | xargs kill -TERM
   ```

---

## 5. TROUBLESHOOTING & COMMON ISSUES

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `Address already in use: 8000` | Previous FastAPI instance still running | Run `lsof -ti :8000 \| xargs kill -TERM` then retry. |
| `Address already in use: 5173` | Previous Vite server still running | Run `lsof -ti :5173 \| xargs kill -TERM` then retry. |
| "Cannot connect to backend server at http://localhost:8000" | FastAPI service is not running | Ensure the backend process is running in Terminal 1. |
| `ModuleNotFoundError` during backend start | Wrong Python environment | Always use `./.venv_mac/bin/uvicorn`, NOT the system Python. |
| `EasyOCR` download error | Offline models missing | Ensure `models/easyocr/craft_mlt_25k.pth` and `english_g2.pth` exist. |
| High rate of `INCONCLUSIVE` | Score in ambiguity band $[0.45, 0.55]$ | Expected behavior under Phase 9 safety guardrails for subtle or ambiguous synthetic edits. |

---

## 6. SYSTEM CONTRACT REFERENCE

- **API Endpoint:** `POST /api/v1/analyze`
- **Request Format:** `multipart/form-data` with form field `image`
- **Response Format:** `application/json` conforming to `AnalyzeResponse`:
  ```json
  {
    "request_id": "req-1793728131309",
    "verdict": "INCONCLUSIVE",
    "confidence": 1.0,
    "manipulation_probability": 0.4997,
    "ocr_data": {
      "transaction_id": "DEMO-UPI-24036454",
      "amount": "{2,400",
      "date": "17 Apr 2025",
      "sender": "Anaya Joshi",
      "receiver": "Sara Rao",
      "status": "Completed"
    },
    "branch_scores": {
      "cnn_score": 0.592743,
      "semantic_score": 0.0,
      "stacker_score": 0.4997
    },
    "pipeline_status": "phase9_complete",
    "component_status": {
      "cnn": "live",
      "ocr": "live",
      "semantic": "live",
      "stacker": "live",
      "verdict_engine": "live"
    },
    "disclaimer": "Notice: Screenshot analysis performs digital artifact and layout integrity checks only. It does not verify bank-side settlement, banking ledger entries, or prove that a financial payment occurred."
  }
  ```
