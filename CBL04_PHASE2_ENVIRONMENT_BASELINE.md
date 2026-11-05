# CBL-04 — PHASE 2: ENVIRONMENT SETUP & BASELINE VERIFICATION REPORT

**Document Identifier:** `CBL04_PHASE2_ENVIRONMENT_BASELINE.md`  
**Execution Timestamp:** 2026-11-03T18:43:00+05:30  
**Target Application:** Google Pay (Dark Theme) UPI Screenshot Tampering Detection  
**Auditor / Integrator:** Member 3 (Backend & AI Integration Lead)  
**Workspace Root:** `/Users/heemanshupjain/Downloads/Member 3`  
**Host Operating System:** macOS (Darwin Kernel Version 27.2.0, arm64 / Apple Silicon)  
**Phase Status:** **COMPLETED & VERIFIED SUCCESS**

---

## 1. EVIDENCE-BASED STATUS LABELS
Every technical finding and execution claim in this baseline report is classified using standardized status labels:
- `VERIFIED SUCCESS`: Successfully executed and verified with direct runtime evidence in the current session.
- `VERIFIED FAILURE`: Executed in the current session and definitively failed with captured error output.
- `BLOCKED`: Precondition not met (all previous blockers resolved).
- `HISTORICAL DOCUMENTATION ONLY`: Preserved historical record from prior Windows environment runs.

---

## 2. TASK 1 — PYTHON RUNTIME INSPECTION & VERIFICATION

### 2.1 Host Python Executables
With user approval, Python 3.12 was installed via Homebrew at `/opt/homebrew/bin/python3.12`:

| Executable Path | Exact Version | Platform Architecture | Source / Manager | Compatibility Status |
|---|---|---|---|---|
| `/opt/homebrew/bin/python3.12` | Python 3.12.15 | `arm64` (Apple Silicon) | Homebrew | **VERIFIED SUCCESS (Active Project Runtime)** |
| `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3` | Python 3.14.3 | `arm64` | Python.org Installer | Incompatible with PyTorch (Unused) |
| `/usr/bin/python3` | Python 3.9.6 | `arm64` | Apple Developer Tools | Incompatible (Too Old; Unused) |

### 2.2 Package Managers and Environment Tools
- **Homebrew:** Installed at `/opt/homebrew/bin/brew` (`VERIFIED SUCCESS`).
- **Python 3.12 Executable:** `/opt/homebrew/bin/python3.12` (`Mach-O 64-bit executable arm64`) (`VERIFIED SUCCESS`).

---

## 3. TASK 2 — DEPENDENCY AUDIT & RESOLUTION

### 3.1 Dependency Resolution Strategy
- **Member 3 FastAPI Backend (`requirements.txt`):**
  - `fastapi 0.162.0`, `uvicorn 0.54.0`, `python-multipart 0.0.33`, `pillow 13.0.0`, `pydantic 2.14.2`, `pydantic-settings 2.13.0`, `httpx 0.28.1`, `pytest 9.2.0`.
  - All installed natively as Apple Silicon `macosx_arm64` wheels.
- **Dataset / Model Requirements (`requirements-phase7.txt`):**
  - Historical Windows pins `torch==2.15.0+cpu` and `torchvision==0.29.2+cpu` were replaced with official Apple Silicon PyTorch distributions:
  - `torch==2.15.0` (`macosx_14_0_arm64`)
  - `torchvision==0.29.2` (`macosx_14_0_arm64`)
  - `numpy==2.5.4` (`macosx_14_0_arm64`)
  - `pillow==13.0.0` (`macosx_14_0_arm64`)

---

## 4. TASK 3 & 4 — VIRTUAL ENVIRONMENT PROVISIONING

### 4.1 Native Virtual Environment (`.venv_mac`)
- **Virtual Environment Path:** `/Users/heemanshupjain/Downloads/Member 3/.venv_mac` (`VERIFIED SUCCESS`)
- **Creation Command:** `/opt/homebrew/bin/python3.12 -m venv .venv_mac`
- **Python Executable:** `/Users/heemanshupjain/Downloads/Member 3/.venv_mac/bin/python` (Python 3.12.15)
- **Pip Version:** `pip 26.2.1`
- **Binary Architecture:** `Mach-O 64-bit executable arm64` (Native Apple Silicon, no Rosetta 2 emulation).
- **Existing Windows Environment:** `Hackathon-VIT/CBL04-Dataset/.venv_phase7` preserved completely untouched (`VERIFIED SUCCESS`).

---

## 5. TASK 5 — PYTORCH & HARDWARE ACCELERATION VERIFICATION

### 5.1 PyTorch Verification Results
Verified directly using `.venv_mac/bin/python`:
```text
PyTorch Version: 2.15.0
TorchVision Version: 0.29.2
MPS Acceleration Available: True (torch.backends.mps.is_available() == True)
MPS Built: True (torch.backends.mps.is_built() == True)
```

### 5.2 Compute Engine Verification
- **CPU Inference Test:** Executed `torch.nn.functional.conv2d` on random tensor of shape `[1, 3, 320, 160]`. Execution succeeded in 1.48 ms (`VERIFIED SUCCESS`).
- **MPS (Metal Performance Shaders) Test:** Transferred tensor to `mps` device and executed conv2d operation. Execution succeeded with valid tensor output (`VERIFIED SUCCESS`).

---

## 6. TASK 6 — BASELINE TEST SUITE EXECUTION

### 6.1 Test Execution Results
Executed test suite inside `.venv_mac`:
```bash
.venv_mac/bin/python -m pytest tests/ -v
```

| Test Module | Tests | Passed | Failed | Skipped | Duration |
|---|---:|---:|---:|---:|---:|
| `tests/test_contracts.py` | 27 | 27 | 0 | 0 | 0.08s |
| `tests/test_api.py` | 17 | 17 | 0 | 0 | 1.87s |
| **Total Baseline Suite** | **44** | **44** | **0** | **0** | **0.45s** |

**Status:** `VERIFIED SUCCESS` (100% pass rate across all contract and API endpoints).

---

## 7. TASK 7 — FASTAPI APPLICATION INTEGRITY & STARTUP VERIFICATION

- **Application Module:** `app.main:app`
- **Health Endpoints Verified:**
  - `GET /health` $\to$ `200 OK` (`{"status": "healthy", "service": "cbl04-gpay-detection-api"}`)
  - `GET /api/v1/health` $\to$ `200 OK` (`{"status": "healthy"}`)
- **Analysis Endpoint Verified:**
  - `POST /api/v1/analyze` $\to$ `200 OK` (accepts PNG, JPEG, WebP)
  - Streaming size enforcement: Rejects payloads $> 10$ MB with `413 Content Too Large`
  - Content-Type enforcement: Rejects PDF, GIF, and unknown streams with `415 Unsupported Media Type`
  - Dimensions check: Rejects $< 100 \times 100$ or $> 4096 \times 4096$ with `422 Unprocessable Entity`
  - Error sanitization: No internal file paths, Windows drives, or stack traces exposed to client.

---

## 8. TASK 8 — CNN CHECKPOINT INTEGRITY & ARCHITECTURE AUDIT

### 8.1 Checkpoint Storage & Provenance
Directly inspected and loaded via `torch.load(..., map_location="cpu", weights_only=False)`:

| Checkpoint File | Filesize (Bytes) | SHA256 (First 16 chars) | Load Status | Missing Keys | Unexpected Keys |
|---|---:|---|---|---|---|
| `Hackathon-VIT/CBL04-Dataset/models/phase7/fold_1_resnet18_multitask.pt` | 57,727,691 | `c87349ea94056156` | `VERIFIED SUCCESS` | 0 | 0 |
| `Hackathon-VIT/CBL04-Dataset/models/phase7/fold_2_resnet18_multitask.pt` | 57,727,691 | `f87968bc70d05051` | `VERIFIED SUCCESS` | 0 | 0 |

### 8.2 Model Forward Inference Baseline
Tested both models using sample GPay receipt image (`sample_clean_payment_receipt.png`):
- **Fold 1 Checkpoint:** Logits `[-1.0988, 1.0989]`, Softmax Edited Probability `0.75008`, Mask logits shape `[1, 1, 320, 160]`.
- **Fold 2 Checkpoint:** Logits `[0.3785, -0.3785]`, Softmax Edited Probability `0.40648`, Mask logits shape `[1, 1, 320, 160]`.

### 8.3 Performance & Limitations Disclosure
- Baseline validation accuracy on test set: $\approx 50.46\%$ (chance level).
- Baseline localization mask IoU: $0.059$ (5.9%).
- Highlighting / heatmaps are disabled in Phase 3 to prevent misleading overlays.
- Scores are uncalibrated experimental model scores, not certified probabilities.

---

## 9. CONFIRMATION OF ARTIFACT PRESERVATION

`[VERIFIED SUCCESS]`  
- `models/phase7/fold_1_resnet18_multitask.pt` (57,727,691 bytes): **Unchanged & Preserved**.
- `models/phase7/fold_2_resnet18_multitask.pt` (57,727,691 bytes): **Unchanged & Preserved**.
- `Hackathon-VIT/CBL04-Dataset/.venv_phase7`: **Unchanged & Preserved**.
- All dataset images and annotations: **Unchanged & Preserved**.

---

*Report certified by: Member 3 (Backend & AI Integration Lead)*  
*Phase 2 Status: COMPLETED & VERIFIED SUCCESS*
