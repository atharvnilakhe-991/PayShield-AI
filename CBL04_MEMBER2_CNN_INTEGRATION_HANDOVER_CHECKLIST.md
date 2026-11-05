# CBL-04 — MEMBER 2 FINAL CNN INTEGRATION HANDOVER CHECKLIST

**Author:** Member 3 — Backend / AI Integration Lead  
**Recipient:** Member 2 — CNN Modeling & Computer Vision Lead  
**Workspace:** `/Users/heemanshupjain/Downloads/Member 3`  
**Current Baseline Checkpoint:** `Hackathon-VIT/CBL04-Dataset/models/phase7/fold_1_resnet18_multitask.pt`  
**Current Checkpoint SHA-256:** `44b654aa846fddfde7be6dab44ceada6687d8dfd35e2c83edbcb60fed909f62a`  
**Integration Service:** `app/services/cnn_inference_service.py`  
**Handover Date:** 2026-11-03  

---

## 1. PURPOSE & ARCHITECTURAL BOUNDARIES

Member 2 is responsible for training, fine-tuning, and evaluating improved CNN architectures for detecting digital tampering in Google Pay transaction screenshots.  
Member 3 is responsible for integrating Member 2's approved model checkpoint into the multi-branch production backend (FastAPI gateway, OCR branch, semantic heuristics, OOF stacker meta-learner, and final tripartite decision engine).

This checklist defines the technical contracts, verification steps, and downstream retraining requirements that must be completed whenever Member 2 delivers a new model checkpoint.

---

## 2. REQUIRED HANDOVER ARTIFACTS

Before any new model checkpoint can be accepted into the backend pipeline, Member 2 must deliver the following **10 mandatory artifacts**:

| # | Handover Artifact | Specification / Requirement | Current Baseline Status |
| :-: | :--- | :--- | :--- |
| **1** | **Final Trained Checkpoint File** | PyTorch checkpoint (`.pt` or `.pth`) containing `state_dict` and `model_config` dictionary. | `fold_1_resnet18_multitask.pt` (57.7 MB) |
| **2** | **Model Architecture & Config** | Specification of backbone (e.g. ResNet-18, EfficientNet) and head dimensions. Saved in `checkpoint["model_config"]`. | ResNet-18 multi-task (classifier + mask decoder) |
| **3** | **Input Preprocessing Spec** | Target dimensions, padding color, interpolation algorithm, mean, and standard deviation tensors. | $320 \times 160$ (H $\times$ W), Lanczos resize, pad RGB `(18, 18, 18)`, ImageNet mean/std |
| **4** | **Class-Label Mapping** | Exact index-to-class alignment for output logits. | Index `0` = Clean / Genuine  <br> Index `1` = Edited / Tampered |
| **5** | **Output Tensor Format** | Expected dictionary keys and tensor shapes from `model.forward(x)`. | `output["logits"]` $\in \mathbb{R}^{B \times 2}$  <br> `output["mask"]` $\in \mathbb{R}^{B \times 1 \times H \times W}$ |
| **6** | **Localization Mask Validation** | Empirical IoU score on test set. If IoU $< 0.50$, mask highlighting must remain disabled. | Baseline IoU: $0.059$ ($5.9\%$) $\longrightarrow$ Mask intentionally suppressed |
| **7** | **Evaluation Report** | Metrics report on validation and held-out test splits (Macro-F1, Precision, Recall, Confusion Matrix). | Documented in `CBL04_PHASE7_BRANCH_VALIDATION_REPORT.md` |
| **8** | **Checkpoint SHA-256 Hash** | Cryptographic hash to verify file transfer integrity and model provenance. | `44b654aa846fddfde7be6dab44ceada6687d8dfd35e2c83edbcb60fed909f62a` |
| **9** | **Dataset Split Provenance** | Clear manifest of training, validation, and test sample IDs used during model training. | 5-fold CV splits codified in `Hackathon-VIT/CBL04-Dataset/splits/` |
| **10** | **Data Separation Confirmation** | Explicit confirmation that the frozen test set (24 samples, `recharge` family) was NOT used during training or tuning. | Confirmed: Test set strictly isolated |

---

## 3. DOWNSTREAM ARTIFACTS IMPACT ANALYSIS

When Member 2 updates the CNN model, the change propagates through the downstream multi-model ensemble. **A new CNN checkpoint CANNOT simply be dropped in without regenerating downstream artifacts.**

```text
[New Member 2 CNN Checkpoint]
            │
            ▼
 1. CNN Inference Service (Updated checkpoint path & weights hash)
            │
            ▼
 2. Out-Of-Fold (OOF) Inference across 5 folds ──► [Regenerate CNN feature columns]
            │
            ▼
 3. Phase 8 Stacker Meta-Learner (models/phase8/stacker_model.json) ──► [MUST RETRAIN]
            │
            ▼
 4. Final Verdict Engine (Operating threshold 0.50 & Ambiguity band [0.45, 0.55]) ──► [RE-EVALUATE]
```

### Why the Phase 8 Stacker MUST Be Retrained:
1. **Covariate Shift in Feature Space:** The Phase 8 logistic meta-learner was trained on 18 numerical features, including `cnn_score` (mean: $0.5607$, std: $0.1484$) and `cnn_pred_class` derived specifically from the Phase 7 ResNet-18 baseline model.
2. **Probability Distribution Mismatch:** A new, more accurate CNN model will have a different probability distribution (e.g. sharper peaks near $0.0$ and $1.0$). Passing these new outputs into the existing Stacker model (`stacker_model.json`) will cause weight miscalibration and false inconclusive triggers.
3. **Requirement:** Member 2 must run 5-fold cross-validated out-of-fold inference with their new model to produce an updated `cnn_oof_predictions.csv`. Member 3 will then retrain `stacker_model.json` on the updated feature matrix.

---

## 4. STEP-BY-STEP TECHNICAL INTEGRATION PROCEDURE

### Step 1: Checkpoint Transfer & Integrity Check
Place the approved checkpoint in `Hackathon-VIT/CBL04-Dataset/models/`:
```bash
# Verify file size and compute SHA-256 hash
shasum -a 256 "Hackathon-VIT/CBL04-Dataset/models/<new_checkpoint>.pt"
```

### Step 2: Architecture Compatibility Verification
Verify that the model loads safely under `torch.load(..., weights_only=True)`:
```python
import torch
ckpt = torch.load("Hackathon-VIT/CBL04-Dataset/models/<new_checkpoint>.pt", weights_only=True)
assert "state_dict" in ckpt
assert "model_config" in ckpt
print("Model config:", ckpt["model_config"])
```

### Step 3: Update Backend Configuration
In `app/core/config.py`, update `CNN_CHECKPOINT_PATH`:
```python
CNN_CHECKPOINT_PATH: str = Field(
    default="Hackathon-VIT/CBL04-Dataset/models/<new_checkpoint>.pt",
    description="Filesystem path to PyTorch CNN model checkpoint (.pt)",
)
```

### Step 4: Validate Preprocessing & Forward Pass
Run the automated CNN unit and integration tests:
```bash
./.venv_mac/bin/pytest tests/test_cnn_inference.py -v
```

### Step 5: Regenerate OOF Predictions & Retrain Stacker
Execute the cross-validated feature generation script and retrain the L2 logistic stacker:
```bash
./.venv_mac/bin/python -m app.services.stacker_service --retrain
./.venv_mac/bin/pytest tests/test_stacker_service.py -v
```

### Step 6: Verify Final Verdict & Tripartite Contract
Run the final decision engine tests:
```bash
./.venv_mac/bin/pytest tests/test_final_verdict_service.py -v
./.venv_mac/bin/pytest tests/test_phase10_frontend_backend_integration.py -v
```

### Step 7: Run Full Workspace Regression
Execute the entire test suite to guarantee 100% backward compatibility:
```bash
./.venv_mac/bin/pytest tests/ -v
cd cbl04-frontend && npm run lint && npm run build
```

---

## 5. COMPATIBILITY & SECURITY SAFEGUARDS

1. **`weights_only=True` Enforcement:** Checkpoints that require arbitrary Python code execution (pickled classes or functions) will be rejected. All weights must load under PyTorch's safe unpickler.
2. **Fallback Degradation:** If a new checkpoint fails to load or encounters a corrupt state, `CNNModelUnavailableError` will trigger graceful degradation or return HTTP 503 rather than crashing the FastAPI worker process.
3. **Mask Localization Rule:** If Member 2's new model does not achieve a certified test segmentation IoU $\ge 0.50$, spatial mask highlighting (`heatmap_url`) MUST remain suppressed (`null`) to uphold explainability standards.
