# CBL-04 — PHASE B READINESS GATE: MEMBER 2 CNN REPLACEMENT VALIDATION

**Project:** CBL-04 — Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Workspace:** `/Users/heemanshupjain/Downloads/Member 3`  
**Role:** Member 3 — Backend / AI Integration Lead  
**Audit Target:** `comparison_inputs/member2_cnn/Member 2/`  
**Date:** 2026-11-04  
**Status:** READ-ONLY READINESS AUDIT COMPLETE  
**Final Decision:** **OPTION D — REJECT THE PROPOSED CHECKPOINT** (With Option B Fair Validation Plan available as an empirical formality)

---

## 1. Executive Summary

Prior to authorizing Phase B (CNN replacement and stacker retraining), this audit performed an independent, read-only forensic inspection of Member 2's model artifacts in `comparison_inputs/member2_cnn/Member 2/`.

### Core Technical Verdict:
1. **The Proposed Checkpoint is a 256x256 Tile Classifier, Not a Screenshot Classifier:**  
   The proposed checkpoint `member2/phase5_model_training/artifacts/best_model.pt` is a patch classifier trained on isolated $256 \times 256$ image tiles. It was never trained on full screenshots.
2. **Catastrophic Screenshot-Level False-Positive Rate (92.0%–100.0%):**  
   Member 2's own Phase 7.1 independent technical audit (`audit_report.md` and `classification_audit.json`) evaluated this architecture across 150 held-out test screenshots (75 clean, 75 tampered) and mathematically proved:
   - **Rule 1 (Any Component $>0\text{ px}$):** Specificity = **0.00%** ($\text{FPR} = \mathbf{100.0\%}$, $\text{FP}=75, \text{TN}=0$).
   - **Rule 2 (Area $\ge 1,000\text{ px}$):** Specificity = **8.00%** ($\text{FPR} = \mathbf{92.0\%}$, $\text{FP}=69, \text{TN}=6$).
   - **Rule 3 (Peak Prob $\ge 0.80$):** Specificity = **0.00%** ($\text{FPR} = \mathbf{100.0\%}$, $\text{FP}=75, \text{TN}=0$).
3. **Severe Tile-Level Deficiencies:**  
   On Member 2's own tile-level test set (`metrics_test.json`), `best_model.pt` achieves a tile precision of only **12.1%** and misses **69.6% of tampered tiles** ($\text{FNR} = 69.6\%$, catching only 7 out of 23 tampered tiles).
4. **Complete Absence of Out-Of-Fold (OOF) Artifacts:**  
   Member 2 used a single $70/15/15$ split. **Zero K-fold training was executed**, zero fold checkpoints exist, and **zero OOF prediction files exist**. The existing Phase 8 stacker cannot be retrained without full cross-validated OOF predictions.
5. **Architectural Mismatch with Member 3 Backend:**  
   Member 3's backend pipeline expects a global CNN that takes a full screenshot ($[1, 3, H, W]$) and produces a calibrated continuous manipulation probability $P \in [0.0, 1.0]$. Member 2's model requires sliding-window tiling, coordinate tracking, and connected-component heuristics that inevitably flag authentic receipts as tampered.

**Final Decision: REJECT THE PROPOSED CHECKPOINT.** Integrating Member 2's checkpoint would directly re-introduce a 92%–100% false-positive rate on clean screenshots, completely destroying the progress achieved in Phase A.

---

## 2. PART 1 — Checkpoint Identity & Architectural Verification

We searched recursively through `comparison_inputs/member2_cnn/Member 2/` and located all stored PyTorch model checkpoints:

| Checkpoint Path | Architecture | Checkpoint Purpose | File Size | SHA-256 Hash |
| :--- | :---: | :--- | :---: | :---: |
| `member2/phase5_model_training/artifacts/best_model.pt` | `resnet18` | 256x256 Tile Classifier | 43.8 MB | `2d47a5eb65ff24f78b2d0d2cbe0ce4fd06dff7ccda7ba2b9d754e9861f28da94` |
| `member2/phase5_model_training/artifacts/last_model.pt` | `resnet18` | Final Epoch Tile Weights | 43.8 MB | `c7f12ab8d84a1c41ff5d8f104e1ab2d8a79fb5dc324df105efed0aece2c2f6e8` |
| `member2/performance_improvement/artifacts/best_improved_model.pt` | `spatial_unet`| Spatial Heatmap U-Net | 1.8 MB | `b6e3234e734c2801a32b98cb58b66328b907bade26645b03469ecd32bdf5765c` |

### 2.1 State-Dict Inspection of `best_model.pt`
Loading `best_model.pt` via `torch.load("...", map_location="cpu")` revealed:
- **Top-Level Keys:** `['epoch', 'model_name', 'state_dict', 'optimizer_state_dict', 'val_loss', 'val_f1', 'val_accuracy', 'val_precision', 'val_recall', 'config', 'training_history']`
- **Model Name:** `"resnet18"`
- **Input Tensor Dimensions:** `(batch_size, 3, 256, 256)`
- **Output Layer Weights:** `backbone.fc.1.weight: torch.Size([2, 512])`
- **Output Layer Bias:** `backbone.fc.1.bias: torch.Size([2])`
- **Output Tensor Shape:** `(batch_size, 2)` (Two logits: Class 0 = Clean Tile, Class 1 = Tampered Tile).
- **Class Ordering:** Index 0 = Clean / Authentic, Index 1 = Tampered / Manipulated.
- **Selection Criterion:** Early stopping on `val_f1` (best at Epoch 3: `val_f1 = 0.2667`, `val_loss = 0.5999`).
- **Training Configuration (`config`):**
  ```json
  {
    "model_architecture": "resnet18",
    "pretrained": false,
    "num_classes": 2,
    "tile_size": 256,
    "stride": 128,
    "tampered_threshold": 16,
    "normalization": "imagenet",
    "batch_size": 32,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "num_epochs": 5,
    "early_stopping_patience": 3,
    "early_stopping_metric": "val_f1",
    "max_train_samples_per_epoch": 1500,
    "split_proportions": {"train": 0.7, "val": 0.15, "test": 0.15}
  }
  ```

---

## 3. PART 2 — Verification of Performance Evidence

We audited all evaluation reports across Member 2's repository:
1. `phase5_model_training/artifacts/metrics_test.json`
2. `phase5_model_training/artifacts/confusion_matrix.json`
3. `phase7_1_evaluation_audit/classification_audit.json`
4. `phase7_1_evaluation_audit/audit_report.md`
5. `performance_improvement/artifacts/baseline_vs_improved_comparison.json`

### 3.1 Tile-Level Classification Metrics (`best_model.pt`)

On Member 2's own test split of 300 tiles (277 clean, 23 tampered):

| Tile Metric | Formula | Value | Status |
| :--- | :---: | :---: | :--- |
| **Tile Accuracy** | $(\text{TP} + \text{TN}) / 300$ | $77.67\%$ | Misleading due to $92.3\%$ class imbalance |
| **Tile Precision** | $\text{TP} / (\text{TP} + \text{FP})$ | **$12.07\%$** | **$87.9\%$ of predicted tampered tiles are false alarms** |
| **Tile Recall** | $\text{TP} / (\text{TP} + \text{FN})$ | **$30.43\%$** | **Misses $69.6\%$ of actual tampered tiles** |
| **Tile F1-Score** | $2 \cdot P \cdot R / (P + R)$ | **$0.1728$** | **Severely underperforming ($17.3\%$)** |
| **Tile False-Positive Rate** | $\text{FP} / \text{Clean Tiles}$ | **$18.41\%$** | 51 false alarms out of 277 clean tiles |
| **Tile False-Negative Rate** | $\text{FN} / \text{Tampered Tiles}$| **$69.57\%$** | Catches only 7 out of 23 tampered tiles |

### 3.2 Screenshot-Level Classification Metrics (`classification_audit.json`)

Evaluated on 150 full held-out test screenshots (75 clean, 75 tampered) in Member 2's Phase 7.1 audit:

| Decision Rule | TP | TN | FP | FN | Accuracy | Sensitivity | Specificity | False-Positive Rate | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule 1 (Any Component $>0\text{ px}$)** | 75 | 0 | 75 | 0 | 50.00% | 100.00% | **0.00%** | **100.00%** | 0.6667 |
| **Rule 2 (Area $\ge 1,000\text{ px}$)** | 70 | 6 | 69 | 5 | 50.67% | 93.33% | **8.00%** | **92.00%** | 0.6542 |
| **Rule 3 (Peak Prob $\ge 0.80$)** | 75 | 0 | 75 | 0 | 50.00% | 100.00% | **0.00%** | **100.00%** | 0.6667 |

### 3.3 Auditor Conclusion in Member 2's Own Report:
> *"Confirmed accurate. The rules are not interchangeable; Rules 1 and 3 suffer complete failure of specificity, while Rule 2 provides minimal specificity (8.00%). The model is a spatial pixel-level localization model, not a standalone global binary screenshot classifier."*  
> — Member 2 Phase 7.1 Independent Technical Audit Report (`audit_report.md`, Section 2)

---

## 4. PART 3 — Fair Evaluation Compatibility Check

Could Member 2's checkpoint theoretically be evaluated on Member 3's frozen validation manifest (`bank_progress`, 108 samples)?

### Compatibility Matrix:
- **Image Compatibility:** **No native compatibility.** Member 3's backend evaluates full screenshots of arbitrary dimensions ($360 \times 740$ to $1080 \times 2400$). Member 2's `best_model.pt` strictly requires a $256 \times 256$ input. To evaluate it, a custom sliding-window tile extractor and spatial stitcher must be inserted.
- **Label Compatibility:** Compatible (Clean = 0, Tampered = 1).
- **Preprocessing Requirements:** ImageNet normalization `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`. Member 3 uses ResNet ImageNet normalization.
- **Duplicate and Group Isolation:** Compliant if using Member 3's existing `validation_manifest.csv`.
- **Layout Representation:** Member 3's validation set is 100% `bank_progress`. In Member 2's tile test (`metrics_test.json`), `bank_progress` achieved tile accuracy of $65.4\%$, tile precision of $12.5\%$, and tile recall of $33.3\%$.

### Standby Evaluation Plan (Option B):
If leadership mandates an empirical benchmark on Member 3's validation set before final closure:
1. Load all 108 frozen validation images from `metadata/phase6/validation_manifest.csv`.
2. Extract $256 \times 256$ sliding-window tiles (stride 128) per image.
3. Pass tiles through `best_model.pt` on CPU.
4. Compute maximum tile softmax probability $P_{\max} = \max_i \text{Softmax}(\text{logits}_i)[1]$.
5. Apply decision threshold $\theta \in [0.50, 0.80]$ to classify full screenshot.
6. Generate confusion matrix and prove whether specificity remains $\le 8.0\%$.

---

## 5. PART 4 — Out-Of-Fold (OOF) Requirements & Audit

To retrain the Phase 8 L2 logistic stacker meta-learner (`stacker_model.json`), what OOF predictions are needed?

### 5.1 Member 3 Stacker Architecture Requirements:
The Phase 8 stacker was trained on **868 synthetic receipts** from Folds 1 and 2. Its input vector requires:
- `cnn_score`: The continuous manipulation probability for each training sample generated while that sample was held out of the CNN's training fold.
- `cnn_pred_class`: The binary thresholded decision ($0.0$ or $1.0$).

### 5.2 Forensic Audit of Member 2 OOF Artifacts:
- Search command: `find "comparison_inputs/member2_cnn" -iname "*oof*"`
- Result: **ZERO MATCHES.**
- K-fold training check: Member 2 only mentioned `GroupKFold` conceptually in READMEs. Member 2's actual training script executed a single $70/15/15$ split (`phase5_model_training/training_config.json`).
- Fold checkpoint count: Exactly **1 checkpoint** (`best_model.pt`), zero fold checkpoints (`fold_1.pt` through `fold_5.pt` do not exist).
- **Conclusion:** Member 2 **never generated Out-Of-Fold predictions**. It is mathematically impossible to retrain Member 3's stacker using Member 2's artifacts without designing and executing a complete 5-fold cross-validation training cycle from scratch.

---

## 6. PART 5 — Score Semantics & Frontend UI Remediation

During Phase A.1 validation, we discovered that `INCONCLUSIVE` receipts displayed `100%` or `99.9%` confidence in the frontend.

### 6.1 The Mathematical Defect:
In `app/services/final_verdict_service.py` (line 203):
```python
confidence = round(1.0 - abs(stacker_prob - 0.50) * 2.0, 4)
```
- At $P = 0.5008$, $|\text{stacker\_prob} - 0.50| = 0.0008$.
- $\text{confidence} = 1.0 - 2(0.0008) = 0.9984$ ($99.8\%$).
- This formula represents an **Uncertainty Index** (where $1.0$ is maximum ignorance at the decision threshold). Returning it in the field named `confidence` causes the frontend to display maximum model ignorance as if it were maximum decision certainty.

### 6.2 Recommended Implementation for Next Phase:
1. **Schema Separation in `app/schemas/response.py`:**
   - For `INCONCLUSIVE` verdicts, set `confidence = 0.50` (or `null`).
   - Add explicit field `uncertainty_index: Optional[float] = Field(description="Normalized entropy / threshold distance index [0.0 to 1.0]")`.
2. **Frontend UI Presentation in `ConfidenceScore.jsx` & `ResultsDashboard.jsx`:**
   - Suppress the large percentage meter when verdict is `INCONCLUSIVE`.
   - Render:
     ```jsx
     <div className="confidence-inconclusive-banner">
       <HelpCircle size={20} />
       <span>Indeterminate Decision Margin (Score: 0.50 ± 0.002)</span>
       <p>The multi-branch ensemble score falls within the indeterminate threshold region [0.45, 0.55]. Manual visual verification is recommended.</p>
     </div>
     ```

---

## 7. PART 6 — Final Decision & Comparative Justification

### The Decision: **OPTION D — REJECT THE PROPOSED CHECKPOINT**

```
              ┌────────────────────────────────────────────────────────┐
              │           MEMBER 2 CHECKPOINT EVALUATION               │
              └──────────────────────────┬─────────────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
        [Architecture Check]                            [Specificity Check]
        best_model.pt is a 256x256                      Audited false-positive
        tile classifier, NOT a                          rate on clean screenshots
        screenshot model.                               is 92.0% to 100.0%.
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                        ┌────────────────▼───────────────┐
                        │     FINAL DECISION: OPTION D   │
                        │  REJECT THE PROPOSED CHECKPOINT│
                        └────────────────────────────────┘
```

### Detailed Justification:
1. **Guaranteed Resurgence of the False-Positive Crisis:**  
   Phase A successfully reduced clean-image false positives from 30.0% to 0.0%. Member 2's model has an audited false-positive rate on clean screenshots of **92.0% to 100.0%**. Integrating it would immediately break the application and falsely accuse nearly all authentic screenshots of tampering.
2. **Poor Forensic Precision on Tiles:**  
   With a tile precision of only $12.1\%$ and a false-negative rate of $69.6\%$, the model is insufficiently accurate even on its native patch domain.
3. **Missing OOF Cross-Validation Data:**  
   Member 2 produced zero OOF cross-validation predictions, making clean retraining of the Phase 8 stacker impossible without an unapproved full training cycle.
4. **Member 3 Pipeline Stability:**  
   Member 3's backend architecture is now clean, regression-free, and handles ambiguity safely. The appropriate path forward is to retain Member 3's ResNet-18 or train a calibrated full-screenshot classifier within Member 3's native pipeline.

---
**Report Author:** Member 3 — Backend / AI Integration Lead  
**Readiness Gate Status:** Evaluated. Checkpoint rejected. Awaiting user review and authorization.

