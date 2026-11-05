# Phase 8 final handover

## Handover checklist

- **Dataset root:** `data/` within `CBL04-Dataset/`; original real screenshots are under `data/real_gpay/raw/`. Keep access restricted and do not publish.
- **Manifests:** `metadata/phase6_manifest.csv`, `metadata/phase6/`, `metadata/phase7/folds/`, `metadata/phase7/fold_manifest.csv`.
- **Original training/evaluation:** `scripts/phase7/`, `metadata/phase7/training_config.json`, `reports/phase7/`.
- **Best available checkpoint:** `models/phase7/fold_1_resnet18_multitask.pt` (validation F1 0.6667; all compact test examples predicted edited, so not reliable). Also preserved: `models/phase7/fold_2_resnet18_multitask.pt`.
- **Phase 8 threshold analysis:** `scripts/phase7/validation_threshold_eval.py`, `reports/phase8/threshold_tuning_metrics.json`, `reports/phase8/threshold_tuning_predictions.csv`. No weights were changed.
- **Application source:** `app/server.py`, `app/index.html`, `app/styles.css`, `app/app.js`.
- **Environment:** `.venv_phase7/` and `requirements-phase7.txt` (Python 3.12.10, CPU PyTorch). Follow README for commands.
- **Demo samples:** `data/synthetic_gpay_phase3/clean/` and `edited/`; each file is synthetic and fictional. The families each have 3 matched pairs.
- **Reports:** `reports/phase8/` and `reports/phase7/`.
- **Presentation:** validated editable deck `reports/phase8/presentation/CBL04_Hackathon_Presentation.pptx`, slide outline `hackathon_presentation.md`, and `demo_script.md`.
- **Presentation validation:** 10 slides; package and layout validators passed with zero findings; each rendered slide was visually reviewed. Native PowerPoint execution/font rendering was not verified.
- **README:** root `README.md`; pre-Phase 8 copy: `backups/phase7/README.pre-phase8.md`.

## Known issues and incomplete work

- Only two of four family-held-out training folds completed; bank-progress and recharge test families are untested.
- The baseline is near chance overall and has severe family-specific errors. Threshold selection improved F1 only by increasing false positives and reducing accuracy.
- No weight-level improvement was demonstrated.
- Region localization is too weak to visualize responsibly; app overlays are disabled.
- The 387 real images are unlabeled and provide no verified real-world performance evidence.
- Application runtime checks passed for startup, browser rendering, all three supported image formats, inference across eight fictional samples, and rejection of corrupt/unsupported files. The 8 sample guesses were all edited, including the clean examples. Large-file boundary handling and browser layout at explicit mobile widths were not exercised.

## Member handoff

**For Member 3 (model):** the handoff was sent and acknowledged. Use this checkpoint, training config, partial evaluation, and limitations together; do not describe the threshold sweep as new training or report the pooled score as a single model. Start future experiments from copied checkpoints/output directories and keep test families untouched.

**For Member 1 (dataset):** retain the organized data and manifests as-is; keep originals restricted and unlabeled. Phase 6 splits and all pilot outputs remain unchanged.

## Future recommendations

Finish folds 3 and 4; increase independently rendered templates per family; compare reference-pair detection with single-image cues; evaluate overlays at original resolution; only tune and calibrate on larger, representative validation groups; and acquire privacy-reviewed ground truth before making claims about genuine payment screenshots.
