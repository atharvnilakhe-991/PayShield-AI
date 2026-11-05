# CBL-04 final project report

## Project

CBL-04 explores detection of field edits in fictional Google Pay-style dark-theme transaction screenshots. The pipeline inventories source material, generates synthetic matched clean/edited pairs with masks and metadata, audits annotations, prepares leakage-conscious family-held-out folds, trains an experimental classifier/localizer, and provides a local browser demo.

## Dataset and lineage

- 387 unique real screenshots were counted after duplicate checks; they remain unlabeled and are not model ground truth.
- Phase 4 generated 500 synthetic clean/edited pairs (1,000 screenshots) with masks and annotations.
- Four synthetic layout families are compact, enriched, bank-progress, and recharge.
- Phase 6 manifests are preserved. Phase 7 family-held-out folds are in `metadata/phase7/folds/`; split files were not changed in Phase 8.
- Earlier pilots remain separated in their existing directories.

## Model and measured results

The best available artifact is ResNet-18 plus a lightweight mask decoder at `models/phase7/fold_1_resnet18_multitask.pt`. It was selected on validation F1=0.6667 but predicted all 218 compact clean examples as edited on its held-out compact test. The partial two-model Phase 7 baseline across compact and enriched has accuracy 50.5%, precision 50.4%, recall 54.6%, F1 52.4%. Threshold-only validation selection gives pooled 50.0% accuracy, 50.0% precision, 64.7% recall, and 56.4% F1 at the cost of 281 false positives. No model weights improved. Folds 3 and 4 are incomplete; bank-progress and recharge have no test scores. No real-world performance claim is made.

## Demonstration

`app/server.py` serves the local interface under `app/`. The demo accepts PNG/JPEG/WebP and reports a model guess plus an uncalibrated score. It does not display a mask overlay because audited localization IoU is only 0.059. The page prominently states that results cannot verify payment or prove fraud. Use only fictional SYNTHETIC / DEMO screenshots.

## Handover status

Application code, metrics, reproducibility configuration, reports, a validated 10-slide presentation, and demo script are present. The local browser flow and PNG/JPEG/WebP API checks passed. The 8 synthetic pilot smoke predictions were all “edited,” including every clean sample, and are documented as a severe demo limitation. See `final_handover.md` for exact paths and remaining work.
