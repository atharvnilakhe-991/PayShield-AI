# Phase 8 model improvement report

## Work performed

Inspected the Phase 6 family/pHash manifests, Phase 7 reports and checkpoints, ResNet-18 + mask decoder, preprocessing, class labels, loss, and saved epoch histories. Preserved both Phase 7 checkpoints. No original image, Phase 6 split, pilot, or Phase 7 report was changed.

### Evidence-based failure review

- **Label mapping:** training maps `clean` to 0 and `edited` to 1; the held-out manifests and stored evaluation rows use the same mapping. No label inversion was found.
- **Input processing:** the training dataset applies aspect-preserving LANCZOS resizing, top-left placement on a 320 × 160 canvas filled with RGB (18,18,18), and ImageNet channel normalization. The demo inference path implements that same transformation.
- **Class balance:** each matched pair contributes one clean and one edited sample. The fold training manifests are pair intact, so nominal image-class imbalance does not explain the all-edited Fold 1 output.
- **Prediction bias/instability:** Fold 1 predicted edited for all 436 compact test samples; Fold 2 detected only 19 of 216 edited enriched samples. Validation F1 varied across epochs, especially Fold 1 (0.578, 0.667, 0.036), consistent with an unstable short run and weak cross-family transfer.
- **Training and loss:** both checkpoints were randomly initialized, trained for only three CPU epochs, with AdamW at 1e-4, unweighted cross-entropy and a jointly optimized weighted BCE + Dice mask loss. This makes undertraining and task competition plausible contributors, not proven causes.
- **Data/architecture:** the four family groups are small and template-bound. Entire-family holdout is strict but makes visual domain shift substantial. A single image also lacks a trusted paired reference for plausible value substitutions. These constraints are evidenced by the split and poor held-out family results.
- **Thresholds:** the validation-only sensitivity experiment selected thresholds 0.6018 (Fold 1, n=108) and 0.4453 (Fold 2, n=24), then applied them to each untouched test family. Pooled accuracy moved from 50.5% to 50.0%, F1 from 52.4% to 56.4%, and false positives from 233 to 281. This changes no weights and is not a new model or calibrated confidence. The small Fold 2 validation set makes its threshold uncertain.

## Model improvement outcome

The only completed weight-bearing experiments remain the original Fold 1 and Fold 2 three-epoch baselines. The completed validation-only threshold experiment is documented in `baseline_vs_improved_comparison.md` and `threshold_tuning_metrics.json`; it is a decision-threshold adjustment, not a new checkpoint. No class-weight, LR sweep, augmentation, pretrained-weight download, or additional fold training was attempted because available CPU resources and the prior time-bounded work make broad experiments unjustified without risking the held-out protocol.

The best checkpoint by its validation F1 remains `models/phase7/fold_1_resnet18_multitask.pt` (Fold 1, selected epoch 2, validation F1 0.6667). It is a preserved research artifact, not a reliable detector. Fold 2 remains at `models/phase7/fold_2_resnet18_multitask.pt`.

## Reproduction

- Environment: `.venv_phase7/`, Python 3.12.10, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu.
- Original training command: `python scripts/phase7/train_evaluate.py` (refuses to overwrite the existing model directory; do not run against the current artifact path).
- Threshold experiment: `python scripts/phase7/validation_threshold_eval.py`.
- Original configuration: `metadata/phase7/training_config.json`; folds: `metadata/phase7/folds/`.
- No real screenshots were assigned labels or used in training or evaluation.

## Conclusion

No demonstrated weight-level model improvement is established. Threshold selection traded lower accuracy and more false positives for higher recall and F1; it does not make the model reliable. Only two folds completed. Fold 3 (bank progress) and Fold 4 (recharge) remain incomplete.
