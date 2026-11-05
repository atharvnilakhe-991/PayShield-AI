# Phase 7 time-bounded completion summary

## Completed

- Preserved Phase 6 and all source data; no images were generated, moved, or changed.
- Prepared four family-held-out fold manifests using seed 20261207 and the approved Phase 6 group assignments.
- Coordinated the ResNet-18 multitask architecture with Member 3.
- Completed 3 training epochs and saved checkpoints for Folds 1 and 2.
- Evaluated both saved checkpoints against their own held-out test manifests: compact and enriched (868 screenshots total).
- Wrote partial training/evaluation records, per-family/per-category metrics, error lists, and plots.

## Incomplete

- Folds 3 (bank-progress test) and 4 (recharge test) training/evaluation were not started.
- Therefore bank-progress and recharge have no held-out test metrics, and a full four-family pooled out-of-fold result does not exist.
- The two-family partial test result is weak (accuracy 50.5%, edited-class F1 52.4%); localization IoU is 5.9% pooled.

## Best saved checkpoint

`models/phase7/fold_1_resnet18_multitask.pt` (57,727,691 bytes), selected at validation F1 0.6667. It predicts every compact test sample as edited, so the checkpoint is preserved as the best available artifact, not recommended as a usable detector. Fold 2's checkpoint is also preserved.

No real-world performance claim is supported. The 387 real screenshots were not labeled or scored. No Phase 6 split was changed. No Phase 8 work was started.
