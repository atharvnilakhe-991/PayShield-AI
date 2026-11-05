# Phase 7 model training report

## Completed work

A CPU-only, single-image multitask baseline was trained with the approved family-held-out folds. The model is a torchvision ResNet-18 encoder with a two-class clean/edited head and a lightweight U-Net-style mask decoder. It was trained from random initialization (`pretrained=false`) using 320×160 aspect-preserving inputs, three epochs per completed fold, batch size 8, AdamW (learning rate 1e-4), cross-entropy classification loss, and weighted BCE + soft Dice localization loss.

Runtime: Python 3.12.10, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, CPU device. Real images were not used. Member 3 reviewed the split limitation and recommended the ResNet-18 plus lightweight decoder architecture; the interface returns `logits` and `mask_logits`.

## Exactly what completed

| Fold | Test family | Training | Best validation epoch/F1 | Checkpoint | Test evaluation |
|---|---|---|---|---|---|
| 1 | compact | 3 epochs complete | Epoch 2 / 0.6667 | `models/phase7/fold_1_resnet18_multitask.pt` | Complete |
| 2 | enriched | 3 epochs complete | Epoch 3 / 0.4000 | `models/phase7/fold_2_resnet18_multitask.pt` | Complete using saved checkpoint after interruption |
| 3 | bank_progress | Not started | — | None | Not evaluated |
| 4 | recharge | Not started | — | None | Not evaluated |

Training was stopped after Fold 2 in response to the time limit. The two checkpoint files and their embedded three-epoch histories were preserved. The additional test pass loaded these checkpoints only; it did not resume training or change any split.

## Best available checkpoint

Best by validation F1: [Fold 1 checkpoint](../../models/phase7/fold_1_resnet18_multitask.pt). Validation F1 was 0.6667 at epoch 2, but this checkpoint predicted every held-out compact screenshot as edited. Its compact test accuracy was 0.50; the validation score therefore does not make it a usable detector. Fold 2's saved checkpoint is also available at `models/phase7/fold_2_resnet18_multitask.pt`.

## Reproducibility

- Architecture: `scripts/phase7/model.py`; architecture config: `templates/phase7/model_config.json`.
- Fold manifests: `metadata/phase7/folds/`; manifest-ordering seed: 20261207; model seed: 20261207.
- Training configuration: `metadata/phase7/training_config.json`.
- Histories: `reports/phase7/partial_training_history.json` and each checkpoint.
- Dependencies: `requirements-phase7.txt`; isolated environment: `.venv_phase7/`.
- Runner: `scripts/phase7/train_evaluate.py`; saved-checkpoint evaluator: `scripts/phase7/evaluate_saved_folds.py`.

The full runner has no resume mode and refuses to overwrite a non-empty `models/phase7/`; finish the remaining folds in a separate worktree/output directory. Phase 6 manifests and source images were not modified.

