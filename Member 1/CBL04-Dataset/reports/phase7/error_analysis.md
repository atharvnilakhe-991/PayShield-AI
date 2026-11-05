# Phase 7 error analysis

## Observed classification errors

The 868 available held-out test screenshots produced 233 false positives and 197 false negatives.

- **Fold 1 / compact:** the model predicted every test screenshot as edited. All 218 clean screenshots became false positives; all 218 edited screenshots were true positives. Accuracy was 50%, despite F1=0.6667.
- **Fold 2 / enriched:** 201 clean examples were true negatives and 15 were false positives. Of 216 edited screenshots, only 19 were detected and 197 were false negatives. Edited recall was 8.8% and F1 was 0.152.
- **Edit fields:** among the evaluated families, false negatives were amount 43/91, date/time 38/91, other/account suffix 32/84, recipient/merchant 46/85, and transaction ID/reference 38/83. Category comparisons share the same clean controls.

Detailed synthetic IDs, probabilities, paths, and error labels are in `available_false_positives.csv` and `available_false_negatives.csv`.

## Localization errors

On the 320×160 evaluation canvas, pooled localization precision was 6.5%, recall 39.4%, Dice 11.2%, and IoU 5.9%. The decoder over-predicted changed pixels relative to the small sparse targets. These coarse-resolution mask scores should not be interpreted as precise original-resolution boxes.

## Why performance is weak

The synthetic edits replace plausible names, amounts, dates, IDs, or account suffixes through the same renderer. A single screenshot often contains no forensic artifact that distinguishes a deliberately replaced plausible value from a clean plausible value. The clean/edited ground truth is counterfactual and defined by the paired source, but the single-image model does not receive that clean reference. This makes several edits underdetermined from the image alone, while the training set's limited templates encourage shortcut learning. Three epochs from random initialization were also a short baseline, and validation selection was unstable (Fold 1 F1: 0.578, 0.667, 0.036 by epoch).

## Limits and next improvements

Do not use this baseline for real transaction decisions. A stronger study needs either a model that explicitly compares a submitted screenshot to a trusted reference, or independently validated forensic cues that can be detected from one screenshot. It also needs more independent layout templates and more verified ground-truth data. Complete the bank-progress and recharge held-out folds, review false positives/negatives, and evaluate at original mask resolution before claiming localization capability. The 387 real screenshots remain unlabeled and were not used for training or scoring.
