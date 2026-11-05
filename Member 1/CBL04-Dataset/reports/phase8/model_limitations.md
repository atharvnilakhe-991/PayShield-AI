# Model limitations

- The two-fold checkpoint is an experimental synthetic-only baseline with weak held-out results (pooled two-model baseline accuracy 50.5%, F1 52.4%). These metrics are not for one final all-family model.
- Fold 1 classified every held-out compact screenshot as edited, including all 218 clean screenshots. Fold 2 missed 197 of 216 edited enriched screenshots.
- Fold 3 (bank-progress test) and Fold 4 (recharge test) were not trained or tested.
- The validation-selected threshold moved pooled accuracy from 50.5% to 50.0%, F1 from 52.4% to 56.4%, and false positives from 233 to 281. It is a tradeoff, not proof of improvement; Fold 2 threshold selection used only 24 validation screenshots.
- Scores are uncalibrated softmax values. Neither score nor argmax is a verified probability, finding of authenticity, or fraud indicator.
- Localization is poor (IoU 0.059 across two test families); the app disables highlighted regions.
- Training used random initialization and three epochs per completed fold, CPU only. Validation behavior was unstable.
- Synthetic examples were created from a small number of renderer families. Plausible field substitutions are counterfactual labels; a single screenshot may not contain evidence that distinguishes a substituted plausible value from a clean one.
- The 387 unique real screenshots have no verified clean/edited labels. They were not used to train, tune, or evaluate the model and do not support real-world claims.
- The app is local demonstration code, not a secure public service. It defaults to loopback and does not retain uploads, but operators should only use fictional demo images and should not expose the process to a network.

## Future work

Complete held-out evaluation for all four families with more independent templates; test counterfactual-pair comparison or verified forensic cues; evaluate localization at original image resolution; investigate a pretrained encoder and conservative augmentation on training groups only; tune thresholds with larger validation groups; calibrate only on representative validation data; and obtain privacy-reviewed, verified ground truth before any real-image claim.
