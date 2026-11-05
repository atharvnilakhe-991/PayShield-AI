# CBL-04 hackathon presentation outline

Use this as a concise slide deck outline. All product language should describe a research demo, not a fraud detector.

## Slide 1 — The problem

UPI payment screenshots are easy to forward and can be altered. A screenshot alone is not a trusted payment record; the real source of truth is the payment provider/bank transaction record.

## Slide 2 — Our research question

Can a computer vision model locate deliberately changed transaction fields in Google Pay-style dark-theme screenshots? Our scope is a synthetic research dataset and an experimental classifier/localizer.

## Slide 3 — Data inventory and privacy

The supplied archive contained 387 unique image contents after exact-duplicate checks. Real screenshots remain restricted, unlabeled, and out of supervised evaluation. They are not presented in the demo.

## Slide 4 — Synthetic dataset

500 matched clean/edited pairs; fictional values; four layout families: compact, merchant/context enriched, bank-progress timeline, recharge/utility detail. Clean/edited pairs share the same rendering pipeline and carry masks, boxes, and metadata. Every synthetic image has a visible DEMO marker.

## Slide 5 — Annotations and split design

Clean masks are empty; edited masks identify changed pixels. Pair and perceptual/template groups stay together. Four whole-family holdouts were planned to test layout transfer rather than random screenshot memorization.

## Slide 6 — Model

ResNet-18 encoder with a binary classification head and light mask decoder. CPU-only, random initialization, three epochs for each of two completed folds. The app runs a preserved checkpoint locally.

## Slide 7 — Results (show honestly)

Completed test families: compact and enriched only. At threshold 0.5, pooled two-model accuracy 50.5%, precision 50.4%, recall 54.6%, F1 52.4%. Compact checkpoint called every sample edited. Enriched checkpoint missed 197/216 edited samples. Bank-progress and recharge held-out tests remain incomplete.

## Slide 8 — Threshold sensitivity

Validation-selected thresholds raised pooled F1 to 56.4% but dropped accuracy to 50.0% and raised false positives from 233 to 281. The thresholds are not calibrated and Fold 2 validation had only 24 images. No weights improved.

## Slide 9 — Application demo

Run the app locally with one clearly marked synthetic example. Show upload, image preview, raw uncalibrated score and experimental guess. Show that region highlighting is disabled because localization IoU is 0.059.

## Slide 10 — Limitations and next steps

No verified real-world performance. More independent templates, completed family folds, paired-reference modeling or validated forensic cues, stronger training, larger validation groups, and real verified ground truth are needed.

## Closing statement

“This is a synthetic research prototype for studying screenshot edits. It cannot verify a payment, prove fraud, or replace a bank or payment-provider record.”
