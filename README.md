# PayShield-AI

## Google Pay (GPay) Payment Screenshot Manipulation Detection

**Revision:** Google Pay–only scope + dataset plan (November 2026)

## 1. Project Scope

CBL-04 will identify manipulated or edited **Google Pay payment screenshots only** in the current hackathon implementation. PhonePe, BHIM, Paytm and other payment applications are explicitly out of scope for this release.

The system will assess whether a submitted GPay screenshot shows signs of image manipulation. It is a screenshot-forensics aid; it does **not** verify a transaction with a bank, UPI network or Google Pay, and a screenshot alone cannot prove that funds were settled.

The design should keep app-specific parsing/layout configuration modular so that future app support can be added without changing the core image-forensics model unnecessarily.

## 2. Dataset Plan — 1,500 Images Total

The team has decided to collect **500 genuine Google Pay receipt screenshots** and generate **1,000 synthetic images**.

### 2.1 Proposed composition of the 1,000 synthetic images

To create source-matched examples and prevent the model from learning “synthetic versus real” instead of “edited versus unedited,” generate:

- **500 synthetic clean GPay-style demo receipts**
- **500 manipulated counterparts** derived from those synthetic clean sources

This gives 500 synthetic clean/edited pairs. Combined with the 500 collected genuine GPay screenshots, the dataset contains 1,500 images:
- 500 authentic collected genuine screenshots (label genuine)
- 500 synthetic clean screenshots (label genuine/clean for the task)
- 500 synthetic manipulated screenshots (label edited)

This produces 1,000 clean/genuine-labeled images and 500 edited-labeled images. Training must address this class imbalance through class weights or carefully documented sampling. Do not silently duplicate samples to claim a larger dataset.

**Dataset caveat:** the collected authentic samples are genuine GPay screenshots, while the synthetic clean/edited samples are controlled mockups. Their provenance must be represented in metadata. The synthetic mockups should use a generic, fictionalized GPay-inspired demo design, not create reusable fraudulent proof-of-payment documents, and must carry a clear internal demo indicator where appropriate. Avoid real names, UPI IDs, account numbers, QR codes, transaction reference identifiers, phone numbers or personal data in synthetic examples.

### 2.2 Genuine image collection and privacy

Collect 500 genuine GPay receipt screenshots only with participant consent and permission to use them for research/hackathon model training. Before storing:
- redact personal names, phone numbers, UPI IDs, account identifiers, balances, QR codes and other sensitive information;
- keep an access-controlled original only if participants explicitly consent and retention is needed;
- store a sanitized copy for training;
- record source provenance and consent status;
- remove identifying metadata where appropriate;
- ensure the collection does not include screenshots obtained without authorization.

Do not represent screenshots as proof of settlement or label them genuine based only on visual appearance. Where possible, participant-provided context should support the genuine label, while noting that the project is not performing bank-side verification.

## 3. Dataset Schema and Grouping

Each record must include:
- `image_id`
- `source_id` (shared by a synthetic clean image and its manipulated counterpart)
- `app_name = "Google Pay"`
- `label` (`0 = clean/genuine appearance`, `1 = edited/manipulated`)
- `provenance` (`collected_real` or `synthetic`)
- `edit_field` (`amount`, `payee`, `status`, `date`, `time`, `reference`, or `none`)
- `edit_method`
- `mask_path` (blank/zero mask for clean samples; region mask for edited samples)
- `bbox` (all altered regions)
- `ocr_fields` / field annotations
- `consent_status` for collected real samples
- `split` (`train`, `validation`, `test`)
- `seed` and generator/template version for synthetic data

Split by **source group**, not by individual image. A synthetic clean image and its manipulated counterpart must never be assigned to different splits. Keep a separate untouched real-world evaluation subset if sufficient consented data can be collected; do not use it for training, threshold tuning or feature selection.

## 4. Synthetic Generation

Do not use AI image-generation tools as the data generator. Render controlled fictional GPay-inspired demo receipt templates with HTML/CSS and Playwright/Chromium; use Python, Pillow and OpenCV for edits, masks and image transforms.

Create multiple variations within the GPay-only layout scope (different screen scale, content, fonts only when naturally part of template variants, field values and screenshot framing). Use matched rendering pipeline for clean and edited counterparts. Avoid an artificial class shortcut where every edited image has different fonts, colors, compression or a conspicuous watermark.

Manipulation categories:
- text-region replacement;
- patch compositing;
- partial-character or local-region edits;
- optional controlled copy-move edits if implementation time permits.

Primary altered fields: amount and payee. Additional controlled fields: status, date/time and reference text. Ground-truth masks must mark all modified regions and annotations must record exact edit type. Do not claim synthetic manipulation fully represents real-world fraud.

## 5. System Architecture

### Pipeline
1. Screenshot upload and file validation
2. Image normalization (orientation, dimensions, color channel handling)
3. **Branch A — Tile Forensics CNN:** local image evidence, tile-level predictions and manipulation heatmap
4. **Branch B — OCR + Semantic Features:** Google Pay-specific field extraction and consistency signals
5. **OOF Stacker:** combine out-of-fold branch outputs/features to obtain final calibrated score
6. Decision layer: `Genuine`, `Suspicious`, `Edited`
7. Explainability output: probability/score, heatmap, affected fields and concise reasons

### Branch A — Tile Forensics CNN
- Train on synthetic clean/edited source pairs and collected genuine clean images.
- Use mask supervision/localization where masks exist; collected real images may have no edit masks and must be handled accordingly (classification-only or weak supervision, not fabricated masks).
- Use source-grouped splits and tile-level labels derived from masks.
- Produce a global image score and heatmap/localized evidence.
- Avoid reporting tile accuracy as screenshot-level accuracy.

### Branch B — OCR + Semantic Features
Use EasyOCR or the chosen implementation from the integration plan. Extract only the GPay fields needed for this scope:
- amount
- payee/recipient
- payment status
- date/time
- transaction/reference text where visible

Features include OCR confidence, field geometry, text-layout anomalies, field presence, and explicitly designed semantic consistency checks. A visible status label is not proof of settlement. Reference/date consistency must only be used if the format has been empirically validated; do not assume all GPay reference identifiers encode dates.

### Stacker and decision layer
- Train the stacker using out-of-fold (OOF) branch outputs to avoid leakage.
- Tune decision thresholds on validation data only.
- Preserve three verdict bands (`Genuine`, `Suspicious`, `Edited`) with the exact thresholds recorded in config.
- Never use test-set results to choose thresholds.

## 6. Evaluation and Validation

Report metrics separately for:
1. collected real GPay screenshots;
2. synthetic clean vs synthetic edited examples;
3. combined dataset;
4. held-out real evaluation samples, if available.

Required metrics:
- ROC-AUC and PR-AUC
- precision, recall, F1
- confusion matrix
- false-positive rate on genuine real GPay receipts
- false-negative rate on manipulated examples
- screenshot-level localization IoU/Dice for samples with masks
- inference latency (target approximately 1 second on CPU, as set by the project plan)

Run:
- source-grouped train/validation/test split;
- class imbalance handling;
- compression/resize robustness tests;
- shortcut probes (provenance-only, template-only, background-only, metadata-only);
- ablation of Tile CNN, OCR/semantic branch and stacker;
- paired-source leakage check;
- manual review of false positives and false negatives.

**Do not claim performance on AI-generated screenshots, other payment apps or bank-verified transactions unless those cases are independently tested.**

## 7. Data and Model Deliverables

- `dataset/real_gpay/` — sanitized, consented collected screenshots and provenance records
- `dataset/synthetic_gpay/clean/` — 500 clean mockup images
- `dataset/synthetic_gpay/edited/` — 500 edited counterparts
- `dataset/annotations/manifest.csv`
- `dataset/annotations/fields/`
- `dataset/annotations/masks/`
- `dataset/splits/source_group_splits.csv`
- synthetic generator and config/version
- trained Tile CNN weights and model card
- OCR/semantic feature schema
- OOF stacker and thresholds config
- evaluation report and error-analysis samples
- demo interface

## 8. Explicit Out-of-Scope Items for Current Hackathon

- PhonePe, BHIM, Paytm and other app screenshot support
- bank/UPI API verification or proof of settlement
- full detection of completely AI-generated receipt images
- production financial fraud decisions
- claiming generalization beyond Google Pay screenshots without testing

## 9. Milestone Order

1. Confirm GPay-only scope, schema and dataset consent/provenance rules.
2. Obtain and sanitize the 500 genuine images.
3. Build the synthetic renderer and generate 500 clean + 500 edited images with masks.
4. Freeze source-group splits and validate data integrity.
5. Implement and train Tile CNN.
6. Implement Google Pay OCR/semantic feature branch.
7. Generate OOF features and train stacker.
8. Integrate verdict and explainability UI.
9. Run metrics, shortcut probes, robustness tests and error analysis.
10. Freeze demo build and report limitations accurately.
