# CBL-04 — Google Pay Dark Screenshot Dataset

This project contains a read-only inventory of the supplied real-image archive and a reproducible pilot generator for fictional Google Pay-style dark transaction screens. Generated examples are clearly marked **SYNTHETIC / DEMO** and are not payment confirmations.

## Current dataset findings

- The supplied ZIP had **394 JPEG files**. All 394 decoded; there were no corrupt files.
- SHA-256 found **7 exact duplicate pairs** (14 files involved, 7 redundant copies), leaving **387 unique usable image contents**. Duplicates remain in `data/real_gpay/raw/`; none were discarded.
- Dimensions vary; common resolutions are listed in [the inventory report](reports/dataset_inventory.md), with per-file details in [dataset_inventory.csv](reports/dataset_inventory.csv).
- Technical usability does not establish Google Pay scope. A visual spot check found mostly dark transaction screens and at least one distinct PhonePe completion presentation. The full in-scope GPay count is not yet verified; entries are marked pending visual review.
- Planning documents that assume 500 real screenshots conflict with the provided archive. The implementation uses the observed archive only.

## Install

Python 3.12 or newer and Pillow are required. From this directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The current pilot was generated and validated with Python 3.12 and Pillow 13.0.0. The renderer uses Pillow directly so the screenshot and pixel masks share one coordinate system; it does not require a browser or Playwright.

## Commands

Audit the extracted real dataset without modifying its images:

```powershell
python scripts/audit_real_dataset.py
```

The second-stage visual labels are recorded in [real_image_classification.csv](reports/real_image_classification.csv). After manually reviewing a changed archive, update that CSV and run `python scripts/second_stage_audit.py` to compile the report. It checks that the unique filenames still match the reviewed set and will stop if images were added or removed.

Generate the default 20 matched pairs (refuses to overwrite an existing manifest or same-name image):

```powershell
python scripts/generate_pilot.py --count 20 --seed 20261103
```

Validate screenshots, masks, annotations, and pair differences:

```powershell
python scripts/validate_dataset.py
```

To create another batch or scale later, use a **new output directory** so existing data stays intact:

```powershell
python scripts/generate_pilot.py --count 100 --seed 20261104 --output .\runs\synthetic-100
python scripts/validate_dataset.py --root .\runs\synthetic-100
```

## Layout

```text
CBL04-Dataset/
├── data/
│   ├── real_gpay/raw/Dataset-Hackathon/     # Extracted originals; untouched
│   └── synthetic_gpay/
│       ├── clean/                            # Pilot clean screenshots
│       ├── edited/                           # Matched edited screenshots
│       └── masks/{clean,edited}/             # Binary PNG masks
├── annotations/
│   ├── bounding_boxes/bounding_boxes.csv
│   └── field_annotations/                    # Per-pair JSON metadata
├── metadata/
│   ├── manifest.csv
│   ├── source_group_splits.csv
│   └── pilot_config.json
├── reports/
│   ├── dataset_inventory.csv
│   ├── dataset_inventory.md
│   └── validation_report.json
├── scripts/
├── templates/
├── requirements.txt
└── README.md
```

## Generation and labels

The template uses a fixed 540 × 1600 canvas inspired by the supplied dark reference: recipient, masked phone, rupee amount, payment-again button, completion status, date/time, fictional bank card, transaction identifiers, and Google Pay-style footer. Names, account suffixes, IDs, and bank names are fictional. Every image has a visible marker at the top and again near the footer.

For each source, the same renderer makes one clean image and one image with a single controlled field replacement (amount, recipient, date/time, or UPI transaction ID). All other layout, typography, seed-derived values, and screenshot dimensions remain fixed. The edited mask is computed by comparing the final clean and edited RGB pixels and marks only pixels that differ. Bounding boxes tightly enclose changed pixels in the edited field. Clean masks are all black. This is synthetic image manipulation data and does not demonstrate real-world fraud detection performance.

The manifest contains both members of every pair and links them by `source_id`. All 20 pilot groups use the `pilot_review` split; the pair is never separated. `field_annotations/<source_id>.json` records field, original and replacement fictional value, edit method, boxes, per-pair seed, and template version.

## Validation result

The generated pilot contains 20 clean images, 20 edited images, 20 clean masks, and 20 edited masks. Validation checks decodability, dimensions, clean-mask emptiness, edited-mask binarity and exact pixel alignment to the image difference, valid boxes, one clean/edited member per source, and complete coverage of changed pixels by boxes. Current result: **passed, 40 images, 20 pairs, 0 errors**. See [validation_report.json](reports/validation_report.json).

## Adding real images later

Place additional provided images under `data/real_gpay/raw/` and run `python scripts/audit_real_dataset.py`. The audit records hashes, dimensions, formats, corruption status, and duplicate references; it does not delete, sanitize, or modify images. Keep sensitive originals access-restricted. Confirm dark-theme Google Pay scope manually before incorporating any real image into a training split.

## Phase 3 family-based visual pilot

The Phase 3 renderer adds four GPay dark-theme structures identified in the archive: `compact`, `enriched`, `bank_progress`, and `recharge`. Family counts and representative source filenames are documented in [layout_family_analysis.md](reports/layout_family_analysis.md), with one row per unique real screenshot in [layout_family_inventory.csv](reports/layout_family_inventory.csv). Phase 3 images are written separately from the original 20-pair pilot under `data/synthetic_gpay_phase3/`.

Generate the reviewed 3-pair-per-family pilot (12 matched pairs, 24 screenshots):

```powershell
python scripts/phase3/generate_phase3_pilot.py --per-family 3 --seed 20261130
```

To generate one family at a time, use an isolated root so the pilot manifest is not overwritten (the generator refuses to overwrite an existing manifest):

```powershell
python scripts/phase3/generate_phase3_pilot.py --families compact --per-family 3 --seed 20261131 --root .\runs\phase3-compact
python scripts/phase3/generate_phase3_pilot.py --families enriched --per-family 3 --seed 20261132 --root .\runs\phase3-enriched
python scripts/phase3/generate_phase3_pilot.py --families bank_progress --per-family 3 --seed 20261133 --root .\runs\phase3-bank-progress
python scripts/phase3/generate_phase3_pilot.py --families recharge --per-family 3 --seed 20261134 --root .\runs\phase3-recharge
```

Keep pilot counts small during review; this command is not authorization to generate a bulk training set. The renderer uses fictional names, services, banks, account suffixes and IDs. Every image has visible `SYNTHETIC / DEMO` markers. These are synthetic examples, not real transaction evidence.

Validate the family pilot and write its JSON summary:

```powershell
python scripts/phase3/validate_phase3.py
```

Create per-family visual review sheets (each row shows clean, edited, and edited mask):

```powershell
python scripts/phase3/create_phase3_contact_sheets.py
```

Outputs are in `reports/phase3_visual_pilot/`; per-pair field annotations and boxes are in `annotations/phase3/`, and the paired manifest is `metadata/phase3_manifest.csv`. The validator checks exact RGB difference masks, black clean masks, tight boxes, allowed field regions, matched geometry, marker presence, decodability, dimensions and family coverage. See [phase3_validation_report.md](reports/phase3_validation_report.md) and the machine-readable `phase3_validation_report.json`.

## Phase 4 bulk synthetic dataset

Phase 4 generated **500 matched pairs** (1,000 screenshots), with 500 clean masks, 500 edited masks, 500 per-pair JSON annotations and a bounding-box CSV. Outputs are separate from the real images and both earlier pilots.

To reproduce the deterministic generation (Python 3 with Pillow installed):

```powershell
python scripts/phase4/generate_bulk.py --seed 20261204 --root .\runs\phase4-repro
```

The script uses five batches of 100 pairs. It validates each batch before proceeding and stops on an error. Its default output is `data/synthetic_gpay_phase4/{clean,edited,masks/clean,masks/edited}/`; annotations are in `annotations/phase4/`, the 1,000-row pair manifest is `metadata/phase4_manifest.csv`, and per-batch manifests are under `metadata/phase4_batches/`. It refuses to overwrite an existing Phase 4 manifest. The generator uses only fictional names, account suffixes, IDs, amounts, dates and service details, with a visible SYNTHETIC / DEMO mark on every screenshot.

Family distribution follows largest-remainder proportional allocation from the 387-image reference set: compact 218, enriched 216, bank progress 54, recharge 12. The five balanced edit categories have 100 pairs each: amount, recipient/merchant, date/time, transaction ID/reference, and other (fictional bank account suffix). Recharge layouts do not expose an account suffix, so the `other` edits are assigned only to families that do.

Run the per-batch validation reports at `reports/phase4/batches/`. The final validation summary, distribution reports, and SHA-256 duplicate audit are in `reports/phase4/`. Create or refresh five-pair-per-family contact sheets with:

```powershell
python scripts/phase4/create_contact_sheets.py
```

Sheets are written to `reports/phase4/visual_audit/`. This is a synthetic dataset for development and evaluation; screenshots are not real payment evidence and no model training is part of this phase.

## Phase 5 dataset audit

Run the read-only inventory, annotation, marker, exact-hash, perceptual-similarity, and contact-sheet audit:

```powershell
python scripts/phase5/audit_dataset.py
```

The audit reads the raw real images and all three synthetic datasets without changing or deleting them. It writes aggregate results to `reports/phase5/audit_results.json` and clean/edited/mask contact sheets for all four families plus a near-similarity example sheet to `reports/phase5/contact_sheets/`. Real screenshots are intentionally excluded from these sheets because they contain sensitive payment information.

Phase 5 reports:

- `reports/phase5/dataset_inventory.md`
- `reports/phase5/annotation_audit.md`
- `reports/phase5/duplicate_similarity_audit.md`
- `reports/phase5/visual_quality_audit.md`
- `reports/phase5/privacy_audit.md`
- `reports/phase5/final_dataset_readiness.md`

The real archive contains names and financial transaction details and must remain access-controlled. No train/validation/test split or model training is part of Phase 5.

## Phase 6 leakage-safe split manifests

Phase 6 creates manifests only; it does not copy, move, generate, or alter images. Pair and group references point back to the existing Phase 4 files. Real images remain separately listed in `restricted_real_holdout` with unknown labels, and the prior pilots remain excluded.

Create and validate the Phase 6 manifests:

```powershell
python scripts/phase6/create_splits.py
python scripts/phase6/validate_splits.py
```

The fixed family-level grouping preserves all pHash groups and template families, so the requested 70/15/15 ratios are not feasible without leakage. The actual synthetic pair split is train 434 (86.8%), validation 54 (10.8%), and test 12 (2.4%). Training contains compact and enriched layouts, validation contains bank-progress layouts, and test contains recharge layouts; metrics will therefore be family-specific. Seed `20261206` and the family assignment are recorded in `metadata/phase6/phase6_config.json`.

The combined index is `metadata/phase6_manifest.csv`; split-specific manifests are in `metadata/phase6/`. The 387 unique real screenshots are listed separately and are not labeled clean, because verified tampering ground truth is absent and the images contain sensitive payment details. See `reports/phase6/` for the split strategy, distribution, leakage audit, class balance, and final validation reports. No train/validation/test files are physically duplicated.

## Phase 8 local application and handover

The `app/` directory provides a local browser demo integrated with the preserved Phase 7 model. It accepts PNG, JPEG, or WebP files up to 15 MB. Images are processed in memory and are not saved. Use fictional `SYNTHETIC / DEMO` screenshots only; do not upload real payment screenshots or expose this local demo to a network.

### Installation

The tested environment is Python 3.12.10 with CPU PyTorch 2.15.0 and torchvision 0.29.2. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv_phase7
.\.venv_phase7\Scripts\Activate.ps1
python -m pip install -r requirements-phase7.txt
```

The project already contains `.venv_phase7/` in the handover environment. Python 3.14 is not the tested runtime for this pinned Torch application. No frontend build step is required; the UI uses local HTML, CSS, and JavaScript assets.

### Start and use

```powershell
python app/server.py
```

Open `http://127.0.0.1:8080`, select a synthetic screenshot, and choose **Analyze screenshot**. The UI shows the saved model's experimental class guess and an uncalibrated edited-class score. A highlighted region is intentionally not shown because audited localization performance was poor. The server defaults to loopback and rejects unsupported, unreadable, oversized, or empty files.

To use the second preserved fold checkpoint:

```powershell
python app/server.py --checkpoint models/phase7/fold_2_resnet18_multitask.pt --port 8081
```

### Model, data, and measured performance

The application uses a ResNet-18 encoder with a classification head and mask decoder. The best available saved artifact is `models/phase7/fold_1_resnet18_multitask.pt`, selected by validation F1. It is not reliable: at the original threshold it classified every held-out compact screenshot as edited. Phase 7 completed only compact and enriched tests (two separate fold models); pooled partial baseline accuracy was 50.5%, precision 50.4%, recall 54.6%, and F1 52.4%. Folds 3 and 4 (bank-progress and recharge test families) were not run. A validation-only threshold sensitivity shifted accuracy to 50.0%, F1 to 56.4%, and false positives from 233 to 281; it is not a weight improvement and thresholds are not calibrated.

There are 387 unique real image contents, but the real screenshots remain unlabeled and do not establish clean examples or real-world performance. Phase 4 contains 500 synthetic matched pairs; earlier pilots remain in separate directories. See [Phase 7 evaluation](reports/phase7/model_evaluation_report.md), [Phase 8 limitations](reports/phase8/model_limitations.md), and [final handover](reports/phase8/final_handover.md).

### Reproducibility and presentation

Fold manifests and training configuration are under `metadata/phase7/`. The threshold-only experiment can be reproduced with `python scripts/phase7/validation_threshold_eval.py`; it reads saved checkpoints and changes no model weights. The full original training runner refuses to overwrite the existing `models/phase7/` directory. Presentation outline and demo talk track are in `reports/phase8/presentation/`.

This is a hackathon research prototype, not a payment verifier or production service. Never claim it can prove fraud or authenticity.
