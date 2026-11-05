# Phase 4 bulk dataset report

## Outcome

Generated 500 matched clean/edited pairs: 500 clean screenshots, 500 edited screenshots, 500 black clean masks, 500 edited masks, 500 per-pair JSON metadata records, and 500 bounding-box sets. The manifest has 1,000 image rows and links each clean/edited pair by `source_id`. Every screenshot is fictional Google Pay dark-theme-style imagery with a visible SYNTHETIC / DEMO marker. The real archive, original 20-pair pilot, and Phase 3 pilot remain separate and unchanged.

## Distribution

- Layout families: compact 218; enriched 216; bank progress 54; recharge 12.
- Edit categories: amount 100; recipient/merchant 100; date/time 100; transaction ID/reference 100; other supported field 100 (fictional bank account suffix).
- Recharge does not display a bank account suffix, so those pairs use the other four edits.

See `layout_family_distribution.json` and `edit_category_distribution.json` for the counts and source proportions.

## Validation

All five 100-pair batch reports passed with zero errors. For each batch, clean masks were all black, edited masks exactly matched clean-to-edited RGB differences, boxes matched changed regions within the intended field, pair metadata and dimensions matched, synthetic markers were found, and no duplicate screenshot hashes were found. The final audit checked 1,000 PNG screenshots and found 0 exact duplicates. See `batches/batch_01_validation.json` through `batch_05_validation.json`, `duplicate_audit.json`, and `final_validation_report.json`.

## Generation history and limits

Early validation-gated runs stopped after discovering field-zone bounds that were too narrow for some randomized scales; those runs did not proceed to later batches. Bounds were corrected and a complete fixed-seed preflight and final five-batch run passed. Intermediate artifacts are retained under `backups/phase4/`; the final dataset is only under `data/synthetic_gpay_phase4/`.

The renderer is a stylized approximation of four real layout families. It does not represent every app/device release, exact bank art, system chrome, or capture crop. The data is synthetic and is not real transaction evidence. No model training was performed.
