# Phase 3 visual pilot validation

## Scope

Validated the 12-pair Phase 3 pilot: three matched pairs for each of four Google Pay dark-theme layout families (compact, enriched, bank progress, recharge). The pilot contains 24 screenshots, 24 masks, per-pair JSON annotations, a manifest, and bounding-box CSV. The earlier 20-pair pilot remains separate and unchanged. No bulk dataset was generated.

## Automated checks

Result: **PASS — 0 errors** (`reports/phase3_validation_report.json`).

- All 24 screenshots and masks decode at the dimensions recorded in the manifest.
- All 12 clean masks are entirely black; edited masks are binary and equal the exact per-pixel RGB difference between each matched clean and edited screenshot.
- Bounding boxes exactly enclose changed pixels within the intended field; pixel changes outside the permitted field regions: 0.
- Each pair retains the same dimensions and layout family; every screenshot contains visible top and bottom SYNTHETIC / DEMO marker pixels.
- Coverage is 3 pairs each for compact, enriched, bank_progress and recharge.

## Variation and edit coverage

The pilot uses seven canvas sizes: 432×1440, 478×1599, 540×1324, 540×1478, 540×1600, 600×1600 and 637×1600. Font scales are 0.94, 1.00 and 1.06; spacing scales are 0.96, 1.00 and 1.04. Optional pay-again controls, contextual chips, and bank progress action strips vary by example. Field edits are balanced: amount 3, recipient 3, date/time 3 and transaction ID 3.

## Visual review and limits

See the four per-family clean / edited / edited-mask sheets in `reports/phase3_visual_pilot/`. The layouts now cover the principal structural groups found in the real archive and retain a consistent dark palette and clear synthetic markers. They are still stylized approximations: typography, bank marks, system status/navigation bars, exact app version details, and screenshot-specific crops do not reproduce the full range of real GPay captures. The pilot is for visual review and pipeline validation; it does not establish fraud-detection accuracy or represent real payment evidence.
