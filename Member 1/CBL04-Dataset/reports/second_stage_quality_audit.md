# CBL-04 second-stage quality audit

## Real dataset

All 387 unique usable images were visually reviewed via 33 contact sheets, then spot-checked at full resolution for variant layouts. The classification CSV contains one row per unique image. Results: **387 Google Pay dark / 0 Google Pay light / 0 other UPI app / 0 uncertain / 0 corrupt**. Seven redundant duplicate copies are omitted from this unique-image classification but retained in the raw archive; there are no unreadable files.

The reviewed set includes standard person-to-person receipts, merchant payment details, bank-specific payment-progress timelines, and recharge/utility-style transaction-detail pages. These variants visibly carry Google Pay branding and Google transaction details, so they were classified as dark Google Pay screens. The dataset is broader than one exact receipt layout.

## Synthetic visual review

The screens are recognizable as dark payment-completion/details pages and carry visible SYNTHETIC / DEMO marks. They are acceptable as a **pilot for pipeline and annotation review**, but are not visually diverse or faithful enough for bulk model training yet. The renderer fixes every screenshot at 540 × 1600 with one layout, one font family, fixed field positions, and simplified generic bank/UPI branding. It does not reproduce the different real GPay layout variants observed in the archive.

The 10-pair visual contact sheet is supplied as five PNG pages: `quality_audit/synthetic_visual_audit_01.png` through `_05.png`.

## Mask and box audit

- Clean masks: 20/20 entirely black.
- Edited masks: 20/20 exactly equal the pixelwise RGB difference between their matched clean/edited images.
- Bounding boxes: every changed pixel is covered; box coordinates are within image dimensions.
- Unrelated UI: no changed pixels outside the named edited field regions were found. Recipient edits include its avatar initial and both displayed name instances; these are part of the recipient field.
- Validation: 40 images, 20 matched pairs, 0 errors; passed = True.

## Edit diversity and layout variation

- Amount: 5
- Recipient name: 5
- Date/time: 5
- Transaction ID: 5
- Other fields: 0

Edit locations span four UI areas: amount near the top, recipient at the avatar/name and card recipient row, date/time in the upper summary, and transaction ID in the details card. Field anchors are fixed across pairs; there is no layout or screen-dimension variation. The 20-pair pilot has balanced field counts but insufficient visual variation for scale.

## Recommendation before bulk generation

Keep the 20-pair pilot as the current review set. Before scaling, add controlled layout and dimension variants based on the real dark GPay families, preserve one renderer-defined exact mask per altered field, and render merchant/recharge/bank-progress variants as distinct template families or explicitly exclude them. Review a revised small batch and audit it again before bulk generation. The pilot does **not** establish real-world fraud-detection performance.
