# Phase 5 final dataset readiness

## Status

The dataset is **technically clean and annotation-consistent for internal review**, but its real-image component is sensitive and the synthetic set has substantial template repetition. This report does not approve public release, production use, train/validation/test splits, or model training.

## Passed checks

- 387 unique real GPay dark-theme screenshots confirmed in 394 readable JPEG files; 7 redundant exact copies were preserved.
- Phase 4 has 500 clean and 500 edited PNG screenshots, 500 clean masks, 500 edited masks, 500 annotation records, and 500 bounding-box sets (707 individual box rows).
- All five Phase 4 batches passed; 0 missing/unreadable files, 0 malformed or out-of-bounds annotations, 0 mask/box mismatches, and 0 exact synthetic screenshot duplicates.
- Markers were detected at both ends of every synthetic image; previous 20-pair and 12-pair pilots remain separate and passed the inventory checks.

## Warnings and limitations

- Treat all real payment screenshots as sensitive. Do not publish or share them outside approved internal access.
- pHash flagged 38,654 of 96,740 same-family/same-role pairs at Hamming distance ≤6. There are no exact copies, but the result and contact sheet show strong repeated-template structure.
- Layouts, typography, logos, crops and system chrome are stylized approximations. Narrow screenshots may make text small. Synthetic data cannot establish real-world fraud detection performance.
- No annotation corrections or data deletions were made. The seven real exact duplicate copies remain in raw storage.

## Organization and next gate

Real images, the original pilot, Phase 3 pilot, and Phase 4 bulk set remain in separate directories, with source IDs linking each synthetic pair across image, mask, annotation and manifest files. No train/validation/test splits were created and no model was trained. Phase 5 is complete; wait for explicit approval before Phase 6.
