# Real dataset inventory

- Extracted image files: 394
- Technically readable files: 394
- Unique usable image content: 387
- Exact duplicate copies: 7
- Corrupt/unreadable files: 0
- Decoded formats: {'JPEG': 394}
- Original files are retained; this audit is read-only.
- Scope review completed for 387 unique images: all were visually classified as dark Google Pay transaction screens. See `real_image_classification.csv`; duplicate copies inherit their canonical image's classification.

## Common dimensions

- 637x1600: 94
- 540x1324: 27
- 624x1600: 25
- 560x1599: 18
- 584x1599: 16
- 540x1501: 12
- 628x1599: 11
- 540x1410: 11
- 437x1600: 10
- 605x1599: 9
- 540x1443: 8
- 462x1600: 8

## Scope notes

The full collection needs visual scope labeling before using it as a Google Pay-only set. Spot checks found mostly dark transaction screens, including at least one visually distinct PhonePe completion presentation; app branding alone is not reliable enough to classify all entries automatically. No entries were discarded.
