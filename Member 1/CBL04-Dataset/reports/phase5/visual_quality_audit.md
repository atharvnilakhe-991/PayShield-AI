# Phase 5 visual quality audit

## Review material

`contact_sheets/compact.png`, `enriched.png`, `bank_progress.png`, and `recharge.png` each show eight Phase 4 clean/edited/mask examples. `contact_sheets/near_similarity_examples.png` shows the closest pHash clean-image pair for each family. No real screenshot or personal data is included in these sheets.

## Findings

- All four requested layout families are present. Dark GPay-style colors, fictional content, and top/bottom SYNTHETIC / DEMO markers remain consistent. The marker was programmatically found in all 1,000 Phase 4 images and in previous pilot images.
- Dimensions span 499 distinct configurations across 500 pairs (279–738 px wide, 1261–1601 px high). Font, spacing, margin, and optional chips/action controls also vary.
- Visual variation is moderate overall but limited in structural terms. Each family reuses one underlying renderer and mostly fixed vertical ordering/field positions. The perceptual scan found roughly 40% of same-family/same-role comparisons within pHash Hamming distance 6; this reflects template similarity, not exact copies.
- The templates remain stylized: common text hierarchy and geometry recur; bank icons and Google Pay marks are simplified; system status/navigation chrome, capture crops, and app-version details are not modeled.
- Some narrow screenshots render small text. The synthetic marker remains present, but text scale/readability should receive human review before using these images as model input.
- Masks contain only actual changed pixels. Amount examples in recharge layouts can alter the displayed total and plan-price rows; those separate edited regions are expected and covered by the annotation boxes.

No visual artifacts were corrected during this audit. The contact sheets are supplied for review; the repetitiveness and stylization are limitations to consider before model training.
