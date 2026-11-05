# Duplicate and perceptual similarity audit

## Exact duplicates (SHA-256)

- Real archive: 394 readable JPEG files, 387 unique contents, 7 duplicate copies across 7 exact-copy groups.
- Original pilot: 0 exact duplicate screenshots.
- Phase 3 pilot: 0 exact duplicate screenshots.
- Phase 4: 0 exact duplicate screenshots among 1,000 PNGs.

No duplicate files were deleted or altered. The seven redundant real files remain preserved in the original archive tree.

## Perceptual similarity (Phase 4)

Compared clean images against clean images and edited against edited images within each family, using a 64-bit grayscale pHash (32×32 resize, low-frequency 8×8 DCT). Across 96,740 same-family/same-role comparisons:

- 7,456 pairs had Hamming distance ≤3.
- 38,654 pairs had Hamming distance ≤6 (about 40.0%).
- 1,145 pairs had identical pHash values (distance 0), while exact SHA-256 duplicates remained zero.

These are similarity flags, not duplicate claims. The hash is intentionally coarse and responds strongly to the common screen skeleton. The contact sheet `contact_sheets/near_similarity_examples.png` shows the closest clean-image pairs per family; their visible values and layouts differ while retaining the same family template. This confirms substantial template repetition and limits effective visual diversity. It does not show byte-identical or pixel-identical screenshots.

The metric is a screening tool rather than a human-quality judgment. Images across different families, and clean-versus-edited cross-role pairs, were not counted in the near-duplicate denominator.
