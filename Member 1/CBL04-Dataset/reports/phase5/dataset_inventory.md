# Phase 5 dataset inventory

## Scope and file checks

The audit traversed the extracted real archive, the original 20-pair pilot, the Phase 3 12-pair visual pilot, and the Phase 4 500-pair dataset. It decoded every image and mask through Pillow and compared recorded dimensions against actual dimensions where manifest fields are present. No files were missing or unreadable. No source image, pilot image, mask, or annotation was edited or removed.

| Dataset | Clean images | Edited images | Clean masks | Edited masks | Pairs | Image format |
|---|---:|---:|---:|---:|---:|---|
| Real archive, physical files | — | — | — | — | 387 unique contents | 394 readable JPEG files; 7 exact duplicate copies |
| Original pilot | 20 | 20 | 20 | 20 | 20 | PNG |
| Phase 3 pilot | 12 | 12 | 12 | 12 | 12 | PNG |
| Phase 4 bulk | 500 | 500 | 500 | 500 | 500 | PNG |

All 394 real files decoded. SHA-256 yielded 387 unique real image contents and 7 redundant exact copies. The prior visual classification found all 387 unique images are Google Pay dark-theme screenshots. No light theme or other payment application was added to the synthetic set.

Synthetic manifest audit results: original pilot 40 rows/20 pairs; Phase 3 24 rows/12 pairs; Phase 4 1,000 rows/500 pairs. Every Phase 4 image and mask is PNG. Phase 4 has 499 distinct width-by-height combinations across 500 pairs; observed dimensions range from 279×1261 to 738×1601. Phase 3 has seven distinct dimensions; the original pilot uses one fixed dimension.

## On-disk organization

```text
data/
  real_gpay/raw/Dataset-Hackathon/  # original archive extraction; untouched
  synthetic_gpay/                  # original 20-pair pilot
  synthetic_gpay_phase3/           # 12-pair four-family visual pilot
  synthetic_gpay_phase4/           # 500-pair bulk synthetic set
    clean/
    edited/
    masks/clean/
    masks/edited/
annotations/
  field_annotations/               # original pilot JSON
  phase3/field_annotations/        # Phase 3 pair JSON and bounding-box CSV
  phase4/field_annotations/        # 500 Phase 4 JSON records
  phase4/bounding_boxes.csv        # 707 box rows for 500 source IDs
metadata/
  manifest.csv                     # original pilot
  phase3_manifest.csv
  phase4_manifest.csv              # 1,000 rows; pair linked by source_id
  phase4_config.json
reports/phase5/                    # this audit, machine summary, contact sheets
```

The previous pilots remain in distinct directories. Phase 4 source IDs (`P4-0001` through `P4-0500`) link clean/edited files, masks, JSON annotations, bounding boxes, and manifest rows. The Phase 2 and Phase 3 source IDs are preserved as well.

Detailed counts and validation evidence: `audit_results.json`, Phase 4 `metadata/phase4_manifest.csv`, and `reports/phase4/`.
