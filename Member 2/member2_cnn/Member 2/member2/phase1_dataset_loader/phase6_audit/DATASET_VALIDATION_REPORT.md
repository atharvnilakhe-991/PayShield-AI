# CBL-04: Phase 1 Dataset & Manifest Validation Report

**Target Manifest:** `phase6_manifest.csv`  
**Dataset Root:** `C:\Users\athar\OneDrive\Desktop\Member 2\Hackathon-VIT\CBL04-Dataset`  
**Validation Status:** ✅ **VALIDATION PASSED**  
**Fatal Errors:** 0  
**Warnings:** 0  

---

## 1. Verified Dataset & Manifest Counts

| Metric | Count | Description / Notes |
|---|---:|---|
| **Manifest Row Count** | 1387 | Total rows loaded from manifest CSV |
| **Unique Image Count** | 1387 | Distinct `image_id` entries registered |
| **Clean Samples** | 500 | Label 0 / 'clean' screenshots |
| **Edited Samples** | 500 | Label 1 / 'edited' screenshots |
| **Unlabelled Samples** | 387 | Raw real screenshots without ground truth labels |
| **Verified Clean-Edited Pairs** | 500 | Complete pairs with matched clean + edited members |
| **Broken / Orphan Pairs** | 0 | Pairs with missing clean or edited counterpart |
| **Masks Registered** | 1000 | Total mask entries referenced in manifest |
| **Missing Images** | 0 | Image file paths that do not exist on disk |
| **Missing Masks** | 0 | Required mask files that do not exist on disk |
| **Duplicate Entries** | 0 | Repeated `image_id` occurrences |
| **Duplicate Image Paths** | 0 | Repeated filesystem paths in manifest |
| **Corrupt / Unreadable Files** | 0 | Files failing PIL header / decode check |
| **Dimension Mismatches** | 0 | Discrepancies between image and mask dimensions |
| **Unsupported Formats** | 0 | Files not in PNG, JPG, JPEG, WEBP |
| **Invalid Mask Pixel Values** | 0 | Clean mask non-zero or edited mask all-zero |
| **Scope Warnings** | 0 | Non-GPay or non-dark-theme flags |

---

## 2. PairMode and SingleMode Interpretation

- **SingleMode:** Evaluates every screenshot individually. Label `0` maps to clean, `1` maps to edited, and unlabelled real screenshots remain `None` (`unlabeled`) without artificial inference.
- **PairMode:** Matches clean and edited screenshots using the shared `source_id` / `pair_id`. Both members of the pair, along with their respective masks and bounding boxes, are linked together in a single `PairSample` record.
- **Leakage-Safe Grouping:** Identifiers are exposed via `loader.get_groups()` so that subsequent splitting phases can perform group-aware partitioning without separating clean and edited counterparts of the same transaction.

---

## 3. Scope Review (Google Pay Dark-Theme Boundary)

- All synthetic screenshots in Phase 4 adhere to Google Pay dark-theme transaction specifications (`theme: dark`, `app_name: Google Pay style (synthetic)`).
- If evaluating Phase 6 (including real screenshots), 387 unique real images are held out in `restricted_real_holdout`. All 387 were visually classified by Member 1 as dark Google Pay transaction screens, with 0 confirmed PhonePe screens retained.

---

## 4. Errors and Warnings Log

### ✅ Fatal Errors: None (0)

### ⚠️ Warnings: None (0)

---

## 5. Verification Sign-off

- **Auditor / Engineer:** Member 2 (Tile Forensics CNN Lead)
- **Result:** Manifest, image paths, masks, bounding boxes, and pair linkages are verified intact.
- **Status for Phase 2 (Tile Extraction):** **UNBLOCKED & READY TO PROCEED**