"""
CBL-04: Dataset Loader and Manifest Verification Module
Phase 1 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module provides a reliable, read-only dataset loader and forensic validation engine
for Google Pay dark-theme transaction screenshot tampering detection.

Key capabilities:
- Robust path resolution relative to dataset root
- Supports SingleMode (individual samples) and PairMode (clean-edited pairs)
- Comprehensive validation (missing files, duplicates, dimension mismatches, mask values)
- Lazy image and mask loading to prevent RAM exhaustion
- Group-aware split protection (preserves source/pair IDs to prevent leakage)
- Preserves GPay-only scope and flags anomalous/unlabelled entries
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)


# Supported image extensions
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class SingleSample:
    """Represents a single screenshot sample (clean, edited, or unlabelled)."""

    image_id: str
    source_id: str
    image_path: Path
    pair_id: Optional[str] = None
    mask_path: Optional[Path] = None
    label: Optional[int] = None  # 0: clean, 1: edited, None: unlabeled/unknown
    label_name: str = "unknown"  # "clean", "edited", "unlabeled", "unknown"
    is_synthetic: bool = True
    layout_family: Optional[str] = None
    edit_category: Optional[str] = None
    bounding_boxes: List[List[int]] = field(default_factory=list)
    width: Optional[int] = None
    height: Optional[int] = None
    sample_mode: str = "SingleMode"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def load_image(self, mode: str = "RGB") -> Image.Image:
        """Lazily load the screenshot PIL image."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Screenshot image not found at: {self.image_path}")
        with Image.open(self.image_path) as img:
            return img.convert(mode)

    def load_mask(self) -> Optional[Image.Image]:
        """Lazily load the manipulation mask PIL image (returns None if no mask path)."""
        if self.mask_path is None or str(self.mask_path).strip() in ("", "None"):
            return None
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask image not found at: {self.mask_path}")
        with Image.open(self.mask_path) as mask:
            return mask.convert("L")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sample metadata to dictionary."""
        return {
            "image_id": self.image_id,
            "source_id": self.source_id,
            "pair_id": self.pair_id,
            "image_path": str(self.image_path),
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "label": self.label,
            "label_name": self.label_name,
            "is_synthetic": self.is_synthetic,
            "layout_family": self.layout_family,
            "edit_category": self.edit_category,
            "bounding_boxes": self.bounding_boxes,
            "width": self.width,
            "height": self.height,
            "sample_mode": self.sample_mode,
        }


@dataclass
class PairSample:
    """Represents a matched clean-edited screenshot pair."""

    pair_id: str
    source_id: str
    clean_sample: SingleSample
    edited_sample: SingleSample
    layout_family: Optional[str] = None
    edit_category: Optional[str] = None
    bounding_boxes: List[List[int]] = field(default_factory=list)
    sample_mode: str = "PairMode"

    @property
    def clean_image_path(self) -> Path:
        return self.clean_sample.image_path

    @property
    def edited_image_path(self) -> Path:
        return self.edited_sample.image_path

    @property
    def clean_mask_path(self) -> Optional[Path]:
        return self.clean_sample.mask_path

    @property
    def edited_mask_path(self) -> Optional[Path]:
        return self.edited_sample.mask_path

    def load_clean_image(self, mode: str = "RGB") -> Image.Image:
        return self.clean_sample.load_image(mode=mode)

    def load_edited_image(self, mode: str = "RGB") -> Image.Image:
        return self.edited_sample.load_image(mode=mode)

    def load_clean_mask(self) -> Optional[Image.Image]:
        return self.clean_sample.load_mask()

    def load_edited_mask(self) -> Optional[Image.Image]:
        return self.edited_sample.load_mask()

    def load_all(self) -> Tuple[Image.Image, Image.Image, Optional[Image.Image], Optional[Image.Image]]:
        """Convenience loader for both images and masks."""
        return (
            self.load_clean_image(),
            self.load_edited_image(),
            self.load_clean_mask(),
            self.load_edited_mask(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pair metadata to dictionary."""
        return {
            "pair_id": self.pair_id,
            "source_id": self.source_id,
            "clean_image_path": str(self.clean_image_path),
            "edited_image_path": str(self.edited_image_path),
            "clean_mask_path": str(self.clean_mask_path) if self.clean_mask_path else None,
            "edited_mask_path": str(self.edited_mask_path) if self.edited_mask_path else None,
            "layout_family": self.layout_family,
            "edit_category": self.edit_category,
            "bounding_boxes": self.bounding_boxes,
            "sample_mode": self.sample_mode,
        }


@dataclass
class ValidationResult:
    """Holds complete results of dataset and manifest integrity verification."""

    manifest_path: str
    dataset_root: str
    manifest_row_count: int = 0
    unique_image_count: int = 0
    clean_sample_count: int = 0
    edited_sample_count: int = 0
    unlabeled_sample_count: int = 0
    pair_count: int = 0
    broken_pair_count: int = 0
    mask_count: int = 0
    missing_image_count: int = 0
    missing_mask_count: int = 0
    duplicate_entry_count: int = 0
    duplicate_path_count: int = 0
    corrupt_file_count: int = 0
    dimension_mismatch_count: int = 0
    unsupported_format_count: int = 0
    invalid_mask_value_count: int = 0
    scope_warning_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Returns True if there are zero fatal errors."""
        return len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "manifest_path": self.manifest_path,
            "dataset_root": self.dataset_root,
            "counts": {
                "manifest_rows": self.manifest_row_count,
                "unique_images": self.unique_image_count,
                "clean_samples": self.clean_sample_count,
                "edited_samples": self.edited_sample_count,
                "unlabeled_samples": self.unlabeled_sample_count,
                "verified_pairs": self.pair_count,
                "broken_pairs": self.broken_pair_count,
                "masks_registered": self.mask_count,
                "missing_images": self.missing_image_count,
                "missing_masks": self.missing_mask_count,
                "duplicate_entries": self.duplicate_entry_count,
                "duplicate_paths": self.duplicate_path_count,
                "corrupt_files": self.corrupt_file_count,
                "dimension_mismatches": self.dimension_mismatch_count,
                "unsupported_formats": self.unsupported_format_count,
                "invalid_mask_values": self.invalid_mask_value_count,
                "scope_warnings": self.scope_warning_count,
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }


class DatasetLoader:
    """
    Robust, read-only dataset loader for CBL-04 GPay screenshot tampering detection.

    Usage:
        loader = DatasetLoader(dataset_root="Hackathon-VIT/CBL04-Dataset")
        result = loader.validate()
        pairs = loader.load(mode="PairMode")
        samples = loader.load(mode="SingleMode")
    """

    def __init__(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        strict: bool = True,
    ) -> None:
        """
        Initialize DatasetLoader.

        Args:
            dataset_root: Path to the dataset directory (e.g. 'Hackathon-VIT/CBL04-Dataset').
                          If None, auto-detected from current and parent directories.
            manifest_path: Path to manifest CSV (defaults to 'metadata/phase4_manifest.csv').
            strict: If True, raises exceptions on fatal required field missing during parse.
        """
        self.dataset_root = self._resolve_dataset_root(dataset_root)
        self.manifest_path = self._resolve_manifest_path(manifest_path)
        self.strict = strict
        self._raw_rows: Optional[List[Dict[str, str]]] = None

    def _resolve_dataset_root(self, root: Optional[Union[str, Path]]) -> Path:
        """Safely find and resolve dataset root."""
        if root is not None:
            return Path(root).resolve()

        # Default fallback candidates when root is not provided
        cwd = Path.cwd()
        candidates = [
            cwd,
            cwd / "Hackathon-VIT" / "CBL04-Dataset",
            cwd / "CBL04-Dataset",
            Path(r"c:\Users\athar\OneDrive\Desktop\Member 2\Hackathon-VIT\CBL04-Dataset"),
        ]

        for c in candidates:
            if c.is_dir() and ((c / "data").is_dir() or (c / "metadata").is_dir()):
                return c.resolve()

        return cwd.resolve()

    def _resolve_manifest_path(self, manifest: Optional[Union[str, Path]]) -> Path:
        """Resolve manifest path relative to dataset root or absolute."""
        if manifest is not None:
            m_path = Path(manifest)
            if m_path.is_absolute() and m_path.exists():
                return m_path
            # Check relative to dataset_root
            if (self.dataset_root / m_path).exists():
                return (self.dataset_root / m_path).resolve()
            return m_path.resolve()

        # Default: metadata/phase4_manifest.csv
        default_manifest = self.dataset_root / "metadata" / "phase4_manifest.csv"
        if default_manifest.exists():
            return default_manifest.resolve()

        # Fallback to metadata/phase6_manifest.csv or manifest.csv
        for alt in ["metadata/phase6_manifest.csv", "metadata/manifest.csv"]:
            alt_path = self.dataset_root / alt
            if alt_path.exists():
                return alt_path.resolve()

        return default_manifest.resolve()

    def _resolve_file_path(self, raw_path: Optional[str]) -> Optional[Path]:
        """Resolves an image or mask path from manifest to a valid filesystem Path."""
        if not raw_path or raw_path.strip() in ("", "None", "nan"):
            return None

        clean_path = raw_path.strip().replace("\\", "/")
        path_obj = Path(clean_path)

        # 1. Direct absolute path
        if path_obj.is_absolute() and path_obj.exists():
            return path_obj

        # 2. Relative to dataset_root
        candidate = (self.dataset_root / path_obj).resolve()
        if candidate.exists():
            return candidate

        # 3. If path contains CBL04-Dataset prefix, try stripping it
        if "CBL04-Dataset/" in clean_path:
            stripped = clean_path.split("CBL04-Dataset/", 1)[1]
            candidate = (self.dataset_root / stripped).resolve()
            if candidate.exists():
                return candidate

        # Return resolved candidate path (even if missing, for validation reporting)
        return candidate

    def get_raw_manifest_rows(self) -> List[Dict[str, str]]:
        """Read and cache raw manifest CSV rows."""
        if self._raw_rows is not None:
            return self._raw_rows

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        rows: List[Dict[str, str]] = []
        with open(self.manifest_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"Manifest CSV is empty or has no header: {self.manifest_path}")
            for row in reader:
                rows.append(row)

        self._raw_rows = rows
        return rows

    def _parse_bounding_boxes(self, val: Any) -> List[List[int]]:
        """Safely parse bounding boxes from JSON string or list representation."""
        if not val or val in ("", "[]", "None", "nan"):
            return []
        if isinstance(val, list):
            return [list(map(int, b)) for b in val if isinstance(b, (list, tuple)) and len(b) == 4]
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [list(map(int, b)) for b in parsed if isinstance(b, (list, tuple)) and len(b) == 4]
        except Exception:
            pass
        return []

    def _parse_label(self, label_val: Any) -> Tuple[Optional[int], str]:
        """Normalize raw manifest label into (int_label, label_name)."""
        if label_val is None:
            return None, "unlabeled"
        s = str(label_val).strip().lower()
        if s in ("0", "clean"):
            return 0, "clean"
        if s in ("1", "edited", "tampered", "manipulated"):
            return 1, "edited"
        if s in ("unlabeled", "unknown", "-1", ""):
            return None, "unlabeled"
        return None, s

    def load(
        self,
        mode: str = "PairMode",
        filter_family: Optional[str] = None,
        filter_split: Optional[str] = None,
    ) -> List[Union[PairSample, SingleSample]]:
        """
        Load samples from the manifest.

        Args:
            mode: "PairMode" (returns List[PairSample]) or "SingleMode" (returns List[SingleSample]).
            filter_family: Optional layout family filter (e.g. "compact").
            filter_split: Optional split filter (e.g. "train", "phase4_batch_01").

        Returns:
            List of PairSample or SingleSample objects.
        """
        raw_rows = self.get_raw_manifest_rows()
        single_samples: List[SingleSample] = []

        for row in raw_rows:
            # Layout family filter
            family = row.get("layout_family") or None
            if filter_family and family != filter_family:
                continue

            # Split filter
            split = row.get("split") or None
            if filter_split and split != filter_split:
                continue

            image_id = row.get("image_id", "")
            source_id = row.get("source_id", "")
            if "pair_id" in row:
                raw_pid = row["pair_id"].strip()
                pair_id = raw_pid if raw_pid else None
            else:
                pair_id = source_id or None

            # Path resolution
            raw_img_path = row.get("image_path", "")
            img_path = self._resolve_file_path(raw_img_path)
            raw_mask_path = row.get("mask_path", "")
            mask_path = self._resolve_file_path(raw_mask_path)

            # Label normalization
            raw_label = row.get("label")
            label_int, label_name = self._parse_label(raw_label)

            # Bounding box
            raw_bbox = row.get("bounding_box") or row.get("bounding_boxes_xyxy") or ""
            boxes = self._parse_bounding_boxes(raw_bbox)

            # Synthetic check
            is_synth_str = str(row.get("is_synthetic", "True")).strip().lower()
            is_synthetic = is_synth_str not in ("false", "0", "no")

            # Dimensions
            w = int(row["width"]) if row.get("width") and str(row["width"]).isdigit() else None
            h = int(row["height"]) if row.get("height") and str(row["height"]).isdigit() else None

            sample = SingleSample(
                image_id=image_id,
                source_id=source_id,
                pair_id=pair_id,
                image_path=img_path,
                mask_path=mask_path,
                label=label_int,
                label_name=label_name,
                is_synthetic=is_synthetic,
                layout_family=family,
                edit_category=row.get("edited_field") or row.get("edit_category"),
                bounding_boxes=boxes,
                width=w,
                height=h,
                sample_mode="SingleMode",
                metadata=row,
            )
            single_samples.append(sample)

        if mode == "SingleMode":
            return single_samples

        if mode == "PairMode":
            # Group into clean and edited pairs by pair_id (or source_id)
            pair_dict: Dict[str, Dict[str, SingleSample]] = {}
            for s in single_samples:
                key = s.pair_id or s.source_id
                if not key:
                    continue
                if key not in pair_dict:
                    pair_dict[key] = {}
                if s.label_name == "clean":
                    pair_dict[key]["clean"] = s
                elif s.label_name == "edited":
                    pair_dict[key]["edited"] = s

            paired_samples: List[PairSample] = []
            for k, members in pair_dict.items():
                if "clean" in members and "edited" in members:
                    c = members["clean"]
                    e = members["edited"]
                    # Guard against accidental pairing of unrelated records
                    if c.source_id and e.source_id and c.source_id != e.source_id:
                        logger.warning(
                            f"Refusing accidental pairing of samples with mismatched source_id: "
                            f"clean={c.source_id} vs edited={e.source_id} (pair_id={k})"
                        )
                        continue
                    paired_samples.append(
                        PairSample(
                            pair_id=k,
                            source_id=c.source_id or e.source_id or k,
                            clean_sample=c,
                            edited_sample=e,
                            layout_family=c.layout_family or e.layout_family,
                            edit_category=e.edit_category,
                            bounding_boxes=e.bounding_boxes,
                            sample_mode="PairMode",
                        )
                    )
            return paired_samples

        raise ValueError(f"Unknown mode: {mode}. Must be 'PairMode' or 'SingleMode'.")

    def get_groups(self) -> Dict[str, List[str]]:
        """
        Returns group mappings (pair_id/source_id -> list of image_ids)
        to prevent clean/edited pair leakage during K-Fold or train/val/test splitting.
        """
        samples = self.load(mode="SingleMode")
        groups: Dict[str, List[str]] = {}
        for s in samples:
            grp = s.pair_id or s.source_id or s.image_id
            if grp not in groups:
                groups[grp] = []
            groups[grp].append(s.image_id)
        return groups

    def validate(
        self,
        check_image_files: bool = True,
        check_mask_values: bool = True,
        max_samples: Optional[int] = None,
    ) -> ValidationResult:
        """
        Forensically validate manifest integrity, pair relationships, and image/mask properties.
        """
        raw_rows = self.get_raw_manifest_rows()
        result = ValidationResult(
            manifest_path=str(self.manifest_path),
            dataset_root=str(self.dataset_root),
            manifest_row_count=len(raw_rows),
        )

        seen_image_ids: Set[str] = set()
        seen_image_paths: Set[str] = set()
        pair_tracker: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for idx, row in enumerate(raw_rows):
            if max_samples and idx >= max_samples:
                break

            img_id = row.get("image_id", "").strip()
            src_id = row.get("source_id", "").strip()
            pair_id = (row.get("pair_id") or src_id).strip()
            raw_img_path = (row.get("image_path") or "").strip()
            raw_mask_path = (row.get("mask_path") or "").strip()
            raw_label = row.get("label")

            # 1. Required field checks
            if not img_id:
                result.errors.append(f"Row {idx}: missing required field 'image_id'")
            if not raw_img_path:
                result.errors.append(f"Row {idx} ('{img_id}'): missing required field 'image_path'")

            # 2. Duplicate checks
            if img_id in seen_image_ids:
                result.duplicate_entry_count += 1
                result.warnings.append(f"Duplicate image_id '{img_id}' at row {idx}")
            seen_image_ids.add(img_id)

            if raw_img_path in seen_image_paths:
                result.duplicate_path_count += 1
                result.warnings.append(f"Duplicate image_path '{raw_img_path}' at row {idx}")
            seen_image_paths.add(raw_img_path)

            # 3. Label validation
            label_int, label_name = self._parse_label(raw_label)
            if label_name == "clean":
                result.clean_sample_count += 1
            elif label_name == "edited":
                result.edited_sample_count += 1
            elif label_name == "unlabeled":
                result.unlabeled_sample_count += 1
            else:
                result.errors.append(f"Row {idx} ('{img_id}'): unrecognized label '{raw_label}'")

            # 4. Scope review check (GPay project scope)
            app_name = row.get("app_name", "").lower()
            theme = row.get("theme", "").lower()
            if app_name and "google pay" not in app_name and "gpay" not in app_name:
                result.scope_warning_count += 1
                result.warnings.append(f"Scope warning: Row {idx} ('{img_id}') app_name='{row.get('app_name')}' outside GPay scope")
            if theme and theme != "dark":
                result.scope_warning_count += 1
                result.warnings.append(f"Scope warning: Row {idx} ('{img_id}') theme='{row.get('theme')}' is not dark")

            # 5. File existence and decoding check
            resolved_img = self._resolve_file_path(raw_img_path)
            img_w, img_h = None, None

            if check_image_files:
                if resolved_img is None or not resolved_img.exists():
                    result.missing_image_count += 1
                    result.errors.append(f"Missing image file: {raw_img_path} (Row {idx}, image_id='{img_id}')")
                else:
                    # Extension format check
                    ext = resolved_img.suffix.lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        result.unsupported_format_count += 1
                        result.warnings.append(f"Unsupported format '{ext}' on {resolved_img.name}")
                    # Decode check
                    try:
                        with Image.open(resolved_img) as im:
                            im.verify()
                        with Image.open(resolved_img) as im:
                            img_w, img_h = im.size
                    except Exception as exc:
                        result.corrupt_file_count += 1
                        result.errors.append(f"Corrupt image file '{resolved_img.name}': {exc}")

            # 6. Mask existence, dimension, and pixel value check
            resolved_mask = self._resolve_file_path(raw_mask_path)
            if raw_mask_path:
                result.mask_count += 1
                if check_image_files:
                    if resolved_mask is None or not resolved_mask.exists():
                        result.missing_mask_count += 1
                        result.errors.append(f"Missing mask file: {raw_mask_path} (Row {idx}, image_id='{img_id}')")
                    else:
                        try:
                            with Image.open(resolved_mask) as mim:
                                mim.verify()
                            with Image.open(resolved_mask) as mim:
                                mask_w, mask_h = mim.size
                                # Dimension check with screenshot
                                if img_w is not None and (img_w != mask_w or img_h != mask_h):
                                    result.dimension_mismatch_count += 1
                                    result.errors.append(
                                        f"Dimension mismatch between image ({img_w}x{img_h}) and mask ({mask_w}x{mask_h}) for '{img_id}'"
                                    )
                                # Mask pixel encoding check
                                if check_mask_values:
                                    extrema = mim.getextrema()
                                    if extrema is not None:
                                        min_v, max_v = extrema
                                        # Clean masks must be all-zero (0, 0)
                                        if label_name == "clean" and (min_v != 0 or max_v != 0):
                                            result.invalid_mask_value_count += 1
                                            result.warnings.append(
                                                f"Clean mask for '{img_id}' contains non-zero pixels (extrema: {extrema})"
                                            )
                                        # Edited masks should contain 255 for altered pixels
                                        if label_name == "edited" and max_v == 0:
                                            result.invalid_mask_value_count += 1
                                            result.warnings.append(
                                                f"Edited mask for '{img_id}' has no positive pixels (extrema: {extrema})"
                                            )
                        except Exception as exc:
                            result.corrupt_file_count += 1
                            result.errors.append(f"Corrupt mask file '{resolved_mask.name}': {exc}")
            else:
                # Mask path missing for an edited sample is an error
                if label_name == "edited":
                    result.missing_mask_count += 1
                    result.errors.append(f"Edited sample '{img_id}' (Row {idx}) is missing required 'mask_path'")

            # 7. Pair tracking for integrity
            if pair_id:
                if pair_id not in pair_tracker:
                    pair_tracker[pair_id] = {}
                pair_tracker[pair_id][label_name] = {
                    "image_id": img_id,
                    "row_idx": idx,
                    "img_path": resolved_img,
                    "mask_path": resolved_mask,
                    "w": img_w,
                    "h": img_h,
                }
            else:
                if label_name in ("clean", "edited"):
                    result.warnings.append(f"Sample '{img_id}' (Row {idx}) has label '{label_name}' but no pair_id/source_id")

        # 8. Pair integrity evaluation
        for pid, members in pair_tracker.items():
            has_clean = "clean" in members
            has_edited = "edited" in members

            if has_clean and has_edited:
                result.pair_count += 1
                # Cross-check dimensions between clean and edited
                c_w, c_h = members["clean"]["w"], members["clean"]["h"]
                e_w, e_h = members["edited"]["w"], members["edited"]["h"]
                if c_w and e_w and (c_w != e_w or c_h != e_h):
                    result.dimension_mismatch_count += 1
                    result.errors.append(
                        f"Pair '{pid}' dimensions mismatch: clean={c_w}x{c_h} vs edited={e_w}x{e_h}"
                    )
            elif has_clean and not has_edited:
                result.broken_pair_count += 1
                result.warnings.append(f"Orphan clean sample in pair '{pid}' (image_id='{members['clean']['image_id']}')")
            elif has_edited and not has_clean:
                result.broken_pair_count += 1
                result.warnings.append(f"Orphan edited sample in pair '{pid}' (image_id='{members['edited']['image_id']}')")

        result.unique_image_count = len(seen_image_ids)
        return result
