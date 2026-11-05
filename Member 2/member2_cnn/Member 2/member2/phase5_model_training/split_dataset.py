"""
CBL-04: Leakage-Safe Group-Aware Dataset Splitting
Phase 5 - Member 2 (Tile Forensics CNN & Heatmap Lead)

This module partitions Google Pay screenshots into Train (70%), Validation (15%),
and Test (15%) splits while strictly preventing data leakage:
- Splitting is performed exclusively at the transaction-pair level using `split_group_id`.
- Both synthetic clean and synthetic edited screenshots of the same transaction pair
  are guaranteed to reside within the exact same split.
- Layout families ('compact', 'enriched', 'bank_progress', 'recharge') are stratified
  to preserve balanced representation across all splits.
- Unlabelled real holdout screenshots are completely segregated from supervised splits.
- Split assignments are deterministic (seed=42) and exportable to JSON and CSV.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
PHASE1_DIR = HERE.parent / "phase1_dataset_loader"
PHASE2_DIR = HERE.parent / "phase2_preprocessing"
PHASE3_DIR = HERE.parent / "phase3_tile_extraction"
PHASE4_DIR = HERE.parent / "phase4_tile_labeling"

for p in [PROJECT_ROOT, HERE, PHASE1_DIR, PHASE2_DIR, PHASE3_DIR, PHASE4_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from member2.phase1_dataset_loader.dataset_loader import DatasetLoader, PairSample, SingleSample
    from member2.phase4_tile_labeling.tile_labeler import LabeledTileMetadata, TileLabeler
except (ImportError, ModuleNotFoundError):
    from dataset_loader import DatasetLoader, PairSample, SingleSample  # type: ignore[import-not-found]
    from tile_labeler import LabeledTileMetadata, TileLabeler  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for group-aware dataset splitting."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    stratify_by_family: bool = True

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
        if any(r <= 0.0 for r in (self.train_ratio, self.val_ratio, self.test_ratio)):
            raise ValueError("All split ratios must be strictly positive")


@dataclass
class DatasetSplit:
    """Container for split assignments, sample lists, and verification statistics."""

    train_groups: List[str]
    val_groups: List[str]
    test_groups: List[str]
    train_samples: List[Any]
    val_samples: List[Any]
    test_samples: List[Any]
    holdout_samples: List[Any]
    config: SplitConfig
    summary: Dict[str, Any] = field(default_factory=dict)

    def verify_no_leakage(self) -> bool:
        """Verify that group IDs across train, val, and test splits are strictly mutually exclusive."""
        train_set = set(self.train_groups)
        val_set = set(self.val_groups)
        test_set = set(self.test_groups)

        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        if train_val_overlap or train_test_overlap or val_test_overlap:
            logger.critical(
                f"LEAKAGE DETECTED: train_val={train_val_overlap}, "
                f"train_test={train_test_overlap}, val_test={val_test_overlap}"
            )
            return False
        return True

    def get_split_name_for_group(self, group_id: str) -> Optional[str]:
        """Return the split name for a given group ID."""
        if group_id in self.train_groups:
            return "train"
        if group_id in self.val_groups:
            return "val"
        if group_id in self.test_groups:
            return "test"
        return None

    def to_manifest_rows(self) -> List[Dict[str, Any]]:
        """Generate manifest rows recording each sample's split assignment and metadata."""
        rows: List[Dict[str, Any]] = []

        def _add_samples(sample_list: List[Any], split_name: str) -> None:
            for s in sample_list:
                group_id = getattr(s, "pair_id", None) or getattr(s, "source_id", None) or f"img_{getattr(s, 'image_id', 'unknown')}"
                im_p = getattr(s, "image_path", None)
                mk_p = getattr(s, "mask_path", None)
                rows.append({
                    "split": split_name,
                    "image_id": getattr(s, "image_id", ""),
                    "source_id": getattr(s, "source_id", ""),
                    "pair_id": getattr(s, "pair_id", ""),
                    "split_group_id": group_id,
                    "is_tampered": getattr(s, "label", 0) if hasattr(s, "label") else None,
                    "layout_family": getattr(s, "layout_family", "unknown"),
                    "edit_category": getattr(s, "edit_category", None),
                    "is_real": getattr(s, "is_real", False),
                    "image_path": str(im_p) if im_p is not None else "",
                    "mask_path": str(mk_p) if mk_p is not None else "",
                })

        _add_samples(self.train_samples, "train")
        _add_samples(self.val_samples, "val")
        _add_samples(self.test_samples, "test")
        _add_samples(self.holdout_samples, "holdout")
        return rows


class GroupSplitter:
    """
    Executes leakage-safe, group-aware partitioning of Google Pay transaction screenshots.
    """

    def __init__(self, config: Optional[SplitConfig] = None) -> None:
        self.config = config or SplitConfig()

    def split_samples(self, samples: List[Any]) -> DatasetSplit:
        """
        Partition a list of SingleSample (or ProcessedSample) objects into train, val, test, and holdout.

        Guarantees:
        - Synthetic pairs sharing split_group_id remain in the same split.
        - Real holdout screenshots are segregated into holdout_samples.
        - Splits are stratified by layout_family if enabled.
        """
        synthetic_samples: List[Any] = []
        holdout_samples: List[Any] = []

        for s in samples:
            # Segregate real holdout images
            is_real = getattr(s, "is_real", False)
            mask_path = getattr(s, "mask_path", None)
            if is_real or mask_path is None or (hasattr(s, "label") and s.label is None):
                holdout_samples.append(s)
            else:
                synthetic_samples.append(s)

        # Group synthetic samples by split_group_id
        # In Phase 1/3/4, clean and edited samples of pair N share pair_id = "pair_XXXX"
        group_to_samples: Dict[str, List[Any]] = defaultdict(list)
        group_to_family: Dict[str, str] = {}

        for s in synthetic_samples:
            group_id = (
                getattr(s, "pair_id", None)
                or getattr(s, "source_id", None)
                or f"img_{getattr(s, 'image_id', 'unknown')}"
            )
            group_to_samples[group_id].append(s)
            fam = getattr(s, "layout_family", "unknown")
            if fam and fam != "unknown":
                group_to_family[group_id] = fam

        # Group partition lists
        train_groups: List[str] = []
        val_groups: List[str] = []
        test_groups: List[str] = []

        rng = random.Random(self.config.random_seed)

        if self.config.stratify_by_family:
            # Partition per layout family to guarantee balanced distribution
            family_to_groups: Dict[str, List[str]] = defaultdict(list)
            for gid in sorted(group_to_samples.keys()):
                fam = group_to_family.get(gid, "unknown")
                family_to_groups[fam].append(gid)

            for fam, gids in sorted(family_to_groups.items()):
                shuffled = list(gids)
                rng.shuffle(shuffled)
                n = len(shuffled)

                n_train = int(round(n * self.config.train_ratio))
                n_val = int(round(n * self.config.val_ratio))
                # Ensure test receives remainder to guarantee exact sum
                n_test = n - n_train - n_val

                # Safety boundary check
                if n_test < 0:
                    n_train = n - n_val
                    n_test = 0

                train_groups.extend(shuffled[:n_train])
                val_groups.extend(shuffled[n_train : n_train + n_val])
                test_groups.extend(shuffled[n_train + n_val :])
        else:
            all_groups = sorted(group_to_samples.keys())
            rng.shuffle(all_groups)
            n = len(all_groups)
            n_train = int(round(n * self.config.train_ratio))
            n_val = int(round(n * self.config.val_ratio))
            train_groups = all_groups[:n_train]
            val_groups = all_groups[n_train : n_train + n_val]
            test_groups = all_groups[n_train + n_val :]

        train_groups.sort()
        val_groups.sort()
        test_groups.sort()

        # Build sample lists
        train_samples: List[Any] = []
        for gid in train_groups:
            train_samples.extend(group_to_samples[gid])

        val_samples: List[Any] = []
        for gid in val_groups:
            val_samples.extend(group_to_samples[gid])

        test_samples: List[Any] = []
        for gid in test_groups:
            test_samples.extend(group_to_samples[gid])

        # Compute summary
        def _family_counts(samples_list: List[Any]) -> Dict[str, int]:
            counts: Dict[str, int] = defaultdict(int)
            for s in samples_list:
                counts[getattr(s, "layout_family", "unknown")] += 1
            return dict(sorted(counts.items()))

        def _label_counts(samples_list: List[Any]) -> Dict[str, int]:
            counts: Dict[str, int] = defaultdict(int)
            for s in samples_list:
                lbl = getattr(s, "label", "none")
                key = "clean" if lbl == 0 else ("tampered" if lbl == 1 else str(lbl))
                counts[key] += 1
            return dict(sorted(counts.items()))

        summary = {
            "total_samples": len(samples),
            "synthetic_samples": len(synthetic_samples),
            "holdout_samples": len(holdout_samples),
            "total_groups": len(group_to_samples),
            "train": {
                "groups": len(train_groups),
                "samples": len(train_samples),
                "labels": _label_counts(train_samples),
                "families": _family_counts(train_samples),
            },
            "val": {
                "groups": len(val_groups),
                "samples": len(val_samples),
                "labels": _label_counts(val_samples),
                "families": _family_counts(val_samples),
            },
            "test": {
                "groups": len(test_groups),
                "samples": len(test_samples),
                "labels": _label_counts(test_samples),
                "families": _family_counts(test_samples),
            },
        }

        split = DatasetSplit(
            train_groups=train_groups,
            val_groups=val_groups,
            test_groups=test_groups,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            holdout_samples=holdout_samples,
            config=self.config,
            summary=summary,
        )

        assert split.verify_no_leakage(), "Data leakage detected during dataset splitting!"
        return split

    def save_split_manifests(
        self, split: DatasetSplit, output_dir: Union[str, Path]
    ) -> Tuple[Path, Path]:
        """Save split assignment manifest as JSON and CSV in output_dir."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_path = out_path / "split_manifest.json"
        csv_path = out_path / "split_manifest.csv"

        rows = split.to_manifest_rows()

        # Save JSON
        manifest_data = {
            "config": asdict(split.config),
            "summary": split.summary,
            "samples": rows,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Save CSV
        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"Saved split manifests to {json_path} and {csv_path}")
        return json_path, csv_path
