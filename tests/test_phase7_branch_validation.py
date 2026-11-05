"""
Tests for Phase 7 Branch Validation and Out-Of-Fold (OOF) Data Preparation.
Validates manifest schema, pair isolation, layout family rotation, test-row exclusion,
checkpoint hash consistency, and zero fabricated predictions.
"""

import csv
import hashlib
import json
from pathlib import Path
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = WORKSPACE_ROOT / "Hackathon-VIT" / "CBL04-Dataset"
META_DIR = DATASET_ROOT / "metadata" / "phase7_branch_validation"
REP_DIR = DATASET_ROOT / "reports" / "phase7_branch_validation"


class TestPhase7ManifestIntegrity:
    """Validates structural integrity of generated Phase 7 manifests."""

    def test_manifest_files_exist(self):
        assert (META_DIR / "branch_output_manifest.csv").exists()
        assert (META_DIR / "valid_oof_training_manifest.csv").exists()
        assert (META_DIR / "validation_manifest.csv").exists()
        assert (META_DIR / "untouched_test_manifest.csv").exists()
        assert (META_DIR / "missing_predictions_manifest.csv").exists()
        assert (REP_DIR / "fold_checkpoint_provenance.json").exists()
        assert (REP_DIR / "leakage_audit_report.md").exists()
        assert (REP_DIR / "branch_evaluation_metrics.json").exists()

    def test_unique_sample_ids_in_branch_manifest(self):
        with open(META_DIR / "branch_output_manifest.csv") as fp:
            rows = list(csv.DictReader(fp))
            assert len(rows) == 1000
            sids = [r["sample_id"] for r in rows]
            assert len(sids) == len(set(sids))

    def test_no_test_rows_in_oof_training_manifest(self):
        """CRITICAL: held-out test rows must NEVER appear in the OOF training manifest."""
        with open(META_DIR / "untouched_test_manifest.csv") as fp:
            test_rows = list(csv.DictReader(fp))
            test_sids = set(r["sample_id"] for r in test_rows)
            test_pids = set(r["pair_id"] for r in test_rows)

        with open(META_DIR / "valid_oof_training_manifest.csv") as fp:
            oof_rows = list(csv.DictReader(fp))
            oof_sids = set(r["sample_id"] for r in oof_rows)
            oof_pids = set(r["pair_id"] for r in oof_rows)

        assert len(test_sids) == 24
        assert len(oof_sids) == 868
        overlap_samples = test_sids.intersection(oof_sids)
        overlap_pairs = test_pids.intersection(oof_pids)
        assert len(overlap_samples) == 0, f"Found leaked test samples in OOF training manifest: {overlap_samples}"
        assert len(overlap_pairs) == 0, f"Found leaked test pairs in OOF training manifest: {overlap_pairs}"

    def test_pair_isolation_in_splits(self):
        """Verifies clean and edited counterparts stay in the same split partition."""
        with open(META_DIR / "branch_output_manifest.csv") as fp:
            rows = list(csv.DictReader(fp))

        pair_splits = {}
        for r in rows:
            pid = r["pair_id"]
            sp = r["split"]
            if pid not in pair_splits:
                pair_splits[pid] = set()
            pair_splits[pid].add(sp)

        # Each pair must belong to exactly one split
        for pid, splits in pair_splits.items():
            assert len(splits) == 1, f"Pair {pid} straddles multiple splits: {splits}"

    def test_family_isolation_in_splits(self):
        """Phase 6 splits must isolate layout families as documented."""
        with open(META_DIR / "branch_output_manifest.csv") as fp:
            rows = list(csv.DictReader(fp))

        split_fams = {}
        for r in rows:
            sp = r["split"]
            fam = r["layout_family"]
            if sp not in split_fams:
                split_fams[sp] = set()
            split_fams[sp].add(fam)

        assert split_fams["test"] == {"recharge"}
        assert split_fams["validation"] == {"bank_progress"}
        assert split_fams["train"] == {"compact", "enriched"}

    def test_oof_provenance_exclusion(self):
        """Every prediction in valid_oof_training_manifest must come from an un-trained fold model."""
        with open(META_DIR / "valid_oof_training_manifest.csv") as fp:
            rows = list(csv.DictReader(fp))

        with open(REP_DIR / "fold_checkpoint_provenance.json") as fp:
            cp_info = json.load(fp)

        for r in rows:
            fam = r["layout_family"]
            cp_used = Path(r["cnn_checkpoint_used"]).name
            meta = cp_info[cp_used]

            # Model must NOT have been trained on this sample's layout family
            assert fam not in meta["train_layout_families"], (
                f"LEAKAGE: Sample {r['sample_id']} (family {fam}) predicted by {cp_used} "
                f"which trained on {meta['train_layout_families']}"
            )
            assert r["cnn_oof_status"] == "valid_oof"

    def test_missing_predictions_manifest_correctness(self):
        """Missing predictions manifest must strictly catalog the 132 samples from Folds 3 and 4."""
        with open(META_DIR / "missing_predictions_manifest.csv") as fp:
            rows = list(csv.DictReader(fp))

        assert len(rows) == 132
        families = set(r["layout_family"] for r in rows)
        assert families == {"bank_progress", "recharge"}
        for r in rows:
            assert r["missing_component"] == "CNN_prediction"

    def test_checkpoint_hash_consistency(self):
        """Validates that baseline checkpoint hashes match physical disk files."""
        with open(REP_DIR / "fold_checkpoint_provenance.json") as fp:
            cp_info = json.load(fp)

        fold1_p = DATASET_ROOT / "models" / "phase7" / "fold_1_resnet18_multitask.pt"
        fold2_p = DATASET_ROOT / "models" / "phase7" / "fold_2_resnet18_multitask.pt"

        def get_hash(p):
            h = hashlib.sha256()
            with open(p, "rb") as fp:
                while c := fp.read(8192):
                    h.update(c)
            return h.hexdigest()

        assert get_hash(fold1_p) == cp_info["fold_1_resnet18_multitask.pt"]["sha256"]
        assert get_hash(fold2_p) == cp_info["fold_2_resnet18_multitask.pt"]["sha256"]

    def test_branch_metrics_contain_no_stacked_score(self):
        """Verifies that Phase 7 metrics report each branch independently without ensemble fusion."""
        with open(REP_DIR / "branch_evaluation_metrics.json") as fp:
            bm = json.load(fp)

        assert "cnn_branch" in bm
        assert "ocr_branch" in bm
        assert "semantic_branch" in bm
        assert bm["branch_independence"]["branches_combined"] is False
        assert bm["branch_independence"]["stacker_implemented"] is False

