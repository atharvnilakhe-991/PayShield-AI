# Phase 7.1: Evaluation Audit & Metric Consistency Verification

**Project:** CBL-04 — Google Pay Dark-Theme Transaction Screenshot Tampering Detection  
**Role:** Member 2 — Tile Forensics CNN & Heatmap Lead  
**Module:** `member2/phase7_1_evaluation_audit`  

---

## 1. Overview

Phase 7.1 is a dedicated audit and reconciliation module that independently verifies:
- Screenshot-level classification decision rules and mathematical metrics.
- The root cause of the minor Dice difference ($0.174612$ on 150 images vs $0.175582$ on 10 images).
- Zero data leakage and exact sample-ID preservation across splits.
- Proper validation-only provenance for $\tau$ and $A_{\min}$ thresholds.
- Actual executed test counts across `unittest discover` and test suites.
- Empirical verification of all 14 robustness conditions.

---

## 2. Audit Artifacts Directory

```
member2/phase7_1_evaluation_audit/
├── audit_report.md                   # Comprehensive Phase 7.1 Audit Report
├── classification_audit.json         # Recomputed classification metrics across 3 rules
├── test_split_consistency.json       # Test split ID matching and leakage verification
├── threshold_leakage_audit.json      # Provenance audit for tau, A_min, and area threshold
├── robustness_reconciliation.json    # Exact mathematical reconciliation of Dice values
├── test_count_verification.json      # Actual test counts across discover and submodules
├── robustness_audit.json             # In-depth analysis of robustness phenomena
├── corrected_phase7_report.md        # Audited, technically grounded Phase 7 report
├── test_phase7_1_audit.py            # Unit tests for audit logic and metrics
└── README.md                         # This guide
```

---

## 3. How to Reproduce the Audit

### Run Phase 7.1 Unit Tests
```bash
python -m unittest member2/phase7_1_evaluation_audit/test_phase7_1_audit.py -v
```

### Run Full Regression Suite (Phases 1 through 7.1)
```bash
python -m unittest discover -s member2 -p "test_*.py" -v
```

### Run Performance Improvement Tests
```bash
python -m unittest member2/performance_improvement/test_improvement_pipeline.py -v
```

