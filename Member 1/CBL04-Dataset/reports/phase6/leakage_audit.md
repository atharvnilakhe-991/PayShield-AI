# Phase 6 leakage audit

## Results

- Pair groups checked: 500; clean/edited pair leakage: **0**.
- Layout-family template groups checked: 4; template groups spanning splits: **0**.
- pHash groups at Hamming ≤6 checked: 6; near-duplicate groups spanning splits: **0**.
- Exact screenshot hashes crossing split boundaries: **0**.
- Image, mask, and annotation references missing: **0**.
- Validation errors overall: **0**.

The pHash groups are based on the Phase 5 64-bit grayscale pHash and are constructed within each layout family. Entire families are assigned to one split, so related screenshots cannot straddle the split boundary. Phase 5's pHash distance ≤6 graph connects each of the three larger families; recharge's three components are merged by its family template group.

## Limit

No family-level split can meet 70/15/15 here. The actual split has a held-out test set consisting only of recharge layouts and a validation set consisting only of bank-progress layouts. Scores on these sets will not measure performance across all four families; a future expansion should add independently designed templates before changing group integrity.
