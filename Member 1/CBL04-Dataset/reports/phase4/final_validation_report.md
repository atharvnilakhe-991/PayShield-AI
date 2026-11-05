# Final validation report

**Result: PASS.** All five 100-pair batches passed before the next batch began.

| Batch | Pairs | Screenshots | Validation errors | Exact duplicate screenshots | Result |
|---|---:|---:|---:|---:|---|
| 1 | 100 | 200 | 0 | 0 | Pass |
| 2 | 100 | 200 | 0 | 0 | Pass |
| 3 | 100 | 200 | 0 | 0 | Pass |
| 4 | 100 | 200 | 0 | 0 | Pass |
| 5 | 100 | 200 | 0 | 0 | Pass |
| **Total** | **500** | **1,000** | **0** | **0** | **Pass** |

For each batch, validation checked image and mask decoding and dimensions; all-black clean masks; binary edited masks exactly matching pixel-level image differences; bounding boxes tightly matching changed pixels; no changed pixels outside the intended field; clean/edited family and dimension matching; metadata field consistency; visible SYNTHETIC / DEMO markers; and exact duplicate PNG hashes. The original real images and prior pilot outputs remain separate. No model training was performed.

Individual JSON results are in `batches/`; the aggregate summary is `final_validation_report.json`.
