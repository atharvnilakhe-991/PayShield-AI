# Application testing report

## Test scope and status

The application is a new local demo because no frontend/backend existed in the CBL-04 dataset repository. It serves a browser interface from Python's standard-library HTTP server and loads the preserved Phase 7 model on CPU.

| Check | Status | Evidence |
|---|---|---|
| Application startup and health endpoint | Passed | Local server on `127.0.0.1:8080`; health returned checkpoint and CPU device |
| Browser page and local asset loading | Passed | HTML/CSS/JS returned HTTP 200; in-app browser rendered preview and result state |
| PNG/JPEG/WebP upload and inference | Passed | All three formats returned a result; 8 PNG smoke cases covered clean and edited samples from all four families |
| Invalid/corrupt and unsupported format rejection | Passed | Corrupt PNG and unsupported BMP each returned JSON 400 |
| Error response behavior | Passed for invalid input | Meaningful 400 JSON; generic 500 path is implemented but not forced in testing |
| Upload persistence | Designed not to persist | Request body remains in memory; no upload output path |
| Confidence presentation | Passed in browser | Raw score labeled uncalibrated and validation threshold displayed separately |
| Localization display | Passed in browser | Highlighting disabled with measured-IoU explanation |
| Responsive layout | CSS implemented; viewport-specific visual test not run | CSS breakpoints at 800px and 480px |

All tests used fictional Phase 3 SYNTHETIC / DEMO pilot examples. They are interface smoke tests, not new model evaluation and not part of the reported held-out metrics. No test-set image was used to tune the model.

## Runtime results

`python -m py_compile app/server.py scripts/phase7/validation_threshold_eval.py scripts/phase8/smoke_test_app.py` completed without syntax errors. The local server loaded the preserved Fold 1 checkpoint and served its page assets. The browser accepted a Phase 3 enriched clean sample, previewed it, ran inference, and displayed the experimental output, uncalibrated score, validation-selected threshold, localization warning, and safety message.

The API smoke runner at `scripts/phase8/smoke_test_app.py` completed with PASS; machine-readable results are in `application_smoke_test_results.json`. It submitted clean and edited pilot examples from each of four families, plus JPEG and WebP conversions. All eight family/label predictions were **edited**, including the four clean images. This small demo-only check confirms severe false-positive behavior persists across these pilot examples; it does not replace Phase 7 held-out metrics. Corrupt PNG and BMP files were rejected with HTTP 400. Size-limit and decompression-bomb boundary cases were not exercised. No uploads were persisted.
