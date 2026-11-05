# System architecture

## Components

```text
Browser (app/index.html, styles.css, app.js)
        │ multipart POST; local loopback only
        ▼
Python stdlib ThreadingHTTPServer (app/server.py)
        │ validates PNG/JPEG/WebP; decodes in memory; 320×160 preprocessing
        ▼
Saved ResNet-18 encoder + binary classifier + mask decoder
        │ CPU inference from preserved Phase 7 checkpoint
        ▼
JSON: raw class guess, uncalibrated edited score, limitations
```

The server defaults to `127.0.0.1:8080`; uploaded bytes are not saved. It accepts images up to 15 MB and 20 megapixels, and supports PNG, JPEG, and WebP. Model input uses the Phase 7 aspect-preserving resize, top-left placement on a 320 × 160 dark canvas, and ImageNet mean/std normalization. Maximum upload checks and decode errors return explicit messages. The UI previews locally using a browser object URL and revokes it when cleared.

## Model and result behavior

The default checkpoint is `models/phase7/fold_1_resnet18_multitask.pt`, selected by best recorded validation F1 (0.6667). Inference is CPU-only. The application reports the uncalibrated softmax score and argmax as an **experimental guess**, not as a verified probability. It always displays research limitations and explicitly prevents a guess from being treated as transaction verification. Region overlays are disabled because audited localization IoU was 0.059 pooled over two synthetic test families.

## Privacy and security boundary

The server binds to loopback unless the operator overrides `--host`. The intended demo uses fictional SYNTHETIC / DEMO screenshots only. Requests are not persisted; request logging omits filenames and content. The server is a local hackathon demo, not a hardened public service. Do not expose it to the network or upload genuine payment screenshots.

## Data lineage

The source dataset and Phase 6 manifests remain in `data/` and `metadata/`. Training/evaluation artifacts are in `models/phase7/` and `reports/phase7/`. Phase 8 source code is isolated under `app/`, with Phase 8 analysis in `reports/phase8/`. The two Phase 7 checkpoints are not overwritten.
