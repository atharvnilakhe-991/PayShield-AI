"""HTTP smoke checks for the local CBL-04 demo; requires its server running."""
from __future__ import annotations

import json
import io
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8080"


def multipart(file_path: Path, boundary: str = "----CBL04SmokeBoundary") -> tuple[bytes, str]:
    payload = file_path.read_bytes()
    mime = "image/png"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{file_path.name}\"\r\n"
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def request(path: str, data: bytes | None = None, content_type: str | None = None):
    headers = {"Content-Type": content_type} if content_type else {}
    with urlopen(Request(BASE + path, data=data, headers=headers), timeout=120) as response:
        return response.status, response.headers.get_content_type(), response.read()


def multipart_content(file_name: str, payload: bytes, mime: str, boundary: str) -> tuple[bytes, str]:
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{file_name}\"\r\n"
            f"Content-Type: {mime}\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> None:
    report = {"server": BASE, "checks": [], "samples": []}
    status, _, body = request("/api/health")
    report["checks"].append({"name": "health", "status": status, "body": json.loads(body)})
    for asset in ("/", "/styles.css", "/app.js"):
        status, mime, _ = request(asset)
        report["checks"].append({"name": f"asset {asset}", "status": status, "mime": mime})

    family_samples = {}
    for family in ("compact", "enriched", "bank_progress", "recharge"):
        for label in ("clean", "edited"):
            folder = ROOT / "data/synthetic_gpay_phase3" / label
            candidates = sorted(folder.glob(f"P3-{family}-*_{label}.png"))
            if not candidates:
                raise FileNotFoundError(f"No Phase 3 {label} sample for {family}")
            family_samples[(family, label)] = candidates[0]
    for (family, label), path in family_samples.items():
        body, content_type = multipart(path)
        status, _, payload = request("/api/analyze", body, content_type)
        result = json.loads(payload)
        if result.get("status") != "experimental_guess" or result.get("localization_available") is not False:
            raise AssertionError(f"Unexpected result for {family}/{label}: {result}")
        report["samples"].append({"family": family, "label": label, "status": status,
                                  "guess": result["model_guess"],
                                  "raw_score_uncalibrated": result["edited_score_uncalibrated"],
                                  "threshold": result["decision_threshold_validation_selected"]})

    png_source = next(path for (family, label), path in family_samples.items() if label == "clean")
    with Image.open(png_source) as src:
        for fmt, mime, suffix in (("JPEG", "image/jpeg", ".jpg"), ("WEBP", "image/webp", ".webp")):
            converted = io.BytesIO()
            src.convert("RGB").save(converted, format=fmt)
            body, content_type = multipart_content("format-check" + suffix, converted.getvalue(), mime,
                                                   "----CBL04FormatBoundary")
            status, _, payload = request("/api/analyze", body, content_type)
            result = json.loads(payload)
            report["checks"].append({"name": f"{fmt} image upload", "status": status,
                                     "result": result.get("status")})

    unsupported, unsupported_type = multipart_content("wrong.bmp", b"BM-invalid", "image/bmp",
                                                     "----CBL04UnsupportedBoundary")
    try:
        request("/api/analyze", unsupported, unsupported_type)
        raise AssertionError("Unsupported BMP was unexpectedly accepted")
    except HTTPError as exc:
        report["checks"].append({"name": "unsupported format rejection", "status": exc.code,
                                 "body": json.loads(exc.read())})
        if exc.code != 400:
            raise

    bad = b"not an image"
    body = (b"------CBL04BadBoundary\r\nContent-Disposition: form-data; name=\"image\"; filename=\"bad.png\"\r\n"
            b"Content-Type: image/png\r\n\r\n" + bad + b"\r\n------CBL04BadBoundary--\r\n")
    try:
        request("/api/analyze", body, "multipart/form-data; boundary=----CBL04BadBoundary")
        raise AssertionError("Invalid image was unexpectedly accepted")
    except HTTPError as exc:
        report["checks"].append({"name": "invalid image rejection", "status": exc.code,
                                 "body": json.loads(exc.read())})
        if exc.code != 400:
            raise
    report["result"] = "PASS"
    (ROOT / "reports/phase8/application_smoke_test_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
