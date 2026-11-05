"""Local-only demo server for the experimental Phase 7 screenshot model."""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import warnings
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT / "scripts" / "phase7"
import sys
sys.path.insert(0, str(sys_path))
from model import build_model  # noqa: E402

HEIGHT, WIDTH = 320, 160
MEAN = torch.tensor([.485, .456, .406]).view(3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(3, 1, 1)
MAX_UPLOAD = 15 * 1024 * 1024
MAX_PIXELS = 20_000_000
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
VALIDATION_THRESHOLDS = {
    "fold_1_resnet18_multitask.pt": 0.60179603099823,
    "fold_2_resnet18_multitask.pt": 0.4452926218509674,
}
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def preprocess(image: Image.Image) -> torch.Tensor:
    scale = min(WIDTH / image.width, HEIGHT / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (18, 18, 18))
    canvas.paste(resized, (0, 0))
    tensor = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).reshape(HEIGHT, WIDTH, 3)
    tensor = tensor.permute(2, 0, 1).float() / 255.0
    return ((tensor - MEAN) / STD).unsqueeze(0)


def extract_upload(body: bytes, content_type: str) -> tuple[str, bytes]:
    parsed = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii", "strict") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    if not parsed.is_multipart():
        raise ValueError("Upload must use multipart form data.")
    for part in parsed.iter_parts():
        if part.get_param("name", header="content-disposition") == "image":
            filename = Path(part.get_filename() or "upload").name
            payload = part.get_payload(decode=True) or b""
            if not payload:
                raise ValueError("The selected file is empty.")
            return filename, payload
    raise ValueError("Choose an image file before analyzing.")


def analyze(model, payload: bytes, threshold: float) -> dict:
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format not in ALLOWED_FORMATS:
                raise ValueError("Supported formats are PNG, JPEG, and WebP.")
            if source.width < 64 or source.height < 64:
                raise ValueError("Image dimensions must be at least 64 × 64 pixels.")
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Image exceeds the 20-megapixel upload limit.")
            image = source.convert("RGB")
            size = image.size
            x = preprocess(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a readable supported image.") from exc
    with torch.inference_mode():
        output = model(x)
        scores = torch.softmax(output["logits"], dim=1)[0]
        raw = float(scores[1].item())
        guess = int(raw >= threshold)
    return {
        "status": "experimental_guess",
        "model_guess": "edited" if guess == 1 else "clean",
        "edited_score_uncalibrated": round(raw, 6),
        "decision_threshold_validation_selected": round(threshold, 6),
        "score_note": "Uncalibrated model score; not a verified probability or fraud likelihood.",
        "image_width": size[0], "image_height": size[1],
        "localization_available": False,
        "localization_note": "Region highlighting is disabled because audited mask localization was unreliable (pooled IoU 0.059 on two synthetic test families).",
        "warning": "Experimental synthetic-only model. This result cannot verify a payment, prove tampering, or establish fraud.",
    }


def make_handler(model, checkpoint_name: str, threshold: float):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CBL04Demo/1.0"

        def _send(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status: int, obj: dict) -> None:
            self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/health":
                self._json(200, {"ok": True, "checkpoint": checkpoint_name, "device": "cpu"})
                return
            file_name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
            if file_name not in {"index.html", "styles.css", "app.js"}:
                self._json(404, {"error": "Not found"})
                return
            file_path = Path(__file__).parent / file_name
            self._send(200, file_path.read_bytes(), mimetypes.guess_type(file_name)[0] + "; charset=utf-8")

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/analyze":
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_UPLOAD + 1024 * 1024:
                    raise ValueError("Upload must be under 15 MB.")
                _, payload = extract_upload(self.rfile.read(length), self.headers.get("Content-Type", ""))
                if len(payload) > MAX_UPLOAD:
                    raise ValueError("Upload must be under 15 MB.")
                self._json(200, analyze(model, payload, threshold))
            except (ValueError, UnicodeError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception:
                self.log_error("Inference failed")
                self._json(500, {"error": "Analysis could not be completed. Check the image and try again."})

        def log_message(self, fmt, *args):
            # Avoid logging user filenames or request contents.
            super().log_message(fmt, *args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/phase7/fold_1_resnet18_multitask.pt")
    parser.add_argument("--host", default="127.0.0.1", help="Local-only by default; avoid exposing payment screenshots on a network.")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    if not checkpoint_path.is_relative_to(ROOT) or not checkpoint_path.is_file():
        raise SystemExit("Checkpoint must be an existing file within this project directory.")
    if checkpoint_path.name not in VALIDATION_THRESHOLDS:
        raise SystemExit("This demo has validation-selected thresholds only for the preserved Fold 1 and Fold 2 checkpoints.")
    threshold = VALIDATION_THRESHOLDS[checkpoint_path.name]
    torch.set_num_threads(2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        model = load_model(checkpoint_path)
    print(f"Loaded preserved checkpoint: {checkpoint_path}")
    print(f"Demo listening at http://{args.host}:{args.port} (local host by default)")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, str(checkpoint_path.relative_to(ROOT)), threshold))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
