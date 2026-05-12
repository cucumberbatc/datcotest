from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_sample_image(path: Path) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.text((32, 56), "PaddleOCR warmup 123", fill="black")
    image.save(path)


def main() -> None:
    from app.config import get_settings
    from app.pdf_pipeline import _get_paddleocr_reader

    settings = get_settings()
    if not settings.ocr_enabled:
        print("OCR is disabled. Set OCR_ENABLED=true to warm up PaddleOCR.")
        return

    image_path = settings.storage_root / "ocr_warmup" / "sample.png"
    _write_sample_image(image_path)

    print("Initializing PaddleOCR. The first run may download model files...")
    reader = _get_paddleocr_reader()
    if reader is None:
        raise RuntimeError("PaddleOCR could not be initialized. Check the installed packages.")

    print("Running one warmup OCR prediction...")
    reader.predict(str(image_path))

    print("PaddleOCR warmup complete. Model files are cached for future runs.")


if __name__ == "__main__":
    main()
