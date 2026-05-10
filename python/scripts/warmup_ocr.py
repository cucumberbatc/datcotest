from __future__ import annotations

from pathlib import Path


def _write_sample_image(path: Path) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.text((32, 56), "EasyOCR warmup 123", fill="black")
    image.save(path)


def main() -> None:
    from app.config import get_settings
    from app.pdf_pipeline import _get_easyocr_reader

    settings = get_settings()
    if not settings.ocr_enabled:
        print("OCR is disabled. Set OCR_ENABLED=true to warm up EasyOCR.")
        return

    image_path = settings.storage_root / "ocr_warmup" / "sample.png"
    _write_sample_image(image_path)

    print("Initializing EasyOCR. The first run may download model files...")
    reader = _get_easyocr_reader()
    if reader is None:
        raise RuntimeError("EasyOCR could not be initialized. Check the installed packages.")

    print("Running one warmup OCR prediction...")
    reader.readtext(str(image_path), detail=1, paragraph=False)

    print("EasyOCR warmup complete. Model files are cached for future runs.")


if __name__ == "__main__":
    main()
