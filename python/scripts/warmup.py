from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def warmup_paddleocr() -> None:
    """Warm up PaddleOCR by initializing it and running one prediction."""
    try:
        from app.config import get_settings
        from app.pdf_pipeline import _get_paddleocr_reader
        from PIL import Image, ImageDraw

        settings = get_settings()
        if not settings.ocr_enabled:
            print("⊘ OCR is disabled. Set OCR_ENABLED=true to warm up PaddleOCR.")
            return

        print("\n▶ Warming up PaddleOCR...")
        image_path = settings.storage_root / "ocr_warmup" / "sample.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)

        # Create sample image
        image = Image.new("RGB", (640, 160), "white")
        draw = ImageDraw.Draw(image)
        draw.text((32, 56), "PaddleOCR warmup 123", fill="black")
        image.save(image_path)

        print("  • Initializing PaddleOCR (downloading models on first run)...")
        reader = _get_paddleocr_reader()
        if reader is None:
            raise RuntimeError("PaddleOCR could not be initialized. Check the installed packages.")

        print("  • Running one OCR prediction...")
        reader.predict(str(image_path))
        print("✓ PaddleOCR warmup complete. Model files cached.")
    except Exception as e:
        print(f"✗ PaddleOCR warmup failed: {e}")
        raise


def warmup_embeddings() -> None:
    """Warm up OpenAI Embeddings by initializing it."""
    try:
        from app.config import get_settings
        from langchain_openai import OpenAIEmbeddings

        settings = get_settings()
        if not settings.openai_api_key:
            print("⊘ OpenAI API key not configured. Skipping embeddings warmup.")
            return

        print("\n▶ Warming up OpenAI Embeddings...")
        print("  • Initializing embeddings model...")
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model or "text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
        # Run one test embedding
        print("  • Running test embedding...")
        embeddings.embed_query("warmup test")
        print("✓ OpenAI Embeddings warmup complete.")
    except Exception as e:
        print(f"✗ OpenAI Embeddings warmup failed: {e}")
        raise


def warmup_reranker() -> None:
    """Warm up CrossEncoder reranker by initializing it."""
    try:
        print("\n▶ Warming up CrossEncoder reranker...")
        print("  • Downloading reranker model (dragonkue/bge-reranker-v2-m3-ko)...")
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder("dragonkue/bge-reranker-v2-m3-ko")
        print("  • Running test reranking...")
        reranker.predict([["test query", "test document"]])
        print("✓ CrossEncoder reranker warmup complete.")
    except Exception as e:
        print(f"✗ CrossEncoder reranker warmup failed: {e}")
        raise


def warmup_kiwi() -> None:
    """Warm up Kiwi Korean tokenizer by initializing it."""
    try:
        print("\n▶ Warming up Kiwi Korean tokenizer...")
        print("  • Initializing Kiwi tokenizer...")
        from kiwipiepy import Kiwi

        kiwi = Kiwi()
        print("  • Running test tokenization...")
        kiwi.tokenize("테스트 문장입니다")
        print("✓ Kiwi tokenizer warmup complete.")
    except Exception as e:
        print(f"✗ Kiwi tokenizer warmup failed: {e}")
        raise


def main() -> None:
    print("\n" + "=" * 60)
    print("  Python RAG Service - Model Warmup")
    print("=" * 60)

    try:
        warmup_paddleocr()
        warmup_embeddings()
        warmup_reranker()
        warmup_kiwi()

        print("\n" + "=" * 60)
        print("✓ All models warmed up successfully!")
        print("=" * 60 + "\n")
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Warmup failed: {e}")
        print("=" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
