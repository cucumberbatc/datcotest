from __future__ import annotations

import logging
from pathlib import Path

from app.config import get_settings
from app.llamaparse_ingest import ingest_pdf_with_llamaparse
from app.pdf_pipeline import ingest_pdf as ingest_pdf_with_pymupdf
from app.store import DocumentRecord

logger = logging.getLogger(__name__)


def ingest_pdf(file_name: str, file_bytes: bytes, output_path: Path) -> DocumentRecord:
    settings = get_settings()
    backend = settings.document_parse_backend.strip().lower()

    if backend != "llamaparse":
        return ingest_pdf_with_pymupdf(file_name, file_bytes, output_path)

    try:
        return ingest_pdf_with_llamaparse(file_name, file_bytes, output_path)
    except Exception as exc:
        if not settings.llama_parse_fallback_to_pymupdf:
            raise

        logger.warning(
            "LlamaParse ingest failed for %s, falling back to PyMuPDF pipeline: %s",
            file_name,
            exc,
        )
        return ingest_pdf_with_pymupdf(file_name, file_bytes, output_path)
