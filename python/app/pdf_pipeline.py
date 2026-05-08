from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
from fastapi import HTTPException, status

from app.store import ChunkRecord, DocumentRecord, PageRecord

BLOCK_TEXT = 0
WHITESPACE = re.compile(r"\s+")


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return safe or "document.pdf"


def normalize_text(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def ingest_pdf(file_name: str, file_bytes: bytes, output_path: Path) -> DocumentRecord:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(file_bytes)

    document_id = uuid.uuid4().hex
    chunks: list[ChunkRecord] = []
    pages: list[PageRecord] = []

    try:
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # pragma: no cover - library-specific error shape
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to open PDF: {file_name}",
        ) from exc

    with pdf:
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            blocks = page.get_text("blocks", sort=True)
            page_paragraphs: list[str] = []
            paragraph_index = 0

            for block in blocks:
                x0, y0, x1, y1, text, _block_no, block_type = block
                if block_type != BLOCK_TEXT:
                    continue

                normalized = normalize_text(text)
                if not normalized:
                    continue

                paragraph_index += 1
                chunk_id = f"{document_id}-p{page_index + 1}-b{paragraph_index}"
                page_paragraphs.append(normalized)
                chunks.append(
                    ChunkRecord(
                        id=chunk_id,
                        document_id=document_id,
                        file_name=file_name,
                        page_number=page_index + 1,
                        paragraph_index=paragraph_index,
                        text=normalized,
                        bbox=(float(x0), float(y0), float(x1), float(y1)),
                    )
                )

            pages.append(PageRecord(page_number=page_index + 1, paragraphs=page_paragraphs))

    return DocumentRecord(
        id=document_id,
        file_name=file_name,
        size_bytes=len(file_bytes),
        uploaded_at=datetime.now(timezone.utc),
        status="ready",
        page_count=len(pages),
        file_path=output_path,
        pages=pages,
        chunks=chunks,
    )


def build_highlighted_pdf(
    source_pdf: Path,
    output_pdf: Path,
    page_number: int,
    bbox: tuple[float, float, float, float],
) -> Path:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with fitz.open(source_pdf) as pdf:
        page = pdf.load_page(page_number - 1)
        rect = fitz.Rect(*bbox)
        annotation = page.add_highlight_annot(rect)
        annotation.update()
        pdf.save(output_pdf, garbage=4, deflate=True)

    return output_pdf
