from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
from fastapi import HTTPException, status

from app.store import ChunkRecord, DocumentRecord, PageRecord

BLOCK_TEXT = 0
WHITESPACE = re.compile(r"\s+")
CHUNK_TARGET_CHARS = 800
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150
MIN_PARAGRAPH_CHARS = 40
DEBUG_EXTRACTED_TEXT = True


@dataclass(slots=True)
class TextLine:
    text: str
    bbox: tuple[float, float, float, float]

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)


@dataclass(slots=True)
class ExtractedParagraph:
    text: str
    rects: list[tuple[float, float, float, float]]
    section_title: str | None = None
    is_section_title: bool = False


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return safe or "document.pdf"


def normalize_text(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def _line_text(line: dict) -> str:
    spans = line.get("spans", [])
    return normalize_text("".join(str(span.get("text", "")) for span in spans))


def _extract_page_lines(page: fitz.Page) -> list[TextLine]:
    page_dict = page.get_text("dict", sort=True)
    lines: list[TextLine] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != BLOCK_TEXT:
            continue

        for line in block.get("lines", []):
            text = _line_text(line)
            if not text:
                continue

            x0, y0, x1, y1 = line["bbox"]
            lines.append(TextLine(text=text, bbox=(float(x0), float(y0), float(x1), float(y1))))

    return lines


def _is_list_marker(text: str) -> bool:
    return bool(re.match(r"^(\(?[0-9A-Za-z]+\)|[0-9]+[.)]|[-*\u2022])\s+", text))


def _looks_like_section_title(text: str) -> bool:
    normalized = normalize_text(text)
    if not 2 <= len(normalized) <= 80:
        return False

    if normalized.endswith((".", "?", "!", ",", ";", ":", "다", "요")):
        return False

    if re.search(r"[.!?]\s+", normalized):
        return False

    if re.match(r"^(\d+(\.\d+)*[.)]?\s+|[IVX]+[.)]\s+|[#]+)\S+", normalized, re.IGNORECASE):
        return True

    title_keywords = (
        "경력",
        "경험",
        "교육",
        "프로젝트",
        "역량",
        "기술",
        "수상",
        "자격",
        "학력",
        "활동",
        "요약",
        "소개",
        "개요",
        "목표",
        "성과",
    )
    if any(keyword in normalized for keyword in title_keywords) and len(normalized.split()) <= 8:
        return True

    return len(normalized.split()) <= 5 and len(normalized) <= 32


def _is_hangul(char: str) -> bool:
    return bool(re.match(r"[\uAC00-\uD7A3]", char))


def _line_joiner(left: str, right: str) -> str:
    if not left or not right:
        return ""

    if left[-1] in "-\u2010\u2011\u2012\u2013\u2014":
        return ""

    if _is_hangul(left[-1]) and _is_hangul(right[0]):
        return ""

    return " "


def _should_start_paragraph(current_lines: list[TextLine], line: TextLine) -> bool:
    if not current_lines:
        return False

    previous = current_lines[-1]
    first = current_lines[0]
    gap = line.y0 - previous.y1
    line_height = max(previous.height, line.height)

    if _is_list_marker(line.text):
        return True

    if gap > line_height * 1.15:
        return True

    if gap > line_height * 0.55 and previous.text.endswith((".", "?", "!", ":", ";", "다.", "요.")):
        return True

    if abs(line.x0 - first.x0) > 36 and previous.text.endswith((".", "?", "!", "다.", "요.")):
        return True

    return False


def _join_paragraph_lines(lines: list[TextLine]) -> str:
    text = ""
    for line in lines:
        if not text:
            text = line.text
            continue

        joiner = _line_joiner(text, line.text)
        if text[-1:] in "-\u2010\u2011\u2012\u2013\u2014":
            text = text[:-1]
        text = f"{text}{joiner}{line.text}"

    return normalize_text(text)


def _extract_page_paragraphs(page: fitz.Page) -> list[ExtractedParagraph]:
    paragraphs: list[ExtractedParagraph] = []
    current_lines: list[TextLine] = []

    for line in _extract_page_lines(page):
        if _should_start_paragraph(current_lines, line):
            text = _join_paragraph_lines(current_lines)
            if text:
                paragraphs.append(ExtractedParagraph(text=text, rects=[item.bbox for item in current_lines]))
            current_lines = []

        current_lines.append(line)

    if current_lines:
        text = _join_paragraph_lines(current_lines)
        if text:
            paragraphs.append(ExtractedParagraph(text=text, rects=[item.bbox for item in current_lines]))

    return paragraphs


def _assign_section_titles(paragraphs: list[ExtractedParagraph]) -> list[ExtractedParagraph]:
    current_section_title: str | None = None
    assigned: list[ExtractedParagraph] = []

    for paragraph in paragraphs:
        is_section_title = _looks_like_section_title(paragraph.text)
        if is_section_title:
            current_section_title = paragraph.text

        assigned.append(
            ExtractedParagraph(
                text=paragraph.text,
                rects=paragraph.rects,
                section_title=current_section_title,
                is_section_title=is_section_title,
            )
        )

    return assigned


def _next_chunk_start(start: int, end: int, paragraphs: list[ExtractedParagraph]) -> int:
    if end >= len(paragraphs):
        return end

    overlap_chars = 0
    index = end
    while index > start + 1 and overlap_chars < CHUNK_OVERLAP_CHARS:
        index -= 1
        overlap_chars += len(paragraphs[index].text)

    return max(index, start + 1)


def _build_page_chunks(
    document_id: str,
    file_name: str,
    page_number: int,
    paragraphs: list[ExtractedParagraph],
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    start = 0

    while start < len(paragraphs):
        chunk_parts: list[str] = []
        chunk_rects: list[tuple[float, float, float, float]] = []
        chunk_paragraph_rects: list[list[tuple[float, float, float, float]]] = []
        section_title = paragraphs[start].section_title if paragraphs else None
        end = start
        char_count = 0

        while end < len(paragraphs):
            paragraph = paragraphs[end]
            if paragraph.section_title:
                section_title = paragraph.section_title
            next_char_count = char_count + len(paragraph.text)
            if chunk_parts and next_char_count > CHUNK_MAX_CHARS:
                break

            chunk_parts.append(paragraph.text)
            chunk_rects.extend(paragraph.rects)
            chunk_paragraph_rects.append(paragraph.rects)
            char_count = next_char_count
            end += 1

            if char_count >= CHUNK_TARGET_CHARS:
                break

        body_text = normalize_text("\n\n".join(chunk_parts))
        text = _chunk_text_with_section(body_text, section_title)
        if text:
            chunk_id = f"{document_id}-p{page_number}-c{len(chunks) + 1}"
            chunks.append(
                ChunkRecord(
                    id=chunk_id,
                    document_id=document_id,
                    file_name=file_name,
                    page_number=page_number,
                    paragraph_index=start + 1,
                    paragraph_end_index=end,
                    section_title=section_title,
                    text=text,
                    rects=chunk_rects,
                    paragraph_rects=chunk_paragraph_rects,
                )
            )

        start = _next_chunk_start(start, end, paragraphs)

    return chunks


def _chunk_text_with_section(text: str, section_title: str | None) -> str:
    if not section_title:
        return text
    return f"Section: {section_title}\n\n{text}"


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
            page_number = page_index + 1
            extracted_paragraphs = _assign_section_titles(_merge_short_paragraphs(_extract_page_paragraphs(page)))
            _write_debug_page_text(output_path, file_name, page_number, extracted_paragraphs)
            page_paragraphs = [paragraph.text for paragraph in extracted_paragraphs]
            chunks.extend(_build_page_chunks(document_id, file_name, page_number, extracted_paragraphs))
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
    rects: list[tuple[float, float, float, float]],
) -> Path:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with fitz.open(source_pdf) as pdf:
        page = pdf.load_page(page_number - 1)
        for bbox in rects:
            rect = fitz.Rect(*bbox)
            annotation = page.add_highlight_annot(rect)
            annotation.update()
        pdf.save(output_pdf, garbage=4, deflate=True)

    return output_pdf

def _merge_short_paragraphs(
    paragraphs: list[ExtractedParagraph],
) -> list[ExtractedParagraph]:
    merged: list[ExtractedParagraph] = []
    buffer_text: list[str] = []
    buffer_rects: list[tuple[float, float, float, float]] = []

    def flush() -> None:
        nonlocal buffer_text, buffer_rects
        if buffer_text:
            merged.append(
                ExtractedParagraph(
                    text=normalize_text(" ".join(buffer_text)),
                    rects=buffer_rects,
                )
            )
            buffer_text = []
            buffer_rects = []

    for paragraph in paragraphs:
        text = normalize_text(paragraph.text)
        if not text:
            continue

        if _looks_like_section_title(text):
            flush()
            merged.append(
                ExtractedParagraph(
                    text=text,
                    rects=paragraph.rects,
                )
            )
            continue

        buffer_text.append(text)
        buffer_rects.extend(paragraph.rects)

        if len(" ".join(buffer_text)) >= MIN_PARAGRAPH_CHARS:
            flush()

    flush()
    return merged

def _write_debug_page_text(
    output_path: Path,
    file_name: str,
    page_number: int,
    paragraphs: list[ExtractedParagraph],
) -> None:
    if not DEBUG_EXTRACTED_TEXT:
        return

    debug_dir = output_path.parent.parent / "debug_extracted"
    debug_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(file_name)
    debug_file = debug_dir / f"{safe_name}_page_{page_number}.txt"

    content = "\n\n".join(
        f"[para {index + 1}]\n{paragraph.text}"
        for index, paragraph in enumerate(paragraphs)
    )
    debug_file.write_text(content, encoding="utf-8")
