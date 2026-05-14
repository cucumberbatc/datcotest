from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from app.config import get_settings
from app.pdf_pipeline import (
    ExtractedParagraph,
    OcrPageResult,
    _assign_section_titles,
    _build_page_chunks,
    _extract_ocr_paragraphs,
    _extract_page_paragraphs,
    _has_large_image_block,
    _merge_paragraph_sources,
    _merge_short_paragraphs,
    _paragraph_char_count,
)
from app.store import DocumentRecord, PageRecord

logger = logging.getLogger(__name__)

WHITESPACE_ONLY = re.compile(r"^\s*$")
NON_SEARCHABLE = re.compile(r"[^0-9A-Za-z\uAC00-\uD7A3]+")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\uAC00-\uD7A3]+")


@dataclass(slots=True)
class AlignmentCandidate:
    paragraph: ExtractedParagraph
    compact_text: str
    tokens: set[str]


def ingest_pdf_with_llamaparse(
    file_name: str,
    file_bytes: bytes,
    output_path: Path,
) -> DocumentRecord:
    settings = get_settings()
    if not settings.llama_cloud_api_key:
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is required when DOCUMENT_PARSE_BACKEND=llamaparse."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(file_bytes)

    client = _create_client(settings.llama_cloud_api_key)
    uploaded_file = client.files.create(file=str(output_path), purpose="parse")
    result = client.parsing.parse(
        file_id=uploaded_file.id,
        tier=settings.llama_parse_tier,
        version=settings.llama_parse_version,
        expand=["markdown"],
    )

    markdown_pages = _markdown_pages_from_result(result)
    document_id = uuid.uuid4().hex
    chunks = []
    pages: list[PageRecord] = []

    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            page_number = page_index + 1
            markdown = markdown_pages[page_index] if page_index < len(markdown_pages) else ""
            alignment_paragraphs = _extract_alignment_paragraphs(
                page=page,
                document_id=document_id,
                page_number=page_number,
            )
            extracted_paragraphs = _markdown_to_paragraphs(
                markdown=markdown,
                page=page,
                alignment_paragraphs=alignment_paragraphs,
            )
            page_paragraphs = [paragraph.text for paragraph in extracted_paragraphs]
            chunks.extend(_build_page_chunks(document_id, file_name, page_number, extracted_paragraphs))
            pages.append(PageRecord(page_number=page_number, paragraphs=page_paragraphs))

    logger.info(
        "Ingested %s with LlamaParse: pages=%s chunks=%s tier=%s version=%s",
        file_name,
        len(pages),
        len(chunks),
        settings.llama_parse_tier,
        settings.llama_parse_version,
    )
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


def _create_client(api_key: str) -> Any:
    try:
        from llama_cloud import LlamaCloud
    except Exception as exc:  # pragma: no cover - import depends on optional package
        raise RuntimeError(
            "llama-cloud is not installed. Run `pipenv install llama-cloud` first."
        ) from exc

    return LlamaCloud(api_key=api_key)


def _markdown_pages_from_result(result: Any) -> list[str]:
    markdown = _read_attr_or_key(result, "markdown")
    pages = _read_attr_or_key(markdown, "pages") if markdown is not None else None
    if not pages:
        raise RuntimeError("LlamaParse returned no markdown pages.")

    extracted: list[str] = []
    for page in pages:
        page_markdown = _read_attr_or_key(page, "markdown")
        extracted.append(str(page_markdown or "").strip())

    return extracted


def _read_attr_or_key(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _extract_alignment_paragraphs(
    *,
    page: fitz.Page,
    document_id: str,
    page_number: int,
) -> list[ExtractedParagraph]:
    settings = get_settings()
    native_paragraphs = _merge_short_paragraphs(_extract_page_paragraphs(page))
    pymupdf_text_chars = _paragraph_char_count(native_paragraphs)
    has_large_image = _has_large_image_block(page)
    ocr_triggered = settings.ocr_enabled and (
        pymupdf_text_chars < settings.ocr_min_text_chars or has_large_image
    )

    ocr_result = OcrPageResult(
        items=[],
        row_paragraphs=[],
        raw_block_paragraphs=[],
        raw_payloads=[],
    )
    extracted_paragraphs = native_paragraphs
    if ocr_triggered:
        ocr_result = _extract_ocr_paragraphs(page, document_id, page_number)
        extracted_paragraphs = _merge_short_paragraphs(
            _merge_paragraph_sources(
                native_paragraphs,
                ocr_result.row_paragraphs,
                ocr_result.raw_block_paragraphs,
            )
        )

    aligned = _assign_section_titles(extracted_paragraphs)
    logger.info(
        "LlamaParse alignment page=%s native_paragraphs=%s ocr_triggered=%s ocr_items=%s aligned_paragraphs=%s",
        page_number,
        len(native_paragraphs),
        ocr_triggered,
        len(ocr_result.items),
        len(aligned),
    )
    return aligned


def _markdown_to_paragraphs(
    markdown: str,
    page: fitz.Page,
    alignment_paragraphs: list[ExtractedParagraph],
) -> list[ExtractedParagraph]:
    if not markdown.strip():
        return [ExtractedParagraph(text="", rects=[])]

    candidates = _alignment_candidates(alignment_paragraphs)
    paragraphs: list[ExtractedParagraph] = []
    current_section_title: str | None = None
    cursor = 0
    for block in _split_markdown_blocks(markdown):
        text = block.strip()
        if not text:
            continue

        if _is_markdown_heading(text):
            current_section_title = _strip_markdown_heading(text)

        rects, matched_until = _match_block_rects(
            text=text,
            candidates=candidates,
            start_index=cursor,
        )
        if matched_until >= cursor:
            cursor = matched_until + 1

        paragraphs.append(
            ExtractedParagraph(
                text=text,
                rects=rects,
                section_title=current_section_title,
            )
        )

    return paragraphs or [ExtractedParagraph(text=markdown.strip(), rects=[])]


def _split_markdown_blocks(markdown: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if WHITESPACE_ONLY.match(line):
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    return blocks


def _is_markdown_heading(text: str) -> bool:
    return text.lstrip().startswith("#")


def _strip_markdown_heading(text: str) -> str:
    stripped = re.sub(r"^#+\s*", "", text).strip()
    return stripped or text.strip()


def _alignment_candidates(paragraphs: list[ExtractedParagraph]) -> list[AlignmentCandidate]:
    candidates: list[AlignmentCandidate] = []
    for paragraph in paragraphs:
        compact_text = _compact_text(paragraph.text)
        tokens = _tokenize(paragraph.text)
        if not compact_text and not tokens:
            continue
        candidates.append(
            AlignmentCandidate(
                paragraph=paragraph,
                compact_text=compact_text,
                tokens=tokens,
            )
        )
    return candidates


def _match_block_rects(
    *,
    text: str,
    candidates: list[AlignmentCandidate],
    start_index: int,
) -> tuple[list[tuple[float, float, float, float]], int]:
    if not candidates:
        return [], -1

    block_compact = _compact_text(_markdown_search_text(text))
    block_tokens = _tokenize(_markdown_search_text(text))
    if not block_compact and not block_tokens:
        return [], -1

    best_score = 0.0
    best_span: tuple[int, int] | None = None
    search_start = min(max(start_index, 0), len(candidates) - 1)
    max_start = min(len(candidates), search_start + 10)

    for candidate_start in range(search_start, max_start):
        joined_compact = ""
        joined_tokens: set[str] = set()
        for candidate_end in range(candidate_start, min(len(candidates), candidate_start + 6)):
            candidate = candidates[candidate_end]
            joined_compact += candidate.compact_text
            joined_tokens.update(candidate.tokens)
            score = _alignment_score(
                block_compact=block_compact,
                block_tokens=block_tokens,
                candidate_compact=joined_compact,
                candidate_tokens=joined_tokens,
            )
            if score > best_score:
                best_score = score
                best_span = (candidate_start, candidate_end)

    if best_span is None or best_score < 0.33:
        return [], -1

    rects: list[tuple[float, float, float, float]] = []
    for index in range(best_span[0], best_span[1] + 1):
        rects.extend(candidates[index].paragraph.rects)

    return rects, best_span[1]


def _alignment_score(
    *,
    block_compact: str,
    block_tokens: set[str],
    candidate_compact: str,
    candidate_tokens: set[str],
) -> float:
    if not candidate_compact and not candidate_tokens:
        return 0.0

    token_overlap = 0.0
    if block_tokens and candidate_tokens:
        token_overlap = len(block_tokens & candidate_tokens) / max(len(block_tokens), 1)

    compact_overlap = 0.0
    if block_compact and candidate_compact:
        if block_compact in candidate_compact or candidate_compact in block_compact:
            shorter = min(len(block_compact), len(candidate_compact))
            longer = max(len(block_compact), len(candidate_compact), 1)
            compact_overlap = shorter / longer
        else:
            compact_overlap = _substring_overlap(block_compact, candidate_compact)

    length_ratio = 0.0
    if block_compact and candidate_compact:
        length_ratio = min(len(block_compact), len(candidate_compact)) / max(
            len(block_compact),
            len(candidate_compact),
            1,
        )

    return (token_overlap * 0.5) + (compact_overlap * 0.35) + (length_ratio * 0.15)


def _substring_overlap(left: str, right: str) -> float:
    if not left or not right:
        return 0.0

    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if len(shorter) < 3:
        return 0.0

    max_window = min(len(shorter), 12)
    for window in range(max_window, 2, -1):
        for index in range(len(shorter) - window + 1):
            if shorter[index : index + window] in longer:
                return window / len(shorter)
    return 0.0


def _markdown_search_text(text: str) -> str:
    without_heading = re.sub(r"^#+\s*", "", text.strip())
    cleaned_lines: list[str] = []
    for line in without_heading.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^\|", "", stripped)
        stripped = re.sub(r"\|$", "", stripped)
        stripped = stripped.replace("|", " ")
        stripped = stripped.replace("`", " ")
        cleaned_lines.append(stripped)
    return " ".join(cleaned_lines)


def _compact_text(text: str) -> str:
    return NON_SEARCHABLE.sub("", text).lower()


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if len(token.strip()) > 1}
