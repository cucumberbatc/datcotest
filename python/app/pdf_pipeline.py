from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from fastapi import HTTPException, status

from app.config import get_settings
from app.store import ChunkRecord, DocumentRecord, PageRecord

BLOCK_TEXT = 0
BLOCK_IMAGE = 1
WHITESPACE = re.compile(r"\s+")
CHUNK_TARGET_CHARS = 800
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150
MIN_PARAGRAPH_CHARS = 40
DEBUG_EXTRACTED_TEXT = True
OCR_SOURCE_TITLE = "OCR extracted text"
LARGE_IMAGE_AREA_RATIO = 0.08
logger = logging.getLogger(__name__)


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


@dataclass(slots=True)
class OcrTextItem:
    text: str
    score: float
    bbox: tuple[float, float, float, float]


@dataclass(slots=True)
class OcrParseResult:
    items: list[OcrTextItem]
    raw_payloads: list[Any]


@dataclass(slots=True)
class OcrPageResult:
    items: list[OcrTextItem]
    row_paragraphs: list[ExtractedParagraph]
    raw_block_paragraphs: list[ExtractedParagraph]
    raw_payloads: list[Any]


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


def _has_large_image_block(page: fitz.Page) -> bool:
    page_dict = page.get_text("dict", sort=True)
    page_area = max(float(page.rect.width) * float(page.rect.height), 1.0)

    for block in page_dict.get("blocks", []):
        if block.get("type") != BLOCK_IMAGE:
            continue

        x0, y0, x1, y1 = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        area = max(float(x1) - float(x0), 0.0) * max(float(y1) - float(y0), 0.0)
        if area / page_area >= LARGE_IMAGE_AREA_RATIO:
            return True

    return False


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


def _looks_like_section_title(text: str) -> bool:
    normalized = normalize_text(text)
    if not 2 <= len(normalized) <= 80:
        return False

    if normalized.endswith((".", "?", "!", ",", ";", ":")):
        return False

    if re.search(r"[.!?]\s+", normalized):
        return False

    if re.match(r"^(\d+(\.\d+)*[.)]?\s+|[IVX]+[.)]\s+|[#]+)\S+", normalized, re.IGNORECASE):
        return True

    if re.match(r"^(chapter|section|part|appendix|overview|summary|agenda|team|members?)\b", normalized, re.IGNORECASE):
        return True

    if ":" in normalized and len(normalized.split()) <= 4:
        return True

    title_keywords = (
        "개요",
        "소개",
        "요약",
        "목적",
        "배경",
        "범위",
        "대상",
        "일정",
        "내용",
        "구성",
        "참가자",
        "참여자",
        "팀원",
        "구성원",
        "결과",
        "문의",
        "예산",
        "산출물",
        "프로젝트",
        "overview",
        "summary",
        "background",
        "scope",
        "members",
        "participants",
    )
    if any(keyword.lower() in normalized.lower() for keyword in title_keywords) and len(normalized.split()) <= 8:
        return True

    return len(normalized.split()) <= 6 and len(normalized) <= 40


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


def _ocr_languages() -> list[str]:
    languages = [
        language.strip()
        for language in get_settings().ocr_languages.split(",")
        if language.strip()
    ]
    return languages or ["ko", "en"]


@lru_cache(maxsize=1)
def _get_paddleocr_reader() -> Any | None:
    settings = get_settings()
    if not settings.ocr_enabled:
        return None

    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        logger.warning("PaddleOCR is not available: %s", exc)
        return None

    try:
        return PaddleOCR(
            lang=_paddleocr_language(),
            ocr_version=settings.ocr_version,
            device="gpu:0" if settings.ocr_gpu else "cpu",
            text_detection_model_name=settings.ocr_det_model_name,
            text_recognition_model_name=settings.ocr_rec_model_name,
            cpu_threads=settings.ocr_cpu_threads,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as exc:
        logger.warning("Failed to initialize PaddleOCR: %s", exc)
        return None


def _paddleocr_language() -> str:
    aliases = {
        "ko": "korean",
        "korean": "korean",
        "kr": "korean",
        "en": "en",
        "english": "en",
        "ch": "ch",
        "zh": "ch",
        "chinese": "ch",
        "japan": "japan",
        "ja": "japan",
        "japanese": "japan",
    }
    normalized = [aliases.get(language.lower(), language.lower()) for language in _ocr_languages()]
    if "korean" in normalized:
        return "korean"
    if "en" in normalized:
        return "en"
    return normalized[0] if normalized else "korean"


def _render_ocr_image(
    page: fitz.Page,
    document_id: str,
    page_number: int,
    zoom: float,
) -> Path:
    image_path = get_settings().ocr_pages_root / document_id / f"page_{page_number}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if image_path.exists():
        return image_path

    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    pixmap.save(image_path)
    return image_path


def _run_paddleocr(image_path: Path) -> OcrParseResult:
    reader = _get_paddleocr_reader()
    if reader is None:
        return OcrParseResult(items=[], raw_payloads=[])

    try:
        raw_result = reader.predict(str(image_path))
        return _parse_paddleocr_result(raw_result)
    except Exception as exc:
        logger.warning("PaddleOCR failed for %s: %s", image_path, exc)
        return OcrParseResult(items=[], raw_payloads=[])

def _parse_paddleocr_result(raw_result: Any) -> OcrParseResult:
    if not raw_result:
        return OcrParseResult(items=[], raw_payloads=[])

    items: list[OcrTextItem] = []
    raw_payloads: list[Any] = []
    for candidate in _iter_paddle_results(raw_result):
        payload = _paddle_result_payload(candidate)
        raw_payloads.append(_json_safe(payload))
        parsed_items = _parse_paddleocr_payload(payload)
        if parsed_items:
            items.extend(parsed_items)

    return OcrParseResult(items=items, raw_payloads=raw_payloads)


def _iter_paddle_results(raw_result: Any) -> Iterable[Any]:
    if isinstance(raw_result, Iterable) and not isinstance(raw_result, (dict, str, bytes)):
        return raw_result
    return [raw_result]


def _paddle_result_payload(candidate: Any) -> Any:
    if isinstance(candidate, dict):
        return candidate.get("res", candidate)

    for attribute_name in ("res", "json"):
        value = getattr(candidate, attribute_name, None)
        if value is None:
            continue
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            return parsed.get("res", parsed)
        if isinstance(value, dict):
            return value.get("res", value)

    return candidate


def _parse_paddleocr_payload(payload: Any) -> list[OcrTextItem]:
    texts = _paddle_field(payload, "rec_texts")
    boxes = _first_paddle_field(payload, "rec_boxes", "rec_polys", "dt_polys")
    scores = _paddle_field(payload, "rec_scores")

    if _has_items(texts) and _has_items(boxes):
        items: list[OcrTextItem] = []
        for index, text in enumerate(texts):
            if index >= len(boxes):
                break
            score = scores[index] if _has_items(scores) and index < len(scores) else 0.0
            parsed_item = _build_ocr_item(str(text), _safe_float(score), boxes[index])
            if parsed_item is not None:
                items.append(parsed_item)
        return items

    text = _paddle_field(payload, "rec_text")
    if text is None:
        return []

    box = _first_paddle_field(payload, "rec_box", "rec_poly", "dt_poly")
    score = _safe_float(_paddle_field(payload, "rec_score"))
    parsed_item = _build_ocr_item(str(text), score, box)
    return [parsed_item] if parsed_item is not None else []


def _paddle_field(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(payload, key, None)


def _first_paddle_field(payload: Any, *keys: str) -> Any:
    for key in keys:
        value = _paddle_field(payload, key)
        if value is not None:
            return value
    return None


def _has_items(value: Any) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_ocr_item(text: str, score: float, box: Any) -> OcrTextItem | None:
    normalized = normalize_text(text)
    if not normalized:
        return None

    try:
        if hasattr(box, "tolist"):
            box = box.tolist()

        if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
            x0, y0, x1, y1 = [float(value) for value in box]
        else:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError, IndexError):
        return None

    return OcrTextItem(text=normalized, score=score, bbox=(x0, y0, x1, y1))


def _filter_ocr_items(items: list[OcrTextItem]) -> list[OcrTextItem]:
    threshold = get_settings().ocr_confidence_threshold
    return [item for item in items if item.score >= threshold]


def _ocr_items_to_raw_block_paragraphs(items: list[OcrTextItem], zoom: float) -> list[ExtractedParagraph]:
    paragraphs: list[ExtractedParagraph] = []
    for item in sorted(items, key=lambda candidate: (candidate.bbox[1], candidate.bbox[0])):
        paragraphs.append(
            ExtractedParagraph(
                text=item.text,
                rects=[
                    (
                        item.bbox[0] / zoom,
                        item.bbox[1] / zoom,
                        item.bbox[2] / zoom,
                        item.bbox[3] / zoom,
                    )
                ],
                section_title=OCR_SOURCE_TITLE,
            )
        )
    return paragraphs


def _ocr_items_to_paragraphs(items: list[OcrTextItem], zoom: float) -> list[ExtractedParagraph]:
    if not items:
        return []

    lines = [
        TextLine(
            text=item.text,
            bbox=(
                item.bbox[0] / zoom,
                item.bbox[1] / zoom,
                item.bbox[2] / zoom,
                item.bbox[3] / zoom,
            ),
        )
        for item in items
    ]
    lines.sort(key=lambda line: (line.y0, line.x0))

    rows: list[list[TextLine]] = []
    for line in lines:
        if not rows:
            rows.append([line])
            continue

        previous_row = rows[-1]
        previous_center = sum((item.y0 + item.y1) / 2 for item in previous_row) / len(previous_row)
        line_center = (line.y0 + line.y1) / 2
        row_height = max(max(item.height for item in previous_row), line.height)

        if abs(line_center - previous_center) <= row_height * 0.65:
            previous_row.append(line)
        else:
            rows.append([line])

    paragraphs: list[ExtractedParagraph] = []
    for row in rows:
        row.sort(key=lambda line: line.x0)
        text = normalize_text(" ".join(line.text for line in row))
        if text:
            paragraphs.append(ExtractedParagraph(text=text, rects=[line.bbox for line in row], section_title=OCR_SOURCE_TITLE))

    return paragraphs


def _extract_ocr_paragraphs(
    page: fitz.Page,
    document_id: str,
    page_number: int,
) -> OcrPageResult:
    settings = get_settings()
    if not settings.ocr_enabled:
        return OcrPageResult(
            items=[],
            row_paragraphs=[],
            raw_block_paragraphs=[],
            raw_payloads=[],
        )

    image_path = _render_ocr_image(page, document_id, page_number, settings.ocr_zoom)
    parsed = _run_paddleocr(image_path)
    filtered_items = _filter_ocr_items(parsed.items)
    return OcrPageResult(
        items=filtered_items,
        row_paragraphs=_ocr_items_to_paragraphs(filtered_items, settings.ocr_zoom),
        raw_block_paragraphs=_ocr_items_to_raw_block_paragraphs(filtered_items, settings.ocr_zoom),
        raw_payloads=parsed.raw_payloads,
    )


def _assign_section_titles(paragraphs: list[ExtractedParagraph]) -> list[ExtractedParagraph]:
    current_section_title: str | None = None
    assigned: list[ExtractedParagraph] = []

    for paragraph in paragraphs:
        is_ocr_content = paragraph.section_title == OCR_SOURCE_TITLE
        is_section_title = not is_ocr_content and _looks_like_section_title(paragraph.text)
        if is_ocr_content:
            current_section_title = OCR_SOURCE_TITLE
        elif is_section_title:
            current_section_title = paragraph.text
        elif paragraph.section_title:
            current_section_title = paragraph.section_title

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


def _merge_paragraph_sources(*paragraph_groups: list[ExtractedParagraph]) -> list[ExtractedParagraph]:
    merged: list[ExtractedParagraph] = []
    seen_texts: set[str] = set()

    for group in paragraph_groups:
        for paragraph in group:
            normalized_text = normalize_text(paragraph.text)
            if not normalized_text:
                continue

            signature = normalized_text.casefold()
            if signature in seen_texts:
                continue

            seen_texts.add(signature)
            merged.append(
                ExtractedParagraph(
                    text=normalized_text,
                    rects=paragraph.rects,
                    section_title=paragraph.section_title,
                    is_section_title=paragraph.is_section_title,
                )
            )

    return merged


def _paragraph_char_count(paragraphs: list[ExtractedParagraph]) -> int:
    return len(normalize_text(" ".join(paragraph.text for paragraph in paragraphs)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
                if pymupdf_text_chars >= settings.ocr_min_text_chars:
                    extracted_paragraphs = native_paragraphs
                else:
                    extracted_paragraphs = _merge_short_paragraphs(
                        _merge_paragraph_sources(
                            native_paragraphs,
                            ocr_result.row_paragraphs,
                        )
                    )

            extracted_paragraphs = _assign_section_titles(extracted_paragraphs)
            ocr_text_chars = _paragraph_char_count(
                ocr_result.row_paragraphs
            )
            logger.info(
                "page=%s pymupdf_text_chars=%s has_large_image=%s ocr_triggered=%s ocr_result_count=%s ocr_text_chars=%s",
                page_number,
                pymupdf_text_chars,
                has_large_image,
                ocr_triggered,
                len(ocr_result.items),
                ocr_text_chars,
            )
            _write_debug_page_text(output_path, file_name, page_number, extracted_paragraphs)
            _write_debug_ocr_result(
                output_path=output_path,
                file_name=file_name,
                page_number=page_number,
                pymupdf_text_chars=pymupdf_text_chars,
                has_large_image=has_large_image,
                ocr_triggered=ocr_triggered,
                ocr_result=ocr_result,
            )
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


def get_page_size(source_pdf: Path, page_number: int) -> tuple[float, float]:
    with fitz.open(source_pdf) as pdf:
        if page_number < 1 or page_number > pdf.page_count:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found.")

        page = pdf.load_page(page_number - 1)
        rect = page.rect
        return float(rect.width), float(rect.height)


def render_page_image(
    source_pdf: Path,
    output_png: Path,
    page_number: int,
    zoom: float = 2.0,
) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if output_png.exists():
        return output_png

    with fitz.open(source_pdf) as pdf:
        if page_number < 1 or page_number > pdf.page_count:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found.")

        page = pdf.load_page(page_number - 1)
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(output_png)

    return output_png

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


def _write_debug_ocr_result(
    output_path: Path,
    file_name: str,
    page_number: int,
    pymupdf_text_chars: int,
    has_large_image: bool,
    ocr_triggered: bool,
    ocr_result: OcrPageResult,
) -> None:
    if not DEBUG_EXTRACTED_TEXT:
        return

    debug_dir = output_path.parent.parent / "debug_ocr"
    debug_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(file_name)
    json_file = debug_dir / f"{safe_name}_page_{page_number}.json"
    txt_file = debug_dir / f"{safe_name}_page_{page_number}.txt"

    row_texts = [paragraph.text for paragraph in ocr_result.row_paragraphs]
    raw_block_texts = [paragraph.text for paragraph in ocr_result.raw_block_paragraphs]
    ocr_text_chars = _paragraph_char_count(
        _merge_paragraph_sources(ocr_result.row_paragraphs, ocr_result.raw_block_paragraphs)
    )

    payload = {
        "pageNumber": page_number,
        "pymupdfTextChars": pymupdf_text_chars,
        "hasLargeImage": has_large_image,
        "ocrTriggered": ocr_triggered,
        "ocrConfidenceThreshold": get_settings().ocr_confidence_threshold,
        "ocrZoom": get_settings().ocr_zoom,
        "ocrResultCount": len(ocr_result.items),
        "ocrTextChars": ocr_text_chars,
        "rowTexts": row_texts,
        "rawBlockTexts": raw_block_texts,
        "items": [
            {
                "text": item.text,
                "score": round(item.score, 4),
                "bbox": [round(value, 2) for value in item.bbox],
            }
            for item in ocr_result.items
        ],
        "rawPayloads": ocr_result.raw_payloads,
    }
    json_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    txt_sections = [
        f"[meta]\npymupdf_text_chars={pymupdf_text_chars}\nhas_large_image={has_large_image}\nocr_triggered={ocr_triggered}\nocr_result_count={len(ocr_result.items)}\nocr_text_chars={ocr_text_chars}\nocr_confidence_threshold={get_settings().ocr_confidence_threshold}\nocr_zoom={get_settings().ocr_zoom}",
        "[row_texts]\n" + ("\n".join(row_texts) if row_texts else "(none)"),
        "[raw_block_texts]\n" + ("\n".join(raw_block_texts) if raw_block_texts else "(none)"),
    ]
    txt_file.write_text("\n\n".join(txt_sections), encoding="utf-8")
