from __future__ import annotations

import mimetypes
import uuid
import logging
from contextlib import contextmanager
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import get_settings
from app.document_ingest import ingest_pdf
from app.pdf_pipeline import build_highlighted_pdf, get_page_size, render_page_image, sanitize_filename
from app.rag_service import rag_service
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentDetailResponse,
    DocumentSummaryResponse,
    HighlightResponse,
    PageMetadataResponse,
    PageResponse,
    RectResponse,
    UploadResponse,
)
from app.store import ChunkRecord, DocumentRecord, store

settings = get_settings()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
logger = logging.getLogger(__name__)
MAX_PAGE_LIKE_RECT_AREA_RATIO = 0.85
MAX_PAGE_LIKE_RECT_DIMENSION_RATIO = 0.95
MAX_UPLOAD_DOCUMENTS = 10
MAX_TOTAL_UPLOAD_BYTES = 200 * 1024 * 1024
PDF_SIGNATURE_SCAN_BYTES = 1024

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings.uploads_root.mkdir(parents=True, exist_ok=True)
settings.highlights_root.mkdir(parents=True, exist_ok=True)
settings.page_images_root.mkdir(parents=True, exist_ok=True)
settings.ocr_pages_root.mkdir(parents=True, exist_ok=True)
document_processing_lock = Lock()


@contextmanager
def acquire_document_processing() -> None:
    if not document_processing_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another document upload or delete is currently being processed. Please try again after it finishes.",
        )
    try:
        yield
    finally:
        document_processing_lock.release()


def to_summary(document: DocumentRecord) -> DocumentSummaryResponse:
    return DocumentSummaryResponse(
        id=document.id,
        fileName=document.file_name,
        sizeBytes=document.size_bytes,
        uploadedAt=store.to_iso(document.uploaded_at),
        status=document.status,
        pageCount=document.page_count,
    )


def to_detail(document: DocumentRecord) -> DocumentDetailResponse:
    return DocumentDetailResponse(
        **to_summary(document).model_dump(),
        downloadUrl=f"/api/documents/{document.id}/file",
        pages=[
            PageResponse(
                pageNumber=page.page_number,
                paragraphs=page.paragraphs,
            )
            for page in document.pages
        ],
    )


def get_document_or_404(document_id: str) -> DocumentRecord:
    document = store.get_document(document_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return document


def _format_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f}MB"


def _validate_pdf_upload(file_name: str, content: bytes) -> None:
    if not file_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PDF 파일만 업로드할 수 있어요: {file_name}",
        )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"빈 파일은 업로드할 수 없어요: {file_name}",
        )
    if not content[:PDF_SIGNATURE_SCAN_BYTES].lstrip().startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"문서를 읽을 수 없습니다. PDF 형식이 아니거나 손상된 파일일 수 있어요: {file_name}",
        )


def _validate_upload_limits(incoming_count: int, incoming_bytes: int) -> None:
    documents = store.list_documents()
    existing_count = len(documents)
    existing_bytes = sum(document.size_bytes for document in documents)

    if incoming_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="업로드할 PDF 파일을 선택해 주세요.",
        )
    if existing_count + incoming_count > MAX_UPLOAD_DOCUMENTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"최대 {MAX_UPLOAD_DOCUMENTS}개의 문서까지 업로드할 수 있어요.",
        )
    if existing_bytes + incoming_bytes > MAX_TOTAL_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"총 {_format_mb(MAX_TOTAL_UPLOAD_BYTES)}까지 업로드할 수 있어요. "
                f"현재 {_format_mb(existing_bytes)}, 추가 {_format_mb(incoming_bytes)}입니다."
            ),
        )


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get(f"{settings.api_prefix}/documents", response_model=list[DocumentSummaryResponse])
def list_documents() -> list[DocumentSummaryResponse]:
    return [to_summary(document) for document in store.list_documents()]


@app.post(f"{settings.api_prefix}/documents", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    uploaded: list[DocumentSummaryResponse] = []
    uploaded_documents: list[DocumentRecord] = []

    with acquire_document_processing():
        uploads: list[tuple[str, bytes]] = []
        for file in files:
            if not file.filename:
                continue

            content = await file.read()
            _validate_pdf_upload(file.filename, content)
            uploads.append((file.filename, content))

        _validate_upload_limits(
            incoming_count=len(uploads),
            incoming_bytes=sum(len(content) for _, content in uploads),
        )

        for file_name, content in uploads:
            safe_name = sanitize_filename(file_name)
            output_path = settings.uploads_root / f"{uuid.uuid4().hex}-{safe_name}"
            try:
                document = ingest_pdf(file_name, content, output_path)
            except HTTPException:
                output_path.unlink(missing_ok=True)
                raise
            except Exception as exc:  # pragma: no cover - defensive boundary
                output_path.unlink(missing_ok=True)
                logger.exception("Document ingest failed for %s", file_name)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="문서 분석 중 오류가 발생했어요. 다른 PDF로 다시 시도해 주세요.",
                ) from exc
            store.upsert_document(document)
            uploaded_documents.append(document)
            uploaded.append(to_summary(document))

        try:
            rag_service.rebuild_index()
        except HTTPException:
            for document in uploaded_documents:
                store.remove_document(document.id)
                document.file_path.unlink(missing_ok=True)
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            for document in uploaded_documents:
                store.remove_document(document.id)
                document.file_path.unlink(missing_ok=True)
            logger.exception("Document indexing failed after upload")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="문서 인덱싱 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요.",
            ) from exc
    return UploadResponse(documents=uploaded)


@app.get(f"{settings.api_prefix}/documents/{{document_id}}", response_model=DocumentDetailResponse)
def get_document(document_id: str) -> DocumentDetailResponse:
    return to_detail(get_document_or_404(document_id))


@app.delete(f"{settings.api_prefix}/documents/{{document_id}}")
def delete_document(document_id: str) -> dict[str, bool]:
    with acquire_document_processing():
        document = store.remove_document(document_id)
        if not document:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

        if document.file_path.exists():
            document.file_path.unlink()

        for file in settings.highlights_root.glob(f"{document_id}-*.pdf"):
            file.unlink(missing_ok=True)

        page_image_dir = settings.page_images_root / document_id
        if page_image_dir.exists():
            for file in page_image_dir.glob("*.png"):
                file.unlink(missing_ok=True)
            page_image_dir.rmdir()

        ocr_page_dir = settings.ocr_pages_root / document_id
        if ocr_page_dir.exists():
            for file in ocr_page_dir.glob("*.png"):
                file.unlink(missing_ok=True)
            ocr_page_dir.rmdir()

        rag_service.rebuild_index()
    return {"ok": True}


@app.get(f"{settings.api_prefix}/documents/{{document_id}}/file")
def open_document_file(document_id: str) -> FileResponse:
    document = get_document_or_404(document_id)
    if not document.file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk.",
        )
    media_type = mimetypes.guess_type(document.file_path.name)[0] or "application/pdf"
    return FileResponse(document.file_path, media_type=media_type, filename=document.file_name)


@app.get(
    f"{settings.api_prefix}/documents/{{document_id}}/pages/{{page_number}}/metadata",
    response_model=PageMetadataResponse,
)
def get_page_metadata(document_id: str, page_number: int) -> PageMetadataResponse:
    document = get_document_or_404(document_id)
    if not document.file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk.",
        )
    width, height = get_page_size(document.file_path, page_number)
    return PageMetadataResponse(
        documentId=document.id,
        pageNumber=page_number,
        width=width,
        height=height,
        imageUrl=f"/api/documents/{document.id}/pages/{page_number}/image",
    )


@app.get(f"{settings.api_prefix}/documents/{{document_id}}/pages/{{page_number}}/image")
def get_page_image(document_id: str, page_number: int) -> FileResponse:
    document = get_document_or_404(document_id)
    if not document.file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk.",
        )
    output_path = settings.page_images_root / document.id / f"page_{page_number}.png"
    render_page_image(document.file_path, output_path, page_number)
    if not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk.",
        )
    return FileResponse(output_path, media_type="image/png", filename=f"{document.id}-page-{page_number}.png")


@app.get(
    f"{settings.api_prefix}/documents/{{document_id}}/chunks/{{chunk_id}}/highlight",
    response_model=HighlightResponse,
)
def get_highlight_metadata(
    document_id: str,
    chunk_id: str,
    paragraphStart: int | None = Query(default=None),
    paragraphEnd: int | None = Query(default=None),
) -> HighlightResponse:
    document = get_document_or_404(document_id)
    chunk = store.get_chunk(chunk_id)
    if not chunk or chunk.document_id != document.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found.")

    return build_highlight_response(document.id, chunk, paragraphStart, paragraphEnd)


@app.get(f"{settings.api_prefix}/documents/{{document_id}}/chunks/{{chunk_id}}/highlighted-file")
def get_highlighted_file(
    document_id: str,
    chunk_id: str,
    paragraphStart: int | None = Query(default=None),
    paragraphEnd: int | None = Query(default=None),
) -> FileResponse:
    document = get_document_or_404(document_id)
    chunk = store.get_chunk(chunk_id)
    if not chunk or chunk.document_id != document.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found.")
    if not document.file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk.",
        )

    highlight_suffix = (
        f"-p{paragraphStart}-{paragraphEnd}"
        if paragraphStart is not None and paragraphEnd is not None
        else ""
    )
    output_path = settings.highlights_root / f"{document.id}-{chunk.id}{highlight_suffix}.pdf"
    build_highlighted_pdf(
        source_pdf=document.file_path,
        output_pdf=output_path,
        page_number=chunk.page_number,
        rects=select_highlight_rects(chunk, paragraphStart, paragraphEnd),
    )
    if not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk.",
        )
    return FileResponse(output_path, media_type="application/pdf", filename=f"{document.file_name}.highlighted.pdf")


def select_highlight_rects(
    chunk: ChunkRecord,
    paragraph_start: int | None,
    paragraph_end: int | None,
) -> list[tuple[float, float, float, float]]:
    page_size = _page_size_for_chunk(chunk)
    if paragraph_start is None or paragraph_end is None:
        return filter_valid_highlight_rects(chunk.rects, page_size=page_size)

    start_offset = max(paragraph_start - chunk.paragraph_index, 0)
    end_offset = min(paragraph_end - chunk.paragraph_index, len(chunk.paragraph_rects) - 1)
    if start_offset > end_offset:
        return []

    selected: list[tuple[float, float, float, float]] = []
    for paragraph_rects in chunk.paragraph_rects[start_offset : end_offset + 1]:
        selected.extend(paragraph_rects)

    return filter_valid_highlight_rects(selected, page_size=page_size)


def _page_size_for_chunk(chunk: ChunkRecord) -> tuple[float, float] | None:
    document = store.get_document(chunk.document_id)
    if document is None:
        return None

    try:
        return get_page_size(document.file_path, chunk.page_number)
    except HTTPException:
        return None


def filter_valid_highlight_rects(
    rects: list[tuple[float, float, float, float]],
    *,
    page_size: tuple[float, float] | None = None,
) -> list[tuple[float, float, float, float]]:
    filtered: list[tuple[float, float, float, float]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for rect in rects:
        if len(rect) != 4:
            continue

        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            continue
        if x0 < 0 or y0 < 0:
            continue

        width = x1 - x0
        height = y1 - y0
        if page_size and _is_page_like_rect(width, height, page_size):
            continue

        rect_key = (
            round(x0 * 1000),
            round(y0 * 1000),
            round(x1 * 1000),
            round(y1 * 1000),
        )
        if rect_key in seen:
            continue

        seen.add(rect_key)
        filtered.append(rect)

    return filtered


def _is_page_like_rect(
    width: float,
    height: float,
    page_size: tuple[float, float],
) -> bool:
    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0:
        return False

    area_ratio = (width * height) / max(page_width * page_height, 1.0)
    width_ratio = width / page_width
    height_ratio = height / page_height
    return (
        area_ratio >= MAX_PAGE_LIKE_RECT_AREA_RATIO
        or (
            width_ratio >= MAX_PAGE_LIKE_RECT_DIMENSION_RATIO
            and height_ratio >= MAX_PAGE_LIKE_RECT_DIMENSION_RATIO
        )
    )


def build_highlight_response(
    document_id: str,
    chunk: ChunkRecord,
    paragraph_start: int | None,
    paragraph_end: int | None,
) -> HighlightResponse:
    return HighlightResponse(
        documentId=document_id,
        chunkId=chunk.id,
        pageNumber=chunk.page_number,
        rects=[
            RectResponse(x0=rect[0], y0=rect[1], x1=rect[2], y1=rect[3])
            for rect in select_highlight_rects(chunk, paragraph_start, paragraph_end)
        ],
    )


@app.post(f"{settings.api_prefix}/chat/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    scoped_documents = (
        [get_document_or_404(document_id) for document_id in request.documentIds]
        if request.documentIds
        else store.list_documents()
    )
    return rag_service.ask(request.question, scoped_documents)

@app.post(f"{settings.api_prefix}/chat/debug-retrieve")
def debug_retrieve(request: AskRequest) -> dict:
    scoped_documents = (
        [get_document_or_404(document_id) for document_id in request.documentIds]
        if request.documentIds
        else store.list_documents()
    )
    return rag_service.debug_retrieve(request.question, scoped_documents)
