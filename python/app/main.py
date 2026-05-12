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
from app.pdf_pipeline import build_highlighted_pdf, get_page_size, ingest_pdf, render_page_image, sanitize_filename
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

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get(f"{settings.api_prefix}/documents", response_model=list[DocumentSummaryResponse])
def list_documents() -> list[DocumentSummaryResponse]:
    return [to_summary(document) for document in store.list_documents()]


@app.post(f"{settings.api_prefix}/documents", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    uploaded: list[DocumentSummaryResponse] = []

    with acquire_document_processing():
        for file in files:
            if not file.filename:
                continue
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Only PDF uploads are supported: {file.filename}",
                )

            content = await file.read()
            safe_name = sanitize_filename(file.filename)
            output_path = settings.uploads_root / f"{uuid.uuid4().hex}-{safe_name}"
            document = ingest_pdf(file.filename, content, output_path)
            store.upsert_document(document)
            uploaded.append(to_summary(document))

        rag_service.rebuild_index()
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
    media_type = mimetypes.guess_type(document.file_path.name)[0] or "application/pdf"
    return FileResponse(document.file_path, media_type=media_type, filename=document.file_name)


@app.get(
    f"{settings.api_prefix}/documents/{{document_id}}/pages/{{page_number}}/metadata",
    response_model=PageMetadataResponse,
)
def get_page_metadata(document_id: str, page_number: int) -> PageMetadataResponse:
    document = get_document_or_404(document_id)
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
    output_path = settings.page_images_root / document.id / f"page_{page_number}.png"
    render_page_image(document.file_path, output_path, page_number)
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
    return FileResponse(output_path, media_type="application/pdf", filename=f"{document.file_name}.highlighted.pdf")


def select_highlight_rects(
    chunk: ChunkRecord,
    paragraph_start: int | None,
    paragraph_end: int | None,
) -> list[tuple[float, float, float, float]]:
    if paragraph_start is None or paragraph_end is None:
        return chunk.rects

    start_offset = max(paragraph_start - chunk.paragraph_index, 0)
    end_offset = min(paragraph_end - chunk.paragraph_index, len(chunk.paragraph_rects) - 1)
    if start_offset > end_offset:
        return chunk.rects

    selected: list[tuple[float, float, float, float]] = []
    for paragraph_rects in chunk.paragraph_rects[start_offset : end_offset + 1]:
        selected.extend(paragraph_rects)

    return selected or chunk.rects


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
