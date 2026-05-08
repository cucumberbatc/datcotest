from __future__ import annotations

import mimetypes
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import get_settings
from app.pdf_pipeline import build_highlighted_pdf, ingest_pdf, sanitize_filename
from app.rag_service import rag_service
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentDetailResponse,
    DocumentSummaryResponse,
    HighlightResponse,
    PageResponse,
    UploadResponse,
)
from app.store import DocumentRecord, store

settings = get_settings()

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
    document = store.remove_document(document_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if document.file_path.exists():
        document.file_path.unlink()

    for file in settings.highlights_root.glob(f"{document_id}-*.pdf"):
        file.unlink(missing_ok=True)

    rag_service.rebuild_index()
    return {"ok": True}


@app.get(f"{settings.api_prefix}/documents/{{document_id}}/file")
def open_document_file(document_id: str) -> FileResponse:
    document = get_document_or_404(document_id)
    media_type = mimetypes.guess_type(document.file_path.name)[0] or "application/pdf"
    return FileResponse(document.file_path, media_type=media_type, filename=document.file_name)


@app.get(
    f"{settings.api_prefix}/documents/{{document_id}}/chunks/{{chunk_id}}/highlight",
    response_model=HighlightResponse,
)
def get_highlight_metadata(document_id: str, chunk_id: str) -> HighlightResponse:
    document = get_document_or_404(document_id)
    chunk = store.get_chunk(chunk_id)
    if not chunk or chunk.document_id != document.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found.")

    return HighlightResponse(
        documentId=document.id,
        chunkId=chunk.id,
        highlightedFileUrl=f"/api/documents/{document.id}/chunks/{chunk.id}/highlighted-file",
    )


@app.get(f"{settings.api_prefix}/documents/{{document_id}}/chunks/{{chunk_id}}/highlighted-file")
def get_highlighted_file(document_id: str, chunk_id: str) -> FileResponse:
    document = get_document_or_404(document_id)
    chunk = store.get_chunk(chunk_id)
    if not chunk or chunk.document_id != document.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found.")

    output_path = settings.highlights_root / f"{document.id}-{chunk.id}.pdf"
    build_highlighted_pdf(
        source_pdf=document.file_path,
        output_pdf=output_path,
        page_number=chunk.page_number,
        bbox=chunk.bbox,
    )
    return FileResponse(output_path, media_type="application/pdf", filename=f"{document.file_name}.highlighted.pdf")


@app.post(f"{settings.api_prefix}/chat/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    scoped_documents = (
        [get_document_or_404(document_id) for document_id in request.documentIds]
        if request.documentIds
        else store.list_documents()
    )
    return rag_service.ask(request.question, scoped_documents)
