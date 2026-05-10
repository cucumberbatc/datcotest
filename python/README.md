# Python RAG Service

This folder contains the Python service for real RAG and PDF highlighting.

## Stack

- `FastAPI` for the REST API
- `LangChain`
  - `ChatOpenAI`
  - `OpenAIEmbeddings`
  - `FAISS`
- `PyMuPDF (fitz)` for PDF text extraction, page image rendering, and highlight generation
- `EasyOCR` for scanned/image-only PDF fallback OCR
- `pipenv` for dependency and virtual environment management

## Why split this into a separate Python service

- LLM, embeddings, vector search, and PDF coordinate work fit the Python ecosystem better.
- `PyMuPDF`, `FAISS`, and `LangChain` are much easier to wire together here than inside Spring.
- It gives us a clean path to add OCR, rerankers, external vector DBs, and async workers later.

## Recommended Python version

- Current `Pipfile` is pinned to `Python 3.10`
- Use `Python 3.10.x` for the least friction with the current lockfile

## Setup with pipenv

```bash
cd python
pipenv install
copy .env.example .env
pipenv run uvicorn app.main:app --reload --port 8000
```

## OCR Model Warmup

EasyOCR downloads its model files the first time it is initialized. To avoid making the first scanned-PDF upload feel slow, run this once during local setup:

```bash
cd python
pipenv run python scripts/warmup_ocr.py
```

This creates a tiny sample image, initializes EasyOCR, runs one OCR prediction, and leaves the downloaded model files in EasyOCR's local cache. Later PDF uploads reuse the cached models.

Recommended OCR-related versions:

```text
numpy==1.26.4
easyocr==1.7.2
```

If you want to open a shell first:

```bash
cd python
pipenv shell
uvicorn app.main:app --reload --port 8000
```

## Frontend connection

Point the frontend at this service:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000/api
```

## API

- `POST /api/documents`
  - Upload PDFs and index them
- `GET /api/documents`
  - List documents
- `GET /api/documents/{document_id}`
  - Document detail with page text
- `DELETE /api/documents/{document_id}`
  - Delete document
- `GET /api/documents/{document_id}/file`
  - Open the original PDF
- `GET /api/documents/{document_id}/pages/{page_number}/image`
  - Return a cached PyMuPDF-rendered PNG page image
- `GET /api/documents/{document_id}/pages/{page_number}/metadata`
  - Return PDF coordinate-space page width/height and image URL
- `GET /api/documents/{document_id}/chunks/{chunk_id}/highlight`
  - Return PDF coordinate-space highlight rectangles for viewer overlay
- `GET /api/documents/{document_id}/chunks/{chunk_id}/highlighted-file`
  - Return a highlighted PDF for the selected evidence chunk
- `POST /api/chat/ask`
  - Run RAG and return answer + citations

## Notes

- `Pipfile` and `Pipfile.lock` are now the source of truth for dependencies.
- `requirements.txt` is kept as a plain reference/export-style list, but day-to-day install should use `pipenv install`.
- If `OPENAI_API_KEY` is not set, the service falls back to a non-LLM extractive answer path.
- OCR runs only as a fallback when a page has very little embedded text. Configure it with `OCR_ENABLED`, `OCR_LANGUAGES`, `OCR_ZOOM`, `OCR_MIN_TEXT_CHARS`, and `OCR_GPU`.
- EasyOCR language codes use values such as `ko,en`. Keep `OCR_GPU=false` for a simple CPU-only setup.

## Next improvements

- Better chunking
- Hybrid retrieval / reranking
- More advanced OCR/table structure parsing
- Persistent vector store such as Chroma, Postgres, or Qdrant
