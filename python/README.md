# Python RAG Service

This folder contains the Python service for real RAG and PDF highlighting.

## Stack

- `FastAPI` for the REST API
- `LangChain`
  - `ChatOpenAI`
  - `OpenAIEmbeddings`
  - `FAISS`
- `PyMuPDF (fitz)` for PDF rendering, page image generation, and coordinate-space highlight generation
- `PaddleOCR` for scanned/image-only PDF fallback OCR
- `LlamaParse` as an optional ingest-time parser for better table/layout-aware RAG input
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

## PaddleOCR CPU install

For CPU-only environments, install PaddlePaddle first in the project virtual environment, then install the rest of the app dependencies.

Official PaddleOCR docs recommend installing `paddlepaddle` before `paddleocr`.

```bash
cd python
pipenv install
pipenv run python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pipenv run python -m pip install paddleocr
```

If you prefer to install from `requirements.txt` after that:

```bash
cd python
pipenv run python -m pip install -r requirements.txt
```

Then refresh the lockfile:

```bash
cd python
pipenv lock
```

## OCR Model Warmup

PaddleOCR downloads its model files the first time it is initialized. To avoid making the first scanned-PDF upload feel slow, run this once during local setup:

```bash
cd python
pipenv run python scripts/warmup_ocr.py
```

This creates a tiny sample image, initializes PaddleOCR, runs one OCR prediction, and leaves the downloaded model files in PaddleOCR's local cache. Later PDF uploads reuse the cached models.

## Optional LlamaParse ingest

If you want to push past plain OCR limits for tables, multi-column pages, or messy scans, you can switch document ingest to LlamaParse while still keeping PyMuPDF for page rendering and PDF highlight generation.

Install the SDK:

```bash
cd python
pipenv install llama-cloud
```

Then configure:

```text
DOCUMENT_PARSE_BACKEND=llamaparse
LLAMA_CLOUD_API_KEY=llx-...
LLAMA_PARSE_TIER=agentic
LLAMA_PARSE_VERSION=latest
LLAMA_PARSE_FALLBACK_TO_PYMUPDF=true
```

Notes:

- LlamaParse is only used during upload/indexing. Question answering still runs against the local FAISS-based RAG pipeline.
- PyMuPDF still handles page images, page sizes, and PDF highlight generation.
- The current LlamaParse integration keeps coordinate highlights by mapping parsed chunks to page-level rectangles. That preserves the highlight API, but the rectangles are broader than the native PyMuPDF paragraph boxes.
- If LlamaParse is unavailable or fails and `LLAMA_PARSE_FALLBACK_TO_PYMUPDF=true`, the service automatically falls back to the existing PyMuPDF + PaddleOCR path.

## OCR model selection

The service does not inspect each uploaded PDF and auto-pick a different OCR model per document. The OCR model choice is fixed at startup from environment settings and reused for every OCR fallback page.

For CPU-first setups, the defaults are pinned to lighter mobile models:

```text
OCR_VERSION=PP-OCRv5
OCR_DET_MODEL_NAME=PP-OCRv5_mobile_det
OCR_REC_MODEL_NAME=korean_PP-OCRv5_mobile_rec
OCR_CPU_THREADS=8
OCR_ZOOM=1.5
```

If you want to try a different fixed combination later, change these values in `.env` and restart the Python service.

Recommended OCR-related versions:

```text
numpy==1.26.4
paddlepaddle==3.2.0
paddleocr>=3.2,<4.0
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

- `Pipfile` is updated for the PaddleOCR swap. After installing the new OCR dependencies, regenerate `Pipfile.lock` with `pipenv lock`.
- `requirements.txt` is kept as a plain reference/export-style list, but day-to-day install should use `pipenv install`.
- If `OPENAI_API_KEY` is not set, the service falls back to a non-LLM extractive answer path.
- OCR runs only as a fallback when a page has very little embedded text. It does not enable PP-Structure or table-structure parsing; it only uses OCR text detection/recognition and then feeds bbox-based row grouping into the existing pipeline.
- Configure OCR with `OCR_ENABLED`, `OCR_LANGUAGES`, `OCR_ZOOM`, `OCR_MIN_TEXT_CHARS`, `OCR_GPU`, `OCR_VERSION`, `OCR_DET_MODEL_NAME`, `OCR_REC_MODEL_NAME`, and `OCR_CPU_THREADS`.
- `OCR_LANGUAGES=ko,en` is supported. Internally this maps to PaddleOCR's Korean OCR pipeline, which also covers English text well for this fallback use case.
- Keep `OCR_GPU=false` for a simple CPU-only setup.
- `DOCUMENT_PARSE_BACKEND=pymupdf` keeps the original local-only pipeline. `DOCUMENT_PARSE_BACKEND=llamaparse` uses LlamaParse during ingest and keeps PyMuPDF for highlight rendering.

## Next improvements

- Better chunking
- Hybrid retrieval / reranking
- More advanced OCR/table structure parsing
- Persistent vector store such as Chroma, Postgres, or Qdrant
