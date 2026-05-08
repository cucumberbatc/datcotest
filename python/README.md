# Python RAG Service

This folder contains the Python service for real RAG and PDF highlighting.

## Stack

- `FastAPI` for the REST API
- `LangChain`
  - `ChatOpenAI`
  - `OpenAIEmbeddings`
  - `FAISS`
- `PyMuPDF (fitz)` for PDF text block extraction and highlight generation
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
- `GET /api/documents/{document_id}/chunks/{chunk_id}/highlighted-file`
  - Return a highlighted PDF for the selected evidence chunk
- `POST /api/chat/ask`
  - Run RAG and return answer + citations

## Notes

- `Pipfile` and `Pipfile.lock` are now the source of truth for dependencies.
- `requirements.txt` is kept as a plain reference/export-style list, but day-to-day install should use `pipenv install`.
- If `OPENAI_API_KEY` is not set, the service falls back to a non-LLM extractive answer path.

## Next improvements

- Better chunking
- Hybrid retrieval / reranking
- PDF.js-based real PDF viewer on the frontend
- OCR for scanned PDFs
- Persistent vector store such as Chroma, Postgres, or Qdrant
