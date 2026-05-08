# datcotest

PDF-grounded Q&A assignment MVP.

## Services

- `frontend`: Next.js 3-pane UI
- `backend`: Spring Boot prototype API
- `python`: main RAG service using LangChain, OpenAI, FAISS, and PyMuPDF

## Recommended architecture

For the actual assignment flow, the recommended path is:

- `frontend` handles UI and user workflow
- `python` handles document ingestion, embeddings, retrieval, LLM answers, and PDF highlighting
- `backend` can remain as a Spring integration layer or supporting API if needed later

In practice:

- Frontend -> Python RAG service
- Optional Spring -> Python proxy/integration

## Run

### 1. Python RAG service

```bash
cd python
pipenv install
copy .env.example .env
pipenv run uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Set the frontend API target to:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000/api
```

### 3. Optional Spring backend

```bash
cd backend
mvn spring-boot:run
```

## API summary

- `POST /api/documents`
- `GET /api/documents`
- `GET /api/documents/{id}`
- `GET /api/documents/{id}/file`
- `DELETE /api/documents/{id}`
- `POST /api/chat/ask`

## Notes

- `python/` is the recommended main implementation path now.
- `PyMuPDF` is used to extract real PDF text blocks and create highlighted PDF outputs.
- The current frontend viewer is still text-view based; a future step is switching it to a real PDF.js viewer.
- `Pipfile` and `Pipfile.lock` are the source of truth for Python dependencies.

See [DESIGN.md](/d:/code/datco/datcotest/DESIGN.md) for architecture notes.
