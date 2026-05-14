# Python RAG 서비스

이 폴더는 실제 RAG 및 PDF 강조 표시를 위한 Python 서비스를 포함하고 있습니다.

## 스택

- `FastAPI` - REST API 서버
- `LangChain`
  - `ChatOpenAI` - LLM
  - `OpenAIEmbeddings` - 임베딩 모델
  - `FAISS` - 벡터 검색
- `PyMuPDF (fitz)` - PDF 렌더링, 페이지 이미지 생성, 하이라이트 좌표 변환
- `PaddleOCR` - 스캔된/이미지 전용 PDF 폴백 OCR
- `LlamaParse` - 선택적 입수 시간 파서 (더 나은 테이블/레이아웃 인식 RAG)
- `Kiwi` - 한글 토크나이저
- `sentence-transformers` - CrossEncoder 리랭커
- `pipenv` - 의존성 및 가상 환경 관리

## Python 서비스를 별도로 분리한 이유

- LLM, 임베딩, 벡터 검색, PDF 좌표 작업은 Python 생태계에 더 적합합니다.
- `PyMuPDF`, `FAISS`, `LangChain`을 Spring 내부보다 여기서 훨씬 쉽게 연결할 수 있습니다.
- OCR, 리랭커, 외부 벡터 DB, 비동기 워커 등을 추가할 수 있는 명확한 경로를 제공합니다.

## 권장 Python 버전

- 현재 `Pipfile`은 `Python 3.10`으로 고정되어 있습니다
- 현재 lockfile과 문제 없이 작동하려면 `Python 3.10.x`를 사용하세요

## Pipenv 설치

### pipenv가 설치되지 않은 경우

#### macOS
```bash
brew install pipenv
```

#### Windows
```bash
pip install pipenv
```

## 설정 및 실행

### 1. 의존성 설치
```bash
cd python
pipenv install
```

### 2. 환경 설정
```bash
copy .env.example .env  # Windows
# 또는
cp .env.example .env    # macOS/Linux
```

### 3. 모델 초기화 (필수)
서버를 처음 실행하기 전에 필요한 모든 모델을 초기화하세요:

```bash
cd python
pipenv run python scripts/warmup.py
```

이 스크립트는 다음을 다운로드 및 초기화합니다:
- **PaddleOCR** - 스캔된 PDF용 OCR 모델
- **OpenAI Embeddings** - 텍스트 임베딩 모델
- **CrossEncoder Reranker** - 검색 결과 재순위 모델 (dragonkue/bge-reranker-v2-m3-ko)
- **Kiwi** - 한글 토크나이저

첫 실행 시 모델 파일을 다운로드하므로 시간이 걸릴 수 있습니다. 이후 실행에서는 로컬 캐시를 사용합니다.

### 4. 서버 실행
```bash
cd python
pipenv run uvicorn app.main:app --port 8000
```

파일 업로드, OCR, 임베딩 및 성능 테스트 시에는 `--reload` 옵션 없이 사용하세요. `--reload`는 요청 진행 중에 서버를 재시작할 수 있습니다.

## PaddleOCR CPU 설치 (선택사항)

CPU 전용 환경에서는 `paddlepaddle`을 먼저 설치한 후 나머지 앱 의존성을 설치하세요:

```bash
cd python
pipenv install
pipenv run python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pipenv run python -m pip install paddleocr
```

그 후 `requirements.txt`에서 설치하려면:

```bash
cd python
pipenv run python -m pip install -r requirements.txt
pipenv lock
```

## LlamaParse 선택적 설정

테이블, 다중 열 페이지 또는 복잡한 스캔을 더 잘 처리하려면 LlamaParse를 사용할 수 있습니다. PyMuPDF는 여전히 페이지 렌더링 및 PDF 강조 표시 생성에 사용됩니다.

SDK 설치:

```bash
cd python
pipenv install llama-cloud
```

그런 다음 `.env` 구성:

```text
DOCUMENT_PARSE_BACKEND=llamaparse
LLAMA_CLOUD_API_KEY=llx-...
LLAMA_PARSE_TIER=agentic
LLAMA_PARSE_VERSION=latest
LLAMA_PARSE_FALLBACK_TO_PYMUPDF=true
```

주의:

- LlamaParse는 업로드/인덱싱 중에만 사용됩니다. 질문 답변은 여전히 로컬 FAISS 기반 RAG 파이프라인에 대해 실행됩니다.
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
uvicorn app.main:app --port 8000
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
