# datcotest

PDF 기반 Q&A 과제 MVP

``` mermaid
flowchart TD
    A[사용자] --> B[Frontend<br/>Next.js UI]

    B --> C[PDF 업로드]
    C --> D[Python RAG Service<br/>FastAPI]

    D --> E[문서 파싱]
    E --> F[텍스트 추출 / 좌표 추출<br/>PyMuPDF]
    F --> G[문단 생성]
    G --> H[RAG Chunk 생성]
    H --> I[임베딩 생성]
    I --> J[FAISS 인덱싱]

    B --> K[질문 입력]
    K --> L[질문 전송]
    L --> D

    D --> M[관련 Chunk 검색<br/>Hybrid Retrieval]
    M --> N[LLM 답변 생성]
    N --> O[근거 Source / bbox 반환]

    O --> P[Frontend]
    P --> Q[답변 표시]
    P --> R[출처 카드 표시]
    P --> S[하이라이트 표시]

    S --> T[문서 뷰어에서 근거 확인]
```

## 서비스

- `frontend`: Next.js 3 페인 UI
- `python`: LangChain, OpenAI, FAISS, PyMuPDF를 사용한 메인 RAG 서비스

## 개발 환경
- feat/frontend 브랜치에서 실행 부탁드립니다
- Python: 3.10.x
- Node.js: 18 이상 권장
- Python dependency manager: Pipenv
- Frontend: Next.js
- Main API: FastAPI on port 8000
- Optional backend: Spring Boot

## 권장 아키텍처

실제 과제 흐름을 위한 권장 경로:

- `frontend`: UI 및 사용자 워크플로우 처리
- `python`: 문서 수집, 임베딩, 검색, LLM 답변, PDF 강조 표시 처리
- `backend`: 필요시 나중에 Spring 통합 계층 또는 지원 API로 사용 가능

실제로는:

- Frontend → Python RAG 서비스
- Optional: Spring → Python 프록시/통합

## 실행 방법

### 0. 사전 준비

이 프로젝트의 Python RAG 서비스는 **Python 3.10.x** 기준으로 실행합니다.

현재 `python/Pipfile`은 Python 3.10으로 고정되어 있으므로, lockfile과 동일한 환경을 사용하려면 Python 3.10.x를 사용하세요.

#### Python 버전 확인

```bash
python --version
```

또는 환경에 따라:

```bash
python3 --version
```

Python 3.10이 없다면 먼저 설치합니다.

#### macOS

```bash
brew install python@3.10
```

설치 후 버전 확인:

```bash
python3.10 --version
```

#### Windows

Python 공식 설치 파일 또는 `winget`을 사용할 수 있습니다.

```powershell
winget install Python.Python.3.10
```

설치 후 버전 확인:

```powershell
python --version
```

---

### 1. Pipenv 설치

Python 의존성 관리는 `pipenv`를 사용합니다.

#### macOS

```bash
brew install pipenv
```

또는 Python 3.10을 명시해서 pip로 설치합니다.

```bash
python3.10 -m pip install --user pipenv
```

#### Windows

```powershell
pip install pipenv
```

또는 Python 3.10을 명시해서 설치합니다.

```powershell
python -m pip install pipenv
```

설치 확인:

```bash
pipenv --version
```

---

### 2. Python 가상환경 및 모델 초기화

```bash
cd python
pipenv --python 3.10
pipenv install
```

만약 `python3.10` 명령어가 별도로 잡혀 있다면 아래처럼 실행할 수 있습니다.

```bash
pipenv --python python3.10
pipenv install
```

`.env` 파일을 생성합니다.

#### Windows

```powershell
copy .env.example .env
```

#### macOS / Linux

```bash
cp .env.example .env
```

필요한 API Key와 설정값을 `.env`에 입력합니다.

```env
OPENAI_API_KEY=your_openai_api_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
```

모든 모델을 미리 다운로드하고 초기화합니다.

```bash
pipenv run python scripts/warmup.py
```

> 파일 업로드, OCR, 임베딩, reranker 모델 로딩은 초기 실행 시 시간이 걸릴 수 있으므로, 처음 실행 전 `warmup.py`를 먼저 실행하는 것을 권장합니다.

---

### 3. Python RAG 서비스 시작

```bash
cd python
pipenv run uvicorn app.main:app --port 8000
```

파일 업로드, OCR, 임베딩 및 성능 테스트 시에는 `--reload` 옵션 없이 사용하는 것을 권장합니다.

`--reload`는 요청 진행 중 서버를 재시작할 수 있어, OCR이나 임베딩처럼 시간이 걸리는 작업에서는 사용하지 않는 편이 안전합니다.

---

### 4. Frontend 시작

```bash
cd frontend
npm install
npm run dev
```

Frontend API 대상 설정:

```env
NEXT_PUBLIC_API_BASE=http://localhost:8000/api
```

`.env.local`을 사용하는 경우 `frontend/.env.local`에 아래 값을 설정합니다.

```env
NEXT_PUBLIC_API_BASE=http://localhost:8000/api
```

---

### 5. Spring Backend 선택 실행

기본적으로 Spring Backend는 파일 구조만 만들어져 있으며, 현재 MVP 실행에는 필요하지 않습니다.

필요시 실행:

```bash
cd backend
mvn spring-boot:run
```

## API 요약

- `POST /api/documents` - 문서 업로드
- `GET /api/documents` - 문서 목록 조회
- `GET /api/documents/{id}` - 문서 상세 조회
- `GET /api/documents/{id}/file` - 원본 PDF 조회
- `DELETE /api/documents/{id}` - 문서 삭제
- `POST /api/chat/ask` - Q&A 질문
- `POST /api/chat/debug-retrieve` - 검색 결과 디버깅
- `GET /api/documents/{id}/pages/{page_number}/metadata` - PDF 페이지 메타데이터 조회
- `GET /api/documents/{id}/pages/{page_number}/image` - PDF 페이지 이미지 조회
- `GET /api/documents/{id}/chunks/{chunk_id}/highlight` - 하이라이트 좌표 조회
- `GET /api/documents/{id}/chunks/{chunk_id}/highlighted-file` - 강조 표시된 PDF 조회

## 주의사항

- `python/` 폴더는 메인 구현 경로입니다.
- `PyMuPDF`는 PDF 텍스트 블록 추출, 페이지 이미지 렌더링, 하이라이트 PDF 생성을 담당합니다.
- OCR은 PyMuPDF로 충분한 텍스트를 추출하지 못하는 스캔본/이미지 기반 PDF에서 주로 사용됩니다.
- `LlamaParse`는 선택적으로 사용할 수 있으며, 실패 시 설정에 따라 PyMuPDF 경로로 fallback할 수 있습니다.
- 현재 MVP에서는 `frontend`가 `python` FastAPI 서버를 직접 호출합니다.
- `backend`는 선택 사항이며, 현재 실행하지 않아도 서비스 핵심 기능은 동작합니다.
