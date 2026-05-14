# datcotest

PDF 기반 Q&A 과제 MVP

## 서비스

- `frontend`: Next.js 3 페인 UI
- `backend`: Spring Boot 프로토타입 API (파일 구조만 생성됨, 실행 불필요)
- `python`: LangChain, OpenAI, FAISS, PyMuPDF를 사용한 메인 RAG 서비스

## 권장 아키텍처

실제 과제 흐름을 위한 권장 경로:

- `frontend`: UI 및 사용자 워크플로우 처리
- `python`: 문서 수집, 임베딩, 검색, LLM 답변, PDF 강조 표시 처리
- `backend`: 필요시 나중에 Spring 통합 계층 또는 지원 API로 사용 가능

실제로는:

- Frontend → Python RAG 서비스
- Optional: Spring → Python 프록시/통합

## 실행 방법

### 준비 작업

먼저 Python 가상환경과 모든 모델을 초기화해야 합니다.

```bash
cd python
pipenv install
copy .env.example .env  # Windows 또는 cp .env.example .env (macOS/Linux)
pipenv run python scripts/warmup.py  # 모든 모델 다운로드 및 초기화
```

### 1. Python RAG 서비스 시작

```bash
cd python
pipenv run uvicorn app.main:app --port 8000
```

파일 업로드, OCR, 임베딩 및 성능 테스트 시에는 `--reload` 옵션 없이 사용하세요.

### 2. Frontend 시작

```bash
cd frontend
npm install
npm run dev
```

Frontend API 대상 설정:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000/api
```

### 3. Spring Backend (선택사항)

기본적으로 파일 구조만 만들어져 있으므로 실행하지 않아도 됩니다.

필요시 실행:

```bash
cd backend
mvn spring-boot:run
```

## API 요약

- `POST /api/documents` - 문서 업로드
- `GET /api/documents` - 문서 목록 조회
- `GET /api/documents/{id}` - 문서 상세 조회
- `GET /api/documents/{id}/file` - 강조 표시된 PDF 다운로드
- `DELETE /api/documents/{id}` - 문서 삭제
- `POST /api/chat/ask` - Q&A 질문

## 주의사항

- `python/` 폴더는 메인 구현 경로입니다.
- `PyMuPDF`는 실제 PDF 텍스트 블록을 추출하고 강조 표시된 PDF를 생성합니다.
- 현재 frontend 뷰어는 텍스트 기반입니다. 향후 PDF.js 뷰어로 전환할 예정입니다.
- `Pipfile`과 `Pipfile.lock`은 Python 의존성의 소스입니다.

아키텍처 참고사항은 [DESIGN.md](DESIGN.md)를 참조하세요.
