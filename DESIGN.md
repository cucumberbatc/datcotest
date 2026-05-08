# 설계 문서

## 1. 목표

이 서비스의 목표는 여러 PDF 문서를 업로드한 뒤, 질문에 대해 문서 근거 기반으로 답변하고, 어떤 파일의 몇 페이지 어느 문단에서 답이 나왔는지를 함께 보여주는 것입니다.

이번 MVP는 면접 과제용 데모에 맞춰 다음 두 가지에 집중했습니다.

- 답변보다 근거 추적이 먼저 보이는 UX
- 설치와 실행이 단순한 구조

## 2. 아키텍처

### 프론트엔드

- Next.js App Router
- 단일 화면 3-pane 레이아웃
  - 좌측: 문서 목록 / 업로드
  - 중앙: 채팅 / 답변 / 출처 비교
  - 우측: 문서 페이지 뷰어 / 근거 문단 강조

### Spring 백엔드

- Spring Boot 3
- Apache PDFBox 기반 PDF 텍스트 추출
- 메모리 기반 문서 인덱스
- 질문 시 간단한 lexical retrieval 수행

### Python RAG 서비스

- FastAPI
- LangChain
  - `langchain-openai`
  - `ChatOpenAI`
  - `OpenAIEmbeddings`
  - `FAISS`
- PyMuPDF(`fitz`)

권장 실서비스 경로는 이 Python 서비스를 중심으로 두는 구조입니다.

- PDF ingestion
- chunk + bbox 추출
- embedding 생성
- vector retrieval
- LLM answer generation
- highlighted PDF 생성

## 3. 백엔드 처리 흐름

### 권장 처리 흐름

1. 사용자가 PDF 업로드
2. Python 서비스가 `PyMuPDF`로 페이지별 text block / bbox 추출
3. chunk metadata와 함께 vector index 적재
4. 질문 입력 시 embedding 기반 retrieval
5. top-k chunk를 컨텍스트로 OpenAI LLM 답변 생성
6. citation에 연결된 bbox를 사용해 하이라이트 PDF 생성 또는 좌표 반환

### 문서 업로드

1. 사용자가 여러 PDF를 업로드
2. 서버가 파일을 로컬 스토리지에 저장
3. PDFBox로 페이지별 텍스트를 추출
4. 페이지 텍스트를 문단 단위로 분리
5. 문서 메타데이터와 페이지/문단 데이터를 메모리에 적재

### 질문 처리

1. 질문을 토큰화
2. 선택된 문서 범위 안에서 문단 후보 생성
3. 질의 토큰과 문단 토큰 간의 lexical score 계산
4. 상위 3개 문단을 근거로 채택
5. 근거 문단을 조합해 답변 생성
6. 파일명 / 페이지 / 문단 번호 / excerpt 반환

### 근거 부족 처리

- 상위 score가 낮거나 중복 토큰이 거의 없으면
- `문서에서 확인 불가` 응답과 가이드 문구 반환

## 4. 프론트엔드 UX 설계

디자인 링크의 핸드오프 번들을 해석해 `Atlas` 톤으로 구현했습니다.

- 차분한 엔터프라이즈 느낌의 라이트 테마
- 좌/중/우 3-pane 고정 구조
- 답변 본문 내 citation 배지 클릭 가능
- 출처 탭을 전환하면 우측 뷰어가 해당 페이지로 이동
- 모바일 폭에서는 pane이 세로로 스택되도록 반응형 처리

## 5. 현재 한계와 확장 포인트

### 현재 한계

- 외부 LLM 연동 없이 추출 기반 답변만 제공
- 실제 PDF 좌표 하이라이트는 미구현
- 서버 재시작 시 인덱스는 메모리에서 사라짐
- OCR 미지원

### 확장 포인트

- OpenAI/Anthropic 연동으로 답변 품질 향상
- BM25 또는 Vector DB 도입
- PDF.js 기반 실제 페이지 렌더링
- bbox 저장 후 좌표 하이라이트
- OCR 및 표 추출
- 사용자/권한 단위 문서 분리
