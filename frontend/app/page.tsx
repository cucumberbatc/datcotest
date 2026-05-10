"use client";

import { type CSSProperties, useEffect, useMemo, useRef, useState } from "react";

type DocumentSummary = {
  id: string;
  fileName: string;
  sizeBytes: number;
  uploadedAt: string;
  status: string;
  pageCount: number;
};

type DocumentPage = {
  pageNumber: number;
  paragraphs: string[];
};

type DocumentDetail = DocumentSummary & {
  downloadUrl: string;
  pages: DocumentPage[];
};

type Source = {
  citationNumber: number;
  documentId: string;
  fileName: string;
  pageNumber: number;
  paragraphIndex: number;
  paragraphEndIndex: number;
  locationLabel: string;
  excerpt: string;
  score: number;
  chunkId?: string;
  highlightFileUrl?: string | null;
};

type AskResponse = {
  question: string;
  answer: string;
  noEvidence: boolean;
  noEvidenceNote: string | null;
  sources: Source[];
  elapsedMs: number;
};

type UserMessage = {
  id: string;
  role: "user";
  text: string;
};

type AssistantMessage = {
  id: string;
  role: "assistant";
  answer: string;
  renderedAnswer: string;
  sources: Source[];
  noEvidence: boolean;
  noEvidenceNote: string | null;
  elapsedMs: number;
  streaming: boolean;
};

type ChatMessage = UserMessage | AssistantMessage;

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000/api";
const API_ORIGIN = API_BASE.replace(/\/api$/, "");

function apiUrl(path: string) {
  return path.startsWith("http") ? path : `${API_ORIGIN}${path}`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatUploadDate(isoDate: string) {
  return new Date(isoDate).toLocaleString("ko-KR", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function makeId(prefix: string) {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function renderAnswer(answer: string, activeCitation: number | null, onActivateCitation: (citation: number) => void) {
  const pieces = answer.split(/(\[\d+\])/g).filter(Boolean);
  return pieces.map((piece, index) => {
    const match = piece.match(/^\[(\d+)\]$/);
    if (!match) {
      return <span key={`${piece}-${index}`}>{piece}</span>;
    }
    const citation = Number(match[1]);
    return (
      <button
        key={`${piece}-${index}`}
        className="cite"
        data-active={activeCitation === citation}
        onClick={() => onActivateCitation(citation)}
        type="button"
      >
        {citation}
      </button>
    );
  });
}

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const chatBodyRef = useRef<HTMLDivElement | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [documentCache, setDocumentCache] = useState<Record<string, DocumentDetail>>({});
  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [viewerDocId, setViewerDocId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [composer, setComposer] = useState("");
  const [scope, setScope] = useState<"all" | "selected">("all");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [activeCitation, setActiveCitation] = useState<number | null>(null);
  const [selectedSource, setSelectedSource] = useState<Source | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const totalBytes = useMemo(
    () => documents.reduce((sum, document) => sum + document.sizeBytes, 0),
    [documents]
  );
  const totalPages = useMemo(
    () => documents.reduce((sum, document) => sum + document.pageCount, 0),
    [documents]
  );

  const activeDocument = viewerDocId ? documentCache[viewerDocId] : null;
  const firstRun = documents.length === 0 && !uploading;

  useEffect(() => {
    void loadDocuments();
  }, []);

  useEffect(() => {
    if (!toast) {
      return;
    }
    const timer = window.setTimeout(() => setToast(null), 3200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (!chatBodyRef.current) {
      return;
    }
    chatBodyRef.current.scrollTo({
      top: chatBodyRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [messages]);

  async function loadDocuments() {
    const response = await fetch(`${API_BASE}/documents`, {
      cache: "no-store"
    });
    if (!response.ok) {
      throw new Error("문서 목록을 불러오지 못했습니다.");
    }
    const payload = (await response.json()) as DocumentSummary[];
    setDocuments(payload);

    if (!payload.length) {
      setActiveDocId(null);
      setViewerDocId(null);
      return;
    }

    if (!activeDocId) {
      setActiveDocId(payload[0].id);
    }
  }

  async function ensureDocumentLoaded(documentId: string) {
    if (documentCache[documentId]) {
      return documentCache[documentId];
    }

    const response = await fetch(`${API_BASE}/documents/${documentId}`, {
      cache: "no-store"
    });
    if (!response.ok) {
      throw new Error("문서 상세 정보를 불러오지 못했습니다.");
    }

    const detail = (await response.json()) as DocumentDetail;
    setDocumentCache((current) => ({
      ...current,
      [documentId]: detail
    }));
    return detail;
  }

  async function handleUpload(files: FileList | null) {
    if (!files || files.length === 0) {
      return;
    }

    const formData = new FormData();
    Array.from(files).forEach((file) => formData.append("files", file));

    setUploading(true);
    try {
      const response = await fetch(`${API_BASE}/documents`, {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      setToast(`${files.length}개 PDF 업로드 및 인덱싱이 완료되었습니다.`);
      await loadDocuments();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "업로드 중 오류가 발생했습니다.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  async function handleDelete(documentId: string) {
    try {
      const response = await fetch(`${API_BASE}/documents/${documentId}`, {
        method: "DELETE"
      });
      if (!response.ok) {
        throw new Error("문서를 삭제하지 못했습니다.");
      }

      setDocumentCache((current) => {
        const next = { ...current };
        delete next[documentId];
        return next;
      });
      if (viewerDocId === documentId) {
        setViewerDocId(null);
        setSelectedSource(null);
        setActiveCitation(null);
      }
      await loadDocuments();
    } catch (error) {
      setToast(error instanceof Error ? error.message : "문서 삭제에 실패했습니다.");
    }
  }

  async function openDocument(documentId: string) {
    setActiveDocId(documentId);
    setViewerDocId(documentId);
    try {
      await ensureDocumentLoaded(documentId);
    } catch (error) {
      setToast(error instanceof Error ? error.message : "문서를 열지 못했습니다.");
    }
  }

  async function activateSource(source: Source) {
    setActiveCitation(source.citationNumber);
    setSelectedSource(source);
    setViewerDocId(source.documentId);
    try {
      await ensureDocumentLoaded(source.documentId);
    } catch (error) {
      setToast(error instanceof Error ? error.message : "근거 문서를 여는 중 오류가 발생했습니다.");
    }
  }

  function activateCitation(citationNumber: number) {
    setActiveCitation(citationNumber);
    const latestAssistant = [...messages]
        .reverse()
        .find((message): message is AssistantMessage => message.role === "assistant" && message.sources.length > 0);

    if (!latestAssistant) {
      return;
    }

    const source = latestAssistant.sources.find((item) => item.citationNumber === citationNumber);
    if (source) {
      void activateSource(source);
    }
  }

  async function askQuestion(nextQuestion?: string) {
    const question = (nextQuestion ?? composer).trim();
    if (!question || busy) {
      return;
    }

    const scopedDocumentIds =
      scope === "selected" && activeDocId ? [activeDocId] : documents.map((document) => document.id);

    const userMessage: UserMessage = {
      id: makeId("user"),
      role: "user",
      text: question
    };

    setMessages((current) => [...current, userMessage]);
    setComposer("");
    setBusy(true);
    setSelectedSource(null);
    setActiveCitation(null);

    try {
      const response = await fetch(`${API_BASE}/chat/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question,
          documentIds: scopedDocumentIds
        })
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload = (await response.json()) as AskResponse;
      const assistantId = makeId("assistant");
      const assistantMessage: AssistantMessage = {
        id: assistantId,
        role: "assistant",
        answer: payload.answer,
        renderedAnswer: payload.noEvidence ? payload.answer : "",
        sources: payload.sources,
        noEvidence: payload.noEvidence,
        noEvidenceNote: payload.noEvidenceNote,
        elapsedMs: payload.elapsedMs,
        streaming: !payload.noEvidence
      };

      setMessages((current) => [...current, assistantMessage]);

      if (payload.noEvidence) {
        return;
      }

      let cursor = 0;
      const answer = payload.answer;
      const intervalId = window.setInterval(() => {
        cursor = Math.min(answer.length, cursor + 12);
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId && message.role === "assistant"
              ? {
                  ...message,
                  renderedAnswer: answer.slice(0, cursor),
                  streaming: cursor < answer.length
                }
              : message
          )
        );

        if (cursor >= answer.length) {
          window.clearInterval(intervalId);
          if (payload.sources[0]) {
            void activateSource(payload.sources[0]);
          }
        }
      }, 24);
    } catch (error) {
      setToast(error instanceof Error ? error.message : "질문 처리 중 오류가 발생했습니다.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept="application/pdf,.pdf"
        multiple
        onChange={(event) => void handleUpload(event.target.files)}
      />

      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">A</div>
          <div>
            <div className="brand-name">Atlas</div>
            <div className="brand-tag">Grounded PDF Assistant</div>
          </div>
        </div>
        <div className="topbar-meta">
          <span>Grounded Q&A</span>
          <span className="sep">/</span>
          <span>OpenAI + LangChain + PyMuPDF</span>
        </div>
        <div className="topbar-actions">
          <button className="btn btn-ghost" onClick={() => fileInputRef.current?.click()} type="button">
            <UploadIcon />
            PDF 업로드
          </button>
        </div>
      </header>

      <aside className="pane pane-left">
        <div className="pane-header">
          <div className="pane-title">
            Documents
            <span className="pill">{documents.length}</span>
          </div>
          <button className="icon-btn" onClick={() => fileInputRef.current?.click()} type="button">
            <PlusIcon />
          </button>
        </div>

        <div
          className="upload-zone"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            void handleUpload(event.dataTransfer.files);
          }}
        >
          <div className="upload-zone-title">여기에 PDF 끌어 놓기</div>
          <div className="upload-zone-sub">또는 클릭해서 선택 · 최대 200MB</div>
        </div>

        <div className="doc-list">
          {documents.map((document) => (
            <div
              key={document.id}
              className="doc-row"
              data-active={activeDocId === document.id}
              onClick={() => void openDocument(document.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  void openDocument(document.id);
                }
              }}
              role="button"
              tabIndex={0}
            >
              <div className="doc-icon">PDF</div>
              <div className="doc-body">
                <div className="doc-name">{document.fileName}</div>
                <div className="doc-meta-line">
                  <span>{document.pageCount}p</span>
                  <span className="dot" />
                  <span>{formatBytes(document.sizeBytes)}</span>
                  <span className="dot" />
                  <span>{formatUploadDate(document.uploadedAt)}</span>
                </div>
              </div>
              <button
                className="icon-btn subtle"
                onClick={(event) => {
                  event.stopPropagation();
                  void handleDelete(document.id);
                }}
                type="button"
              >
                <CloseIcon />
              </button>
            </div>
          ))}
        </div>

        <div className="left-footer">
          <div className="quota-line">
            <span>Total size</span>
            <span>{formatBytes(totalBytes)} / 200MB</span>
          </div>
          <div className="quota-bar">
            <div style={{ width: `${Math.min(100, (totalBytes / (200 * 1024 * 1024)) * 100)}%` }} />
          </div>
          <div className="quota-line muted">
            <span>{totalPages.toLocaleString("ko-KR")} pages indexed</span>
            <span>{documents.length} docs</span>
          </div>
        </div>
      </aside>

      <main className="pane pane-center">
        <div className="pane-header">
          <div className="pane-title">
            새 대화
            <span className="pill accent">{scope === "all" ? "all docs" : "selected"}</span>
          </div>
          <div className="pane-actions">
            <button className="btn btn-ghost" onClick={() => setScope(scope === "all" ? "selected" : "all")} type="button">
              {scope === "all" ? "선택 문서만" : "전체 문서"}
            </button>
            <button
              className="btn btn-ghost"
              onClick={() => {
                setMessages([]);
                setSelectedSource(null);
                setActiveCitation(null);
              }}
              type="button"
            >
              새로 시작
            </button>
          </div>
        </div>

        <div className="chat-body" ref={chatBodyRef}>
          {firstRun ? (
            <section className="empty-state">
              <div className="empty-badge">Atlas</div>
              <h1>먼저 PDF를 업로드해 주세요</h1>
              <p>
                여러 개의 PDF를 업로드한 뒤 질문하면, 답변과 함께 파일명, 페이지 번호, 문단 위치까지
                함께 제시합니다.
              </p>
              <div className="suggest-grid">
                <button className="suggest-card" onClick={() => fileInputRef.current?.click()} type="button">
                  <strong>PDF 업로드 시작</strong>
                  <span>여러 문서를 한 번에 올리고 바로 인덱싱할 수 있습니다.</span>
                </button>
                <div className="suggest-card static-card">
                  <strong>답변 방식</strong>
                  <span>LLM이 검색된 근거를 바탕으로 답변하고, 출처를 함께 보여줍니다.</span>
                </div>
              </div>
            </section>
          ) : messages.length === 0 ? (
            <section className="empty-state">
              <div className="empty-badge">Workspace</div>
              <h1>업로드한 문서에 물어보세요</h1>
              <p>
                질문을 입력하면 검색된 근거 문단을 바탕으로 답변을 생성하고, 우측 패널에서 관련 페이지와
                문단을 바로 확인할 수 있습니다.
              </p>
              <div className="suggest-grid">
                <div className="suggest-card static-card">
                  <strong>질문 팁</strong>
                  <span>수치, 일정, 정책 범위, 비교 요청처럼 문서에 명시된 내용을 물어보면 가장 정확합니다.</span>
                </div>
                <div className="suggest-card static-card">
                  <strong>근거 확인</strong>
                  <span>답변의 citation 또는 출처 카드를 누르면 연결된 문서 위치로 이동합니다.</span>
                </div>
              </div>
            </section>
          ) : (
            messages.map((message) =>
              message.role === "user" ? (
                <div key={message.id} className="message-row user">
                  <div className="message-avatar user">나</div>
                  <div className="message-bubble user">{message.text}</div>
                </div>
              ) : (
                <div key={message.id} className="message-row">
                  <div className="message-avatar">A</div>
                  <div className="message-content">
                    {message.streaming ? (
                      <div className="answer-text streaming">
                        {renderAnswer(message.renderedAnswer, activeCitation, activateCitation)}
                        <span className="caret" />
                      </div>
                    ) : message.noEvidence ? (
                      <div className="no-evidence-card">
                        <div className="no-evidence-title">문서에서 확인 불가</div>
                        <p>{message.noEvidenceNote}</p>
                        <div className="card-actions">
                          <button className="btn" onClick={() => fileInputRef.current?.click()} type="button">
                            <UploadIcon />
                            관련 문서 추가
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="answer-text">
                        {renderAnswer(message.answer, activeCitation, activateCitation)}
                      </div>
                    )}

                    {!message.noEvidence && message.sources.length > 0 && (
                      <div className="source-panel">
                        <div className="source-tabs">
                          {message.sources.map((source) => (
                            <button
                              key={source.citationNumber}
                              className="source-tab"
                              data-active={activeCitation === source.citationNumber}
                              onClick={() => void activateSource(source)}
                              type="button"
                            >
                              <span className="source-number">{source.citationNumber}</span>
                              <span className="source-file">{source.fileName}</span>
                              <span className="source-page">p.{source.pageNumber}</span>
                            </button>
                          ))}
                        </div>

                        {(() => {
                          const activeSource =
                            message.sources.find((source) => source.citationNumber === activeCitation) ??
                            message.sources[0];

                          return (
                            <div className="source-card">
                              <div className="source-card-head">
                                <span className="doc-icon small">PDF</span>
                                <strong>{activeSource.fileName}</strong>
                                <span className="pill">{activeSource.locationLabel}</span>
                              </div>
                              <p>{activeSource.excerpt}</p>
                              <div className="source-card-foot">
                                <div className="card-actions">
                                  <button className="btn" onClick={() => void activateSource(activeSource)} type="button">
                                    뷰어에서 열기
                                  </button>
                                  {activeSource.highlightFileUrl ? (
                                    <a
                                      className="btn btn-ghost"
                                      href={apiUrl(activeSource.highlightFileUrl)}
                                      rel="noreferrer"
                                      target="_blank"
                                    >
                                      하이라이트 PDF
                                    </a>
                                  ) : null}
                                </div>
                                <span className="confidence">score {activeSource.score.toFixed(2)}</span>
                              </div>
                            </div>
                          );
                        })()}
                    </div>
                    )}

                    {!message.noEvidence && !message.streaming && (
                      <div className="message-meta">{(message.elapsedMs / 1000).toFixed(1)}s</div>
                    )}
                  </div>
                </div>
              )
            )
          )}
        </div>

        <div className="composer-wrap">
          <div className="composer">
            <textarea
              rows={1}
              placeholder="문서에 대해 무엇이든 물어보세요…"
              value={composer}
              onChange={(event) => setComposer(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void askQuestion();
                }
              }}
            />
            <button className="send-btn" disabled={busy || !composer.trim()} onClick={() => void askQuestion()} type="button">
              {busy ? <SpinnerIcon /> : <SendIcon />}
            </button>
          </div>
          <div className="composer-meta">
            <span>{scope === "all" ? `전 문서(${documents.length}) 검색` : "선택 문서만 검색"}</span>
            <span>{uploading ? "업로드 중…" : busy ? "답변 생성 중…" : "Ready"}</span>
          </div>
        </div>
      </main>

      <aside className="pane pane-right">
        <div className="pane-header">
          <div className="pane-title">Source viewer</div>
          {activeDocument ? (
            <a className="btn btn-ghost" href={apiUrl(activeDocument.downloadUrl)} rel="noreferrer" target="_blank">
              원본 PDF
            </a>
          ) : null}
        </div>

        {!activeDocument ? (
          <div className="viewer-empty">
            <div className="viewer-empty-box" />
            <strong>출처를 선택하면 여기에 열립니다</strong>
            <p>좌측 문서를 클릭하거나 답변의 출처 카드를 선택해 보세요.</p>
          </div>
        ) : (
          <ViewerPane document={activeDocument} selectedSource={selectedSource} />
        )}
      </aside>

      {toast ? <div className="toast">{toast}</div> : null}
    </div>
  );
}

function ViewerPane({
  document,
  selectedSource
}: {
  document: DocumentDetail;
  selectedSource: Source | null;
}) {
  const [zoom, setZoom] = useState(100);
  const [currentPage, setCurrentPage] = useState(1);

  useEffect(() => {
    if (!selectedSource || selectedSource.documentId !== document.id) {
      return;
    }

    setCurrentPage(selectedSource.pageNumber);
  }, [document.id, selectedSource]);

  useEffect(() => {
    setCurrentPage((page) => Math.min(Math.max(page, 1), document.pageCount));
  }, [document.pageCount]);

  const page = document.pages.find((item) => item.pageNumber === currentPage) ?? document.pages[0];
  const isActiveSourcePage = selectedSource?.documentId === document.id && selectedSource.pageNumber === page.pageNumber;
  const canGoPrev = currentPage > 1;
  const canGoNext = currentPage < document.pageCount;
  const pageTextStyle = {
    "--page-text-scale": zoom / 100
  } as CSSProperties;

  function goToPrevPage() {
    setCurrentPage((pageNumber) => Math.max(1, pageNumber - 1));
  }

  function goToNextPage() {
    setCurrentPage((pageNumber) => Math.min(document.pageCount, pageNumber + 1));
  }

  return (
    <>
      <div className="viewer-toolbar">
        <div className="viewer-toolbar-group">
          <button className="icon-btn" onClick={() => setZoom((current) => Math.max(70, current - 10))} type="button">
            -
          </button>
          <span>{zoom}%</span>
          <button className="icon-btn" onClick={() => setZoom((current) => Math.min(150, current + 10))} type="button">
            +
          </button>
        </div>
      </div>

      <div className="viewer-canvas">
        <div className="viewer-stage">
          <button
            aria-label="Previous page"
            className="viewer-side-nav"
            disabled={!canGoPrev}
            onClick={goToPrevPage}
            type="button"
          >
            <span>&lt;</span>
          </button>

          <div className="viewer-page-shell">
            <div className="pdf-page" data-active={isActiveSourcePage}>
              <div className="pdf-page-inner" style={pageTextStyle}>
                <div className="pdf-heading">Page {page.pageNumber}</div>
                {page.paragraphs.map((paragraph, index) => {
                  const paragraphNumber = index + 1;
                  const highlighted =
                    isActiveSourcePage &&
                    paragraphNumber >= selectedSource.paragraphIndex &&
                    paragraphNumber <= selectedSource.paragraphEndIndex;
                  return (
                    <p key={`${page.pageNumber}-${index}`} className="pdf-paragraph" data-highlight={highlighted}>
                      {paragraph}
                    </p>
                  );
                })}
              </div>
              <div className="pdf-footer">
                <span>{document.fileName}</span>
                <span>{page.pageNumber}</span>
              </div>
            </div>
          </div>

          <button
            aria-label="Next page"
            className="viewer-side-nav"
            disabled={!canGoNext}
            onClick={goToNextPage}
            type="button"
          >
            <span>&gt;</span>
          </button>
        </div>

        <div className="viewer-pagination">
          <button
            aria-label="Previous page"
            className="icon-btn"
            disabled={!canGoPrev}
            onClick={goToPrevPage}
            type="button"
          >
            &lt;
          </button>
          <div className="viewer-page-indicator">
            <span>{currentPage}</span>
            <span>/</span>
            <span>{document.pageCount}</span>
          </div>
          <button
            aria-label="Next page"
            className="icon-btn"
            disabled={!canGoNext}
            onClick={goToNextPage}
            type="button"
          >
            &gt;
          </button>
        </div>
      </div>
    </>
  );
}

function UploadIcon() {
  return (
    <svg className="icon" viewBox="0 0 16 16" fill="none">
      <path d="M8 11V3M8 3L5 6M8 3L11 6M3 11.5V13h10v-1.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg className="icon" viewBox="0 0 16 16" fill="none">
      <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg className="icon" viewBox="0 0 16 16" fill="none">
      <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg className="icon" viewBox="0 0 16 16" fill="none">
      <path d="M14 2L2 7.2l4.4 1.4L7.8 13 14 2z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg className="icon spin" viewBox="0 0 16 16" fill="none">
      <path d="M8 2a6 6 0 106 6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}
