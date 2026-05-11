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
  sectionTitle?: string | null;
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

type PdfRect = {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
};

type PageMetadata = {
  documentId: string;
  pageNumber: number;
  width: number;
  height: number;
  imageUrl: string;
};

type HighlightMetadata = {
  documentId: string;
  chunkId: string;
  pageNumber: number;
  rects: PdfRect[];
};

type DebugRetrieveHit = {
  rank: number;
  score: number;
  chunkId: string;
  documentId: string;
  fileName: string;
  pageNumber: number;
  paragraphIndex: number;
  paragraphEndIndex: number;
  sectionTitle?: string | null;
  text: string;
};

type DebugRetrieveResult = {
  question: string;
  hits: DebugRetrieveHit[];
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
const LEFT_PANE_WIDTH = 280;
const VIEWER_MIN_WIDTH = 520;
const VIEWER_DEFAULT_WIDTH = 960;
const CENTER_MIN_WIDTH = 420;
const SPLITTER_WIDTH = 12;

function apiUrl(path: string) {
  return path.startsWith("http") ? path : `${API_ORIGIN}${path}`;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
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
  const appShellRef = useRef<HTMLDivElement | null>(null);
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
  const [debugResult, setDebugResult] = useState<DebugRetrieveResult | null>(null);
  const [showDebugModal, setShowDebugModal] = useState(false);
  const [viewerBaseWidth, setViewerBaseWidth] = useState(VIEWER_DEFAULT_WIDTH);
  const [viewerZoom, setViewerZoom] = useState(100);
  const [shellWidth, setShellWidth] = useState(0);
  const [resizingViewer, setResizingViewer] = useState(false);

  const modalOverlayStyle: CSSProperties = {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.6)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 9999
  };

  const modalStyle: CSSProperties = {
    backgroundColor: "#fff",
    borderRadius: "8px",
    boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
    maxWidth: "600px",
    width: "90%",
    maxHeight: "80vh",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden"
  };

  const modalHeaderStyle: CSSProperties = {
    padding: "16px 20px",
    borderBottom: "1px solid #e5e7eb",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center"
  };

  const modalContentStyle: CSSProperties = {
    overflowY: "auto",
    padding: "16px 20px",
    flex: 1
  };

  const hitCardStyle: CSSProperties = {
    marginBottom: "16px",
    padding: "12px",
    backgroundColor: "#f9fafb",
    borderRadius: "6px",
    border: "1px solid #e5e7eb"
  };

  const hitHeaderStyle: CSSProperties = {
    display: "flex",
    gap: "12px",
    alignItems: "center",
    marginBottom: "8px",
    fontSize: "13px"
  };

  const hitTextStyle: CSSProperties = {
    margin: 0,
    fontSize: "13px",
    lineHeight: "1.5",
    color: "#374151"
  };

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
  const viewerZoomMultiplier = viewerZoom > 100 ? 1 + ((viewerZoom - 100) / 100) * 0.8 : 1;
  const maxViewerWidth = Math.max(VIEWER_MIN_WIDTH, shellWidth - LEFT_PANE_WIDTH - SPLITTER_WIDTH - CENTER_MIN_WIDTH);
  const viewerPaneWidth = clamp(Math.round(viewerBaseWidth * viewerZoomMultiplier), VIEWER_MIN_WIDTH, maxViewerWidth);
  const isStackedLayout = shellWidth > 0 && shellWidth <= 1100;

  useEffect(() => {
    void loadDocuments();
  }, []);

  useEffect(() => {
    if (!appShellRef.current) {
      return;
    }

    const shell = appShellRef.current;
    const updateShellWidth = () => setShellWidth(shell.clientWidth);
    updateShellWidth();

    const observer = new ResizeObserver(updateShellWidth);
    observer.observe(shell);
    return () => observer.disconnect();
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

  useEffect(() => {
    if (!resizingViewer) {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      return;
    }

    function handlePointerMove(event: PointerEvent) {
      if (!appShellRef.current) {
        return;
      }

      const rect = appShellRef.current.getBoundingClientRect();
      const desiredWidth = rect.right - event.clientX;
      setViewerBaseWidth(clamp(Math.round(desiredWidth / viewerZoomMultiplier), VIEWER_MIN_WIDTH, maxViewerWidth));
    }

    function stopResizing() {
      setResizingViewer(false);
    }

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopResizing);
    window.addEventListener("pointercancel", stopResizing);

    return () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopResizing);
      window.removeEventListener("pointercancel", stopResizing);
    };
  }, [maxViewerWidth, resizingViewer, viewerZoomMultiplier]);

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

  async function debugRetrieve() {
    const question = composer.trim();
    if (!question || busy) {
      return;
    }

    const scopedDocumentIds =
      scope === "selected" && activeDocId ? [activeDocId] : documents.map((document) => document.id);

    try {
      const response = await fetch(`${API_BASE}/chat/debug-retrieve`, {
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

      const payload = (await response.json()) as DebugRetrieveResult;
      setDebugResult(payload);
      setShowDebugModal(true);
    } catch (error) {
      setToast(error instanceof Error ? error.message : "디버그 검색 중 오류가 발생했습니다.");
    }
  }

  return (
    <div
      ref={appShellRef}
      className="app-shell"
      style={
        {
          "--viewer-pane-width": `${viewerPaneWidth}px`,
          "--viewer-splitter-width": `${isStackedLayout ? 0 : SPLITTER_WIDTH}px`
        } as CSSProperties
      }
    >
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
            <div className="brand-name">DocQ</div>

          </div>
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
            <div style={{ display: "flex", gap: "8px" }}>
              <button className="send-btn" disabled={busy || !composer.trim()} onClick={() => void debugRetrieve()} title="검색 결과 디버깅" type="button">
                {busy ? "..." : "🔍"}
              </button>
              <button className="send-btn" disabled={busy || !composer.trim()} onClick={() => void askQuestion()} type="button">
                {busy ? <SpinnerIcon /> : <SendIcon />}
              </button>
            </div>
          </div>
          <div className="composer-meta">
            <span>{scope === "all" ? `전 문서(${documents.length}) 검색` : "선택 문서만 검색"}</span>
            <span>{uploading ? "업로드 중…" : busy ? "답변 생성 중…" : "Ready"}</span>
          </div>
        </div>
      </main>

      <div
        aria-hidden={isStackedLayout}
        className="pane-splitter"
        data-active={resizingViewer}
        onPointerDown={(event) => {
          if (isStackedLayout) {
            return;
          }
          event.preventDefault();
          setResizingViewer(true);
        }}
        onKeyDown={(event) => {
          if (isStackedLayout) {
            return;
          }
          if (event.key === "ArrowLeft") {
            event.preventDefault();
            setViewerBaseWidth((current) => clamp(current + 32, VIEWER_MIN_WIDTH, maxViewerWidth));
          }
          if (event.key === "ArrowRight") {
            event.preventDefault();
            setViewerBaseWidth((current) => clamp(current - 32, VIEWER_MIN_WIDTH, maxViewerWidth));
          }
        }}
        role="separator"
        tabIndex={isStackedLayout ? -1 : 0}
      />

      <aside className="pane pane-right">
        <div className="pane-header">
          <div className="pane-title">Source viewer</div>
          <div className="pane-actions viewer-header-actions">
            <div className="viewer-toolbar-group">
              <button
                className="icon-btn"
                disabled={!activeDocument || viewerZoom <= 100}
                onClick={() => setViewerZoom((current) => Math.max(50, current - 10))}
                type="button"
              >
                -
              </button>
              <span className="viewer-header-zoom-label">{viewerZoom}%</span>
              <button
                className="icon-btn"
                disabled={!activeDocument}
                onClick={() => setViewerZoom((current) => Math.min(220, current + 10))}
                type="button"
              >
                +
              </button>
              <button className="btn btn-ghost viewer-fit-btn" disabled={!activeDocument} onClick={() => setViewerZoom(100)} type="button">
                Fit
              </button>
            </div>
            {activeDocument ? (
              <a className="btn btn-ghost viewer-header-link" href={apiUrl(activeDocument.downloadUrl)} rel="noreferrer" target="_blank">
                원본 PDF
              </a>
            ) : null}
          </div>
          {false && activeDocument ? (
            <a className="btn btn-ghost" href={apiUrl(activeDocument.downloadUrl)} rel="noreferrer" target="_blank">
              원본 PDF
            </a>
          ) : null}
        </div>

        {!activeDocument ? (
          <div className="viewer-empty">
            <strong>출처를 선택하면 여기에 열립니다</strong>
            <button className="btn" onClick={() => fileInputRef.current?.click()} type="button">
              <UploadIcon />
              PDF 업로드
            </button>
            <p>좌측 문서를 클릭하거나 답변의 출처 카드를 선택해 보세요.</p>
          </div>
        ) : (
          <ViewerPane
            document={activeDocument}
            imageZoom={viewerZoom}
            selectedSource={selectedSource}
          />
        )}
      </aside>

      {toast ? <div className="toast">{toast}</div> : null}

      {showDebugModal && debugResult ? (
        <div style={modalOverlayStyle} onClick={() => setShowDebugModal(false)}>
          <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
            <div style={modalHeaderStyle}>
              <h2 style={{ margin: 0 }}>검색 디버그: "{debugResult.question}"</h2>
              <button
                onClick={() => setShowDebugModal(false)}
                style={{ background: "none", border: "none", cursor: "pointer", fontSize: "20px" }}
                type="button"
              >
                ✕
              </button>
            </div>
            <div style={modalContentStyle}>
              {debugResult.hits.length === 0 ? (
                <p style={{ color: "#999" }}>검색 결과 없음</p>
              ) : (
                debugResult.hits.map((hit) => (
                  <div key={hit.chunkId} style={hitCardStyle}>
                    <div style={hitHeaderStyle}>
                      <span style={{ fontWeight: "bold" }}>#{hit.rank}</span>
                      <span style={{ color: "#f59e0b", fontWeight: "bold" }}>{hit.score.toFixed(4)}</span>
                      <span style={{ color: "#666", fontSize: "12px" }}>{hit.fileName}</span>
                      <span style={{ color: "#666", fontSize: "12px" }}>p.{hit.pageNumber}</span>
                    </div>
                    <p style={hitTextStyle}>{hit.text}</p>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function ViewerPane({
  document,
  imageZoom,
  selectedSource
}: {
  document: DocumentDetail;
  imageZoom: number;
  selectedSource: Source | null;
}) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageMetadata, setPageMetadata] = useState<PageMetadata | null>(null);
  const [highlightMetadata, setHighlightMetadata] = useState<HighlightMetadata | null>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [viewerError, setViewerError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedSource || selectedSource.documentId !== document.id) {
      return;
    }

    setCurrentPage(selectedSource.pageNumber);
  }, [document.id, selectedSource]);

  useEffect(() => {
    setCurrentPage((page) => Math.min(Math.max(page, 1), document.pageCount));
  }, [document.pageCount]);

  useEffect(() => {
    let ignore = false;

    async function loadPageMetadata() {
      setViewerError(null);
      const response = await fetch(`${API_BASE}/documents/${document.id}/pages/${currentPage}/metadata`, {
        cache: "no-store"
      });
      if (!response.ok) {
        throw new Error("페이지 이미지를 불러오지 못했습니다.");
      }

      const payload = (await response.json()) as PageMetadata;
      if (!ignore) {
        setPageMetadata(payload);
        setImageSize({ width: 0, height: 0 });
      }
    }

    void loadPageMetadata().catch((error) => {
      if (!ignore) {
        setViewerError(error instanceof Error ? error.message : "페이지 이미지를 불러오지 못했습니다.");
        setPageMetadata(null);
      }
    });

    return () => {
      ignore = true;
    };
  }, [currentPage, document.id]);

  useEffect(() => {
    let ignore = false;

    async function loadHighlightMetadata() {
      if (!selectedSource?.chunkId || selectedSource.documentId !== document.id || selectedSource.pageNumber !== currentPage) {
        setHighlightMetadata(null);
        return;
      }

      const params = new URLSearchParams({
        paragraphStart: String(selectedSource.paragraphIndex),
        paragraphEnd: String(selectedSource.paragraphEndIndex)
      });
      const response = await fetch(
        `${API_BASE}/documents/${document.id}/chunks/${selectedSource.chunkId}/highlight?${params.toString()}`,
        { cache: "no-store" }
      );
      if (!response.ok) {
        throw new Error("하이라이트 좌표를 불러오지 못했습니다.");
      }

      const payload = (await response.json()) as HighlightMetadata;
      if (!ignore) {
        setHighlightMetadata(payload);
      }
    }

    void loadHighlightMetadata().catch((error) => {
      if (!ignore) {
        setViewerError(error instanceof Error ? error.message : "하이라이트 좌표를 불러오지 못했습니다.");
        setHighlightMetadata(null);
      }
    });

    return () => {
      ignore = true;
    };
  }, [currentPage, document.id, selectedSource]);

  useEffect(() => {
    if (!imgRef.current) {
      return;
    }

    const image = imgRef.current;
    const updateImageSize = () => {
      setImageSize({
        width: image.clientWidth,
        height: image.clientHeight
      });
    };

    updateImageSize();
    const observer = new ResizeObserver(updateImageSize);
    observer.observe(image);
    return () => observer.disconnect();
  }, [pageMetadata, imageZoom]);

  const isActiveSourcePage = selectedSource?.documentId === document.id && selectedSource.pageNumber === currentPage;
  const canGoPrev = currentPage > 1;
  const canGoNext = currentPage < document.pageCount;
  const scaleX = pageMetadata && imageSize.width > 0 ? imageSize.width / pageMetadata.width : 0;
  const scaleY = pageMetadata && imageSize.height > 0 ? imageSize.height / pageMetadata.height : 0;
  const highlightRects = isActiveSourcePage ? highlightMetadata?.rects ?? [] : [];
  const pageImageUrl = pageMetadata ? apiUrl(pageMetadata.imageUrl) : null;

  function goToPrevPage() {
    setCurrentPage((pageNumber) => Math.max(1, pageNumber - 1));
  }

  function goToNextPage() {
    setCurrentPage((pageNumber) => Math.min(document.pageCount, pageNumber + 1));
  }

  return (
    <>
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
            <div className="pdf-page image-page" data-active={isActiveSourcePage} style={{ width: `${imageZoom}%` }}>
              {viewerError ? <div className="viewer-error">{viewerError}</div> : null}
              {pageImageUrl ? (
                <div className="pdf-image-layer">
                  <img
                    ref={imgRef}
                    alt={`${document.fileName} page ${currentPage}`}
                    className="pdf-page-image"
                    onLoad={() => {
                      if (imgRef.current) {
                        setImageSize({
                          width: imgRef.current.clientWidth,
                          height: imgRef.current.clientHeight
                        });
                      }
                    }}
                    src={pageImageUrl}
                  />
                  <div className="pdf-highlight-layer" aria-hidden="true">
                    {highlightRects.map((rect, index) => (
                      <div
                        key={`${rect.x0}-${rect.y0}-${index}`}
                        className="pdf-highlight-rect"
                        style={{
                          left: rect.x0 * scaleX,
                          top: rect.y0 * scaleY,
                          width: (rect.x1 - rect.x0) * scaleX,
                          height: (rect.y1 - rect.y0) * scaleY
                        }}
                      />
                    ))}
                  </div>
                </div>
              ) : (
                <div className="viewer-loading">Loading page image...</div>
              )}
              <div className="pdf-footer">
                <span>{document.fileName}</span>
                <span>{currentPage}</span>
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
