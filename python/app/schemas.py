from pydantic import BaseModel, Field


class DocumentSummaryResponse(BaseModel):
    id: str
    fileName: str
    sizeBytes: int
    uploadedAt: str
    status: str
    pageCount: int


class PageResponse(BaseModel):
    pageNumber: int
    paragraphs: list[str]


class DocumentDetailResponse(DocumentSummaryResponse):
    downloadUrl: str
    pages: list[PageResponse]


class UploadResponse(BaseModel):
    documents: list[DocumentSummaryResponse]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    documentIds: list[str] | None = None


class SourceResponse(BaseModel):
    citationNumber: int
    documentId: str
    fileName: str
    pageNumber: int
    paragraphIndex: int
    locationLabel: str
    excerpt: str
    score: float
    chunkId: str
    highlightFileUrl: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    noEvidence: bool
    noEvidenceNote: str | None = None
    sources: list[SourceResponse]
    elapsedMs: int


class HighlightResponse(BaseModel):
    documentId: str
    chunkId: str
    highlightedFileUrl: str


class LlmAnswerPayload(BaseModel):
    answer: str
    citations: list[int] = Field(default_factory=list)
    no_evidence: bool = False
    no_evidence_reason: str | None = None
