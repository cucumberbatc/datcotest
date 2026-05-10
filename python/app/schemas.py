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
    paragraphEndIndex: int
    sectionTitle: str | None = None
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


class PageMetadataResponse(BaseModel):
    documentId: str
    pageNumber: int
    width: float
    height: float
    imageUrl: str


class RectResponse(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class HighlightResponse(BaseModel):
    documentId: str
    chunkId: str
    pageNumber: int
    rects: list[RectResponse]


class LlmAnswerPayload(BaseModel):
    answer: str
    citations: list[int] = Field(default_factory=list)
    no_evidence: bool = False
    no_evidence_reason: str | None = None
