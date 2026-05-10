from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    id: str
    document_id: str
    file_name: str
    page_number: int
    paragraph_index: int
    paragraph_end_index: int
    section_title: str | None
    text: str
    rects: list[tuple[float, float, float, float]]
    paragraph_rects: list[list[tuple[float, float, float, float]]]


@dataclass(slots=True)
class PageRecord:
    page_number: int
    paragraphs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DocumentRecord:
    id: str
    file_name: str
    size_bytes: int
    uploaded_at: datetime
    status: str
    page_count: int
    file_path: Path
    pages: list[PageRecord]
    chunks: list[ChunkRecord]


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self.documents: dict[str, DocumentRecord] = {}
        self.chunk_lookup: dict[str, ChunkRecord] = {}
        self.vector_store: Any | None = None

    def list_documents(self) -> list[DocumentRecord]:
        return sorted(
            self.documents.values(),
            key=lambda item: item.uploaded_at,
            reverse=True,
        )

    def upsert_document(self, document: DocumentRecord) -> None:
        self.documents[document.id] = document
        for chunk in document.chunks:
            self.chunk_lookup[chunk.id] = chunk

    def remove_document(self, document_id: str) -> DocumentRecord | None:
        removed = self.documents.pop(document_id, None)
        if removed:
            for chunk in removed.chunks:
                self.chunk_lookup.pop(chunk.id, None)
        return removed

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        return self.chunk_lookup.get(chunk_id)

    @staticmethod
    def to_iso(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()


store = InMemoryDocumentStore()
