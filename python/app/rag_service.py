from __future__ import annotations

import re
from time import perf_counter

from fastapi import HTTPException, status
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import get_settings
from app.schemas import AskResponse, LlmAnswerPayload, SourceResponse
from app.store import ChunkRecord, DocumentRecord, store

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\uAC00-\uD7A3]+")


class RagService:
    def __init__(self) -> None:
        self._settings = get_settings

    def rebuild_index(self) -> None:
        settings = self._settings()
        chunks: list[ChunkRecord] = []
        for document in store.list_documents():
            chunks.extend(document.chunks)

        if not chunks or not settings.openai_api_key:
            store.vector_store = None
            return

        embedding = self._embeddings()
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "file_name": chunk.file_name,
                    "page_number": chunk.page_number,
                    "paragraph_index": chunk.paragraph_index,
                    "bbox": list(chunk.bbox),
                },
            )
            for chunk in chunks
        ]
        ids = [chunk.id for chunk in chunks]
        store.vector_store = FAISS.from_documents(documents, embedding=embedding, ids=ids)

    def ask(self, question: str, scoped_documents: list[DocumentRecord]) -> AskResponse:
        settings = self._settings()
        started_at = perf_counter()
        if not scoped_documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No indexed documents available.",
            )

        retrieved = self._retrieve(question, scoped_documents)
        if not retrieved:
            return AskResponse(
                question=question,
                answer="",
                noEvidence=True,
                noEvidenceNote=self._no_evidence_note(len(scoped_documents)),
                sources=[],
                elapsedMs=int((perf_counter() - started_at) * 1000),
            )

        source_candidates = [
            self._to_source(index + 1, chunk, score)
            for index, (chunk, score) in enumerate(retrieved)
        ]

        if settings.openai_api_key:
            payload = self._answer_with_llm(question, source_candidates)
        else:
            payload = self._fallback_answer(source_candidates)

        if payload.no_evidence:
            return AskResponse(
                question=question,
                answer="",
                noEvidence=True,
                noEvidenceNote=payload.no_evidence_reason or self._no_evidence_note(len(scoped_documents)),
                sources=[],
                elapsedMs=int((perf_counter() - started_at) * 1000),
            )

        selected_numbers = set(payload.citations or [1])
        selected_sources = [
            source for source in source_candidates if source.citationNumber in selected_numbers
        ]
        if not selected_sources:
            selected_sources = source_candidates[: settings.rag_answer_top_k]

        return AskResponse(
            question=question,
            answer=payload.answer.strip(),
            noEvidence=False,
            noEvidenceNote=None,
            sources=selected_sources,
            elapsedMs=int((perf_counter() - started_at) * 1000),
        )

    def _retrieve(
        self,
        question: str,
        scoped_documents: list[DocumentRecord],
    ) -> list[tuple[ChunkRecord, float]]:
        settings = self._settings()
        allowed_ids = {document.id for document in scoped_documents}
        vector_store = store.vector_store

        if vector_store is None:
            return self._lexical_retrieve(question, scoped_documents)

        raw_hits = vector_store.similarity_search_with_score(
            question,
            k=max(settings.rag_retrieval_k * 2, settings.rag_retrieval_k),
        )

        ranked: list[tuple[ChunkRecord, float]] = []
        seen_chunk_ids: set[str] = set()
        for doc, score in raw_hits:
            metadata = doc.metadata
            document_id = str(metadata["document_id"])
            chunk_id = str(metadata["chunk_id"])

            if document_id not in allowed_ids or chunk_id in seen_chunk_ids:
                continue

            chunk = store.get_chunk(chunk_id)
            if chunk is None:
                continue

            seen_chunk_ids.add(chunk_id)
            normalized_score = 1 / (1 + float(score))
            ranked.append((chunk, normalized_score))
            if len(ranked) >= settings.rag_answer_top_k:
                break

        if ranked:
            return ranked
        return self._lexical_retrieve(question, scoped_documents)

    def _lexical_retrieve(
        self,
        question: str,
        scoped_documents: list[DocumentRecord],
    ) -> list[tuple[ChunkRecord, float]]:
        settings = self._settings()
        query_tokens = self._tokenize(question)
        if not query_tokens:
            return []

        ranked: list[tuple[ChunkRecord, float]] = []
        for document in scoped_documents:
            for chunk in document.chunks:
                chunk_tokens = self._tokenize(chunk.text)
                overlap = len(set(query_tokens) & set(chunk_tokens))
                if overlap == 0:
                    continue

                score = overlap / max(len(set(query_tokens)), 1)
                if any(char.isdigit() for char in question) and any(char.isdigit() for char in chunk.text):
                    score += 0.15
                ranked.append((chunk, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[: settings.rag_answer_top_k]

    def _answer_with_llm(
        self,
        question: str,
        sources: list[SourceResponse],
    ) -> LlmAnswerPayload:
        settings = self._settings()
        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            temperature=settings.openai_temperature,
        ).with_structured_output(LlmAnswerPayload, method="json_schema")

        context = "\n\n".join(
            (
                f"[{source.citationNumber}] file={source.fileName} "
                f"page={source.pageNumber} paragraph={source.paragraphIndex}\n"
                f"{source.excerpt}"
            )
            for source in sources
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a grounded RAG assistant. Use only the supplied context. "
                        "If the context is insufficient, set no_evidence=true. "
                        "Keep cited source numbers aligned with the provided context."
                    ),
                ),
                ("human", "Question:\n{question}\n\nContext:\n{context}"),
            ]
        )

        payload = llm.invoke(prompt.format_messages(question=question, context=context))
        if not payload.answer and not payload.no_evidence:
            return self._fallback_answer(sources)
        return payload

    def _fallback_answer(self, sources: list[SourceResponse]) -> LlmAnswerPayload:
        settings = self._settings()
        if not sources:
            return LlmAnswerPayload(
                answer="",
                citations=[],
                no_evidence=True,
                no_evidence_reason=self._no_evidence_note(0),
            )

        chosen = sources[: settings.rag_answer_top_k]
        sentences: list[str] = []
        for index, source in enumerate(chosen):
            prefix = "Grounded evidence:" if index == 0 else "Additional evidence:"
            sentences.append(f"{prefix} {source.excerpt} [{source.citationNumber}]")

        return LlmAnswerPayload(
            answer=" ".join(sentences),
            citations=[source.citationNumber for source in chosen],
            no_evidence=False,
            no_evidence_reason=None,
        )

    def _to_source(self, citation_number: int, chunk: ChunkRecord, score: float) -> SourceResponse:
        return SourceResponse(
            citationNumber=citation_number,
            documentId=chunk.document_id,
            fileName=chunk.file_name,
            pageNumber=chunk.page_number,
            paragraphIndex=chunk.paragraph_index,
            locationLabel=f"p.{chunk.page_number} para {chunk.paragraph_index}",
            excerpt=self._abbreviate(chunk.text, 320),
            score=round(score, 4),
            chunkId=chunk.id,
            highlightFileUrl=f"/api/documents/{chunk.document_id}/chunks/{chunk.id}/highlighted-file",
        )

    def _embeddings(self) -> OpenAIEmbeddings:
        settings = self._settings()
        if not settings.openai_api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OPENAI_API_KEY is not configured.",
            )

        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text) if len(token.strip()) > 1]

    @staticmethod
    def _abbreviate(text: str, max_length: int) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= max_length:
            return normalized
        return normalized[: max_length - 1].rstrip() + "..."

    @staticmethod
    def _no_evidence_note(scoped_document_count: int) -> str:
        return (
            f"No grounded evidence was found across {scoped_document_count} document(s). "
            "Try refining the question or upload more relevant PDFs."
        )


rag_service = RagService()
