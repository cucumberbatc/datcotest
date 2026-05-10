from __future__ import annotations

import re
import logging
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
HANGUL_RE = re.compile(r"[\uAC00-\uD7A3]")
ALNUM_HANGUL_RE = re.compile(r"[^0-9A-Za-z\uAC00-\uD7A3]+")
BRACKET_LABEL_RE = re.compile(r"^\[[^\]]+\]")
logger = logging.getLogger("rag.retrieval")
SOURCE_EXCERPT_CHARS = 2200
KOREAN_PARTICLE_SUFFIXES = ("으로", "에서", "에게", "까지", "부터", "처럼", "보다", "은", "는", "이", "가", "을", "를", "의", "에", "도", "만")

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
                    "paragraph_end_index": chunk.paragraph_end_index,
                    "section_title": chunk.section_title,
                    "rects": [list(rect) for rect in chunk.rects],
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
        min_score = max(settings.rag_min_score, 0.0)
        if not retrieved or (min_score > 0 and retrieved[0][1] < min_score):
            return AskResponse(
                question=question,
                answer="",
                noEvidence=True,
                noEvidenceNote=self._no_evidence_note(len(scoped_documents)),
                sources=[],
                elapsedMs=int((perf_counter() - started_at) * 1000),
            )

        source_candidates = [
            self._to_source(index + 1, chunk, score, question)
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

        retrieval_k = getattr(settings, "rag_retrieval_k", 12)
        answer_top_k = getattr(settings, "rag_answer_top_k", 5)

        lexical_hits = self._lexical_retrieve(question, scoped_documents)
        self._log_retrieval_hits("LEXICAL HITS", question, lexical_hits)

        if vector_store is None:
            self._log_retrieval_hits("FINAL HITS - LEXICAL ONLY", question, lexical_hits[:answer_top_k])
            return lexical_hits[:answer_top_k]

        raw_hits = vector_store.similarity_search_with_score(
            question,
            k=max(retrieval_k * 3, 15),
        )

        vector_hits: list[tuple[ChunkRecord, float]] = []
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

            # FAISS returns a distance-like score, so smaller is usually better.
            normalized_score = 1 / (1 + float(score))
            vector_hits.append((chunk, normalized_score))

            if len(vector_hits) >= retrieval_k:
                break

        self._log_retrieval_hits("VECTOR HITS", question, vector_hits)

        merged = self._merge_retrieval_hits(
            vector_hits=vector_hits,
            lexical_hits=lexical_hits,
            limit=retrieval_k,
        )

        self._log_retrieval_hits("MERGED HITS", question, merged)

        final_hits = merged[:answer_top_k]
        self._log_retrieval_hits("FINAL CONTEXT HITS", question, final_hits)

        return final_hits

    def _lexical_retrieve(
        self,
        question: str,
        scoped_documents: list[DocumentRecord],
    ) -> list[tuple[ChunkRecord, float]]:
        settings = self._settings()
        query_tokens = self._tokenize(question)
        if not query_tokens:
            return []

        query_token_set = set(query_tokens)
        query_compact = self._compact_text(question)
        query_terms = self._query_terms(question)
        ranked: list[tuple[ChunkRecord, float]] = []
        for document in scoped_documents:
            for chunk in document.chunks:
                chunk_tokens = self._tokenize(chunk.text)
                chunk_token_set = set(chunk_tokens)
                overlap = len(query_token_set & chunk_token_set)
                compact_bonus = self._compact_match_score(query_compact, query_terms, chunk.text)

                token_score = overlap / max(len(query_token_set), 1) if overlap > 0 else 0.0
                if token_score == 0 and compact_bonus == 0:
                    continue

                if compact_bonus > token_score:
                    score = compact_bonus * 1.2
                else:
                    score = max(token_score, compact_bonus)

                if any(char.isdigit() for char in question) and any(char.isdigit() for char in chunk.text):
                    score += 0.15
                ranked.append((chunk, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[: settings.rag_answer_top_k]

    def _merge_retrieval_hits(
        self,
        vector_hits: list[tuple[ChunkRecord, float]],
        lexical_hits: list[tuple[ChunkRecord, float]],
        limit: int,
    ) -> list[tuple[ChunkRecord, float]]:
        merged: dict[str, tuple[ChunkRecord, float, set[str]]] = {}

        for chunk, score in vector_hits:
            merged[chunk.id] = (chunk, score, {"vector"})

        for chunk, score in lexical_hits:
            lexical_score = min(score * 0.75, 0.9)

            existing = merged.get(chunk.id)
            if existing is None:
                merged[chunk.id] = (chunk, lexical_score, {"lexical"})
            else:
                existing_chunk, existing_score, sources = existing
                combined_score = min(max(existing_score, lexical_score) + 0.15, 1.0)
                sources.add("lexical")
                merged[chunk.id] = (existing_chunk, combined_score, sources)

        ranked = sorted(
            merged.values(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [(chunk, score) for chunk, score, _sources in ranked[:limit]]

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
                        "You are a PDF-grounded question-answering assistant. "
                        "Use only the supplied context. Do not infer facts outside the context. "
                        "If the context is insufficient, set no_evidence=true. "
                        "Answer in Korean. Include only the citation numbers actually used."
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

    def _to_source(
        self,
        citation_number: int,
        chunk: ChunkRecord,
        score: float,
        question: str,
    ) -> SourceResponse:
        evidence_start, evidence_end, evidence_text = self._best_evidence_span(question, chunk)
        location_label = (
            f"p.{chunk.page_number} para {evidence_start}"
            if evidence_start == evidence_end
            else f"p.{chunk.page_number} paras {evidence_start}-{evidence_end}"
        )
        return SourceResponse(
            citationNumber=citation_number,
            documentId=chunk.document_id,
            fileName=chunk.file_name,
            pageNumber=chunk.page_number,
            paragraphIndex=evidence_start,
            paragraphEndIndex=evidence_end,
            sectionTitle=chunk.section_title,
            locationLabel=location_label,
            excerpt=self._abbreviate(evidence_text, SOURCE_EXCERPT_CHARS),
            score=round(score, 4),
            chunkId=chunk.id,
            highlightFileUrl=(
                f"/api/documents/{chunk.document_id}/chunks/{chunk.id}/highlighted-file"
                f"?paragraphStart={evidence_start}&paragraphEnd={evidence_end}"
            ),
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

    def _log_retrieval_hits(
        self,
        label: str,
        question: str,
        hits: list[tuple[ChunkRecord, float]],
        limit: int = 10,
    ) -> None:
        settings = self._settings()
        if not getattr(settings, "rag_debug", True):
            return

        logger.info("========== %s ==========", label)
        logger.info("question=%s", question)
        logger.info("hit_count=%s", len(hits))

        for index, (chunk, score) in enumerate(hits[:limit], start=1):
            preview = self._abbreviate(chunk.text, 240)
            logger.info(
                "#%s score=%.4f file=%s page=%s para=%s-%s chunk_id=%s text=%s",
                index,
                score,
                chunk.file_name,
                chunk.page_number,
                chunk.paragraph_index,
                chunk.paragraph_end_index,
                chunk.id,
                preview,
            )

    def _best_evidence_span(self, question: str, chunk: ChunkRecord) -> tuple[int, int, str]:
        document = store.get_document(chunk.document_id)
        if document is None or chunk.page_number > len(document.pages):
            return chunk.paragraph_index, chunk.paragraph_index, chunk.text

        page = document.pages[chunk.page_number - 1]
        start = max(chunk.paragraph_index, 1)
        end = min(chunk.paragraph_end_index, len(page.paragraphs))
        if start > end:
            return chunk.paragraph_index, chunk.paragraph_index, chunk.text

        query_tokens = set(self._tokenize(question))
        query_compact = self._compact_text(question)
        query_terms = self._query_terms(question)

        best_index = start
        best_score = -1.0
        scored_paragraphs: list[tuple[int, float]] = []
        for paragraph_index in range(start, end + 1):
            paragraph = page.paragraphs[paragraph_index - 1]
            paragraph_score = self._paragraph_match_score(query_tokens, query_compact, query_terms, paragraph)
            scored_paragraphs.append((paragraph_index, paragraph_score))
            if paragraph_score > best_score:
                best_score = paragraph_score
                best_index = paragraph_index

        if self._is_label_paragraph(page.paragraphs[best_index - 1]):
            label_end = self._label_section_end(page.paragraphs, best_index, end)
            evidence_text = self._evidence_text(page.paragraphs, chunk.section_title, best_index, label_end)
            return best_index, label_end, evidence_text

        if chunk.section_title and page.paragraphs[best_index - 1] == chunk.section_title and best_index < end:
            best_index += 1

        related_indices = [index for index, score in scored_paragraphs if score > 0]
        if len(related_indices) > 1:
            span_start = min(related_indices)
            span_end = max(related_indices)
            evidence_text = self._evidence_text(page.paragraphs, chunk.section_title, span_start, span_end)
            if len(evidence_text) <= SOURCE_EXCERPT_CHARS:
                return span_start, span_end, evidence_text

        evidence_text = page.paragraphs[best_index - 1]
        if chunk.section_title and chunk.section_title not in evidence_text:
            evidence_text = f"Section: {chunk.section_title}\n\n{evidence_text}"

        return best_index, best_index, evidence_text

    def _paragraph_match_score(
        self,
        query_tokens: set[str],
        query_compact: str,
        query_terms: list[str],
        paragraph: str,
    ) -> float:
        paragraph_tokens = set(self._tokenize(paragraph))
        overlap_score = len(query_tokens & paragraph_tokens) / max(len(query_tokens), 1)
        compact_score = self._compact_match_score(query_compact, query_terms, paragraph)
        return max(overlap_score, compact_score)

    @staticmethod
    def _is_label_paragraph(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip()
        return bool(BRACKET_LABEL_RE.match(normalized))

    def _label_section_end(self, paragraphs: list[str], label_index: int, chunk_end: int) -> int:
        section_end = label_index
        for paragraph_index in range(label_index + 1, chunk_end + 1):
            paragraph = paragraphs[paragraph_index - 1]
            if self._is_label_paragraph(paragraph):
                break
            section_end = paragraph_index

        return section_end

    @staticmethod
    def _evidence_text(
        paragraphs: list[str],
        section_title: str | None,
        start: int,
        end: int,
    ) -> str:
        selected = "\n".join(paragraphs[index - 1] for index in range(start, end + 1))
        if section_title and section_title not in selected:
            return f"Section: {section_title}\n\n{selected}"
        return selected

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text) if len(token.strip()) > 1]

    @staticmethod
    def _compact_text(text: str) -> str:
        return ALNUM_HANGUL_RE.sub("", text).lower()

    @classmethod
    def _query_terms(cls, text: str) -> list[str]:
        terms: set[str] = set()
        for token in TOKEN_RE.findall(text):
            compact_token = cls._compact_text(token)
            if not compact_token:
                continue
            terms.add(compact_token)
            stripped_token = cls._strip_korean_particle(compact_token)
            if stripped_token:
                terms.add(stripped_token)

        compact = cls._compact_text(text)
        if HANGUL_RE.search(compact):
            terms.add(compact)
        return [term for term in terms if len(term) > 1]

    @staticmethod
    def _strip_korean_particle(term: str) -> str:
        if not HANGUL_RE.search(term):
            return term

        for suffix in KOREAN_PARTICLE_SUFFIXES:
            if term.endswith(suffix) and len(term) > len(suffix) + 1:
                return term[: -len(suffix)]

        return term

    @classmethod
    def _compact_match_score(cls, query_compact: str, query_terms: list[str], chunk_text: str) -> float:
        chunk_compact = cls._compact_text(chunk_text)
        if not query_compact or not chunk_compact:
            return 0.0

        score = 0.0
        if len(query_compact) > 2 and query_compact in chunk_compact:
            score += 1.0
            return score

        matched_terms = [
            term
            for term in query_terms
            if len(term) > 1 and term in chunk_compact
        ]
        if matched_terms:
            if len(matched_terms) == len(query_terms):
                score += 0.95
            else:
                score += (len(matched_terms) / max(len(query_terms), 1)) * 0.7

        if len(query_compact) > 3:
            for i in range(len(chunk_compact) - len(query_compact) + 1):
                chunk_substring = chunk_compact[i : i + len(query_compact)]
                matches = sum(1 for a, b in zip(query_compact, chunk_substring) if a == b)
                if matches >= len(query_compact) * 0.7:
                    score = max(score, 0.6 * (matches / len(query_compact)))

        return min(score, 1.0)

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

    def debug_retrieve(
        self,
        question: str,
        scoped_documents: list[DocumentRecord],
    ) -> dict:
        hits = self._retrieve(question, scoped_documents)

        return {
            "question": question,
            "hits": [
                {
                    "rank": index,
                    "score": round(score, 4),
                    "chunkId": chunk.id,
                    "documentId": chunk.document_id,
                    "fileName": chunk.file_name,
                    "pageNumber": chunk.page_number,
                    "paragraphIndex": chunk.paragraph_index,
                    "paragraphEndIndex": chunk.paragraph_end_index,
                    "sectionTitle": chunk.section_title,
                    "text": self._abbreviate(chunk.text, 600),
                }
                for index, (chunk, score) in enumerate(hits, start=1)
            ],
        }

rag_service = RagService()
