from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from app.rag_service import RagService
from app.schemas import LlmAnswerPayload, SourceResponse
from app.store import ChunkRecord, DocumentRecord, PageRecord, store


def _make_chunk(
    *,
    chunk_id: str = "chunk-1",
    text: str = "Section: Specs\n\nIP67 waterproof rated.",
    section_title: str | None = "Specs",
) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        document_id="doc-1",
        file_name="sample.pdf",
        page_number=1,
        paragraph_index=1,
        paragraph_end_index=2,
        section_title=section_title,
        text=text,
        rects=[],
        paragraph_rects=[],
    )


def _make_document(chunk: ChunkRecord) -> DocumentRecord:
    return DocumentRecord(
        id=chunk.document_id,
        file_name=chunk.file_name,
        size_bytes=123,
        uploaded_at=datetime.now(timezone.utc),
        status="ready",
        page_count=1,
        file_path=Path("sample.pdf"),
        pages=[PageRecord(page_number=1, paragraphs=["Specs", "IP67 waterproof rated."])],
        chunks=[chunk],
    )


def _make_source(
    *,
    citation_number: int = 1,
    score: float,
    excerpt: str = "IP67 waterproof rated.",
    chunk_id: str = "chunk-1",
) -> SourceResponse:
    return SourceResponse(
        citationNumber=citation_number,
        documentId="doc-1",
        fileName="sample.pdf",
        pageNumber=1,
        paragraphIndex=2,
        paragraphEndIndex=2,
        sectionTitle="Specs",
        locationLabel="p.1 para 2",
        excerpt=excerpt,
        score=score,
        chunkId=chunk_id,
        highlightFileUrl=None,
    )


class RagServiceAnswerFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._documents = dict(store.documents)
        self._chunks = dict(store.chunk_lookup)
        self._vector_store = store.vector_store

    def tearDown(self) -> None:
        store.documents = self._documents
        store.chunk_lookup = self._chunks
        store.vector_store = self._vector_store

    def test_source_context_for_llm_uses_full_chunk_text(self) -> None:
        service = RagService()
        chunk = _make_chunk(
            text="Section: Specs\n\nFeature summary paragraph.\nIP67 waterproof rated.",
        )
        store.chunk_lookup[chunk.id] = chunk
        source = _make_source(score=0.72, excerpt="Feature summary paragraph.", chunk_id=chunk.id)

        context = service._source_context_for_llm(source)

        self.assertIn("IP67 waterproof rated.", context)
        self.assertNotEqual(context, source.excerpt)

    def test_strong_retrieval_overrides_llm_no_evidence(self) -> None:
        service = RagService()
        settings = SimpleNamespace(
            openai_api_key="test-key",
            rag_min_score=0.0,
            rag_answer_top_k=3,
            rag_strong_evidence_score=0.65,
        )
        chunk = _make_chunk()
        document = _make_document(chunk)
        source = _make_source(score=0.72, chunk_id=chunk.id)

        service._settings = lambda: settings
        service._retrieve = Mock(return_value=[(chunk, 0.72)])
        service._to_source = Mock(return_value=source)
        service._answer_with_llm = Mock(
            return_value=LlmAnswerPayload(
                answer="",
                citations=[],
                no_evidence=True,
                no_evidence_reason="LLM judged too conservatively",
            )
        )

        response = service.ask("Is it waterproof?", [document])

        self.assertFalse(response.noEvidence)
        self.assertIsNone(response.noEvidenceNote)
        self.assertIn("검색된 문서 근거에서는 다음 내용을 확인할 수 있습니다.", response.answer)
        self.assertIn("IP67 waterproof rated.", response.answer)
        self.assertEqual([source.citationNumber], [item.citationNumber for item in response.sources])

    def test_weak_retrieval_keeps_no_evidence(self) -> None:
        service = RagService()
        settings = SimpleNamespace(
            openai_api_key="test-key",
            rag_min_score=0.0,
            rag_answer_top_k=3,
            rag_strong_evidence_score=0.65,
        )
        chunk = _make_chunk()
        document = _make_document(chunk)
        source = _make_source(score=0.55, chunk_id=chunk.id)

        service._settings = lambda: settings
        service._retrieve = Mock(return_value=[(chunk, 0.55)])
        service._to_source = Mock(return_value=source)
        service._answer_with_llm = Mock(
            return_value=LlmAnswerPayload(
                answer="",
                citations=[],
                no_evidence=True,
                no_evidence_reason="No grounded evidence",
            )
        )

        response = service.ask("Is it waterproof?", [document])

        self.assertTrue(response.noEvidence)
        self.assertEqual("", response.answer)
        self.assertEqual([], response.sources)
        self.assertEqual("No grounded evidence", response.noEvidenceNote)

    def test_source_context_for_llm_adds_section_when_missing(self) -> None:
        service = RagService()
        chunk = _make_chunk(
            text="IP67 waterproof rated at 700mA.",
            section_title="Electrical Specs",
        )
        store.chunk_lookup[chunk.id] = chunk
        source = _make_source(score=0.81, chunk_id=chunk.id)

        context = service._source_context_for_llm(source)

        self.assertTrue(context.startswith("Section: Electrical Specs"))
        self.assertIn("IP67 waterproof rated at 700mA.", context)

    def test_fallback_grounded_answer_uses_top_two_excerpts_only(self) -> None:
        service = RagService()
        sources = [
            _make_source(citation_number=1, score=0.91, excerpt="First grounded evidence", chunk_id="chunk-1"),
            SourceResponse(
                citationNumber=2,
                documentId="doc-1",
                fileName="sample.pdf",
                pageNumber=2,
                paragraphIndex=1,
                paragraphEndIndex=2,
                sectionTitle="Details",
                locationLabel="p.2 paras 1-2",
                excerpt="Second grounded evidence",
                score=0.83,
                chunkId="chunk-2",
                highlightFileUrl=None,
            ),
            SourceResponse(
                citationNumber=3,
                documentId="doc-1",
                fileName="sample.pdf",
                pageNumber=3,
                paragraphIndex=4,
                paragraphEndIndex=4,
                sectionTitle="Extra",
                locationLabel="p.3 para 4",
                excerpt="Third grounded evidence",
                score=0.79,
                chunkId="chunk-3",
                highlightFileUrl=None,
            ),
        ]

        payload = service._fallback_grounded_answer("question", sources)

        self.assertFalse(payload.no_evidence)
        self.assertEqual([1, 2], payload.citations)
        self.assertIn("First grounded evidence [1]", payload.answer)
        self.assertIn("Second grounded evidence [2]", payload.answer)
        self.assertNotIn("Third grounded evidence [3]", payload.answer)

    def test_selected_sources_follow_llm_citations(self) -> None:
        service = RagService()
        settings = SimpleNamespace(
            openai_api_key="test-key",
            rag_min_score=0.0,
            rag_answer_top_k=3,
            rag_strong_evidence_score=0.65,
        )
        chunk = _make_chunk()
        document = _make_document(chunk)
        first_source = _make_source(citation_number=1, score=0.88, excerpt="First source", chunk_id="chunk-1")
        second_source = SourceResponse(
            citationNumber=2,
            documentId="doc-1",
            fileName="sample.pdf",
            pageNumber=2,
            paragraphIndex=3,
            paragraphEndIndex=3,
            sectionTitle="Details",
            locationLabel="p.2 para 3",
            excerpt="Second source",
            score=0.77,
            chunkId="chunk-2",
            highlightFileUrl=None,
        )

        service._settings = lambda: settings
        service._retrieve = Mock(return_value=[(chunk, 0.88), (chunk, 0.77)])
        service._to_source = Mock(side_effect=[first_source, second_source])
        service._answer_with_llm = Mock(
            return_value=LlmAnswerPayload(
                answer="Answer grounded in the second source.",
                citations=[2],
                no_evidence=False,
                no_evidence_reason=None,
            )
        )

        response = service.ask("question", [document])

        self.assertFalse(response.noEvidence)
        self.assertEqual("Answer grounded in the second source.", response.answer)
        self.assertEqual([2], [source.citationNumber for source in response.sources])

    def test_debug_retrieve_returns_expanded_queries_and_hits(self) -> None:
        service = RagService()
        chunk = _make_chunk()
        document = _make_document(chunk)

        service._generate_queries = Mock(return_value=["original question", "expanded question"])
        service._retrieve = Mock(return_value=[(chunk, 0.8765)])

        payload = service.debug_retrieve("original question", [document])

        self.assertEqual("original question", payload["question"])
        self.assertEqual(["original question", "expanded question"], payload["expandedQueries"])
        self.assertEqual(1, payload["retrievedCount"])
        self.assertEqual(1, len(payload["hits"]))
        self.assertEqual(1, payload["hits"][0]["rank"])
        self.assertEqual("doc-1", payload["hits"][0]["documentId"])
        self.assertEqual("sample.pdf", payload["hits"][0]["fileName"])
        self.assertEqual("chunk-1", payload["hits"][0]["chunkId"])
        self.assertEqual("Specs", payload["hits"][0]["sectionTitle"])
        self.assertIn("textPreview", payload["hits"][0])


if __name__ == "__main__":
    unittest.main()
