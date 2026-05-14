from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import HTTPException
from langchain_core.documents import Document

from app.rag_service import RagService
from app.schemas import LlmAnswerPayload, SourceResponse
from app.rag_service import PageSelection
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

    def test_source_context_for_llm_limits_full_ocr_table_chunks(self) -> None:
        service = RagService()
        long_tail = " ".join(f"tail-{index}" for index in range(500))
        chunk = _make_chunk(
            text=(
                "Section: OCR extracted text\n\n"
                "[OCR Table]\n"
                "Part | Temp | Package\n"
                "LM124N | -55 to +125 | DIP\n"
                "LM224D | -25 to +85 | SO\n"
                f"{long_tail}"
            ),
            section_title="OCR extracted text",
        )
        store.chunk_lookup[chunk.id] = chunk
        source = _make_source(
            score=0.91,
            excerpt="LM124N | -55 to +125 | DIP",
            chunk_id=chunk.id,
        )

        context = service._source_context_for_llm(source)

        self.assertIn("LM124N | -55 to +125 | DIP", context)
        self.assertLessEqual(len(context), 1610)
        self.assertTrue(context.endswith("..."))

    def test_rerank_text_for_chunk_limits_ocr_table_only(self) -> None:
        service = RagService()
        settings = SimpleNamespace(rag_table_rerank_context_chars=80)
        table_chunk = _make_chunk(
            text="[OCR Table]\nPart | Value\nLM124N | -55 to +125\n" + ("tail " * 100),
            section_title="OCR extracted text",
        )
        normal_chunk = _make_chunk(
            text="Regular paragraph " + ("tail " * 100),
            section_title="Specs",
        )

        table_text = service._rerank_text_for_chunk(table_chunk, settings)
        normal_text = service._rerank_text_for_chunk(normal_chunk, settings)

        self.assertLess(len(table_text), len(table_chunk.text))
        self.assertTrue(table_text.endswith("..."))
        self.assertEqual(normal_chunk.text, normal_text)

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

    def test_llm_timeout_is_returned_as_user_facing_http_error(self) -> None:
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
        service._build_llm_context_blocks = Mock(return_value=[])
        service._answer_with_llm = Mock(side_effect=TimeoutError("request timed out"))

        with self.assertRaises(HTTPException) as context:
            service.ask("Is it waterproof?", [document])

        self.assertEqual(504, context.exception.status_code)
        self.assertIn("답변 생성이 지연", context.exception.detail)

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

    def test_clean_source_excerpt_strips_html_table_markup(self) -> None:
        service = RagService()

        cleaned = service._clean_source_excerpt(
            "<table><tr><th>항목</th><th>사양</th></tr>"
            "<tr><td>방수방진</td><td>IP67<br/>Indoor &amp; Outdoor</td></tr></table>"
        )

        self.assertNotIn("<table>", cleaned)
        self.assertNotIn("<td>", cleaned)
        self.assertIn("방수방진", cleaned)
        self.assertIn("IP67", cleaned)
        self.assertIn("Indoor & Outdoor", cleaned)
        self.assertIn("\n", cleaned)

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

    def test_lexical_retrieve_skips_empty_token_corpus(self) -> None:
        service = RagService()
        service._settings = lambda: SimpleNamespace(rag_answer_top_k=3, rag_retrieval_k=6)
        chunk = _make_chunk(text="!!! ... ///")
        document = _make_document(chunk)
        document.pages = [PageRecord(page_number=1, paragraphs=["!!! ... ///"])]

        self.assertEqual([], service._lexical_retrieve(["소비전력에 따른 모듈개수좀"], [document]))
        self.assertEqual([], service._lexical_page_retrieve(["소비전력에 따른 모듈개수좀"], [document]))

    def test_lexical_retrieve_matches_compound_korean_terms(self) -> None:
        service = RagService()
        service._settings = lambda: SimpleNamespace(rag_answer_top_k=3, rag_retrieval_k=6)
        matching_chunk = _make_chunk(
            chunk_id="chunk-match",
            text="소비 전력 400W 기준 모듈 개수는 8개입니다.",
        )
        other_chunk = _make_chunk(
            chunk_id="chunk-other-1",
            text="방수 등급은 IP67이며 실내용 제품입니다.",
        )
        third_chunk = _make_chunk(
            chunk_id="chunk-other-2",
            text="색온도는 6500K이고 광속 정보가 포함됩니다.",
        )
        document = _make_document(matching_chunk)
        document.chunks = [matching_chunk, other_chunk, third_chunk]
        document.pages = [
            PageRecord(
                page_number=1,
                paragraphs=[
                    "소비 전력 400W 기준 모듈 개수는 8개입니다.",
                    "방수 등급은 IP67이며 실내용 제품입니다.",
                    "색온도는 6500K이고 광속 정보가 포함됩니다.",
                ],
            )
        ]

        hits = service._lexical_retrieve(["소비전력에 따른 모듈개수좀"], [document])

        self.assertTrue(hits)
        self.assertEqual("chunk-match", hits[0][0].id)

    def test_retrieve_keeps_global_vector_candidates_outside_page_filter(self) -> None:
        service = RagService()
        service._settings = lambda: SimpleNamespace(
            rag_debug=False,
            rag_page_candidate_k=1,
            rag_retrieval_k=2,
            rag_answer_top_k=1,
        )
        wrong_chunk = _make_chunk(
            chunk_id="chunk-wrong",
            text="서비스 기획 배경과 시장 현황",
        )
        right_chunk = _make_chunk(
            chunk_id="chunk-right",
            text="팀장 김진하 Back-end 강채은 AI 서민석 임민경 김예지 박지혁 장현욱",
        )
        right_chunk.page_number = 4
        right_chunk.paragraph_index = 1
        right_chunk.paragraph_end_index = 3
        document = _make_document(wrong_chunk)
        document.page_count = 4
        document.chunks = [wrong_chunk, right_chunk]
        document.pages = [
            PageRecord(page_number=1, paragraphs=["서비스 기획 배경"]),
            PageRecord(page_number=4, paragraphs=["팀장 김진하", "Back-end 강채은", "AI 서민석 임민경 김예지 박지혁 장현욱"]),
        ]
        store.chunk_lookup[wrong_chunk.id] = wrong_chunk
        store.chunk_lookup[right_chunk.id] = right_chunk
        store.vector_store = object()
        service._generate_queries = Mock(return_value=["참여자 누구야?"])
        service._select_relevant_pages = Mock(
            return_value=PageSelection(
                page_candidates=[],
                selected_pages={(document.id, 1)},
                expanded_pages={(document.id, 1)},
                page_score_lookup={},
                filtering_enabled=True,
            )
        )
        service._lexical_retrieve = Mock(return_value=[])
        service._search_raw_vector_hits = Mock(
            return_value=[
                (
                    Document(
                        page_content=wrong_chunk.text,
                        metadata={
                            "document_id": wrong_chunk.document_id,
                            "chunk_id": wrong_chunk.id,
                            "page_number": wrong_chunk.page_number,
                        },
                    ),
                    1.0,
                ),
                (
                    Document(
                        page_content=right_chunk.text,
                        metadata={
                            "document_id": right_chunk.document_id,
                            "chunk_id": right_chunk.id,
                            "page_number": right_chunk.page_number,
                        },
                    ),
                    1.2,
                ),
            ]
        )
        service._get_reranker = Mock(return_value=Mock(predict=Mock(return_value=[0.1, 0.9])))

        hits = service._retrieve("참여자 누구야?", [document])

        self.assertEqual([(right_chunk, 0.9)], hits)


if __name__ == "__main__":
    unittest.main()
