from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from app.main import (
    MAX_TOTAL_UPLOAD_BYTES,
    MAX_UPLOAD_DOCUMENTS,
    get_highlighted_file,
    get_page_image,
    get_page_metadata,
    open_document_file,
    select_highlight_rects,
    _validate_pdf_upload,
    _validate_upload_limits,
)
from app.store import ChunkRecord, DocumentRecord, PageRecord


def _make_document(file_path: Path) -> DocumentRecord:
    return DocumentRecord(
        id="doc-1",
        file_name="sample.pdf",
        size_bytes=123,
        uploaded_at=datetime.now(timezone.utc),
        status="ready",
        page_count=1,
        file_path=file_path,
        pages=[PageRecord(page_number=1, paragraphs=["sample"])],
        chunks=[],
    )


def _make_chunk() -> ChunkRecord:
    return ChunkRecord(
        id="chunk-1",
        document_id="doc-1",
        file_name="sample.pdf",
        page_number=1,
        paragraph_index=1,
        paragraph_end_index=1,
        section_title="Section",
        text="sample",
        rects=[(1.0, 1.0, 2.0, 2.0)],
        paragraph_rects=[[(1.0, 1.0, 2.0, 2.0)]],
    )


class MainRouteFileChecksTests(unittest.TestCase):
    def test_open_document_file_raises_404_when_source_file_is_missing(self) -> None:
        document = _make_document(Path("missing-source.pdf"))

        with patch("app.main.get_document_or_404", return_value=document):
            with self.assertRaises(HTTPException) as context:
                open_document_file("doc-1")

        self.assertEqual(404, context.exception.status_code)
        self.assertEqual("문서 파일을 찾을 수 없어요.", context.exception.detail)


class UploadValidationTests(unittest.TestCase):
    def test_validate_pdf_upload_rejects_non_pdf_extension(self) -> None:
        with self.assertRaises(HTTPException) as context:
            _validate_pdf_upload("sample.txt", b"%PDF-1.7")

        self.assertEqual(400, context.exception.status_code)
        self.assertIn("PDF 파일만", context.exception.detail)

    def test_validate_pdf_upload_rejects_empty_file(self) -> None:
        with self.assertRaises(HTTPException) as context:
            _validate_pdf_upload("sample.pdf", b"")

        self.assertEqual(400, context.exception.status_code)
        self.assertIn("빈 파일", context.exception.detail)

    def test_validate_pdf_upload_rejects_invalid_pdf_signature(self) -> None:
        with self.assertRaises(HTTPException) as context:
            _validate_pdf_upload("sample.pdf", b"not a pdf")

        self.assertEqual(400, context.exception.status_code)
        self.assertIn("손상된 파일", context.exception.detail)

    def test_validate_upload_limits_rejects_too_many_documents(self) -> None:
        with patch("app.main.store.list_documents", return_value=[]):
            with self.assertRaises(HTTPException) as context:
                _validate_upload_limits(MAX_UPLOAD_DOCUMENTS + 1, 100)

        self.assertEqual(400, context.exception.status_code)
        self.assertIn("최대 10개", context.exception.detail)

    def test_validate_upload_limits_rejects_total_size_over_200mb(self) -> None:
        with patch("app.main.store.list_documents", return_value=[]):
            with self.assertRaises(HTTPException) as context:
                _validate_upload_limits(1, MAX_TOTAL_UPLOAD_BYTES + 1)

        self.assertEqual(413, context.exception.status_code)
        self.assertIn("총 200.0MB", context.exception.detail)

    def test_get_page_metadata_raises_404_when_source_file_is_missing(self) -> None:
        document = _make_document(Path("missing-source.pdf"))

        with patch("app.main.get_document_or_404", return_value=document):
            with self.assertRaises(HTTPException) as context:
                get_page_metadata("doc-1", 1)

        self.assertEqual(404, context.exception.status_code)
        self.assertEqual("문서 파일을 찾을 수 없어요.", context.exception.detail)

    def test_get_page_image_raises_404_when_source_file_is_missing(self) -> None:
        document = _make_document(Path("missing-source.pdf"))

        with patch("app.main.get_document_or_404", return_value=document):
            with self.assertRaises(HTTPException) as context:
                get_page_image("doc-1", 1)

        self.assertEqual(404, context.exception.status_code)
        self.assertEqual("문서 파일을 찾을 수 없어요.", context.exception.detail)

    def test_get_page_image_returns_user_facing_error_when_rendering_fails(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "sample.pdf"
            source_path.write_bytes(b"%PDF-1.7")
            document = _make_document(source_path)

            with patch("app.main.get_document_or_404", return_value=document), patch(
                "app.main.render_page_image",
                side_effect=RuntimeError("render failed"),
            ):
                with self.assertRaises(HTTPException) as context:
                    get_page_image("doc-1", 1)

        self.assertEqual(500, context.exception.status_code)
        self.assertIn("페이지 이미지를 생성하지 못했어요", context.exception.detail)

    def test_get_highlighted_file_raises_404_when_source_file_is_missing(self) -> None:
        document = _make_document(Path("missing-source.pdf"))
        chunk = _make_chunk()

        with patch("app.main.get_document_or_404", return_value=document), patch(
            "app.main.store.get_chunk",
            return_value=chunk,
        ):
            with self.assertRaises(HTTPException) as context:
                get_highlighted_file("doc-1", "chunk-1")

        self.assertEqual(404, context.exception.status_code)
        self.assertEqual("문서 파일을 찾을 수 없어요.", context.exception.detail)


class HighlightRectSelectionTests(unittest.TestCase):
    def test_select_highlight_rects_filters_page_like_rects(self) -> None:
        document = _make_document(Path("sample.pdf"))
        chunk = _make_chunk()
        chunk.rects = [
            (0.0, 0.0, 100.0, 100.0),
            (10.0, 12.0, 24.0, 20.0),
        ]

        with patch("app.main.store.get_document", return_value=document), patch(
            "app.main.get_page_size",
            return_value=(100.0, 100.0),
        ):
            rects = select_highlight_rects(chunk, None, None)

        self.assertEqual([(10.0, 12.0, 24.0, 20.0)], rects)

    def test_select_highlight_rects_returns_empty_for_invalid_paragraph_range(self) -> None:
        chunk = _make_chunk()

        rects = select_highlight_rects(chunk, 3, 1)

        self.assertEqual([], rects)

    def test_select_highlight_rects_does_not_fallback_when_selected_rects_are_empty(self) -> None:
        document = _make_document(Path("sample.pdf"))
        chunk = _make_chunk()
        chunk.rects = [(0.0, 0.0, 100.0, 100.0)]
        chunk.paragraph_rects = [[]]

        with patch("app.main.store.get_document", return_value=document), patch(
            "app.main.get_page_size",
            return_value=(100.0, 100.0),
        ):
            rects = select_highlight_rects(chunk, 1, 1)

        self.assertEqual([], rects)


if __name__ == "__main__":
    unittest.main()
