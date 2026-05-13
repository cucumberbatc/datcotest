from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app.main import (
    get_highlighted_file,
    get_page_image,
    get_page_metadata,
    open_document_file,
    select_highlight_rects,
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
        self.assertEqual("File not found on disk.", context.exception.detail)

    def test_get_page_metadata_raises_404_when_source_file_is_missing(self) -> None:
        document = _make_document(Path("missing-source.pdf"))

        with patch("app.main.get_document_or_404", return_value=document):
            with self.assertRaises(HTTPException) as context:
                get_page_metadata("doc-1", 1)

        self.assertEqual(404, context.exception.status_code)
        self.assertEqual("File not found on disk.", context.exception.detail)

    def test_get_page_image_raises_404_when_source_file_is_missing(self) -> None:
        document = _make_document(Path("missing-source.pdf"))

        with patch("app.main.get_document_or_404", return_value=document):
            with self.assertRaises(HTTPException) as context:
                get_page_image("doc-1", 1)

        self.assertEqual(404, context.exception.status_code)
        self.assertEqual("File not found on disk.", context.exception.detail)

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
        self.assertEqual("File not found on disk.", context.exception.detail)


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
