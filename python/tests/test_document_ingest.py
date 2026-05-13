from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.document_ingest import ingest_pdf


class DocumentIngestTests(unittest.TestCase):
    def test_pymupdf_backend_uses_existing_pipeline(self) -> None:
        expected = Mock()
        settings = SimpleNamespace(
            document_parse_backend="pymupdf",
            llama_parse_fallback_to_pymupdf=True,
        )

        with patch("app.document_ingest.get_settings", return_value=settings), patch(
            "app.document_ingest.ingest_pdf_with_pymupdf",
            return_value=expected,
        ) as pymupdf_ingest, patch(
            "app.document_ingest.ingest_pdf_with_llamaparse"
        ) as llamaparse_ingest:
            result = ingest_pdf("sample.pdf", b"pdf", Path("sample.pdf"))

        self.assertIs(result, expected)
        pymupdf_ingest.assert_called_once()
        llamaparse_ingest.assert_not_called()

    def test_llamaparse_backend_uses_llamaparse_pipeline(self) -> None:
        expected = Mock()
        settings = SimpleNamespace(
            document_parse_backend="llamaparse",
            llama_parse_fallback_to_pymupdf=True,
        )

        with patch("app.document_ingest.get_settings", return_value=settings), patch(
            "app.document_ingest.ingest_pdf_with_llamaparse",
            return_value=expected,
        ) as llamaparse_ingest, patch(
            "app.document_ingest.ingest_pdf_with_pymupdf"
        ) as pymupdf_ingest:
            result = ingest_pdf("sample.pdf", b"pdf", Path("sample.pdf"))

        self.assertIs(result, expected)
        llamaparse_ingest.assert_called_once()
        pymupdf_ingest.assert_not_called()

    def test_llamaparse_failure_falls_back_when_enabled(self) -> None:
        expected = Mock()
        settings = SimpleNamespace(
            document_parse_backend="llamaparse",
            llama_parse_fallback_to_pymupdf=True,
        )

        with patch("app.document_ingest.get_settings", return_value=settings), patch(
            "app.document_ingest.ingest_pdf_with_llamaparse",
            side_effect=RuntimeError("parse failed"),
        ) as llamaparse_ingest, patch(
            "app.document_ingest.ingest_pdf_with_pymupdf",
            return_value=expected,
        ) as pymupdf_ingest:
            result = ingest_pdf("sample.pdf", b"pdf", Path("sample.pdf"))

        self.assertIs(result, expected)
        llamaparse_ingest.assert_called_once()
        pymupdf_ingest.assert_called_once()

    def test_llamaparse_failure_raises_when_fallback_disabled(self) -> None:
        settings = SimpleNamespace(
            document_parse_backend="llamaparse",
            llama_parse_fallback_to_pymupdf=False,
        )

        with patch("app.document_ingest.get_settings", return_value=settings), patch(
            "app.document_ingest.ingest_pdf_with_llamaparse",
            side_effect=RuntimeError("parse failed"),
        ), patch("app.document_ingest.ingest_pdf_with_pymupdf") as pymupdf_ingest:
            with self.assertRaisesRegex(RuntimeError, "parse failed"):
                ingest_pdf("sample.pdf", b"pdf", Path("sample.pdf"))

        pymupdf_ingest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
