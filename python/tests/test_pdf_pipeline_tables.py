import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from app.pdf_pipeline import (
    OCR_TABLE_PREFIX,
    ExtractedParagraph,
    OcrTextItem,
    ingest_pdf,
    _merge_short_paragraphs,
    _ocr_items_to_table_paragraphs,
    _run_paddleocr,
)


class OcrTableParagraphTests(unittest.TestCase):
    def test_reconstructs_table_text_from_ocr_boxes(self) -> None:
        items = [
            OcrTextItem("Power", 0.99, (20, 20, 80, 40)),
            OcrTextItem("Modules", 0.99, (120, 20, 180, 40)),
            OcrTextItem("Flux", 0.99, (220, 20, 280, 40)),
            OcrTextItem("50W", 0.99, (20, 60, 80, 80)),
            OcrTextItem("2", 0.99, (120, 60, 180, 80)),
            OcrTextItem("5700", 0.99, (220, 60, 280, 80)),
            OcrTextItem("100W", 0.99, (20, 100, 80, 120)),
            OcrTextItem("4", 0.99, (120, 100, 180, 120)),
            OcrTextItem("11400", 0.99, (220, 100, 280, 120)),
        ]

        paragraphs = _ocr_items_to_table_paragraphs(items, zoom=2.0)

        self.assertEqual(1, len(paragraphs))
        self.assertTrue(paragraphs[0].text.startswith(OCR_TABLE_PREFIX))
        self.assertIn("Power | Modules | Flux", paragraphs[0].text)
        self.assertIn("50W | 2 | 5700", paragraphs[0].text)
        self.assertIn("100W | 4 | 11400", paragraphs[0].text)

    def test_skips_single_column_ocr_text(self) -> None:
        items = [
            OcrTextItem("First line", 0.99, (20, 20, 180, 40)),
            OcrTextItem("Second line", 0.99, (20, 60, 180, 80)),
        ]

        paragraphs = _ocr_items_to_table_paragraphs(items, zoom=2.0)

        self.assertEqual([], paragraphs)

    def test_table_paragraph_is_not_merged_with_neighbor_text(self) -> None:
        table = ExtractedParagraph(
            text=f"{OCR_TABLE_PREFIX}\nA | B\n1 | 2",
            rects=[(0, 0, 10, 10)],
        )
        other = ExtractedParagraph(
            text="short",
            rects=[(0, 20, 10, 30)],
        )

        paragraphs = _merge_short_paragraphs([table, other])

        self.assertEqual(table.text, paragraphs[0].text)
        self.assertEqual("short", paragraphs[1].text)


class PdfIngestValidationTests(unittest.TestCase):
    def test_ingest_pdf_rejects_password_protected_pdf(self) -> None:
        pdf = MagicMock()
        pdf.needs_pass = True
        pdf.__enter__.return_value = pdf
        pdf.__exit__.return_value = None

        with patch("app.pdf_pipeline.fitz.open", return_value=pdf):
            with self.assertRaises(HTTPException) as context:
                ingest_pdf("locked.pdf", b"%PDF-1.7", Path("locked.pdf"))

        self.assertEqual(400, context.exception.status_code)
        self.assertIn("암호가 걸린 PDF", context.exception.detail)


class OcrFailureHandlingTests(unittest.TestCase):
    def test_run_paddleocr_raises_when_reader_is_unavailable(self) -> None:
        with patch("app.pdf_pipeline._get_paddleocr_reader", return_value=None):
            with self.assertRaises(HTTPException) as context:
                _run_paddleocr(Path("page.png"))

        self.assertEqual(503, context.exception.status_code)
        self.assertIn("OCR 엔진", context.exception.detail)

    def test_run_paddleocr_raises_when_prediction_fails(self) -> None:
        reader = MagicMock()
        reader.predict.side_effect = RuntimeError("ocr failed")

        with patch("app.pdf_pipeline._get_paddleocr_reader", return_value=reader):
            with self.assertRaises(HTTPException) as context:
                _run_paddleocr(Path("page.png"))

        self.assertEqual(500, context.exception.status_code)
        self.assertIn("OCR 처리 중 오류", context.exception.detail)


if __name__ == "__main__":
    unittest.main()
