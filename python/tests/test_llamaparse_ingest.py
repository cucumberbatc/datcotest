from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.llamaparse_ingest import _markdown_to_paragraphs
from app.pdf_pipeline import ExtractedParagraph


class LlamaParseIngestAlignmentTests(unittest.TestCase):
    def test_markdown_blocks_reuse_matching_paragraph_rects(self) -> None:
        page = SimpleNamespace(rect=SimpleNamespace(width=600, height=800))
        first_rect = (10.0, 10.0, 200.0, 30.0)
        second_rect = (10.0, 60.0, 300.0, 90.0)
        third_rect = (10.0, 100.0, 320.0, 130.0)
        alignment_paragraphs = [
            ExtractedParagraph(text="제품 개요", rects=[first_rect]),
            ExtractedParagraph(text="방수 등급은 IP67 입니다", rects=[second_rect]),
            ExtractedParagraph(text="정격 전류는 700mA 입니다", rects=[third_rect]),
        ]

        paragraphs = _markdown_to_paragraphs(
            markdown="# 제품 개요\n\n방수 등급은 IP67 입니다\n\n정격 전류는 700mA 입니다",
            page=page,
            alignment_paragraphs=alignment_paragraphs,
        )

        self.assertEqual([first_rect], paragraphs[0].rects)
        self.assertEqual([second_rect], paragraphs[1].rects)
        self.assertEqual([third_rect], paragraphs[2].rects)

    def test_markdown_table_can_match_multiple_candidate_rects(self) -> None:
        page = SimpleNamespace(rect=SimpleNamespace(width=600, height=800))
        row_one = (10.0, 150.0, 330.0, 180.0)
        row_two = (10.0, 185.0, 330.0, 215.0)
        alignment_paragraphs = [
            ExtractedParagraph(text="모델 A IP67", rects=[row_one]),
            ExtractedParagraph(text="모델 B IP68", rects=[row_two]),
        ]

        paragraphs = _markdown_to_paragraphs(
            markdown="| 모델 | 등급 |\n| --- | --- |\n| A | IP67 |\n| B | IP68 |",
            page=page,
            alignment_paragraphs=alignment_paragraphs,
        )

        self.assertEqual([row_one, row_two], paragraphs[0].rects)

    def test_unmatched_block_falls_back_to_page_rect(self) -> None:
        page = SimpleNamespace(rect=SimpleNamespace(width=600, height=800))
        page_rect = (0.0, 0.0, 600.0, 800.0)
        alignment_paragraphs = [
            ExtractedParagraph(text="제품 개요", rects=[(10.0, 10.0, 200.0, 30.0)]),
        ]

        paragraphs = _markdown_to_paragraphs(
            markdown="완전히 다른 텍스트",
            page=page,
            alignment_paragraphs=alignment_paragraphs,
        )

        self.assertEqual([page_rect], paragraphs[0].rects)


if __name__ == "__main__":
    unittest.main()
