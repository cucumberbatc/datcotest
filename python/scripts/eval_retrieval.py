#!/usr/bin/env python3
"""
eval_retrieval.py

Small evaluation runner for the DocQ / DatcoTest FastAPI RAG service.

Checks:
- debug-retrieve top-k recall
- answer keyword inclusion
- source page / source text match
- optional highlight rect availability

Usage:
  python eval_retrieval.py --base-url http://localhost:8000 --eval-set eval_questions_sample.json
  python eval_retrieval.py --base-url http://localhost:8000 --eval-set eval_questions_sample.json --out eval_results.json
  python eval_retrieval.py --base-url http://localhost:8000 --eval-set eval_questions_sample.json --document-id <DOC_ID>
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests


@dataclass
class EvalCase:
    id: str
    question: str
    expected_pages: list[int]
    expected_keywords: list[str]
    qtype: str = "general"
    document_ids: list[str] | None = None
    notes: str = ""


@dataclass
class EvalResult:
    id: str
    question: str
    qtype: str
    expected_pages: list[int]
    expected_keywords: list[str]
    retrieved_pages: list[int]
    retrieved_chunks: list[str]
    answer: str
    source_pages: list[int]
    source_chunks: list[str]
    recall_top3: bool
    recall_top5: bool
    answer_keyword_hit: bool
    source_page_hit: bool
    source_keyword_hit: bool
    highlight_rects_present: bool | None
    elapsed_ms: int | None
    status: str
    debug_elapsed_ms: int | None = None
    ask_elapsed_ms: int | None = None
    error: str = ""


def normalize_text(text: str) -> str:
    return "".join(str(text).lower().split())


def keyword_hit(text: str, keywords: list[str], min_hits: int = 1) -> bool:
    if not keywords:
        return True
    normalized = normalize_text(text)
    hits = sum(1 for keyword in keywords if normalize_text(keyword) in normalized)
    return hits >= min_hits


def load_eval_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = raw.get("cases", raw) if isinstance(raw, dict) else raw

    return [
        EvalCase(
            id=str(item["id"]),
            question=str(item["question"]),
            expected_pages=[int(p) for p in item.get("expected_pages", [])],
            expected_keywords=[str(k) for k in item.get("expected_keywords", [])],
            qtype=str(item.get("qtype", "general")),
            document_ids=item.get("document_ids"),
            notes=str(item.get("notes", "")),
        )
        for item in raw_cases
    ]


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    response = requests.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_json(base_url: str, path: str, timeout: int = 60) -> dict[str, Any]:
    response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def build_payload(case: EvalCase, default_document_ids: list[str] | None) -> dict[str, Any]:
    document_ids = case.document_ids if case.document_ids is not None else default_document_ids
    payload: dict[str, Any] = {"question": case.question}
    if document_ids:
        payload["documentIds"] = document_ids
    return payload


def extract_debug_hits(debug_response: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("hits", "finalContextHits", "results"):
        value = debug_response.get(key)
        if isinstance(value, list):
            return value
    return []


def get_page(obj: dict[str, Any]) -> int | None:
    for key in ("pageNumber", "page", "page_number"):
        value = obj.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def get_chunk_id(obj: dict[str, Any]) -> str:
    for key in ("chunkId", "chunk_id", "id"):
        value = obj.get(key)
        if value is not None:
            return str(value)
    return ""


def extract_answer_sources(answer_response: dict[str, Any]) -> list[dict[str, Any]]:
    sources = answer_response.get("sources")
    return sources if isinstance(sources, list) else []


def source_to_text(source: dict[str, Any]) -> str:
    return " ".join(
        str(source.get(field, ""))
        for field in ("excerpt", "textPreview", "text", "sectionTitle", "fileName")
    )


def check_highlight_rects(base_url: str, source: dict[str, Any]) -> bool | None:
    document_id = source.get("documentId") or source.get("document_id")
    chunk_id = source.get("chunkId") or source.get("chunk_id")
    if not document_id or not chunk_id:
        return None

    params = []
    if source.get("paragraphIndex") is not None:
        params.append(f"paragraphStart={source['paragraphIndex']}")
    if source.get("paragraphEndIndex") is not None:
        params.append(f"paragraphEnd={source['paragraphEndIndex']}")

    query = f"?{'&'.join(params)}" if params else ""
    path = f"/api/documents/{document_id}/chunks/{chunk_id}/highlight{query}"

    try:
        response = get_json(base_url, path)
    except Exception:
        return None

    rects = response.get("rects")
    return len(rects) > 0 if isinstance(rects, list) else None


def evaluate_case(
    base_url: str,
    case: EvalCase,
    default_document_ids: list[str] | None,
    check_highlights: bool,
) -> EvalResult:
    payload = build_payload(case, default_document_ids)
    expected_pages = set(case.expected_pages)
    started = time.time()

    debug_started = time.time()
    debug = post_json(base_url, "/api/chat/debug-retrieve", payload)
    debug_elapsed_ms = int((time.time() - debug_started) * 1000)
    hits = extract_debug_hits(debug)

    retrieved_pages = [page for page in (get_page(hit) for hit in hits) if page is not None]
    retrieved_chunks = [get_chunk_id(hit) for hit in hits]

    ask_started = time.time()
    answer_response = post_json(base_url, "/api/chat/ask", payload)
    ask_elapsed_ms = int((time.time() - ask_started) * 1000)
    elapsed_ms = int((time.time() - started) * 1000)

    answer = str(answer_response.get("answer", ""))
    sources = extract_answer_sources(answer_response)

    source_pages = [page for page in (get_page(source) for source in sources) if page is not None]
    source_chunks = [get_chunk_id(source) for source in sources]

    recall_top3 = bool(expected_pages & set(retrieved_pages[:3])) if expected_pages else True
    recall_top5 = bool(expected_pages & set(retrieved_pages[:5])) if expected_pages else True
    answer_keyword_hit = keyword_hit(answer, case.expected_keywords)
    source_page_hit = bool(expected_pages & set(source_pages)) if expected_pages else True
    source_keyword_hit = keyword_hit(" ".join(source_to_text(source) for source in sources), case.expected_keywords)

    highlight_rects_present: bool | None = None
    if check_highlights and sources:
        checks = [check_highlight_rects(base_url, source) for source in sources[:3]]
        known = [value for value in checks if value is not None]
        highlight_rects_present = any(known) if known else None

    return EvalResult(
        id=case.id,
        question=case.question,
        qtype=case.qtype,
        expected_pages=case.expected_pages,
        expected_keywords=case.expected_keywords,
        retrieved_pages=retrieved_pages,
        retrieved_chunks=retrieved_chunks,
        answer=answer,
        source_pages=source_pages,
        source_chunks=source_chunks,
        recall_top3=recall_top3,
        recall_top5=recall_top5,
        answer_keyword_hit=answer_keyword_hit,
        source_page_hit=source_page_hit,
        source_keyword_hit=source_keyword_hit,
        highlight_rects_present=highlight_rects_present,
        elapsed_ms=elapsed_ms,
        debug_elapsed_ms=debug_elapsed_ms,
        ask_elapsed_ms=ask_elapsed_ms,
        status="ok",
    )


def print_summary(results: list[EvalResult]) -> None:
    total = len(results)
    ok_results = [r for r in results if r.status == "ok"]
    ok_total = len(ok_results)

    def rate(field: str) -> float:
        if ok_total == 0:
            return 0.0
        return sum(1 for r in ok_results if getattr(r, field)) / ok_total * 100

    print("\n=== Evaluation Summary ===")
    print(f"Total cases: {total}")
    print(f"OK cases:    {ok_total}")
    print(f"Errors:      {total - ok_total}")
    print(f"Recall@3:    {rate('recall_top3'):.1f}%")
    print(f"Recall@5:    {rate('recall_top5'):.1f}%")
    print(f"Answer hit:  {rate('answer_keyword_hit'):.1f}%")
    print(f"Source page: {rate('source_page_hit'):.1f}%")
    print(f"Source text: {rate('source_keyword_hit'):.1f}%")

    print("\n=== Failed / Suspicious Cases ===")
    for r in results:
        suspicious = r.status != "ok" or not (
            r.recall_top3 and r.answer_keyword_hit and r.source_page_hit and r.source_keyword_hit
        )
        if suspicious:
            print(f"- [{r.id}] {r.question}")
            print(f"  status={r.status}, error={r.error}")
            print(f"  type={r.qtype}")
            print(f"  expected_pages={r.expected_pages}, retrieved_pages={r.retrieved_pages[:5]}, source_pages={r.source_pages}")
            print(f"  expected_keywords={r.expected_keywords}")
            print(f"  answer={r.answer[:250]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--eval-set", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--document-id", action="append", dest="document_ids")
    parser.add_argument("--check-highlights", action="store_true")
    args = parser.parse_args()

    cases = load_eval_cases(args.eval_set)
    results: list[EvalResult] = []

    for case in cases:
        try:
            result = evaluate_case(args.base_url, case, args.document_ids, args.check_highlights)
        except Exception as exc:
            result = EvalResult(
                id=case.id,
                question=case.question,
                qtype=case.qtype,
                expected_pages=case.expected_pages,
                expected_keywords=case.expected_keywords,
                retrieved_pages=[],
                retrieved_chunks=[],
                answer="",
                source_pages=[],
                source_chunks=[],
                recall_top3=False,
                recall_top5=False,
                answer_keyword_hit=False,
                source_page_hit=False,
                source_keyword_hit=False,
                highlight_rects_present=None,
                elapsed_ms=None,
                debug_elapsed_ms=None,
                ask_elapsed_ms=None,
                status="error",
                error=repr(exc),
            )

        results.append(result)
        passed = (
            result.status == "ok"
            and result.recall_top3
            and result.answer_keyword_hit
            and result.source_page_hit
            and result.source_keyword_hit
        )
        print(f"[{'PASS' if passed else 'CHECK'}] {case.id} {case.question}")

    print_summary(results)

    if args.out:
        args.out.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved results to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
