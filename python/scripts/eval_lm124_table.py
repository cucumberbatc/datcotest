#!/usr/bin/env python3
"""
eval_lm124_table.py

LM124 원본 PDF + 스캔본 PDF를 대상으로 표 기반 질의 정확도를 점검하는 작은 평가 스크립트입니다.

전제:
- FastAPI 서버가 http://localhost:8000 에서 실행 중
- 원본 LM124.PDF와 LM124_image_only_no_text.pdf를 둘 다 업로드한 상태
- /api/chat/debug-retrieve, /api/chat/ask 엔드포인트 사용 가능
- documentIds를 지정하지 않으면 현재 서버에 올라간 전체 문서를 대상으로 평가합니다.

실행:
  python eval_lm124_table.py --eval-set lm124_table_eval_cases.json --base-url http://localhost:8000 --out lm124_eval_results.json

특정 문서만 평가:
  python eval_lm124_table.py --eval-set lm124_table_eval_cases.json --document-id <DOC_ID>

원본/스캔본 각각 따로 평가하고 싶으면 document-id를 하나씩 바꿔서 실행하세요.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class EvalCase:
    id: str
    question: str
    expected_pages: list[int]
    expected_keywords: list[str]
    qtype: str = "table"
    document_ids: list[str] | None = None


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
    source_document_ids: list[str]
    source_file_names: list[str]
    recall_top3: bool | None
    recall_top5: bool | None
    answer_keyword_hit: bool
    source_page_hit: bool
    source_keyword_hit: bool
    highlight_rects_present: bool | None
    source_count: int
    source_text_chars: int
    server_elapsed_ms: int | None
    retrieval_elapsed_ms: int | None
    query_expansion_elapsed_ms: int | None
    vector_search_elapsed_ms: int | None
    page_selection_elapsed_ms: int | None
    lexical_search_elapsed_ms: int | None
    retrieval_merge_elapsed_ms: int | None
    rerank_elapsed_ms: int | None
    context_build_elapsed_ms: int | None
    llm_elapsed_ms: int | None
    context_chars: int | None
    context_block_count: int | None
    elapsed_ms: int | None
    debug_elapsed_ms: int | None
    ask_elapsed_ms: int | None
    status: str
    error: str = ""


def normalize(text: str) -> str:
    # OCR/파싱에서 공백과 줄바꿈이 흔들리므로 공백 제거 후 비교
    return "".join(str(text).lower().replace("℃", "°c").split())


def contains_keywords(text: str, keywords: list[str], min_hits: int | None = None) -> bool:
    if not keywords:
        return True

    if min_hits is None:
        # 표 질의에서는 OCR이 일부 깨질 수 있어서 키워드 2개 이상이면 2개 hit를 기본 통과로 둠
        min_hits = 1 if len(keywords) == 1 else min(2, len(keywords))

    norm_text = normalize(text)
    hits = sum(1 for keyword in keywords if normalize(keyword) in norm_text)
    return hits >= min_hits


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    response = requests.post(f"{base_url.rstrip('/')}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_json(base_url: str, path: str, timeout: int = 60) -> dict[str, Any]:
    response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("cases", raw) if isinstance(raw, dict) else raw
    return [
        EvalCase(
            id=str(row["id"]),
            question=str(row["question"]),
            expected_pages=[int(p) for p in row.get("expected_pages", [])],
            expected_keywords=[str(k) for k in row.get("expected_keywords", [])],
            qtype=str(row.get("qtype", "table")),
            document_ids=row.get("document_ids"),
        )
        for row in rows
    ]


def extract_hits(debug_response: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("hits", "finalContextHits", "results"):
        value = debug_response.get(key)
        if isinstance(value, list):
            return value
    return []


def extract_sources(answer_response: dict[str, Any]) -> list[dict[str, Any]]:
    value = answer_response.get("sources")
    return value if isinstance(value, list) else []


def get_page(obj: dict[str, Any]) -> int | None:
    for key in ("pageNumber", "page", "page_number"):
        if obj.get(key) is not None:
            try:
                return int(obj[key])
            except (TypeError, ValueError):
                return None
    return None


def get_chunk_id(obj: dict[str, Any]) -> str:
    for key in ("chunkId", "chunk_id", "id"):
        value = obj.get(key)
        if value is not None:
            return str(value)
    return ""


def get_document_id(obj: dict[str, Any]) -> str:
    for key in ("documentId", "document_id"):
        value = obj.get(key)
        if value is not None:
            return str(value)
    return ""


def get_file_name(obj: dict[str, Any]) -> str:
    for key in ("fileName", "file_name"):
        value = obj.get(key)
        if value is not None:
            return str(value)
    return ""


def source_text(source: dict[str, Any]) -> str:
    fields = ("excerpt", "textPreview", "text", "sectionTitle", "fileName")
    return " ".join(str(source.get(field, "")) for field in fields)


def check_highlight_rects(base_url: str, source: dict[str, Any]) -> bool | None:
    document_id = get_document_id(source)
    chunk_id = get_chunk_id(source)
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


def make_payload(case: EvalCase, default_document_ids: list[str] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"question": case.question}
    document_ids = case.document_ids if case.document_ids is not None else default_document_ids
    if document_ids:
        payload["documentIds"] = document_ids
    return payload


def evaluate_case(
    base_url: str,
    case: EvalCase,
    default_document_ids: list[str] | None,
    skip_debug: bool,
    check_highlights: bool,
) -> EvalResult:
    payload = make_payload(case, default_document_ids)
    expected_pages = set(case.expected_pages)
    started = time.time()

    debug_elapsed_ms: int | None = None
    hits: list[dict[str, Any]] = []
    if not skip_debug:
        debug_started = time.time()
        debug = post_json(base_url, "/api/chat/debug-retrieve", payload)
        debug_elapsed_ms = int((time.time() - debug_started) * 1000)
        hits = extract_hits(debug)

    retrieved_pages = [page for page in (get_page(hit) for hit in hits) if page is not None]
    retrieved_chunks = [get_chunk_id(hit) for hit in hits]

    ask_started = time.time()
    answer_response = post_json(base_url, "/api/chat/ask", payload)
    ask_elapsed_ms = int((time.time() - ask_started) * 1000)
    elapsed_ms = int((time.time() - started) * 1000)

    answer = str(answer_response.get("answer", ""))
    sources = extract_sources(answer_response)
    source_pages = [page for page in (get_page(source) for source in sources) if page is not None]
    source_chunks = [get_chunk_id(source) for source in sources]
    source_document_ids = [get_document_id(source) for source in sources]
    source_file_names = [get_file_name(source) for source in sources]
    joined_source_text = " ".join(source_text(source) for source in sources)
    highlight_rects_present: bool | None = None
    if check_highlights and sources:
        checks = [check_highlight_rects(base_url, source) for source in sources[:3]]
        known = [value for value in checks if value is not None]
        highlight_rects_present = any(known) if known else None

    recall_top3 = None if skip_debug else (bool(expected_pages & set(retrieved_pages[:3])) if expected_pages else True)
    recall_top5 = None if skip_debug else (bool(expected_pages & set(retrieved_pages[:5])) if expected_pages else True)

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
        source_document_ids=source_document_ids,
        source_file_names=source_file_names,
        recall_top3=recall_top3,
        recall_top5=recall_top5,
        answer_keyword_hit=contains_keywords(answer, case.expected_keywords),
        source_page_hit=bool(expected_pages & set(source_pages)) if expected_pages else True,
        source_keyword_hit=contains_keywords(joined_source_text, case.expected_keywords),
        highlight_rects_present=highlight_rects_present,
        source_count=len(sources),
        source_text_chars=len(joined_source_text),
        server_elapsed_ms=answer_response.get("elapsedMs"),
        retrieval_elapsed_ms=answer_response.get("retrievalElapsedMs"),
        query_expansion_elapsed_ms=answer_response.get("queryExpansionElapsedMs"),
        vector_search_elapsed_ms=answer_response.get("vectorSearchElapsedMs"),
        page_selection_elapsed_ms=answer_response.get("pageSelectionElapsedMs"),
        lexical_search_elapsed_ms=answer_response.get("lexicalSearchElapsedMs"),
        retrieval_merge_elapsed_ms=answer_response.get("retrievalMergeElapsedMs"),
        rerank_elapsed_ms=answer_response.get("rerankElapsedMs"),
        context_build_elapsed_ms=answer_response.get("contextBuildElapsedMs"),
        llm_elapsed_ms=answer_response.get("llmElapsedMs"),
        context_chars=answer_response.get("contextChars"),
        context_block_count=answer_response.get("contextBlockCount"),
        elapsed_ms=elapsed_ms,
        debug_elapsed_ms=debug_elapsed_ms,
        ask_elapsed_ms=ask_elapsed_ms,
        status="ok",
    )


def print_summary(results: list[EvalResult]) -> None:
    ok = [result for result in results if result.status == "ok"]

    def rate_text(field: str) -> str:
        known = [result for result in ok if getattr(result, field) is not None]
        if not known:
            return "n/a"
        rate = sum(1 for result in known if getattr(result, field)) / len(known) * 100
        return f"{rate:.1f}%"

    def speed_text(field: str) -> str:
        values = [int(getattr(result, field)) for result in ok if getattr(result, field) is not None]
        if not values:
            return "n/a"
        average = sum(values) / len(values)
        over_10s = sum(1 for value in values if value > 10000)
        return f"avg={average:.0f}ms min={min(values)}ms max={max(values)}ms >10s={over_10s}/{len(values)}"

    def count_text(field: str) -> str:
        values = [int(getattr(result, field)) for result in ok if getattr(result, field) is not None]
        if not values:
            return "n/a"
        average = sum(values) / len(values)
        return f"avg={average:.0f} min={min(values)} max={max(values)}"

    def passed(result: EvalResult) -> bool:
        retrieval_ok = result.recall_top3 is not False
        highlight_ok = result.highlight_rects_present is not False
        return (
            result.status == "ok"
            and retrieval_ok
            and result.answer_keyword_hit
            and result.source_page_hit
            and result.source_keyword_hit
            and highlight_ok
        )

    print("\n=== LM124 Table Evaluation Summary ===")
    print(f"Total:       {len(results)}")
    print(f"OK:          {len(ok)}")
    print(f"Errors:      {len(results) - len(ok)}")
    print(f"Strict pass: {sum(1 for result in ok if passed(result))}/{len(ok)}")
    print(f"Recall@3:    {rate_text('recall_top3')}")
    print(f"Recall@5:    {rate_text('recall_top5')}")
    print(f"Answer hit:  {rate_text('answer_keyword_hit')}")
    print(f"Source page: {rate_text('source_page_hit')}")
    print(f"Source text: {rate_text('source_keyword_hit')}")
    print(f"Highlights:  {rate_text('highlight_rects_present')}")
    print("\n=== Timing ===")
    print(f"Ask client:  {speed_text('ask_elapsed_ms')}")
    print(f"Ask server:  {speed_text('server_elapsed_ms')}")
    print(f"Retrieval:   {speed_text('retrieval_elapsed_ms')}")
    print(f"Query exp:   {speed_text('query_expansion_elapsed_ms')}")
    print(f"Vector:      {speed_text('vector_search_elapsed_ms')}")
    print(f"Page select: {speed_text('page_selection_elapsed_ms')}")
    print(f"Lexical:     {speed_text('lexical_search_elapsed_ms')}")
    print(f"Merge:       {speed_text('retrieval_merge_elapsed_ms')}")
    print(f"Rerank:      {speed_text('rerank_elapsed_ms')}")
    print(f"Ctx build:   {speed_text('context_build_elapsed_ms')}")
    print(f"LLM:         {speed_text('llm_elapsed_ms')}")
    print(f"Debug:       {speed_text('debug_elapsed_ms')}")
    print(f"Total:       {speed_text('elapsed_ms')}")
    print(f"Ctx chars:   {count_text('context_chars')}")
    print(f"Ctx blocks:  {count_text('context_block_count')}")

    print("\n=== Check Cases ===")
    for result in results:
        if not passed(result):
            print(f"- [{result.id}] {result.question}")
            print(f"  status={result.status}, error={result.error}")
            print(f"  expected_pages={result.expected_pages}, retrieved_pages={result.retrieved_pages[:5]}, source_pages={result.source_pages}")
            print(f"  expected_keywords={result.expected_keywords}")
            print(f"  source_chunks={result.source_chunks[:3]}, highlight={result.highlight_rects_present}")
            print(f"  answer={result.answer[:250]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--eval-set", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--document-id", action="append", dest="document_ids")
    parser.add_argument("--skip-debug", action="store_true", help="Only call /api/chat/ask for faster latency benchmarking.")
    parser.add_argument("--check-highlights", action="store_true", help="Check highlight rects for returned sources.")
    args = parser.parse_args()

    cases = load_cases(args.eval_set)
    results: list[EvalResult] = []

    for case in cases:
        try:
            result = evaluate_case(
                args.base_url,
                case,
                args.document_ids,
                args.skip_debug,
                args.check_highlights,
            )
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
                source_document_ids=[],
                source_file_names=[],
                recall_top3=None if args.skip_debug else False,
                recall_top5=None if args.skip_debug else False,
                answer_keyword_hit=False,
                source_page_hit=False,
                source_keyword_hit=False,
                highlight_rects_present=None,
                source_count=0,
                source_text_chars=0,
                server_elapsed_ms=None,
                retrieval_elapsed_ms=None,
                query_expansion_elapsed_ms=None,
                vector_search_elapsed_ms=None,
                page_selection_elapsed_ms=None,
                lexical_search_elapsed_ms=None,
                retrieval_merge_elapsed_ms=None,
                rerank_elapsed_ms=None,
                context_build_elapsed_ms=None,
                llm_elapsed_ms=None,
                context_chars=None,
                context_block_count=None,
                elapsed_ms=None,
                debug_elapsed_ms=None,
                ask_elapsed_ms=None,
                status="error",
                error=repr(exc),
            )

        retrieval_ok = result.recall_top3 is not False
        highlight_ok = result.highlight_rects_present is not False
        passed = (
            result.status == "ok"
            and retrieval_ok
            and result.answer_keyword_hit
            and result.source_page_hit
            and result.source_keyword_hit
            and highlight_ok
        )
        timing = f"ask={result.ask_elapsed_ms}ms"
        if result.server_elapsed_ms is not None:
            timing += f", server={result.server_elapsed_ms}ms"
        if result.llm_elapsed_ms is not None:
            timing += f", llm={result.llm_elapsed_ms}ms"
        if result.retrieval_elapsed_ms is not None:
            timing += f", retrieval={result.retrieval_elapsed_ms}ms"
        if result.rerank_elapsed_ms is not None:
            timing += f", rerank={result.rerank_elapsed_ms}ms"
        if result.query_expansion_elapsed_ms is not None:
            timing += f", qexp={result.query_expansion_elapsed_ms}ms"
        if result.debug_elapsed_ms is not None:
            timing += f", debug={result.debug_elapsed_ms}ms"
        print(f"[{'PASS' if passed else 'CHECK'}] {case.id}: {case.question} ({timing})")
        results.append(result)

    print_summary(results)

    if args.out:
        args.out.write_text(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved results to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
