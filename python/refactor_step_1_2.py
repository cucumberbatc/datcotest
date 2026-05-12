import re

with open("app/rag_service.py", "r") as f:
    content = f.read()

# Add imports
imports = """
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
"""
content = re.sub(r'(from langchain_openai import ChatOpenAI, OpenAIEmbeddings)', r'\1\n' + imports.strip(), content)

# Add tokenizer
tokenizer_code = """
kiwi = Kiwi()

def kiwi_tokenize(text: str) -> list[str]:
    tokens = kiwi.tokenize(text)
    return [t.form for t in tokens if t.tag.startswith('N') or t.tag.startswith('V') or t.tag.startswith('S')]

"""
content = re.sub(r'(@dataclass\(slots=True\)\nclass PageHit:)', tokenizer_code + r'\1', content)

# Update RagService._retrieve
retrieve_old = """    def _retrieve(
        self,
        question: str,
        scoped_documents: list[DocumentRecord],
    ) -> list[tuple[ChunkRecord, float]]:
        settings = self._settings()
        allowed_ids = {document.id for document in scoped_documents}
        vector_store = store.vector_store

        page_candidate_k = max(getattr(settings, "rag_page_candidate_k", 3), 1)
        retrieval_k = getattr(settings, "rag_retrieval_k", 12)
        answer_top_k = getattr(settings, "rag_answer_top_k", 5)
        raw_hits = self._search_raw_vector_hits(question, retrieval_k, vector_store)"""

retrieve_new = """    def _generate_queries(self, question: str) -> list[str]:
        settings = self._settings()
        if not settings.openai_api_key:
            return [question]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 AI 검색 어시스턴트입니다. 사용자의 질문을 분석하여, 검색 엔진에서 문서를 찾기 좋은 형태의 다양한 동의어/유사 질문으로 3개를 만들어주세요. 원본 질문과 의미가 같아야 하며, 검색 키워드가 다채로워야 합니다. 결과는 줄바꿈으로 구분된 텍스트로 반환하세요. 번호 매기기나 추가 설명은 하지 마세요."),
            ("human", "{question}")
        ])
        llm = ChatOpenAI(api_key=settings.openai_api_key, model=settings.openai_chat_model, temperature=0.3)
        try:
            response = llm.invoke(prompt.format_messages(question=question))
            queries = [q.strip("- ") for q in response.content.split("\\n") if q.strip()]
            if not queries:
                return [question]
            return [question] + queries[:3]
        except Exception as e:
            logger.error("Query expansion failed: %s", e)
            return [question]

    def _retrieve(
        self,
        question: str,
        scoped_documents: list[DocumentRecord],
    ) -> list[tuple[ChunkRecord, float]]:
        settings = self._settings()
        allowed_ids = {document.id for document in scoped_documents}
        vector_store = store.vector_store

        # 1. Query Expansion
        expanded_queries = self._generate_queries(question)
        if getattr(settings, "rag_debug", True):
            logger.info("========== EXPANDED QUERIES ==========")
            logger.info("\\n".join(expanded_queries))

        page_candidate_k = max(getattr(settings, "rag_page_candidate_k", 3), 1)
        retrieval_k = getattr(settings, "rag_retrieval_k", 12)
        answer_top_k = getattr(settings, "rag_answer_top_k", 5)
        
        # 2. Vector Search (Multi-query)
        raw_hits = []
        seen_vector_docs = set()
        for q in expanded_queries:
            q_hits = self._search_raw_vector_hits(q, retrieval_k, vector_store)
            for doc, score in q_hits:
                doc_key = (doc.metadata.get("document_id"), doc.metadata.get("chunk_id"))
                if doc_key not in seen_vector_docs:
                    seen_vector_docs.add(doc_key)
                    raw_hits.append((doc, score))
"""
content = content.replace(retrieve_old, retrieve_new)

# Update lexical_retrieve call
lexical_call_old = """        lexical_hits = self._lexical_retrieve(
            question,
            scoped_documents,
            allowed_pages=allowed_pages,
        )"""
lexical_call_new = """        lexical_hits = self._lexical_retrieve(
            expanded_queries,
            scoped_documents,
            allowed_pages=allowed_pages,
        )"""
content = content.replace(lexical_call_old, lexical_call_new)

# Update rank_pages call
select_old = """        page_selection = self._select_relevant_pages(
            question,
            scoped_documents,
            raw_hits=raw_hits,
            page_candidate_k=page_candidate_k,
            answer_top_k=answer_top_k,
        )"""
select_new = """        page_selection = self._select_relevant_pages(
            expanded_queries,
            scoped_documents,
            raw_hits=raw_hits,
            page_candidate_k=page_candidate_k,
            answer_top_k=answer_top_k,
        )"""
content = content.replace(select_old, select_new)

# Update _select_relevant_pages signature
def_select_old = """    def _select_relevant_pages(
        self,
        question: str,"""
def_select_new = """    def _select_relevant_pages(
        self,
        expanded_queries: list[str],"""
content = content.replace(def_select_old, def_select_new)

def_rank_pages_old = """    def _rank_pages(
        self,
        question: str,"""
def_rank_pages_new = """    def _rank_pages(
        self,
        expanded_queries: list[str],"""
content = content.replace(def_rank_pages_old, def_rank_pages_new)

# Inside _select_relevant_pages and _rank_pages
content = content.replace("self._rank_pages(\n            question,", "self._rank_pages(\n            expanded_queries,")
content = content.replace("self._lexical_page_retrieve(question,", "self._lexical_page_retrieve(expanded_queries,")
content = content.replace("self._log_page_hits(\"PAGE LEXICAL HITS\", question", "self._log_page_hits(\"PAGE LEXICAL HITS\", expanded_queries[0]")
content = content.replace("self._log_page_hits(\"PAGE VECTOR HITS\", question", "self._log_page_hits(\"PAGE VECTOR HITS\", expanded_queries[0]")

# Update _lexical_page_retrieve
lex_page_old = """    def _lexical_page_retrieve(
        self,
        question: str,
        scoped_documents: list[DocumentRecord],
    ) -> list[PageHit]:
        query_terms = self._search_terms(question)
        query_token_set = set(query_terms)
        if not query_token_set:
            return []

        query_compact = self._compact_text(question)
        query_ngrams = self._search_ngrams(question)
        hits: list[PageHit] = []

        for document in scoped_documents:
            chunks_by_page: dict[int, list[ChunkRecord]] = {}
            for chunk in document.chunks:
                chunks_by_page.setdefault(chunk.page_number, []).append(chunk)

            for page in document.pages:
                page_text = self._page_search_text(
                    page,
                    chunks_by_page.get(page.page_number, []),
                )
                score = self._text_match_score(
                    question=question,
                    query_terms=query_terms,
                    query_token_set=query_token_set,
                    query_compact=query_compact,
                    query_ngrams=query_ngrams,
                    target_text=page_text,
                )
                if score == 0:
                    continue

                hits.append(
                    PageHit(
                        document_id=document.id,
                        file_name=document.file_name,
                        page_number=page.page_number,
                        score=score,
                    )
                )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits"""

lex_page_new = """    def _lexical_page_retrieve(
        self,
        expanded_queries: list[str],
        scoped_documents: list[DocumentRecord],
    ) -> list[PageHit]:
        allowed_pages_text = []
        page_refs = []
        for document in scoped_documents:
            chunks_by_page: dict[int, list[ChunkRecord]] = {}
            for chunk in document.chunks:
                chunks_by_page.setdefault(chunk.page_number, []).append(chunk)

            for page in document.pages:
                page_text = self._page_search_text(
                    page,
                    chunks_by_page.get(page.page_number, []),
                )
                allowed_pages_text.append(page_text)
                page_refs.append((document.id, document.file_name, page.page_number))

        if not allowed_pages_text:
            return []

        tokenized_corpus = [kiwi_tokenize(text) for text in allowed_pages_text]
        bm25 = BM25Okapi(tokenized_corpus)
        
        page_scores = {i: 0.0 for i in range(len(allowed_pages_text))}
        
        for q in expanded_queries:
            tokenized_query = kiwi_tokenize(q)
            if not tokenized_query: continue
            scores = bm25.get_scores(tokenized_query)
            for i, score in enumerate(scores):
                page_scores[i] = max(page_scores[i], score)

        hits: list[PageHit] = []
        max_score = max(page_scores.values()) if page_scores else 0
        
        for i, score in page_scores.items():
            if score > 0:
                normalized_score = score / max_score if max_score > 0 else 0
                doc_id, file_name, page_number = page_refs[i]
                hits.append(
                    PageHit(
                        document_id=doc_id,
                        file_name=file_name,
                        page_number=page_number,
                        score=normalized_score,
                    )
                )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits"""
content = content.replace(lex_page_old, lex_page_new)

# Update _lexical_retrieve
lex_old = """    def _lexical_retrieve(
        self,
        question: str,
        scoped_documents: list[DocumentRecord],
        allowed_pages: set[tuple[str, int]] | None = None,
    ) -> list[tuple[ChunkRecord, float]]:
        settings = self._settings()
        query_terms = self._search_terms(question)
        query_token_set = set(query_terms)
        if not query_token_set:
            return []

        query_compact = self._compact_text(question)
        query_ngrams = self._search_ngrams(question)
        ranked: list[tuple[ChunkRecord, float]] = []
        for document in scoped_documents:
            for chunk in document.chunks:
                page_key = (chunk.document_id, chunk.page_number)
                if allowed_pages and page_key not in allowed_pages:
                    continue

                score = self._text_match_score(
                    question=question,
                    query_terms=query_terms,
                    query_token_set=query_token_set,
                    query_compact=query_compact,
                    query_ngrams=query_ngrams,
                    target_text=chunk.text,
                )
                if score == 0:
                    continue
                ranked.append((chunk, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[: max(settings.rag_answer_top_k, settings.rag_retrieval_k)]"""

lex_new = """    def _lexical_retrieve(
        self,
        expanded_queries: list[str],
        scoped_documents: list[DocumentRecord],
        allowed_pages: set[tuple[str, int]] | None = None,
    ) -> list[tuple[ChunkRecord, float]]:
        settings = self._settings()
        
        allowed_chunks = []
        for document in scoped_documents:
            for chunk in document.chunks:
                page_key = (chunk.document_id, chunk.page_number)
                if allowed_pages and page_key not in allowed_pages:
                    continue
                allowed_chunks.append(chunk)

        if not allowed_chunks:
            return []

        tokenized_corpus = [kiwi_tokenize(chunk.text) for chunk in allowed_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        
        chunk_scores = {chunk.id: 0.0 for chunk in allowed_chunks}
        
        for q in expanded_queries:
            tokenized_query = kiwi_tokenize(q)
            if not tokenized_query: continue
            scores = bm25.get_scores(tokenized_query)
            for chunk, score in zip(allowed_chunks, scores):
                chunk_scores[chunk.id] = max(chunk_scores[chunk.id], score)

        ranked = []
        max_score = max(chunk_scores.values()) if chunk_scores else 0
        for chunk in allowed_chunks:
            score = chunk_scores[chunk.id]
            if score > 0:
                normalized_score = score / max_score if max_score > 0 else 0
                ranked.append((chunk, normalized_score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[: max(settings.rag_answer_top_k, settings.rag_retrieval_k)]"""
content = content.replace(lex_old, lex_new)

# Update _best_evidence_span to use kiwi_tokenize instead of custom text match score
best_ev_old = """        query_terms = self._search_terms(question)
        query_tokens = set(query_terms)
        query_compact = self._compact_text(question)
        query_ngrams = self._search_ngrams(question)

        best_index = start
        best_score = -1.0
        scored_paragraphs: list[tuple[int, float]] = []
        for paragraph_index in range(start, end + 1):
            paragraph = page.paragraphs[paragraph_index - 1]
            paragraph_score = self._paragraph_match_score(
                query_tokens,
                query_compact,
                query_terms,
                query_ngrams,
                paragraph,
            )"""
best_ev_new = """        query_tokens = set(kiwi_tokenize(question))

        best_index = start
        best_score = -1.0
        scored_paragraphs: list[tuple[int, float]] = []
        for paragraph_index in range(start, end + 1):
            paragraph = page.paragraphs[paragraph_index - 1]
            paragraph_tokens = set(kiwi_tokenize(paragraph))
            overlap = len(query_tokens & paragraph_tokens)
            paragraph_score = overlap / max(len(query_tokens), 1) if overlap > 0 else 0.0
"""
content = content.replace(best_ev_old, best_ev_new)

# Also fix the definition of _best_evidence_span since _paragraph_match_score is gone
content = re.sub(r'    def _paragraph_match_score\([\s\S]*?    def _should_include_neighboring_context', '    def _should_include_neighboring_context', content)

with open("app/rag_service.py", "w") as f:
    f.write(content)
