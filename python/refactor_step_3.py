import re

with open("app/rag_service.py", "r") as f:
    content = f.read()

rebuild_old = """    def rebuild_index(self) -> None:
        settings = self._settings()
        chunks: list[ChunkRecord] = []
        for document in store.list_documents():
            chunks.extend(document.chunks)

        if not chunks or not settings.openai_api_key:
            store.vector_store = None
            return

        embedding = self._embeddings()
        documents = [
            Document(
                page_content=self._embedding_search_text(chunk.text),
                metadata={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "file_name": chunk.file_name,
                    "page_number": chunk.page_number,
                    "paragraph_index": chunk.paragraph_index,
                    "paragraph_end_index": chunk.paragraph_end_index,
                    "section_title": chunk.section_title,
                    "rects": [list(rect) for rect in chunk.rects],
                },
            )
            for chunk in chunks
        ]
        ids = [chunk.id for chunk in chunks]
        store.vector_store = FAISS.from_documents(documents, embedding=embedding, ids=ids)"""

rebuild_new = """    def rebuild_index(self) -> None:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        settings = self._settings()
        chunks: list[ChunkRecord] = []
        for document in store.list_documents():
            chunks.extend(document.chunks)

        if not chunks or not settings.openai_api_key:
            store.vector_store = None
            return

        embedding = self._embeddings()
        
        # Parent-Child Chunking: Split large logical chunks into smaller child texts for dense retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\\n\\n", "\\n", ".", "?", "!", " ", ""],
        )

        documents = []
        ids = []
        for chunk in chunks:
            child_texts = text_splitter.split_text(chunk.text)
            for i, child_text in enumerate(child_texts):
                child_id = f"{chunk.id}-child-{i}"
                documents.append(
                    Document(
                        page_content=self._embedding_search_text(child_text),
                        metadata={
                            "parent_chunk_id": chunk.id,
                            "chunk_id": chunk.id, # Keep this to map back to the parent ChunkRecord in _retrieve
                            "document_id": chunk.document_id,
                            "file_name": chunk.file_name,
                            "page_number": chunk.page_number,
                            "paragraph_index": chunk.paragraph_index,
                            "paragraph_end_index": chunk.paragraph_end_index,
                            "section_title": chunk.section_title,
                            "rects": [list(rect) for rect in chunk.rects],
                        },
                    )
                )
                ids.append(child_id)
                
        store.vector_store = FAISS.from_documents(documents, embedding=embedding, ids=ids)"""

content = content.replace(rebuild_old, rebuild_new)

with open("app/rag_service.py", "w") as f:
    f.write(content)
