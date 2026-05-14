from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "datcotest-python-rag"
    api_prefix: str = "/api"
    python_service_port: int = 8000
    python_storage_root: str = "./storage"
    openai_api_key: str | None = None
    openai_chat_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_temperature: float = 0.0
    rag_page_candidate_k: int = 3
    rag_page_score_boost: float = 0.12
    rag_retrieval_k: int = 6
    rag_answer_top_k: int = 3
    rag_candidate_pool_k: int = 12
    rag_rerank_top_k: int = 7
    rag_query_expansion_k: int = 2
    rag_min_score: float = 0.0
    rag_strong_evidence_score: float = 0.65
    ocr_enabled: bool = True
    ocr_languages: str = "ko,en"
    ocr_zoom: float = 3.0
    ocr_min_text_chars: int = 40
    ocr_confidence_threshold: float = 0.15
    ocr_gpu: bool = False
    ocr_version: str = "PP-OCRv5"
    ocr_det_model_name: str = "PP-OCRv5_mobile_det"
    ocr_rec_model_name: str = "korean_PP-OCRv5_mobile_rec"
    ocr_cpu_threads: int = 8
    document_parse_backend: str = "pymupdf"
    llama_cloud_api_key: str | None = None
    llama_parse_tier: str = "agentic"
    llama_parse_version: str = "latest"
    llama_parse_fallback_to_pymupdf: bool = True

    @property
    def storage_root(self) -> Path:
        return Path(self.python_storage_root).resolve()

    @property
    def uploads_root(self) -> Path:
        return self.storage_root / "uploads"

    @property
    def highlights_root(self) -> Path:
        return self.storage_root / "highlights"

    @property
    def page_images_root(self) -> Path:
        return self.storage_root / "page_images"

    @property
    def ocr_pages_root(self) -> Path:
        return self.storage_root / "ocr_pages"


@lru_cache
def get_settings() -> Settings:
    return Settings()
