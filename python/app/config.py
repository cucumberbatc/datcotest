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
    openai_chat_model: str = Field(...)
    openai_embedding_model: str = Field(...)
    openai_temperature: float = 0.0
    rag_retrieval_k: int = 6
    rag_answer_top_k: int = 3
    rag_min_score: float = 0.0
    ocr_enabled: bool = True
    ocr_languages: str = "ko,en"
    ocr_zoom: float = 2.0
    ocr_min_text_chars: int = 40
    ocr_gpu: bool = False

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
