"""Application configuration loaded from environment variables / `.env` via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["mock", "openai", "anthropic"]
EmbeddingProviderName = Literal["hashing", "sentence-transformers", "openai"]
VectorStoreName = Literal["chroma", "qdrant", "memory"]
RerankerName = Literal["cross-encoder", "lexical", "none"]
RetrievalMode = Literal["hybrid", "dense", "bm25"]


class Settings(BaseSettings):
    """All runtime configuration. Every field maps to an upper-case env var of the same name."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Service -----------------------------------------------------------------------------
    app_name: str = "Hybrid RAG"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"
    data_dir: Path = Path("./data")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_upload_mb: int = Field(default=25, ge=1)
    auto_seed_dir: Path | None = None  # ingest this directory on startup if the index is empty

    # --- LLM ---------------------------------------------------------------------------------
    llm_provider: LLMProviderName = "mock"
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    llm_max_tokens: int = Field(default=8192, ge=256)
    llm_timeout_s: float = Field(default=120.0, gt=0)
    mock_stream_delay_s: float = Field(default=0.012, ge=0)

    # --- Embeddings --------------------------------------------------------------------------
    embedding_provider: EmbeddingProviderName = "hashing"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    hashing_dim: int = Field(default=768, ge=64)
    embedding_batch_size: int = Field(default=64, ge=1)

    # --- Vector store ------------------------------------------------------------------------
    vector_store: VectorStoreName = "chroma"
    collection_prefix: str = "docs"
    qdrant_url: str | None = None  # e.g. http://qdrant:6333 ; unset -> embedded local mode
    qdrant_api_key: SecretStr | None = None

    # --- Chunking ----------------------------------------------------------------------------
    chunk_size_tokens: int = Field(default=320, ge=32)
    chunk_overlap_tokens: int = Field(default=48, ge=0)
    chunk_min_tokens: int = Field(default=24, ge=1)
    tokenizer: Literal["tiktoken", "simple"] = "tiktoken"
    tiktoken_encoding: str = "cl100k_base"

    # --- Retrieval ---------------------------------------------------------------------------
    retrieval_mode: RetrievalMode = "hybrid"
    dense_top_k: int = Field(default=30, ge=1)
    bm25_top_k: int = Field(default=30, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=1.0, ge=0)
    bm25_weight: float = Field(default=1.0, ge=0)
    rerank_candidates: int = Field(default=20, ge=1)
    final_top_k: int = Field(default=5, ge=1, le=20)
    reranker: RerankerName = "lexical"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Verification ------------------------------------------------------------------------
    verify_citations: bool = True
    verifier_threshold: float = Field(default=0.55, ge=0, le=1)
    verifier_lexical_weight: float = Field(default=0.5, ge=0, le=1)
    verifier_llm_judge: bool = False
    strip_unsupported_citations: bool = True

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.llm_provider == "openai" and self.openai_api_key is None:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        if self.llm_provider == "anthropic" and self.anthropic_api_key is None:
            raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        if self.embedding_provider == "openai" and self.openai_api_key is None:
            raise ValueError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        return self

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "docstore.sqlite3"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def llm_model_name(self) -> str:
        return {
            "mock": "mock-extractive",
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
        }[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
