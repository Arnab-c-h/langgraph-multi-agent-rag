"""Environment-backed application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platformdirs import user_data_path
from pydantic import AliasChoices, Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings for the supported local full and test profiles."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_profile: Literal["full", "test"] = Field(
        default="full", validation_alias=AliasChoices("APP_PROFILE", "app_profile")
    )
    model_name: str = Field(default="gemini-3.5-flash", validation_alias=AliasChoices("MODEL_NAME", "model_name"))
    embedding_model: str = Field(
        default="gemini-embedding-2", validation_alias=AliasChoices("EMBEDDING_MODEL", "embedding_model")
    )
    embedding_dimensions: Literal[768] = Field(
        default=768, validation_alias=AliasChoices("EMBEDDING_DIMENSIONS", "embedding_dimensions")
    )
    query_concurrency: int = Field(
        default=2, ge=1, le=2, validation_alias=AliasChoices("QUERY_CONCURRENCY", "query_concurrency")
    )
    query_timeout_seconds: int = Field(default=120, ge=1, le=300)
    dependency_timeout_seconds: int = Field(default=30, ge=1, le=60)
    query_rate_limit_per_minute: int = Field(default=60, ge=1, le=600)
    api_max_request_bytes: int = Field(default=26 * 1024 * 1024, ge=64, le=26 * 1024 * 1024)
    provider_max_retries: int = Field(default=2, ge=0, le=3)
    trust_proxy_headers: bool = False
    public_base_url: HttpUrl = Field(default=HttpUrl("http://localhost:8000"))

    pdf_max_bytes: int = Field(default=25 * 1024 * 1024, ge=1, le=25 * 1024 * 1024)
    pdf_max_pages: int = Field(default=200, ge=1, le=200)
    web_max_bytes: int = Field(default=5 * 1024 * 1024, ge=1, le=5 * 1024 * 1024)
    web_timeout_seconds: int = Field(default=30, ge=1, le=30)
    web_max_urls: int = Field(default=1, ge=1, le=1)
    github_max_files: int = Field(default=1000, ge=1, le=1000)
    github_max_bytes: int = Field(default=50 * 1024 * 1024, ge=1, le=50 * 1024 * 1024)
    tot_branches: int = Field(default=3, ge=1, le=3)
    react_max_actions: int = Field(default=6, ge=1, le=6)
    reflection_revisions: int = Field(default=1, ge=0, le=1)
    consistency_candidates: int = Field(default=3, ge=1, le=3)
    decomposition_questions: int = Field(default=4, ge=1, le=4)
    recursive_max_depth: int = Field(default=2, ge=0, le=2)
    recursive_fanout: int = Field(default=3, ge=1, le=3)

    rag_data_dir: Path = Field(
        default_factory=lambda: _resolve_data_dir(None),
        validation_alias=AliasChoices("RAG_DATA_DIR", "rag_data_dir"),
    )
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "gemini_api_key")
    )
    admin_token: SecretStr | None = Field(default=None, validation_alias=AliasChoices("ADMIN_TOKEN", "admin_token"))

    @field_validator("rag_data_dir", mode="before")
    @classmethod
    def normalize_rag_data_dir(cls, value: str | Path | None) -> Path:
        return _resolve_data_dir(value)

    def validate_runtime_credentials(self) -> None:
        """Fail closed at lifespan startup while keeping ASGI imports inert."""
        if self.app_profile == "test":
            return
        credentials = {
            "GEMINI_API_KEY": self.gemini_api_key,
            "ADMIN_TOKEN": self.admin_token,
        }
        missing = [name for name, value in credentials.items() if value is None or not value.get_secret_value().strip()]
        if missing:
            raise ValueError(f"{self.app_profile.title()} profile requires: {', '.join(missing)}")

    def safe_summary(self) -> dict[str, str | int | bool]:
        """Return diagnostic configuration without exposing secret material."""
        return {
            "app_profile": self.app_profile,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "query_concurrency": self.query_concurrency,
            "query_timeout_seconds": self.query_timeout_seconds,
            "trust_proxy_headers": self.trust_proxy_headers,
            "gemini_configured": self.gemini_api_key is not None,
            "admin_token_configured": self.admin_token is not None,
        }


def _resolve_data_dir(value: str | Path | None) -> Path:
    """Normalize the local data root without creating filesystem state."""
    candidate = str(value).strip() if value is not None else ""
    base = Path(candidate) if candidate else user_data_path("LangGraphMultiAgentRAG", appauthor=False)
    return base.expanduser().resolve()
