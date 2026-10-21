"""Typed records shared by source loaders, persistence, and ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from rag_app.contracts import SourceType


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SourceRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source_type: SourceType
    canonical_uri: str = Field(min_length=1, max_length=4096)
    title: str = Field(min_length=1, max_length=512)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class ChunkRecord(BaseModel):
    id: UUID
    source_id: UUID
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class IngestionJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    status: Literal["started", "completed", "failed", "unchanged"]
    error_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None


class IngestionResult(BaseModel):
    source: SourceRecord
    job: IngestionJob
    chunks: list[ChunkRecord] = Field(default_factory=list)
    status: Literal["ingested", "unchanged"]
    chunk_count: int | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.chunk_count is None:
            self.chunk_count = len(self.chunks)


class DocumentSegment(BaseModel):
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedDocument(BaseModel):
    """Loader output before its content is hashed, embedded, and persisted."""

    source_type: SourceType
    canonical_uri: str = Field(min_length=1, max_length=4096)
    title: str = Field(min_length=1, max_length=512)
    content: str = Field(min_length=1)
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    segments: list[DocumentSegment] = Field(default_factory=list)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("content must not be blank")
        return normalized
