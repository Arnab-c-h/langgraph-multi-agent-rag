"""HTTP-only schemas that keep persistence details private."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl

from rag_app.contracts import SourceType
from rag_app.ingestion.models import IngestionResult, SourceRecord


class SourceSummary(BaseModel):
    id: UUID
    source_type: SourceType
    title: str
    canonical_uri: str

    @classmethod
    def from_record(cls, source: SourceRecord) -> SourceSummary:
        return cls(
            id=source.id,
            source_type=source.source_type,
            title=source.title,
            canonical_uri=source.canonical_uri,
        )


class SourceUrlRequest(BaseModel):
    url: HttpUrl


class IngestionResponse(BaseModel):
    source_id: UUID
    job_id: UUID
    status: Literal["ingested", "unchanged"]
    chunk_count: int

    @classmethod
    def from_result(cls, result: IngestionResult) -> IngestionResponse:
        return cls(
            source_id=result.source.id,
            job_id=result.job.id,
            status=result.status,
            chunk_count=result.chunk_count or 0,
        )


class DeleteResponse(BaseModel):
    source_id: UUID
    deleted: bool = True


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    profile: str
    ready: bool
    dependencies: dict[str, bool] = Field(default_factory=dict)
