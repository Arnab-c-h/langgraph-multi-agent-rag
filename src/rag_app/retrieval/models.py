"""Typed records passed from retrieval to the bounded workflows."""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from pydantic import BaseModel, Field

from rag_app.contracts import Citation, SourceType


class RetrievedChunk(BaseModel):
    """A stored chunk plus its source and retrieval metadata."""

    chunk_id: UUID
    source_id: UUID
    source_type: SourceType
    source_title: str = Field(min_length=1, max_length=512)
    content: str = Field(min_length=1)
    supporting_excerpt: str = Field(min_length=1, max_length=4000)
    locator: str = Field(min_length=1, max_length=512)
    source_url: str | None = Field(default=None, max_length=4096)
    source_path: str | None = Field(default=None, max_length=4096)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    similarity: float = Field(ge=-1, le=1)

    def citation(self) -> Citation:
        return Citation(
            source_id=self.source_id,
            chunk_id=self.chunk_id,
            source_title=self.source_title,
            locator=self.locator,
            source_url=cast(Any, self.source_url),
            source_path=self.source_path,
            supporting_excerpt=self.supporting_excerpt,
        )


class EvidenceAssessment(BaseModel):
    """A relevance decision with only an externally safe summary."""

    chunk: RetrievedChunk
    relevant: bool
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(default="", max_length=512)


class EvidenceBundle(BaseModel):
    """Validated, positively relevant evidence for a workflow to use."""

    evidence: list[EvidenceAssessment] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    used_source_ids: list[UUID] = Field(default_factory=list)
    retrieval_summary: dict[str, int] = Field(default_factory=dict)
    validation_summary: dict[str, int] = Field(default_factory=dict)
