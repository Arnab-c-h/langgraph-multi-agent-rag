"""Fake-friendly async boundaries for retrieval and evidence gates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable
from uuid import UUID

from rag_app.contracts import Citation, QueryRequest
from rag_app.retrieval.models import EvidenceAssessment, RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, request: QueryRequest) -> list[RetrievedChunk]: ...


@runtime_checkable
class EvidenceGrader(Protocol):
    async def grade(self, question: str, candidates: Sequence[RetrievedChunk]) -> list[EvidenceAssessment]: ...


@runtime_checkable
class CitationValidator(Protocol):
    def validate(
        self, evidence: Sequence[EvidenceAssessment], citation_chunk_ids: Sequence[UUID] | None = None
    ) -> list[Citation]: ...
