"""Citation validation backed only by retrieved stored evidence."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from rag_app.contracts import Citation
from rag_app.errors import AppError, ErrorCode
from rag_app.retrieval.models import EvidenceAssessment


class StrictCitationValidator:
    """Maps approved evidence to citations without trusting model-created text."""

    def validate(
        self, evidence: Sequence[EvidenceAssessment], citation_chunk_ids: Sequence[UUID] | None = None
    ) -> list[Citation]:
        known = {item.chunk.chunk_id: item for item in evidence if item.relevant}
        requested = list(citation_chunk_ids) if citation_chunk_ids is not None else list(known)
        citations: list[Citation] = []
        seen: set[UUID] = set()
        for chunk_id in requested:
            if chunk_id in seen:
                continue
            assessment = known.get(chunk_id)
            if assessment is None:
                raise AppError(ErrorCode.NO_SUPPORTING_EVIDENCE, "Unknown citation reference")
            citations.append(assessment.chunk.citation())
            seen.add(chunk_id)
        if not citations:
            raise AppError(ErrorCode.NO_SUPPORTING_EVIDENCE, "No validated citations")
        return citations
