"""Strict evidence orchestration shared by every reasoning workflow."""

from __future__ import annotations

from uuid import UUID

from rag_app.contracts import QueryRequest
from rag_app.errors import AppError, ErrorCode
from rag_app.retrieval.models import EvidenceAssessment, EvidenceBundle, RetrievedChunk
from rag_app.retrieval.protocols import CitationValidator, EvidenceGrader, Retriever


class StrictRagService:
    """Retrieve, grade, and validate evidence without generating an answer."""

    def __init__(self, retriever: Retriever, grader: EvidenceGrader, citation_validator: CitationValidator) -> None:
        self._retriever = retriever
        self._grader = grader
        self._citation_validator = citation_validator

    async def prepare(self, request: QueryRequest) -> EvidenceBundle:
        candidates = await self._retriever.retrieve(request)
        if not candidates:
            raise AppError(ErrorCode.NO_SUPPORTING_EVIDENCE, "No retrieved evidence")
        assessments = await self._grader.grade(request.question, candidates)
        candidates_by_id = {candidate.chunk_id: candidate for candidate in candidates}
        first_decision_by_id: dict[UUID, EvidenceAssessment] = {}
        for assessment in assessments:
            chunk_id = assessment.chunk.chunk_id
            if chunk_id in candidates_by_id and chunk_id not in first_decision_by_id:
                first_decision_by_id[chunk_id] = assessment
        positive = [
            _canonical_assessment(candidate, first_decision_by_id[candidate.chunk_id])
            for candidate in _unique_candidates_in_order(candidates)
            if candidate.chunk_id in first_decision_by_id and first_decision_by_id[candidate.chunk_id].relevant
        ]
        if not positive:
            raise AppError(ErrorCode.NO_SUPPORTING_EVIDENCE, "No relevant evidence")
        citations = self._citation_validator.validate(positive)
        used_source_ids = list(dict.fromkeys(item.chunk.source_id for item in positive))
        return EvidenceBundle(
            evidence=positive,
            citations=citations,
            used_source_ids=used_source_ids,
            retrieval_summary={"candidate_count": len(candidates), "accepted_count": len(positive)},
            validation_summary={"citation_count": len(citations), "rejected_count": len(candidates) - len(positive)},
        )


def _unique_candidates_in_order(candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Return one stored candidate per chunk ID in retrieval order."""
    unique: list[RetrievedChunk] = []
    seen: set[UUID] = set()
    for candidate in candidates:
        if candidate.chunk_id not in seen:
            unique.append(candidate)
            seen.add(candidate.chunk_id)
    return unique


def _canonical_assessment(candidate: RetrievedChunk, decision: EvidenceAssessment) -> EvidenceAssessment:
    """Keep grader decision fields but never a model-controlled chunk record."""
    return EvidenceAssessment(
        chunk=candidate,
        relevant=decision.relevant,
        confidence=decision.confidence,
        summary=decision.summary,
    )
