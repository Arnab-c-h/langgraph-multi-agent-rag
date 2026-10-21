from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest

from rag_app.config import Settings
from rag_app.contracts import QueryRequest, SourceType
from rag_app.errors import AppError, ErrorCode
from rag_app.retrieval.models import EvidenceAssessment, RetrievedChunk
from rag_app.retrieval.protocols import CitationValidator, EvidenceGrader, Retriever
from rag_app.retrieval.service import StrictRagService


def _chunk(
    *,
    content: str = "Retrieved evidence.",
    source_type: SourceType = SourceType.WEB,
    source_id: UUID | None = None,
    chunk_id: UUID | None = None,
    source_url: str | None = "https://example.com/rag",
    source_path: str | None = None,
    locator: str = "section Evidence",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid4(),
        source_id=source_id or uuid4(),
        source_type=source_type,
        source_title="Evidence guide",
        content=content,
        supporting_excerpt=content,
        locator=locator,
        source_url=source_url,
        source_path=source_path,
        metadata={"heading": "Evidence"},
        provenance={"final_url": "https://example.com/rag"},
        similarity=0.92,
    )


class _FakeRetriever:
    def __init__(self, chunks: Sequence[RetrievedChunk]) -> None:
        self._chunks = list(chunks)

    async def retrieve(self, _request: QueryRequest) -> list[RetrievedChunk]:
        return self._chunks


class _FakeGrader:
    def __init__(self, decisions: Sequence[bool]) -> None:
        self._decisions = list(decisions)

    async def grade(self, _question: str, candidates: Sequence[RetrievedChunk]) -> list[EvidenceAssessment]:
        return [
            EvidenceAssessment(chunk=chunk, relevant=decision, confidence=0.9 if decision else 0.1)
            for chunk, decision in zip(candidates, self._decisions, strict=True)
        ]


class _StaticGrader:
    def __init__(self, assessments: Sequence[EvidenceAssessment]) -> None:
        self._assessments = list(assessments)

    async def grade(self, _question: str, _candidates: Sequence[RetrievedChunk]) -> list[EvidenceAssessment]:
        return self._assessments


class _AlwaysValidCitationValidator:
    def __init__(self) -> None:
        from rag_app.retrieval.citations import StrictCitationValidator

        self._delegate = StrictCitationValidator()

    def validate(
        self, evidence: Sequence[EvidenceAssessment], citation_chunk_ids: Sequence[UUID] | None = None
    ) -> list[object]:
        return self._delegate.validate(evidence, citation_chunk_ids)


@pytest.mark.asyncio
async def test_strict_rag_returns_only_relevant_evidence_with_exact_stored_citation_excerpt() -> None:
    accepted = _chunk(content="The corpus requires retrieved evidence.")
    rejected = _chunk(content="An unrelated note.")
    service = StrictRagService(
        retriever=_FakeRetriever([accepted, rejected]),
        grader=_FakeGrader([True, False]),
        citation_validator=_AlwaysValidCitationValidator(),
    )

    bundle = await service.prepare(QueryRequest(question="What does the corpus require?"))

    assert [item.chunk.chunk_id for item in bundle.evidence] == [accepted.chunk_id]
    assert bundle.citations[0].supporting_excerpt == accepted.content
    assert bundle.citations[0].locator == accepted.locator
    assert str(bundle.citations[0].source_url) == accepted.source_url
    assert bundle.used_source_ids == [accepted.source_id]
    assert bundle.retrieval_summary == {"candidate_count": 2, "accepted_count": 1}


@pytest.mark.asyncio
async def test_strict_rag_refuses_when_no_positive_evidence_survives_grading() -> None:
    service = StrictRagService(
        retriever=_FakeRetriever([_chunk()]),
        grader=_FakeGrader([False]),
        citation_validator=_AlwaysValidCitationValidator(),
    )

    with pytest.raises(AppError) as raised:
        await service.prepare(QueryRequest(question="Unsupported question"))

    assert raised.value.code is ErrorCode.NO_SUPPORTING_EVIDENCE
    assert "Unsupported" not in str(raised.value)


@pytest.mark.asyncio
async def test_strict_rag_canonicalizes_forged_grader_chunk_fields_by_known_chunk_id() -> None:
    stored = _chunk(content="Canonical stored evidence", locator="page 2")
    forged = _chunk(
        chunk_id=stored.chunk_id,
        source_id=uuid4(),
        content="Forged model-controlled content",
        locator="forged locator",
        source_url="https://attacker.invalid",
    )
    service = StrictRagService(
        retriever=_FakeRetriever([stored]),
        grader=_StaticGrader([EvidenceAssessment(chunk=forged, relevant=True, confidence=0.9, summary="safe")]),
        citation_validator=_AlwaysValidCitationValidator(),
    )

    bundle = await service.prepare(QueryRequest(question="Question"))

    assert bundle.evidence[0].chunk == stored
    assert bundle.evidence[0].summary == "safe"
    assert bundle.citations[0].supporting_excerpt == "Canonical stored evidence"
    assert bundle.citations[0].locator == "page 2"


@pytest.mark.asyncio
async def test_strict_rag_deduplicates_grader_assessments_and_restores_candidate_order() -> None:
    first = _chunk(content="First")
    second = _chunk(content="Second")
    service = StrictRagService(
        retriever=_FakeRetriever([first, second]),
        grader=_StaticGrader(
            [
                EvidenceAssessment(chunk=second, relevant=True, confidence=0.8),
                EvidenceAssessment(chunk=first, relevant=True, confidence=0.7),
                EvidenceAssessment(chunk=first, relevant=True, confidence=0.6),
                EvidenceAssessment(chunk=_chunk(), relevant=True, confidence=1.0),
            ]
        ),
        citation_validator=_AlwaysValidCitationValidator(),
    )

    bundle = await service.prepare(QueryRequest(question="Question"))

    assert [assessment.chunk.chunk_id for assessment in bundle.evidence] == [first.chunk_id, second.chunk_id]
    assert [assessment.confidence for assessment in bundle.evidence] == [0.7, 0.8]
    assert [citation.chunk_id for citation in bundle.citations] == [first.chunk_id, second.chunk_id]


def test_citation_validator_rejects_unknown_chunk_references_and_deduplicates_known_references() -> None:
    from rag_app.retrieval.citations import StrictCitationValidator

    chunk = _chunk(content="Exact stored evidence")
    assessment = EvidenceAssessment(chunk=chunk, relevant=True, confidence=0.95)
    validator = StrictCitationValidator()

    citations = validator.validate([assessment], [chunk.chunk_id, chunk.chunk_id])

    assert len(citations) == 1
    assert citations[0].supporting_excerpt == "Exact stored evidence"
    with pytest.raises(AppError) as raised:
        validator.validate([assessment], [uuid4()])
    assert raised.value.code is ErrorCode.NO_SUPPORTING_EVIDENCE


def test_protocols_accept_fake_friendly_async_services() -> None:
    assert isinstance(_FakeRetriever([]), Retriever)
    assert isinstance(_FakeGrader([]), EvidenceGrader)
    assert isinstance(_AlwaysValidCitationValidator(), CitationValidator)


@pytest.mark.asyncio
async def test_gemini_grader_uses_structured_output_and_maps_malformed_or_provider_errors_to_a_safe_code() -> None:
    from rag_app.retrieval.grading import GeminiEvidenceGrader

    chunk = _chunk()

    class StructuredModel:
        def __init__(self, result: object) -> None:
            self.result = result
            self.schemas: list[type[object]] = []

        def with_structured_output(self, schema: type[object]) -> StructuredModel:
            self.schemas.append(schema)
            return self

        async def ainvoke(self, _payload: object) -> object:
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    model = StructuredModel({"relevant": True, "confidence": 0.8, "summary": "Question is supported."})
    graded = await GeminiEvidenceGrader(model=model).grade("Question", [chunk])

    assert graded == [EvidenceAssessment(chunk=chunk, relevant=True, confidence=0.8, summary="Question is supported.")]
    assert model.schemas
    for bad_result in ({"relevant": "maybe"}, RuntimeError("token=private")):
        with pytest.raises(AppError) as raised:
            await GeminiEvidenceGrader(model=StructuredModel(bad_result)).grade("Question", [chunk])
        assert raised.value.code is ErrorCode.PROVIDER_RATE_LIMITED
        assert "private" not in str(raised.value)


def test_gemini_grader_uses_configured_model_name_through_a_settings_factory() -> None:
    from rag_app.retrieval.grading import GeminiEvidenceGrader

    class StructuredModel:
        def with_structured_output(self, _schema: type[object]) -> StructuredModel:
            return self

    grader = GeminiEvidenceGrader.from_settings(
        Settings(app_profile="test", model_name="gemini-custom-test"), model=StructuredModel()
    )

    assert grader.model_name == "gemini-custom-test"
