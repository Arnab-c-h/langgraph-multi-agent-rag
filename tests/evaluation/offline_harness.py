"""Deterministic end-to-end evaluation over a small synthetic corpus."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from rag_app.config import Settings
from rag_app.contracts import ErrorEvent, FinalEvent, QueryRequest, ReasoningMode, SourceType, StreamEvent
from rag_app.retrieval.citations import StrictCitationValidator
from rag_app.retrieval.models import EvidenceAssessment, RetrievedChunk
from rag_app.retrieval.service import StrictRagService
from rag_app.workflows import ModelOperation, ModelRequest, ModelResult, WorkflowRegistry, WorkflowRunner

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "corpus.json"
THRESHOLDS = {
    "refusal_accuracy": 1.0,
    "citation_validity": 1.0,
    "citation_completeness": 1.0,
    "citation_precision": 1.0,
    "filter_correctness": 1.0,
    "stream_lifecycle": 1.0,
    "safety": 1.0,
}

_WORD = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
}


class DeterministicEmbeddings:
    """Stable 768-dimensional token hashing; no model or network access."""

    dimensions = 768

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


class FixtureRetriever:
    def __init__(self, chunks: Sequence[RetrievedChunk], embeddings: DeterministicEmbeddings | None = None) -> None:
        self.chunks = list(chunks)
        self.embeddings = embeddings or DeterministicEmbeddings()

    async def retrieve(self, request: QueryRequest) -> list[RetrievedChunk]:
        query_vector = self.embeddings.embed_query(request.question)
        allowed_types = set(request.source_types or SourceType)
        allowed_ids = set(request.source_ids) if request.source_ids else None
        ranked: list[RetrievedChunk] = []
        for chunk in self.chunks:
            if chunk.source_type not in allowed_types or (
                allowed_ids is not None and chunk.source_id not in allowed_ids
            ):
                continue
            similarity = _cosine(query_vector, self.embeddings.embed_query(chunk.content))
            if similarity > 0:
                ranked.append(chunk.model_copy(update={"similarity": similarity}))
        ranked.sort(key=lambda item: (-item.similarity, str(item.chunk_id)))
        return ranked[: request.top_k]


class DeterministicEvidenceGrader:
    async def grade(
        self, question: str, candidates: Sequence[RetrievedChunk]
    ) -> list[EvidenceAssessment]:
        question_tokens = _tokens(question)
        return [
            EvidenceAssessment(
                chunk=chunk,
                relevant=bool(question_tokens & _tokens(chunk.content)),
                confidence=min(1.0, max(0.0, chunk.similarity)),
                summary="deterministic lexical support",
            )
            for chunk in candidates
        ]


class DeterministicWorkflowModel:
    """Returns public artifacts and exact fixture citation IDs only."""

    async def complete(self, request: ModelRequest) -> ModelResult:
        chunk_ids = [assessment.chunk.chunk_id for assessment in request.evidence.evidence]
        answer = " ".join(
            f"{assessment.chunk.content} [chunk:{assessment.chunk.chunk_id}]"
            for assessment in request.evidence.evidence
        )
        if request.operation in {ModelOperation.PLAN, ModelOperation.DECOMPOSE}:
            return ModelResult(items=[request.question] * request.max_items)
        if request.operation is ModelOperation.APPROACH:
            return ModelResult(text="Compare the accepted evidence and cite each supporting chunk.")
        if request.operation is ModelOperation.RANK:
            return ModelResult(selected_index=0, items=["Selected by accepted evidence coverage."])
        if request.operation is ModelOperation.ACT:
            return ModelResult(text=request.question, complete=True)
        if request.operation in {ModelOperation.CRITIQUE, ModelOperation.VALIDATE}:
            return ModelResult(approved=True)
        return ModelResult(text=answer, citation_chunk_ids=chunk_ids)


@dataclass(frozen=True)
class EvaluationReport:
    metrics: dict[str, float]
    thresholds: dict[str, float]
    modes: tuple[str, ...]
    cases: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(self.metrics[name] >= threshold for name, threshold in self.thresholds.items())


def load_fixture_corpus() -> list[RetrievedChunk]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return [RetrievedChunk.model_validate({**item, "similarity": 0.0}) for item in payload]


def build_runner(chunks: Sequence[RetrievedChunk] | None = None) -> tuple[WorkflowRunner, FixtureRetriever]:
    corpus = list(chunks or load_fixture_corpus())
    retriever = FixtureRetriever(corpus)
    validator = StrictCitationValidator()
    rag = StrictRagService(retriever, DeterministicEvidenceGrader(), validator)
    registry = WorkflowRegistry(
        rag=rag,
        model=DeterministicWorkflowModel(),
        citation_validator=validator,
        settings=Settings(app_profile="test"),
    )
    return WorkflowRunner(registry), retriever


async def run_offline_evaluation() -> EvaluationReport:
    corpus = load_fixture_corpus()
    runner, retriever = build_runner(corpus)
    by_id = {chunk.chunk_id: chunk for chunk in corpus}
    cases = (
        (
            "pdf_retention",
            QueryRequest(question="Where are original uploaded PDF binaries retained?", source_types=[SourceType.PDF]),
            {UUID("00000000-0000-0000-0000-000000000101")},
        ),
        (
            "web_safety",
            QueryRequest(
                question="What private network and redirect protections apply to web ingestion?",
                source_types=[SourceType.WEB],
            ),
            {UUID("00000000-0000-0000-0000-000000000102")},
        ),
        (
            "multi_source",
            QueryRequest(question="What ingestion protections cover web URLs and repository files?"),
            {
                UUID("00000000-0000-0000-0000-000000000102"),
                UUID("00000000-0000-0000-0000-000000000103"),
            },
        ),
    )
    validity: list[bool] = []
    completeness: list[bool] = []
    precision: list[float] = []
    safety: list[bool] = []
    case_names: list[str] = []
    for name, request, expected_ids in cases:
        events = [event async for event in runner.stream(request)]
        final = next((event for event in events if isinstance(event, FinalEvent)), None)
        case_names.append(name)
        if final is None:
            validity.append(False)
            completeness.append(False)
            precision.append(0.0)
            safety.append(False)
            continue
        cited_ids = {citation.chunk_id for citation in final.citations}
        completeness.append(expected_ids == cited_ids)
        precision.append(len(expected_ids & cited_ids) / len(cited_ids) if cited_ids else 0.0)
        validity.append(
            all(
                citation.chunk_id in by_id
                and citation.supporting_excerpt == by_id[citation.chunk_id].supporting_excerpt
                and citation.locator == by_id[citation.chunk_id].locator
                for citation in final.citations
            )
        )
        safety.append(_safe_events(events))

    filters = [
        (
            QueryRequest(question="What ingestion protections apply?", source_types=[SourceType.WEB]),
            {SourceType.WEB},
            None,
        ),
        (
            QueryRequest(
                question="What provider retry policy applies?",
                source_ids=[UUID("00000000-0000-0000-0000-000000000203")],
            ),
            None,
            {UUID("00000000-0000-0000-0000-000000000203")},
        ),
    ]
    filter_results: list[bool] = []
    for request, expected_types, expected_ids in filters:
        retrieved = await retriever.retrieve(request)
        filter_results.append(
            bool(retrieved)
            and (expected_types is None or {chunk.source_type for chunk in retrieved} <= expected_types)
            and (expected_ids is None or {chunk.source_id for chunk in retrieved} <= expected_ids)
        )

    unsupported = [event async for event in runner.stream(QueryRequest(question="What is the capital of Mars?"))]
    refusal = len(unsupported) == 1 and isinstance(unsupported[0], ErrorEvent)
    if refusal:
        refusal = unsupported[0].code.value == "NO_SUPPORTING_EVIDENCE"

    lifecycle_results: list[bool] = []
    for mode in ReasoningMode:
        request = QueryRequest(
            question="Where are original PDF binaries retained? Authorization: Bearer evaluation-secret-value",
            reasoning_mode=mode,
            source_types=[SourceType.PDF],
        )
        events = [event async for event in runner.stream(request)]
        types = [event.type for event in events]
        lifecycle_results.append(
            bool(types)
            and types[0] == "status"
            and types[-1] == "final"
            and "token" in types
            and "citation" in types
            and types.index("token") < types.index("citation") < types.index("final")
            and _safe_events(events)
        )

    metrics = {
        "refusal_accuracy": float(refusal),
        "citation_validity": _mean(validity),
        "citation_completeness": _mean(completeness),
        "citation_precision": sum(precision) / len(precision) if precision else 0.0,
        "filter_correctness": _mean(filter_results),
        "stream_lifecycle": _mean(lifecycle_results),
        "safety": _mean(safety + lifecycle_results),
    }
    return EvaluationReport(
        metrics=metrics,
        thresholds=dict(THRESHOLDS),
        modes=tuple(mode.value for mode in ReasoningMode),
        cases=tuple(case_names),
    )


def _tokens(value: str) -> set[str]:
    return {token for token in _WORD.findall(value.casefold()) if token not in _STOP_WORDS and len(token) > 1}


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _safe_events(events: Sequence[StreamEvent]) -> bool:
    serialized = " ".join(json.dumps(event.model_dump(mode="json"), sort_keys=True) for event in events).casefold()
    forbidden = (
        "evaluation-secret-value",
        "chain-of-thought",
        "private reasoning",
        "raw reasoning",
        "scratchpad",
        "internal prompt",
    )
    return not any(value in serialized for value in forbidden)


def _mean(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0
