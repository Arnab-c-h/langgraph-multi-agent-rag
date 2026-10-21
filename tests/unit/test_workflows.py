from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from rag_app.config import Settings
from rag_app.contracts import QueryRequest, ReasoningMode, SourceType, StreamEvent
from rag_app.errors import AppError, ErrorCode
from rag_app.retrieval.citations import StrictCitationValidator
from rag_app.retrieval.models import EvidenceAssessment, EvidenceBundle, RetrievedChunk
from rag_app.threads import ThreadTurn
from rag_app.workflows import (
    GeminiWorkflowModel,
    ModelOperation,
    ModelRequest,
    ModelResult,
    WorkflowRegistry,
    WorkflowRunner,
)


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID("00000000-0000-0000-0000-000000000101"),
        source_id=UUID("00000000-0000-0000-0000-000000000201"),
        source_type=SourceType.WEB,
        source_title="Grounded guide",
        content="LangGraph workflows must use retrieved evidence.",
        supporting_excerpt="LangGraph workflows must use retrieved evidence.",
        locator="Grounding",
        source_url="https://example.com/grounding",
        similarity=0.98,
    )


class FakeRag:
    def __init__(self, *, supported: bool = True) -> None:
        self.supported = supported
        self.requests: list[QueryRequest] = []
        self.chunk = _chunk()

    async def prepare(self, request: QueryRequest) -> EvidenceBundle:
        self.requests.append(request)
        if not self.supported:
            raise AppError(ErrorCode.NO_SUPPORTING_EVIDENCE, "private provider detail")
        assessment = EvidenceAssessment(chunk=self.chunk, relevant=True, confidence=0.99, summary="supported")
        return EvidenceBundle(
            evidence=[assessment],
            citations=[self.chunk.citation()],
            used_source_ids=[self.chunk.source_id],
            retrieval_summary={"candidate_count": 1, "accepted_count": 1},
            validation_summary={"citation_count": 1, "rejected_count": 0},
        )


class FakeModel:
    def __init__(self, chunk_id: UUID, *, invalid_citation: bool = False, fail: Exception | None = None) -> None:
        self.chunk_id = chunk_id
        self.invalid_citation = invalid_citation
        self.fail = fail
        self.calls: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResult:
        self.calls.append(request)
        if self.fail is not None:
            raise self.fail
        citation_ids = [uuid4()] if self.invalid_citation else [self.chunk_id]
        if request.operation in {ModelOperation.PLAN, ModelOperation.DECOMPOSE}:
            return ModelResult(items=[f"public step {index}" for index in range(request.max_items + 2)])
        if request.operation is ModelOperation.APPROACH:
            return ModelResult(text=f"approach {len(self.calls)}")
        if request.operation is ModelOperation.RANK:
            return ModelResult(text="approach 1", selected_index=0, items=["ranked by evidence"])
        if request.operation is ModelOperation.ACT:
            return ModelResult(text="inspect retrieved source", complete=False)
        if request.operation is ModelOperation.CRITIQUE:
            return ModelResult(text="citation coverage checked", approved=False)
        if request.operation is ModelOperation.JUDGE:
            return ModelResult(
                text=f"Grounded answer [chunk:{citation_ids[0]}]",
                citation_chunk_ids=citation_ids,
                selected_index=0,
            )
        if request.operation is ModelOperation.VALIDATE:
            return ModelResult(approved=True)
        return ModelResult(text=f"Grounded answer [chunk:{citation_ids[0]}]", citation_chunk_ids=citation_ids)


def _runner(
    mode: ReasoningMode,
    *,
    supported: bool = True,
    invalid_citation: bool = False,
    fail: Exception | None = None,
    settings: Settings | None = None,
    checkpointer: object | None = None,
    thread_store: object | None = None,
) -> tuple[WorkflowRunner, FakeRag, FakeModel, WorkflowRegistry]:
    rag = FakeRag(supported=supported)
    model = FakeModel(rag.chunk.chunk_id, invalid_citation=invalid_citation, fail=fail)
    registry = WorkflowRegistry(
        rag=rag,
        model=model,
        citation_validator=StrictCitationValidator(),
        settings=settings or Settings(app_profile="test"),
        checkpointer=checkpointer,
    )
    assert registry.graph_for(mode) is not None
    return WorkflowRunner(registry, thread_store=thread_store), rag, model, registry


async def _events(
    runner: WorkflowRunner, request: QueryRequest, cancel: asyncio.Event | None = None
) -> list[StreamEvent]:
    return [event async for event in runner.stream(request, cancel_event=cancel)]


def test_registry_has_nine_distinct_compiled_state_graphs() -> None:
    _, _, _, registry = _runner(ReasoningMode.STANDARD)
    graphs = [registry.graph_for(mode) for mode in ReasoningMode]

    assert len(graphs) == 9
    assert len({id(graph) for graph in graphs}) == 9
    assert {graph.name for graph in graphs} == {f"workflow_{mode.value}" for mode in ReasoningMode}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(ReasoningMode))
async def test_all_modes_stream_safe_ordered_events_and_validated_citations(mode: ReasoningMode) -> None:
    runner, _, _, _ = _runner(mode)
    request = QueryRequest(question="What is required?", reasoning_mode=mode, thread_id=uuid4())

    events = await _events(runner, request)

    event_types = [event.type for event in events]
    assert event_types[0] == "status"
    assert event_types[-1] == "final"
    assert "token" in event_types
    assert "citation" in event_types
    assert event_types.index("token") < event_types.index("citation") < event_types.index("final")
    final = events[-1]
    assert final.type == "final"
    assert final.mode is mode
    assert final.run_id is not None
    assert final.used_source_ids == [UUID("00000000-0000-0000-0000-000000000201")]
    assert final.trace_summary["validation_passed"] is True
    serialized = " ".join(str(event.model_dump(mode="json")) for event in events).lower()
    assert "chain-of-thought" not in serialized
    assert "scratchpad" not in serialized
    assert "internal prompt" not in serialized
    assert "private provider" not in serialized


@pytest.mark.asyncio
async def test_strict_refusal_is_exactly_one_public_error_event() -> None:
    runner, _, _, _ = _runner(ReasoningMode.STANDARD, supported=False)

    events = await _events(runner, QueryRequest(question="Unsupported"))

    assert len(events) == 1
    assert events[0].type == "error"
    assert events[0].code is ErrorCode.NO_SUPPORTING_EVIDENCE
    assert "private" not in events[0].message


@pytest.mark.asyncio
async def test_completed_thread_turns_flow_across_modes_into_public_model_context() -> None:
    class MemoryThreads:
        def __init__(self) -> None:
            self.turns: dict[UUID, list[ThreadTurn]] = {}

        async def load(self, thread_id: UUID, limit: int = 10) -> list[ThreadTurn]:
            return self.turns.get(thread_id, [])[-limit:]

        async def append(self, thread_id: UUID, turn: ThreadTurn) -> None:
            self.turns.setdefault(thread_id, []).append(turn)

    threads = MemoryThreads()
    runner, _, model, _ = _runner(ReasoningMode.STANDARD, thread_store=threads)
    thread_id = uuid4()

    first = await _events(runner, QueryRequest(question="First question", thread_id=thread_id))
    second = await _events(
        runner,
        QueryRequest(question="Follow-up", reasoning_mode=ReasoningMode.SELF_CONSISTENCY, thread_id=thread_id),
    )

    assert first[-1].type == "final" and second[-1].type == "final"
    assert len(threads.turns[thread_id]) == 2
    candidate_calls = [call for call in model.calls if call.operation is ModelOperation.CANDIDATE]
    assert candidate_calls
    assert all(
        any("First question" in item and "Grounded answer" in item for item in call.context)
        for call in candidate_calls
    )


@pytest.mark.asyncio
async def test_filters_and_thread_configuration_are_propagated_and_checkpointed() -> None:
    saver = InMemorySaver()
    runner, rag, _, _ = _runner(ReasoningMode.STANDARD, checkpointer=saver)
    thread_id = UUID("00000000-0000-0000-0000-000000000301")
    source_id = UUID("00000000-0000-0000-0000-000000000401")
    request = QueryRequest(
        question="Filtered?",
        thread_id=thread_id,
        source_types=[SourceType.WEB],
        source_ids=[source_id],
    )

    events = await _events(runner, request)
    from rag_app.workflows.runner import _storage_thread_id

    checkpoints = list(
        saver.list({"configurable": {"thread_id": _storage_thread_id(thread_id, ReasoningMode.STANDARD)}})
    )

    assert rag.requests[0].source_types == [SourceType.WEB]
    assert rag.requests[0].source_ids == [source_id]
    assert checkpoints
    assert events[-1].type == "final" and events[-1].thread_id == thread_id
    checkpoint_text = str(checkpoints[0].checkpoint).lower()
    assert "internal prompt" not in checkpoint_text
    assert "scratchpad" not in checkpoint_text


@pytest.mark.asyncio
async def test_reusing_a_thread_resets_per_run_bounds_instead_of_reusing_old_branches() -> None:
    saver = InMemorySaver()
    runner, _, model, _ = _runner(ReasoningMode.TOT, checkpointer=saver)
    thread_id = uuid4()
    request = QueryRequest(question="First", reasoning_mode=ReasoningMode.TOT, thread_id=thread_id)

    await _events(runner, request)
    first_count = sum(call.operation is ModelOperation.APPROACH for call in model.calls)
    await _events(runner, request.model_copy(update={"question": "Second"}))
    second_count = sum(call.operation is ModelOperation.APPROACH for call in model.calls) - first_count

    assert first_count == 3
    assert second_count == 3


@pytest.mark.asyncio
async def test_pending_thread_checkpoint_resumes_without_restarting_retrieval() -> None:
    from rag_app.workflows.runner import _initial_state, _run_config

    saver = InMemorySaver()
    runner, rag, _, registry = _runner(ReasoningMode.STANDARD, checkpointer=saver)
    request = QueryRequest(question="Resume", thread_id=uuid4())
    original_run_id = uuid4()
    config = _run_config(request.thread_id, request.reasoning_mode, original_run_id)
    assert config["configurable"]["thread_id"] == f"{request.thread_id}:standard"
    assert config["configurable"]["checkpoint_ns"] == ""
    graph = registry.graph_for(ReasoningMode.STANDARD)
    await graph.ainvoke(_initial_state(request, original_run_id), config=config, interrupt_after=["retrieve"])
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("answer",)

    events = await _events(runner, request)

    assert events[-1].type == "final" and events[-1].run_id == original_run_id
    assert len(rag.requests) == 1


@pytest.mark.asyncio
async def test_one_public_thread_resumes_pending_modes_from_independent_storage_keys() -> None:
    from rag_app.workflows.runner import _initial_state, _run_config

    saver = InMemorySaver()
    rag = FakeRag()
    model = FakeModel(rag.chunk.chunk_id)
    registry = WorkflowRegistry(
        rag=rag,
        model=model,
        citation_validator=StrictCitationValidator(),
        settings=Settings(app_profile="test"),
        checkpointer=saver,
    )
    runner = WorkflowRunner(registry)
    thread_id = uuid4()
    requests = [
        QueryRequest(question="Standard pending", thread_id=thread_id, reasoning_mode=ReasoningMode.STANDARD),
        QueryRequest(question="CoT pending", thread_id=thread_id, reasoning_mode=ReasoningMode.COT),
    ]
    original_runs = [uuid4(), uuid4()]
    for request, run_id in zip(requests, original_runs, strict=True):
        config = _run_config(thread_id, request.reasoning_mode, run_id)
        await registry.graph_for(request.reasoning_mode).ainvoke(
            _initial_state(request, run_id), config=config, interrupt_after=["retrieve"]
        )

    finals = [(await _events(runner, request))[-1] for request in requests]

    assert all(final.type == "final" and final.thread_id == thread_id for final in finals)
    assert [final.run_id for final in finals] == original_runs
    assert len(rag.requests) == 2


@pytest.mark.asyncio
async def test_pending_checkpoint_with_stale_question_or_filters_starts_fresh() -> None:
    saver = InMemorySaver()
    rag = FakeRag()

    class CancelOnceModel(FakeModel):
        def __init__(self, chunk_id: UUID) -> None:
            super().__init__(chunk_id)
            self.started = asyncio.Event()
            self.block_once = True

        async def complete(self, request: ModelRequest) -> ModelResult:
            if request.operation is ModelOperation.ANSWER and self.block_once:
                self.block_once = False
                self.started.set()
                await asyncio.Event().wait()
            return await super().complete(request)

    model = CancelOnceModel(rag.chunk.chunk_id)
    registry = WorkflowRegistry(
        rag=rag,
        model=model,
        citation_validator=StrictCitationValidator(),
        settings=Settings(app_profile="test"),
        checkpointer=saver,
    )
    runner = WorkflowRunner(registry)
    thread_id = uuid4()
    old_source = uuid4()
    new_source = uuid4()
    old = QueryRequest(question="Old cancelled question", thread_id=thread_id, source_ids=[old_source])
    cancellation = asyncio.Event()
    old_task = asyncio.create_task(_events(runner, old, cancellation))
    await model.started.wait()
    cancellation.set()
    cancelled_events = await asyncio.wait_for(old_task, timeout=1)
    assert cancelled_events[-1].type == "error" and cancelled_events[-1].code is ErrorCode.RUN_CANCELLED

    new = old.model_copy(update={"question": "New question", "source_ids": [new_source]})
    events = await _events(runner, new)

    assert events[-1].type == "final"
    assert [item.question for item in rag.requests] == ["Old cancelled question", "New question"]
    assert rag.requests[-1].source_ids == [new_source]


@pytest.mark.asyncio
async def test_invalid_model_citation_is_rejected_without_leaking_reference() -> None:
    runner, _, _, _ = _runner(ReasoningMode.STANDARD, invalid_citation=True)

    events = await _events(runner, QueryRequest(question="Question"))

    assert events[-1].type == "error" and events[-1].code is ErrorCode.NO_SUPPORTING_EVIDENCE
    assert all(event.type not in {"citation", "final"} for event in events)


@pytest.mark.asyncio
async def test_pre_cancelled_run_returns_stable_cancelled_error() -> None:
    runner, _, _, _ = _runner(ReasoningMode.STANDARD)
    cancel = asyncio.Event()
    cancel.set()

    events = await _events(runner, QueryRequest(question="Question"), cancel)

    assert len(events) == 1
    assert events[0].type == "error" and events[0].code is ErrorCode.RUN_CANCELLED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        AppError(ErrorCode.PROVIDER_RATE_LIMITED, "secret"),
        AppError(ErrorCode.PROVIDER_RATE_LIMITED, "api_key=private"),
    ],
)
async def test_provider_failures_are_mapped_to_stable_public_errors(failure: Exception) -> None:
    runner, _, _, _ = _runner(ReasoningMode.STANDARD, fail=failure)

    events = await _events(runner, QueryRequest(question="Question"))

    assert events[-1].type == "error" and events[-1].code is ErrorCode.PROVIDER_RATE_LIMITED
    assert "private" not in events[-1].message


@pytest.mark.asyncio
async def test_unexpected_internal_failure_is_not_misreported_as_rate_limit() -> None:
    runner, _, _, _ = _runner(ReasoningMode.STANDARD, fail=RuntimeError("api_key=private"))

    events = await _events(runner, QueryRequest(question="Question"))

    assert events[-1].type == "error" and events[-1].code is ErrorCode.RUN_FAILED
    assert "private" not in events[-1].message


@pytest.mark.asyncio
async def test_hard_mode_bounds_cannot_be_exceeded() -> None:
    settings = Settings(
        app_profile="test",
        tot_branches=3,
        react_max_actions=6,
        reflection_revisions=1,
        consistency_candidates=3,
        decomposition_questions=4,
        recursive_max_depth=2,
        recursive_fanout=3,
    )
    expected = {
        ReasoningMode.TOT: (ModelOperation.APPROACH, 3),
        ReasoningMode.REACT: (ModelOperation.ACT, 6),
        ReasoningMode.SELF_REFLECTION: (ModelOperation.REVISE, 1),
        ReasoningMode.SELF_CONSISTENCY: (ModelOperation.CANDIDATE, 3),
    }
    for mode, (operation, maximum) in expected.items():
        runner, _, model, _ = _runner(mode, settings=settings)
        await _events(runner, QueryRequest(question="Question", reasoning_mode=mode))
        assert sum(call.operation is operation for call in model.calls) == maximum

    for mode in (ReasoningMode.COT, ReasoningMode.DECOMPOSED, ReasoningMode.LEAST_TO_MOST):
        runner, _, model, _ = _runner(mode, settings=settings)
        await _events(runner, QueryRequest(question="Question", reasoning_mode=mode))
        planning = [call for call in model.calls if call.operation in {ModelOperation.PLAN, ModelOperation.DECOMPOSE}]
        assert planning and planning[0].max_items <= 4

    runner, _, model, _ = _runner(ReasoningMode.RECURSIVE, settings=settings)
    await _events(runner, QueryRequest(question="Question", reasoning_mode=ReasoningMode.RECURSIVE))
    recursive_calls = [call for call in model.calls if call.operation is ModelOperation.DECOMPOSE]
    leaf_calls = [call for call in model.calls if call.operation is ModelOperation.SUBANSWER]
    assert len(recursive_calls) == 4
    assert len(leaf_calls) == 9
    assert len(recursive_calls) + len(leaf_calls) == 13
    assert all(call.max_items <= 3 for call in recursive_calls)


@pytest.mark.asyncio
async def test_react_executes_strict_filtered_retrieval_for_every_allowlisted_action() -> None:
    runner, rag, _, _ = _runner(ReasoningMode.REACT)
    source_id = uuid4()
    request = QueryRequest(
        question="Question",
        reasoning_mode=ReasoningMode.REACT,
        source_types=[SourceType.WEB],
        source_ids=[source_id],
    )

    events = await _events(runner, request)

    assert events[-1].type == "final"
    assert len(rag.requests) == 7
    assert all(item.source_types == [SourceType.WEB] and item.source_ids == [source_id] for item in rag.requests)
    assert [item.question for item in rag.requests[1:]] == ["inspect retrieved source"] * 6


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [ReasoningMode.DECOMPOSED, ReasoningMode.LEAST_TO_MOST, ReasoningMode.RECURSIVE])
async def test_invalid_subanswer_citation_fails_before_accumulation(mode: ReasoningMode) -> None:
    class InvalidSubanswerModel(FakeModel):
        async def complete(self, request: ModelRequest) -> ModelResult:
            if request.operation is ModelOperation.SUBANSWER:
                unknown = uuid4()
                return ModelResult(text=f"Unsupported [chunk:{unknown}]", citation_chunk_ids=[unknown])
            return await super().complete(request)

    rag = FakeRag()
    model = InvalidSubanswerModel(rag.chunk.chunk_id)
    registry = WorkflowRegistry(
        rag=rag,
        model=model,
        citation_validator=StrictCitationValidator(),
        settings=Settings(app_profile="test"),
    )

    events = await _events(WorkflowRunner(registry), QueryRequest(question="Question", reasoning_mode=mode))

    assert events[-1].type == "error" and events[-1].code is ErrorCode.NO_SUPPORTING_EVIDENCE
    assert all(event.type != "final" for event in events)


@pytest.mark.asyncio
async def test_runtime_cancellation_interrupts_an_inflight_model_call() -> None:
    class BlockingModel(FakeModel):
        async def complete(self, request: ModelRequest) -> ModelResult:
            self.calls.append(request)
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    rag = FakeRag()
    model = BlockingModel(rag.chunk.chunk_id)
    registry = WorkflowRegistry(
        rag=rag,
        model=model,
        citation_validator=StrictCitationValidator(),
        settings=Settings(app_profile="test"),
    )
    runner = WorkflowRunner(registry)
    cancel = asyncio.Event()

    async def collect() -> list[StreamEvent]:
        return [event async for event in runner.stream(QueryRequest(question="Question"), cancel_event=cancel)]

    task = asyncio.create_task(collect())
    while not model.calls:
        await asyncio.sleep(0)
    cancel.set()
    events = await asyncio.wait_for(task, timeout=1)

    assert events[-1].type == "error" and events[-1].code is ErrorCode.RUN_CANCELLED


@pytest.mark.asyncio
async def test_model_controlled_private_reasoning_labels_are_redacted_from_stream_and_checkpoint() -> None:
    class UnsafeModel(FakeModel):
        async def complete(self, request: ModelRequest) -> ModelResult:
            result = await super().complete(request)
            if request.operation in {ModelOperation.ANSWER, ModelOperation.PLAN}:
                return result.model_copy(
                    update={
                        "text": (
                            "BEGIN PRIVATE REASONING\n"
                            + "reasoning-body-" * 1000
                            + "\nEND PRIVATE REASONING "
                            + f"[chunk:{rag.chunk.chunk_id}]"
                        ),
                        "items": ["<think>" + "scratch-body-" * 1000 + "</think>"],
                    }
                )
            return result

    saver = InMemorySaver()
    rag = FakeRag()
    model = UnsafeModel(rag.chunk.chunk_id)
    registry = WorkflowRegistry(
        rag=rag,
        model=model,
        citation_validator=StrictCitationValidator(),
        settings=Settings(app_profile="test"),
        checkpointer=saver,
    )
    thread_id = uuid4()

    events = await _events(
        WorkflowRunner(registry),
        QueryRequest(
            question="Explain chain-of-thought and system prompt reasoning: api_key=ultrasecret",
            reasoning_mode=ReasoningMode.COT,
            thread_id=thread_id,
        ),
    )

    public_text = str([event.model_dump(mode="json") for event in events]).lower()
    from rag_app.workflows.runner import _storage_thread_id

    checkpoint_text = str(
        next(saver.list({"configurable": {"thread_id": _storage_thread_id(thread_id, ReasoningMode.COT)}})).checkpoint
    ).lower()
    for forbidden in ("begin private reasoning", "<think>", "reasoning-body", "scratch-body"):
        assert forbidden not in public_text
    for forbidden_body in ("ultrasecret", "reasoning-body", "scratch-body"):
        assert forbidden_body not in checkpoint_text
    assert "chain-of-thought" in checkpoint_text
    assert "system prompt" in checkpoint_text
    assert "reasoning:" in checkpoint_text
    assert rag.requests[0].question.startswith("Explain chain-of-thought and system prompt reasoning:")


@pytest.mark.asyncio
async def test_legitimate_reasoning_topic_model_answer_is_preserved_and_validated() -> None:
    class TopicModel(FakeModel):
        async def complete(self, request: ModelRequest) -> ModelResult:
            result = await super().complete(request)
            if request.operation is ModelOperation.ANSWER:
                return result.model_copy(
                    update={
                        "text": (
                            "Chain-of-thought and system prompts are reasoning: topics in this answer "
                            f"[chunk:{self.chunk_id}]"
                        )
                    }
                )
            return result

    rag = FakeRag()
    registry = WorkflowRegistry(
        rag=rag,
        model=TopicModel(rag.chunk.chunk_id),
        citation_validator=StrictCitationValidator(),
        settings=Settings(app_profile="test"),
    )

    events = await _events(WorkflowRunner(registry), QueryRequest(question="Explain reasoning topics"))

    assert events[-1].type == "final"
    assert "Chain-of-thought and system prompts are reasoning: topics" in events[-1].answer


def test_inline_redaction_handles_auth_dsn_provider_keys_jwts_and_labels_without_censoring_topics() -> None:
    from rag_app.redaction import redact_inline_secrets

    secrets = [
        "Authorization: Bearer bearer-secret-1234567890",
        "postgresql://dbuser:dbpassword@db.example.com:5432/postgres",
        "AIzaSyA12345678901234567890123456789012",
        "sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue123",
        "password=hunter2",
        "api-key: provider-secret",
        "access_token = access-secret",
    ]
    text = " | ".join(secrets) + " | Explain chain-of-thought, system prompt, and reasoning: safely."

    redacted = redact_inline_secrets(text)

    for secret in secrets:
        assert secret not in redacted
    assert "chain-of-thought, system prompt, and reasoning: safely" in redacted


@pytest.mark.asyncio
async def test_inline_credentials_from_model_output_and_status_never_enter_checkpoint() -> None:
    class CredentialModel(FakeModel):
        async def complete(self, request: ModelRequest) -> ModelResult:
            result = await super().complete(request)
            if request.operation is ModelOperation.PLAN:
                return result.model_copy(update={"items": ["Authorization: Bearer model-bearer-secret"]})
            if request.operation is ModelOperation.ANSWER:
                return result.model_copy(
                    update={
                        "text": (
                            "Use postgresql://user:pass@host/db and sk-proj-abcdefghijklmnopqrstuvwxyz123456 "
                            f"[chunk:{rag.chunk.chunk_id}]"
                        )
                    }
                )
            return result

    saver = InMemorySaver()
    rag = FakeRag()
    registry = WorkflowRegistry(
        rag=rag,
        model=CredentialModel(rag.chunk.chunk_id),
        citation_validator=StrictCitationValidator(),
        settings=Settings(app_profile="test"),
        checkpointer=saver,
    )
    thread_id = uuid4()

    events = await _events(
        WorkflowRunner(registry),
        QueryRequest(question="Question", reasoning_mode=ReasoningMode.COT, thread_id=thread_id),
    )
    combined = str([event.model_dump(mode="json") for event in events]) + str(
        next(saver.list({"configurable": {"thread_id": f"{thread_id}:{ReasoningMode.COT.value}"}})).checkpoint
    )

    for secret in ("model-bearer-secret", "user:pass", "sk-proj-abcdefghijklmnopqrstuvwxyz123456"):
        assert secret not in combined


@pytest.mark.asyncio
async def test_tot_and_consistency_candidates_are_independent_and_consensus_wins_deterministically() -> None:
    class ConsensusModel(FakeModel):
        def __init__(self, chunk_id: UUID) -> None:
            super().__init__(chunk_id)
            self.candidate_index = 0

        async def complete(self, request: ModelRequest) -> ModelResult:
            self.calls.append(request)
            marker = f"[chunk:{self.chunk_id}]"
            if request.operation is ModelOperation.CANDIDATE:
                answers = [f"Answer A {marker}", f"Answer B {marker}", f"Answer A {marker}"]
                answer = answers[self.candidate_index]
                self.candidate_index += 1
                return ModelResult(text=answer, citation_chunk_ids=[self.chunk_id])
            if request.operation is ModelOperation.JUDGE:
                return ModelResult(text=f"Answer B {marker}", selected_index=1, items=["safe ranking"])
            return await super().complete(request)

    rag = FakeRag()
    model = ConsensusModel(rag.chunk.chunk_id)
    registry = WorkflowRegistry(
        rag=rag,
        model=model,
        citation_validator=StrictCitationValidator(),
        settings=Settings(app_profile="test"),
    )
    events = await _events(
        WorkflowRunner(registry),
        QueryRequest(question="Question", reasoning_mode=ReasoningMode.SELF_CONSISTENCY),
    )
    final = events[-1]

    assert final.type == "final" and final.answer.startswith("Answer A")
    candidate_calls = [call for call in model.calls if call.operation is ModelOperation.CANDIDATE]
    assert all(call.context == [] for call in candidate_calls)

    tot_runner, _, tot_model, _ = _runner(ReasoningMode.TOT)
    await _events(tot_runner, QueryRequest(question="Question", reasoning_mode=ReasoningMode.TOT))
    approach_calls = [call for call in tot_model.calls if call.operation is ModelOperation.APPROACH]
    assert all(call.context == [] for call in approach_calls)


@pytest.mark.asyncio
async def test_gemini_adapter_uses_configured_model_and_structured_grounded_output() -> None:
    rag = FakeRag()
    bundle = await rag.prepare(QueryRequest(question="Question"))

    class StructuredModel:
        def __init__(self) -> None:
            self.schema: type[object] | None = None
            self.payload = ""

        def with_structured_output(self, schema: type[object]) -> StructuredModel:
            self.schema = schema
            return self

        async def ainvoke(self, payload: object) -> object:
            self.payload = str(payload)
            return {"text": "Grounded", "citation_chunk_ids": [str(rag.chunk.chunk_id)]}

    provider = StructuredModel()
    adapter = GeminiWorkflowModel.from_settings(
        Settings(app_profile="test", model_name="gemini-test-model"), model=provider
    )

    result = await adapter.complete(ModelRequest(operation=ModelOperation.ANSWER, question="Question", evidence=bundle))

    assert adapter.model_name == "gemini-test-model"
    assert result.citation_chunk_ids == [rag.chunk.chunk_id]
    assert f"[chunk:{rag.chunk.chunk_id}]" in provider.payload
    assert "hidden reasoning" in provider.payload
