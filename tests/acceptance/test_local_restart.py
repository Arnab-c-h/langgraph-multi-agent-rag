from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory
from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    ListTasksRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)

from rag_app import composition
from rag_app.api.app import create_app
from rag_app.config import Settings
from rag_app.contracts import Citation, FinalEvent, QueryRequest, ReasoningMode, SourceType
from rag_app.ingestion.models import DocumentSegment, NormalizedDocument
from rag_app.workflows.model import ModelOperation, ModelRequest, ModelResult
from rag_app.workflows.runner import WorkflowRunner, _request_fingerprint
from tests.evaluation.offline_harness import DeterministicEmbeddings, DeterministicEvidenceGrader

EXPECTED_REASONING_MODE_IDENTIFIERS = (
    "standard",
    "cot",
    "tot",
    "react",
    "self_reflection",
    "self_consistency",
    "decomposed",
    "least_to_most",
    "recursive",
)


class ProviderFreeEmbeddings(DeterministicEmbeddings):
    def __init__(self, _api_key: str) -> None:
        super().__init__()


class ProviderFreeWorkflowModel:
    def __init__(self) -> None:
        self.cancellation_started = asyncio.Event()
        self.cancellation_received = asyncio.Event()
        self.cancellation_finished = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResult:
        if "cancel" in request.question.casefold():
            self.cancellation_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancellation_received.set()
                raise
            finally:
                self.cancellation_finished.set()
        chunk_ids = [assessment.chunk.chunk_id for assessment in request.evidence.evidence]
        answer = " ".join(
            f"{assessment.chunk.content} [chunk:{assessment.chunk.chunk_id}]"
            for assessment in request.evidence.evidence
        )
        if request.operation in {ModelOperation.PLAN, ModelOperation.DECOMPOSE}:
            return ModelResult(items=[request.question] * request.max_items)
        if request.operation is ModelOperation.APPROACH:
            return ModelResult(text="Use the accepted local evidence.")
        if request.operation is ModelOperation.RANK:
            return ModelResult(selected_index=0, items=["Selected by evidence coverage."])
        if request.operation is ModelOperation.ACT:
            return ModelResult(text=request.question, complete=True)
        if request.operation in {ModelOperation.CRITIQUE, ModelOperation.VALIDATE}:
            return ModelResult(approved=True)
        return ModelResult(text=answer, citation_chunk_ids=chunk_ids)


def _message(text: str, *, return_immediately: bool = False) -> SendMessageRequest:
    request = SendMessageRequest(
        message=Message(message_id=str(uuid4()), role=Role.ROLE_USER, parts=[Part(text=text)])
    )
    request.configuration.return_immediately = return_immediately
    return request


async def _client(app: Any, *, streaming: bool) -> AsyncIterator[Any]:
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    factory = ClientFactory(
        ClientConfig(httpx_client=http, supported_protocol_bindings=["JSONRPC"], streaming=streaming)
    )
    client = await factory.create_from_url("http://test")
    try:
        yield client
    finally:
        await client.close()
        await http.aclose()


async def _collect(iterator: AsyncIterator[Any]) -> list[Any]:
    async with asyncio.timeout(20):
        return [item async for item in iterator]


def _final(events: list[object]) -> FinalEvent:
    final = next((event for event in events if isinstance(event, FinalEvent)), None)
    assert final is not None, events
    return final


def _assert_exact_citation(citation: Citation, *, source_id: UUID, chunk_id: UUID) -> None:
    assert citation.model_dump(mode="json") == {
        "source_id": str(source_id),
        "chunk_id": str(chunk_id),
        "source_title": "Restart-safe local storage",
        "locator": "https://example.com/local-storage — Persistence",
        "source_url": "https://example.com/local-storage",
        "source_path": None,
        "supporting_excerpt": "SQLite stores authoritative restart evidence while Chroma indexes embeddings.",
    }


class BlockingUpdateIterator:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.finished = asyncio.Event()
        self.closed = asyncio.Event()
        self.anext_active = False
        self.close_while_active = False
        self.next_task: asyncio.Task[object] | None = None

    def __aiter__(self) -> BlockingUpdateIterator:
        return self

    async def __anext__(self) -> dict[str, object]:
        self.next_task = asyncio.current_task()
        self.anext_active = True
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        finally:
            self.anext_active = False
            self.finished.set()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close_while_active = self.anext_active
        self.closed.set()


class BlockingGraph:
    def __init__(self, iterator: BlockingUpdateIterator) -> None:
        self._iterator = iterator

    def astream(self, *_args: object, **_kwargs: object) -> BlockingUpdateIterator:
        return self._iterator


class BlockingRegistry:
    has_checkpointer = False

    def __init__(self, graph: BlockingGraph) -> None:
        self._graph = graph

    def graph_for(self, _mode: ReasoningMode) -> BlockingGraph:
        return self._graph


@pytest.mark.asyncio
async def test_workflow_runner_cooperative_cancellation_drains_then_closes_iterator() -> None:
    iterator = BlockingUpdateIterator()
    runner = WorkflowRunner(BlockingRegistry(BlockingGraph(iterator)))  # type: ignore[arg-type]
    cancellation = asyncio.Event()

    consume = asyncio.create_task(
        _collect(runner.stream(QueryRequest(question="cancel cooperatively"), cancel_event=cancellation))
    )
    await asyncio.wait_for(iterator.started.wait(), timeout=1)
    cancellation.set()
    events = await asyncio.wait_for(consume, timeout=1)

    assert len(events) == 1
    assert events[0].type == "error" and events[0].code.value == "RUN_CANCELLED"
    assert iterator.cancelled.is_set()
    assert iterator.finished.is_set()
    assert iterator.closed.is_set()
    assert iterator.close_while_active is False
    assert iterator.next_task is not None and iterator.next_task.done()


@pytest.mark.asyncio
async def test_workflow_runner_parent_cancellation_drains_then_closes_iterator() -> None:
    iterator = BlockingUpdateIterator()
    runner = WorkflowRunner(BlockingRegistry(BlockingGraph(iterator)))  # type: ignore[arg-type]
    consume = asyncio.create_task(_collect(runner.stream(QueryRequest(question="cancel parent"))))
    await asyncio.wait_for(iterator.started.wait(), timeout=1)

    consume.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consume, timeout=1)
    finally:
        if iterator.next_task is not None and not iterator.next_task.done():
            iterator.next_task.cancel()
            await asyncio.gather(iterator.next_task, return_exceptions=True)

    assert iterator.cancelled.is_set()
    assert iterator.finished.is_set()
    assert iterator.closed.is_set()
    assert iterator.close_while_active is False
    assert iterator.next_task is not None and iterator.next_task.done()


def _assert_mode_specific_checkpoint(mode: ReasoningMode, values: dict[str, Any]) -> None:
    if mode is ReasoningMode.STANDARD:
        assert values["status"] == "Validated answer citations and evidence support."
    elif mode is ReasoningMode.COT:
        assert len(values["public_plan"]) == 4
    elif mode is ReasoningMode.TOT:
        assert len(values["approaches"]) == 3
        assert values["branch_rankings"] == ["Selected by evidence coverage."]
    elif mode is ReasoningMode.REACT:
        assert values["action_count"] == 1
        assert len(values["tool_actions"]) == 1
    elif mode is ReasoningMode.SELF_REFLECTION:
        assert values["critique_approved"] is True
    elif mode is ReasoningMode.SELF_CONSISTENCY:
        assert len(values["candidates"]) == 3
        assert len(values["candidate_citations"]) == 3
    elif mode in {ReasoningMode.DECOMPOSED, ReasoningMode.LEAST_TO_MOST}:
        assert values["subquestion_index"] == 4
        assert len(values["subanswers"]) == 4
    else:
        assert mode is ReasoningMode.RECURSIVE
        assert values["recursive_node_count"] == 13
        assert len(values["recursive_nodes"]) == 13


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_free_runtime_and_official_a2a_survive_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models: list[ProviderFreeWorkflowModel] = []

    def create_model(_settings: Settings) -> ProviderFreeWorkflowModel:
        model = ProviderFreeWorkflowModel()
        models.append(model)
        return model

    monkeypatch.setattr(composition, "GeminiEmbeddingAdapter", ProviderFreeEmbeddings)
    monkeypatch.setattr(
        composition.GeminiEvidenceGrader,
        "from_settings",
        lambda _settings: DeterministicEvidenceGrader(),
    )
    monkeypatch.setattr(composition.GeminiWorkflowModel, "from_settings", create_model)
    settings = Settings(
        app_profile="full",
        rag_data_dir=tmp_path,
        gemini_api_key="provider-free-test-key",
        admin_token="provider-free-test-admin",
        public_base_url="http://test",
        query_timeout_seconds=20,
        dependency_timeout_seconds=5,
    )
    thread_id = uuid4()
    completed_task_id = ""
    first_runtime_ids: tuple[int, int, int, int] | None = None

    async with composition.compose_runtime(settings) as services:
        assert isinstance(services.runner, WorkflowRunner)
        repository = services.sources._repository  # type: ignore[attr-defined]
        ingestion = services.sources._ingestion  # type: ignore[attr-defined]
        result = await ingestion.ingest(
            NormalizedDocument(
                source_type=SourceType.WEB,
                canonical_uri="https://example.com/local-storage",
                title="Restart-safe local storage",
                content="SQLite stores authoritative restart evidence while Chroma indexes embeddings.",
                provenance={"fixture": "provider-free-restart"},
                segments=[
                    DocumentSegment(
                        content="SQLite stores authoritative restart evidence while Chroma indexes embeddings.",
                        metadata={"heading": "Persistence"},
                    )
                ],
            )
        )
        source = result.source
        chunk = result.chunks[0]
        first_runtime_ids = (
            id(repository._sqlite._database),  # type: ignore[attr-defined]
            id(repository._chroma),  # type: ignore[attr-defined]
            id(services.runner._registry.graph_for(ReasoningMode.STANDARD).checkpointer),  # type: ignore[attr-defined]
            id(services.a2a_task_store),
        )

        assert len(EXPECTED_REASONING_MODE_IDENTIFIERS) == 9
        assert set(EXPECTED_REASONING_MODE_IDENTIFIERS) == {
            "standard",
            "cot",
            "tot",
            "react",
            "self_reflection",
            "self_consistency",
            "decomposed",
            "least_to_most",
            "recursive",
        }
        assert tuple(mode.value for mode in ReasoningMode) == EXPECTED_REASONING_MODE_IDENTIFIERS
        reasoning_modes = tuple(ReasoningMode(identifier) for identifier in EXPECTED_REASONING_MODE_IDENTIFIERS)

        for mode in reasoning_modes:
            events = [
                event
                async for event in services.runner.stream(
                    QueryRequest(
                        question="Where is authoritative restart evidence stored?",
                        reasoning_mode=mode,
                        thread_id=thread_id,
                        source_types=[SourceType.WEB],
                        source_ids=[source.id],
                    )
                )
            ]
            final = _final(events)
            assert final.mode is mode
            assert final.thread_id == thread_id
            assert len(final.citations) == 1
            _assert_exact_citation(final.citations[0], source_id=source.id, chunk_id=chunk.id)

        turns = await services.runner._thread_store.load(thread_id)  # type: ignore[attr-defined]
        assert [turn.mode for turn in turns] == list(reasoning_modes)

        app = create_app(settings=settings, services=services)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as http:
                health = (await http.get("/healthz")).json()
            assert health == {
                "status": "ok",
                "profile": "full",
                "ready": True,
                "dependencies": {
                    "sqlite": True,
                    "chroma": True,
                    "checkpoints": True,
                    "sources": True,
                    "runner": True,
                    "a2a_store": True,
                },
            }

            async for client in _client(app, streaming=False):
                sent = await _collect(client.send_message(_message("Where is authoritative restart evidence stored?")))
                completed_task_id = sent[-1].task.id
                assert sent[-1].task.status.state == TaskState.TASK_STATE_COMPLETED
                fetched = await client.get_task(GetTaskRequest(id=completed_task_id))
                listing = await client.list_tasks(ListTasksRequest(include_artifacts=True))
                assert fetched.status.state == TaskState.TASK_STATE_COMPLETED
                assert completed_task_id in {task.id for task in listing.tasks}

            async for client in _client(app, streaming=True):
                streamed = await _collect(
                    client.send_message(_message("Where is authoritative restart evidence stored?"))
                )
                assert len(streamed) > 1
                terminal = next(
                    response.status_update
                    for response in reversed(streamed)
                    if response.HasField("status_update")
                    and response.status_update.status.state == TaskState.TASK_STATE_COMPLETED
                )
                assert terminal.status.state == TaskState.TASK_STATE_COMPLETED
                assert any(response.HasField("artifact_update") for response in streamed)

            async for client in _client(app, streaming=False):
                pending = await _collect(
                    client.send_message(
                        _message("Where is authoritative restart evidence stored? cancel", return_immediately=True)
                    )
                )
                cancellation_task_id = pending[0].task.id
                await asyncio.wait_for(models[0].cancellation_started.wait(), timeout=5)
                cancelled = await asyncio.wait_for(
                    client.cancel_task(CancelTaskRequest(id=cancellation_task_id)), timeout=5
                )
                await asyncio.wait_for(models[0].cancellation_received.wait(), timeout=5)
                await asyncio.wait_for(models[0].cancellation_finished.wait(), timeout=5)
                assert cancelled.status.state == TaskState.TASK_STATE_CANCELED

    assert completed_task_id
    assert first_runtime_ids is not None

    async with composition.compose_runtime(settings) as restarted:
        restarted_repository = restarted.sources._repository  # type: ignore[attr-defined]
        restarted_ids = (
            id(restarted_repository._sqlite._database),  # type: ignore[attr-defined]
            id(restarted_repository._chroma),  # type: ignore[attr-defined]
            id(restarted.runner._registry.graph_for(ReasoningMode.STANDARD).checkpointer),  # type: ignore[attr-defined]
            id(restarted.a2a_task_store),
        )
        assert all(
            restarted_id != first_id
            for restarted_id, first_id in zip(restarted_ids, first_runtime_ids, strict=True)
        )
        assert len(models) == 2

        sources = await restarted.sources.list_sources()
        assert [item.id for item in sources] == [source.id]
        chunks = await restarted_repository.list_chunks(source.id)
        assert [item.id for item in chunks] == [chunk.id]
        assert chunks[0].content == chunk.content
        turns = await restarted.runner._thread_store.load(thread_id)  # type: ignore[attr-defined]
        assert [turn.mode for turn in turns] == list(reasoning_modes)

        graph_checkpointer = restarted.runner._registry.graph_for(  # type: ignore[attr-defined]
            ReasoningMode.STANDARD
        ).checkpointer
        recovered_fingerprints: set[str] = set()
        for mode in reasoning_modes:
            recovered = await graph_checkpointer.aget_tuple(
                {"configurable": {"thread_id": f"{thread_id}:{mode.value}", "checkpoint_ns": ""}}
            )
            assert recovered is not None
            assert recovered.config["configurable"]["thread_id"] == f"{thread_id}:{mode.value}"
            values = recovered.checkpoint["channel_values"]
            assert values["request"] == {
                "question": "Where is authoritative restart evidence stored?",
                "reasoning_mode": mode.value,
                "thread_id": str(thread_id),
                "top_k": 6,
                "source_types": ["web"],
                "source_ids": [str(source.id)],
            }
            fingerprint = values["request_fingerprint"]
            assert isinstance(fingerprint, str) and len(fingerprint) == 64
            expected_request = QueryRequest(
                question="Where is authoritative restart evidence stored?",
                reasoning_mode=mode,
                thread_id=thread_id,
                top_k=6,
                source_types=[SourceType.WEB],
                source_ids=[source.id],
            )
            assert fingerprint == _request_fingerprint(expected_request)
            recovered_fingerprints.add(fingerprint)
            assert values["validation_passed"] is True
            assert values["trace_summary"] == {
                "accepted_evidence": 1,
                "candidate_evidence": 1,
                "citation_count": 1,
                "validation_passed": True,
            }
            _assert_mode_specific_checkpoint(mode, values)
        assert len(recovered_fingerprints) == 9

        restarted_events = [
            event
            async for event in restarted.runner.stream(
                QueryRequest(
                    question="Where is authoritative restart evidence stored?",
                    reasoning_mode=ReasoningMode.STANDARD,
                    thread_id=thread_id,
                    source_types=[SourceType.WEB],
                    source_ids=[source.id],
                )
            )
        ]
        restarted_final = _final(restarted_events)
        _assert_exact_citation(restarted_final.citations[0], source_id=source.id, chunk_id=chunk.id)

        restarted_app = create_app(settings=settings, services=restarted)
        async with restarted_app.router.lifespan_context(restarted_app):
            async for client in _client(restarted_app, streaming=False):
                recovered_task = await client.get_task(GetTaskRequest(id=completed_task_id))
                listing = await client.list_tasks(ListTasksRequest(include_artifacts=True))
                assert recovered_task.status.state == TaskState.TASK_STATE_COMPLETED
                assert completed_task_id in {task.id for task in listing.tasks}
