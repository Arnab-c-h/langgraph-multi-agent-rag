from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from rag_app.api.app import AppServices, create_app
from rag_app.api.limits import QueryCapacity
from rag_app.api.security import AdminAuthorizer
from rag_app.config import Settings
from rag_app.contracts import (
    Citation,
    CitationEvent,
    FinalEvent,
    QueryRequest,
    ReasoningMode,
    StatusEvent,
    StreamEvent,
    TokenEvent,
)
from rag_app.errors import AppError, ErrorCode
from rag_app.ingestion.models import IngestionJob, IngestionResult, SourceRecord
from rag_app.ui.app import UiController, _summarize_a2a_responses, build_ui


class FakeRunner:
    def __init__(self) -> None:
        self.requests: list[QueryRequest] = []
        self.cancelled = False

    async def stream(
        self, request: QueryRequest, *, cancel_event: asyncio.Event | None = None
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        citation = Citation(
            source_id=uuid4(),
            chunk_id=uuid4(),
            source_title='<script>alert("x")</script>',
            locator="page 3",
            source_url="https://example.com/doc?q=<unsafe>",
            supporting_excerpt="Evidence <img src=x onerror=alert(1)>",
        )
        yield StatusEvent(message="Retrieving <b>evidence</b>")
        yield TokenEvent(token="Grounded ")
        yield CitationEvent(citation=citation)
        if cancel_event is not None and cancel_event.is_set():
            self.cancelled = True
        yield FinalEvent(
            answer="Grounded answer",
            citations=[citation],
            thread_id=request.thread_id,
            mode=request.reasoning_mode,
            run_id=uuid4(),
            used_source_ids=[citation.source_id],
            trace_summary={"validator": "passed"},
        )


class FakeSources:
    def __init__(self) -> None:
        self.source = SourceRecord(
            source_type="web",
            canonical_uri="https://example.com/<script>",
            title="Example <script>",
            content_hash="a" * 64,
            metadata={"status": "ready"},
        )
        self.mutations: list[str] = []

    async def list_sources(self) -> list[SourceRecord]:
        return [self.source]

    def _result(self) -> IngestionResult:
        return IngestionResult(
            source=self.source,
            job=IngestionJob(source_id=self.source.id, status="completed"),
            status="ingested",
            chunk_count=2,
        )

    async def ingest_pdf(self, filename: str, data: bytes) -> IngestionResult:
        self.mutations.append(f"pdf:{filename}:{len(data)}")
        return self._result()

    async def ingest_web(self, url: str) -> IngestionResult:
        self.mutations.append(f"web:{url}")
        return self._result()

    async def ingest_github(self, url: str) -> IngestionResult:
        self.mutations.append(f"github:{url}")
        return self._result()

    async def reindex(self, source_id: UUID) -> IngestionResult:
        self.mutations.append(f"reindex:{source_id}")
        return self._result()

    async def delete(self, source_id: UUID) -> None:
        self.mutations.append(f"delete:{source_id}")


class FakeA2ATester:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def send(self, text: str, mode: ReasoningMode) -> tuple[str, str]:
        return "task-123", f"completed:{text}:{mode.value}"

    async def stream(self, text: str, mode: ReasoningMode) -> AsyncIterator[tuple[str, str]]:
        yield "task-123", f"working:{text}:{mode.value}"
        yield "task-123", "completed"

    async def cancel(self, task_id: str) -> str:
        self.cancelled.append(task_id)
        return "cancelled"


def _controller(profile: str = "test") -> tuple[UiController, FakeRunner, FakeSources]:
    runner = FakeRunner()
    sources = FakeSources()
    settings = Settings(app_profile=profile)  # type: ignore[arg-type]
    controller = UiController(
        settings=settings,
        services=AppServices(runner=runner, sources=sources),
        authorizer=AdminAuthorizer("secret"),
        a2a_tester=FakeA2ATester(),
        readiness_provider=lambda: {
            "sqlite": True,
            "chroma": True,
            "checkpoints": True,
            "runner": True,
            "sources": True,
            "a2a_store": True,
        },
    )
    return controller, runner, sources


def test_ui_constructs_exactly_three_tabs_and_all_reasoning_modes() -> None:
    controller, _, _ = _controller()
    ui = build_ui(controller)
    config = ui.get_config_file()
    tabs = [
        component["props"].get("label")
        for component in config["components"]
        if component["type"] == "tabitem"
    ]
    mode_choices = [
        component["props"].get("choices")
        for component in config["components"]
        if component["props"].get("label") == "Reasoning mode"
    ]

    assert tabs == ["Chat", "Knowledge Base", "System / A2A"]
    assert mode_choices
    assert {choice[1] for choice in mode_choices[0]} == {mode.value for mode in ReasoningMode}


@pytest.mark.asyncio
async def test_chat_stream_maps_filters_threads_tokens_and_safe_citations() -> None:
    controller, runner, _ = _controller()
    source_id = uuid4()
    thread_id = uuid4()

    updates = [
        update
        async for update in controller.stream_chat(
            "Question",
            ReasoningMode.TOT.value,
            ["web", "pdf"],
            str(source_id),
            str(thread_id),
        )
    ]

    assert runner.requests == [
        QueryRequest(
            question="Question",
            reasoning_mode=ReasoningMode.TOT,
            source_types=["web", "pdf"],
            source_ids=[source_id],
            thread_id=thread_id,
        )
    ]
    assert any("Grounded " in update.answer for update in updates)
    assert updates[-1].answer == "Grounded answer"
    assert "<script>" not in updates[-1].citations_html
    assert "&lt;script&gt;" in updates[-1].citations_html
    assert "onerror=alert" not in updates[-1].citations_html
    assert updates[-1].thread_id == str(thread_id)
    assert updates[-1].error == ""


@pytest.mark.asyncio
async def test_chat_cancellation_sets_the_active_runner_event() -> None:
    class BlockingRunner(FakeRunner):
        async def stream(
            self, request: QueryRequest, *, cancel_event: asyncio.Event | None = None
        ) -> AsyncIterator[StreamEvent]:
            self.requests.append(request)
            yield StatusEvent(message="Working")
            assert cancel_event is not None
            await cancel_event.wait()
            self.cancelled = True

    controller, _, sources = _controller()
    runner = BlockingRunner()
    controller = UiController(
        settings=Settings(app_profile="test"),
        services=AppServices(runner=runner, sources=sources),
        authorizer=AdminAuthorizer("secret"),
        a2a_tester=FakeA2ATester(),
    )
    stream = controller.stream_chat("Question", "standard", [], "", "")
    first = await anext(stream)

    message = controller.cancel_chat(first.run_key)
    with pytest.raises(StopAsyncIteration):
        await anext(stream)

    assert message == "RUN_CANCELLED"
    assert runner.cancelled is True


@pytest.mark.asyncio
async def test_full_admin_callbacks_authorize_and_use_shared_source_service(tmp_path: Path) -> None:
    controller, _, sources = _controller()
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.7\nfixture")
    assert "UNAUTHORIZED" in await controller.ingest_web("https://example.com", "wrong")
    assert not sources.mutations

    pdf_result = await controller.ingest_pdf(pdf, "secret")
    result = await controller.ingest_web("https://example.com", "secret")
    github_result = await controller.ingest_github("https://github.com/example/repo", "secret")
    await controller.reindex(str(sources.source.id), "secret")
    await controller.delete(str(sources.source.id), "secret")

    assert all("ingested" in item for item in (pdf_result, result, github_result))
    assert sources.mutations == [
        "pdf:source.pdf:16",
        "web:https://example.com",
        "github:https://github.com/example/repo",
        f"reindex:{sources.source.id}",
        f"delete:{sources.source.id}",
    ]


@pytest.mark.asyncio
async def test_ui_uses_the_local_only_admin_authorizer_contract() -> None:
    class LocalOnlyAuthorizer:
        def authorize(self, authorization: str | None) -> tuple[bool, int]:
            return (authorization == "Bearer secret", 200 if authorization == "Bearer secret" else 401)

    runner = FakeRunner()
    sources = FakeSources()
    controller = UiController(
        settings=Settings(app_profile="full"),
        services=AppServices(runner=runner, sources=sources),
        authorizer=LocalOnlyAuthorizer(),  # type: ignore[arg-type]
        a2a_tester=FakeA2ATester(),
    )

    result = await controller.ingest_web("https://example.com", "secret")

    assert "ingested" in result
    assert sources.mutations == ["web:https://example.com"]


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_sources_and_a2a_tester_render_safe_results() -> None:
    controller, _, _ = _controller()
    rows = await controller.list_sources()
    task_id, result = await controller.a2a_send("hello", "react")
    streamed = [item async for item in controller.a2a_stream("hello", "react")]
    cancelled = await controller.a2a_cancel(task_id)

    assert rows[0][2] == "Example &lt;script&gt;"
    assert "<script>" not in rows[0][3]
    assert rows[0][4] == "ready"
    assert result == "completed:hello:react"
    assert streamed[-1] == ("task-123", "completed")
    assert cancelled == "cancelled"


@pytest.mark.asyncio
async def test_source_status_is_blank_when_persistence_does_not_return_one() -> None:
    controller, _, sources = _controller()
    sources.source.metadata = {}

    rows = await controller.list_sources()

    assert rows[0][4] == ""


def test_system_summary_and_agent_card_are_safe_and_advertise_local_ingestion() -> None:
    controller, _, _ = _controller("full")
    summary = json.loads(controller.system_summary())
    card = json.loads(controller.agent_card_summary())

    assert summary["readiness"] == {
        "sqlite": True,
        "chroma": True,
        "checkpoints": True,
        "runner": True,
        "sources": True,
        "a2a_store": True,
    }
    assert summary["health_path"] == "/healthz"
    assert "secret" not in controller.system_summary().lower()
    assert card["supportedInterfaces"][0]["url"] == "http://localhost:8000/a2a"
    assert {skill["id"] for skill in card["skills"]} == {"query", "list_sources", "ingestion"}


@pytest.mark.asyncio
async def test_ingestion_progress_is_visible_and_finishes_with_persisted_result() -> None:
    controller, _, sources = _controller()

    updates = [item async for item in controller.ingest_web_progress("https://example.com", "secret")]

    assert updates[0] == "Starting web ingestion"
    assert len(updates) == 2
    assert "ingested" in updates[-1]
    assert sources.mutations == ["web:https://example.com"]


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_ui_admin_mutations_use_shared_provider_retry_policy() -> None:
    class RateLimitedSources(FakeSources):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        async def ingest_web(self, url: str) -> IngestionResult:
            self.attempts += 1
            if self.attempts == 1:
                raise AppError(ErrorCode.PROVIDER_RATE_LIMITED, "private provider response")
            return await super().ingest_web(url)

    sources = RateLimitedSources()
    controller = UiController(
        settings=Settings(app_profile="test", provider_max_retries=1),
        services=AppServices(runner=FakeRunner(), sources=sources),
        authorizer=AdminAuthorizer("secret"),
        a2a_tester=FakeA2ATester(),
        capacity=QueryCapacity(1),
    )

    result = await controller.ingest_web("https://example.com", "secret")

    assert "ingested" in result
    assert sources.attempts == 2


@pytest.mark.asyncio
async def test_ui_query_uses_shared_capacity() -> None:
    class ControlledRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()
            self.started = 0
            self.release = asyncio.Event()

        async def stream(
            self, request: QueryRequest, *, cancel_event: asyncio.Event | None = None
        ) -> AsyncIterator[StreamEvent]:
            del cancel_event
            self.requests.append(request)
            self.started += 1
            yield StatusEvent(message="started")
            await self.release.wait()

    runner = ControlledRunner()
    sources = FakeSources()
    controller = UiController(
        settings=Settings(app_profile="test"),
        services=AppServices(runner=runner, sources=sources),
        authorizer=AdminAuthorizer("secret"),
        a2a_tester=FakeA2ATester(),
        capacity=QueryCapacity(1),
    )
    first = controller.stream_chat("one", "standard", [], "", "")
    second = controller.stream_chat("two", "standard", [], "", "")
    await anext(first)
    blocked = asyncio.create_task(anext(second))
    await asyncio.sleep(0)

    assert runner.started == 1
    assert not blocked.done()

    runner.release.set()
    with pytest.raises(StopAsyncIteration):
        await anext(first)
    await blocked
    await second.aclose()


@pytest.mark.asyncio
async def test_ui_query_timeout_cancels_runner_and_returns_stable_error() -> None:
    class SlowRunner(FakeRunner):
        cancel_event: asyncio.Event | None = None

        async def stream(
            self, request: QueryRequest, *, cancel_event: asyncio.Event | None = None
        ) -> AsyncIterator[StreamEvent]:
            del request
            self.cancel_event = cancel_event
            await asyncio.sleep(2)
            yield StatusEvent(message="late")

    runner = SlowRunner()
    _, _, sources = _controller()
    controller = UiController(
        settings=Settings(app_profile="test", query_timeout_seconds=1),
        services=AppServices(runner=runner, sources=sources),
        authorizer=AdminAuthorizer("secret"),
        a2a_tester=FakeA2ATester(),
        capacity=QueryCapacity(1),
    )

    update = await anext(controller.stream_chat("slow", "standard", [], "", ""))

    assert update.error == ErrorCode.RUN_FAILED.value
    assert runner.cancel_event is not None and runner.cancel_event.is_set()


def test_official_a2a_response_summary_includes_state_and_artifact_content() -> None:
    from a2a.types import Artifact, Part, StreamResponse, Task, TaskState, TaskStatus

    response = StreamResponse(
        task=Task(
            id="task-123",
            context_id="context-123",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            artifacts=[Artifact(artifact_id="answer", name="result", parts=[Part(text='{"answer":"Grounded"}')])],
        )
    )

    task_id, summary = _summarize_a2a_responses([response])

    assert task_id == "task-123"
    assert "TASK_STATE_COMPLETED" in summary
    assert '{"answer":"Grounded"}' in summary


@pytest.mark.asyncio
async def test_fastapi_mounts_gradio_without_import_time_network() -> None:
    controller, runner, sources = _controller()
    del controller
    app: FastAPI = create_app(
        settings=Settings(app_profile="test", public_base_url="http://test"),
        services=AppServices(runner=runner, sources=sources),
    )
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/ui" in paths

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ui/")
    assert response.status_code == 200
    assert "LangGraph Multi-Agent RAG" in response.text
    assert app.state.ui_controller._capacity is app.state.query_capacity
    app.state.readiness = {"runner": False, "sources": True, "a2a_store": True}
    assert json.loads(app.state.ui_controller.system_summary())["readiness"]["runner"] is False


def test_local_ui_config_constructs_admin_locked_mutation_controls() -> None:
    controller, _, _ = _controller("full")
    config = build_ui(controller).get_config_file()
    labels = {
        component["props"].get("label") or component["props"].get("value")
        for component in config["components"]
    }

    assert "Upload PDF" in labels
    assert "Ingest web URL" in labels
    assert "Ingest GitHub repository" in labels
    assert "Reindex source ID" in labels
    assert "Delete source ID" in labels
    assert "Agent Card" in labels
    assert "Refresh system status" in labels
