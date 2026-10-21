from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from a2a.server.context import ServerCallContext
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import Task, TaskState, TaskStatus

from rag_app.api.app import AppServices, BindableTaskStore, create_app
from rag_app.api.security import AdminAuthorizer
from rag_app.config import Settings
from rag_app.contracts import (
    Citation,
    CitationEvent,
    ErrorEvent,
    FinalEvent,
    QueryRequest,
    SourceType,
    StatusEvent,
    StreamEvent,
    TokenEvent,
)
from rag_app.errors import ErrorCode
from rag_app.ingestion.models import IngestionJob, IngestionResult, SourceRecord


class FakeRunner:
    def __init__(self, events: list[StreamEvent] | None = None) -> None:
        self.requests: list[QueryRequest] = []
        self.cancel_events: list[asyncio.Event | None] = []
        self.events = events or _success_events()

    async def stream(
        self, request: QueryRequest, *, cancel_event: asyncio.Event | None = None
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        self.cancel_events.append(cancel_event)
        for event in self.events:
            yield event


class FakeSources:
    def __init__(self) -> None:
        self.source = SourceRecord(
            id=uuid4(),
            source_type=SourceType.WEB,
            canonical_uri="https://example.com/",
            title="Example",
            content_hash="a" * 64,
            provenance={"private": "not public"},
        )
        self.deleted: list[UUID] = []
        self.reindexed: list[UUID] = []

    async def list_sources(self) -> list[SourceRecord]:
        return [self.source]

    async def ingest_pdf(self, filename: str, data: bytes) -> IngestionResult:
        return self._result()

    async def ingest_web(self, url: str) -> IngestionResult:
        return self._result()

    async def ingest_github(self, url: str) -> IngestionResult:
        return self._result()

    async def reindex(self, source_id: UUID) -> IngestionResult:
        self.reindexed.append(source_id)
        return self._result()

    async def delete(self, source_id: UUID) -> None:
        self.deleted.append(source_id)

    def _result(self) -> IngestionResult:
        return IngestionResult(
            source=self.source,
            job=IngestionJob(source_id=self.source.id, status="completed"),
            status="ingested",
        )


def _success_events() -> list[StreamEvent]:
    source_id = uuid4()
    citation = Citation(
        source_id=source_id,
        chunk_id=uuid4(),
        source_title="Example",
        locator="page 1",
        supporting_excerpt="Evidence",
    )
    run_id = uuid4()
    thread_id = uuid4()
    return [
        StatusEvent(message="Retrieving"),
        TokenEvent(token="Answer"),
        CitationEvent(citation=citation),
        FinalEvent(
            answer="Answer",
            citations=[citation],
            thread_id=thread_id,
            run_id=run_id,
            used_source_ids=[source_id],
        ),
    ]


def _app(
    *,
    runner: FakeRunner | None = None,
    sources: FakeSources | None = None,
    settings: Settings | None = None,
    authorizer: AdminAuthorizer | None = None,
) -> tuple[Any, FakeRunner, FakeSources]:
    actual_runner = runner or FakeRunner()
    actual_sources = sources or FakeSources()
    app = create_app(
        settings=settings or Settings(app_profile="test", admin_token="test-secret", query_rate_limit_per_minute=20),
        services=AppServices(runner=actual_runner, sources=actual_sources),
        admin_authorizer=authorizer,
    )
    return app, actual_runner, actual_sources


@pytest.mark.asyncio
async def test_health_query_and_public_sources_have_stable_safe_schemas() -> None:
    app, runner, sources = _app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        health = await client.get("/healthz")
        response = await client.post(
            "/api/v1/query",
            json={"question": "Question", "source_ids": [str(sources.source.id)]},
        )
        listing = await client.get("/api/v1/sources")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "profile": "test",
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
    assert response.status_code == 200 and response.json()["answer"] == "Answer"
    assert runner.requests[0].source_ids == [sources.source.id]
    assert listing.status_code == 200
    assert listing.json()[0]["title"] == "Example"
    assert "content_hash" not in listing.text and "private" not in listing.text


@pytest.mark.asyncio
async def test_query_error_is_mapped_without_secret_or_traceback() -> None:
    runner = FakeRunner(
        [ErrorEvent(code=ErrorCode.PROVIDER_RATE_LIMITED, message="Provider request could not be completed.")]
    )
    app, _, _ = _app(runner=runner)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/query", json={"question": "api_key=private"})

    assert response.status_code == 429
    assert response.json() == {
        "error": {
            "code": "PROVIDER_RATE_LIMITED",
            "message": "Provider request could not be completed.",
        }
    }
    assert "private" not in response.text.lower()


@pytest.mark.asyncio
async def test_stream_is_ordered_sse_and_stops_at_terminal_event() -> None:
    events = _success_events() + [TokenEvent(token="must-not-leak")]
    app, _, _ = _app(runner=FakeRunner(events))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/query/stream", json={"question": "Question"})

    assert response.status_code == 200
    payloads = [
        json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert [payload["type"] for payload in payloads] == ["status", "token", "citation", "final"]
    assert "must-not-leak" not in response.text


@pytest.mark.asyncio
async def test_admin_uses_constant_time_bearer_authorization_for_local_ingestion() -> None:
    comparisons: list[tuple[str, str]] = []

    def compare(received: str, expected: str) -> bool:
        comparisons.append((received, expected))
        return received == expected

    authorizer = AdminAuthorizer("test-secret", comparer=compare)
    assert authorizer.authorize("Bearer test-secret") == (True, 200)
    comparisons.clear()
    app, _, sources = _app(authorizer=authorizer)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        denied = await client.post("/api/v1/admin/sources/web", json={"url": "https://example.com"})
        allowed = await client.post(
            "/api/v1/admin/sources/web",
            json={"url": "https://example.com"},
            headers={"Authorization": "Bearer test-secret"},
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200 and allowed.json()["source_id"] == str(sources.source.id)
    assert comparisons == [("", "test-secret"), ("test-secret", "test-secret")]



@pytest.mark.asyncio
async def test_request_and_rate_limits_fail_before_runner() -> None:
    settings = Settings(
        app_profile="test",
        admin_token="secret",
        api_max_request_bytes=64,
        query_rate_limit_per_minute=1,
    )
    app, runner, _ = _app(settings=settings)
    size_transport = httpx.ASGITransport(app=app, client=("203.0.113.3", 1234))
    async with httpx.AsyncClient(transport=size_transport, base_url="http://test") as client:
        too_large = await client.post("/api/v1/query", content=b"x" * 65, headers={"content-type": "application/json"})
    rate_transport = httpx.ASGITransport(app=app, client=("203.0.113.4", 1234))
    async with httpx.AsyncClient(transport=rate_transport, base_url="http://test") as client:
        first = await client.post("/api/v1/query", json={"question": "first"})
        limited = await client.post("/api/v1/query", json={"question": "second"})
    assert too_large.status_code == 413
    assert first.status_code == 200
    assert limited.status_code == 429
    assert len(runner.requests) == 1


def test_settings_fail_closed_for_missing_admin_token_outside_test() -> None:
    settings = Settings(
        app_profile="full",
        gemini_api_key="key",
        admin_token=None,
    )

    with pytest.raises(ValueError, match="ADMIN_TOKEN"):
        settings.validate_runtime_credentials()


@pytest.mark.asyncio
async def test_default_app_fails_closed_at_lifespan_before_composition_without_runtime_secrets() -> None:
    composed = False

    @asynccontextmanager
    async def composer(_settings: Settings) -> AsyncIterator[AppServices]:
        nonlocal composed
        composed = True
        yield AppServices(runner=FakeRunner(), sources=FakeSources())

    app = create_app(
        settings=Settings(app_profile="full", gemini_api_key=None, admin_token=None),
        composer=composer,
    )

    with pytest.raises(ValueError, match="GEMINI_API_KEY, ADMIN_TOKEN"):
        async with app.router.lifespan_context(app):
            pass
    assert composed is False


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_default_app_binds_lazy_composition_only_inside_lifespan() -> None:
    runner = FakeRunner()
    sources = FakeSources()
    entered = False
    exited = False

    @asynccontextmanager
    async def composer(_settings: Settings) -> AsyncIterator[AppServices]:
        nonlocal entered, exited
        entered = True
        try:
            yield AppServices(runner=runner, sources=sources)
        finally:
            exited = True

    app = create_app(
        settings=Settings(app_profile="test", admin_token="secret"),
        composer=composer,
    )
    assert entered is False
    assert app.state.readiness == {
        "sqlite": False,
        "chroma": False,
        "checkpoints": False,
        "sources": False,
        "runner": False,
        "a2a_store": False,
    }
    async with app.router.lifespan_context(app):
        assert entered is True
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            health = await client.get("/healthz")
            answer = await client.post("/api/v1/query", json={"question": "Question"})
        assert health.json()["ready"] is True
        assert answer.status_code == 200 and answer.json()["answer"] == "Answer"
    assert exited is True
    assert all(ready is False for ready in app.state.readiness.values())


@pytest.mark.asyncio
async def test_shutdown_drains_gateway_before_runtime_resources_are_closed_or_unbound() -> None:
    backing_store = InMemoryTaskStore()
    runtime_closed = False
    task_id = str(uuid4())
    context = ServerCallContext()

    @asynccontextmanager
    async def composer(_settings: Settings) -> AsyncIterator[AppServices]:
        nonlocal runtime_closed
        try:
            yield AppServices(runner=FakeRunner(), sources=FakeSources(), a2a_task_store=backing_store)
        finally:
            runtime_closed = True

    app = create_app(settings=Settings(app_profile="test"), composer=composer)
    proxy = app.state.services.a2a_task_store
    assert isinstance(proxy, BindableTaskStore)
    original_close = app.state.a2a_gateway.handler.aclose

    async def drain_active_task() -> None:
        assert proxy.ready is True
        assert runtime_closed is False
        await proxy.save(
            Task(
                id=task_id,
                context_id=str(uuid4()),
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            ),
            context,
        )
        await original_close()

    app.state.a2a_gateway.handler.aclose = drain_active_task

    async with app.router.lifespan_context(app):
        assert proxy.ready is True

    persisted = await backing_store.get(task_id, context)
    assert persisted is not None
    assert persisted.status.state == TaskState.TASK_STATE_COMPLETED
    assert runtime_closed is True
    assert proxy.ready is False


@pytest.mark.asyncio
async def test_chunked_body_without_content_length_is_rejected_before_parsing() -> None:
    app, runner, _ = _app(
        settings=Settings(
            app_profile="test",
            admin_token="secret",
            api_max_request_bytes=64,
        )
    )
    sent: list[dict[str, Any]] = []
    chunks = iter(
        [
            {"type": "http.request", "body": b"{" + b"x" * 39, "more_body": True},
            {"type": "http.request", "body": b"y" * 40, "more_body": False},
        ]
    )

    async def receive() -> dict[str, Any]:
        return next(chunks)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/query",
            "raw_path": b"/api/v1/query",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("203.0.113.5", 1234),
            "server": ("test", 80),
        },
        receive,
        send,
    )
    assert next(item for item in sent if item["type"] == "http.response.start")["status"] == 413
    assert len(runner.requests) == 0


@pytest.mark.asyncio
async def test_rate_limited_chunked_request_is_rejected_without_consuming_body() -> None:
    app, _, _ = _app(
        settings=Settings(
            app_profile="test",
            admin_token="secret",
            query_rate_limit_per_minute=1,
        )
    )
    client_ip = ("203.0.113.19", 1234)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=client_ip), base_url="http://test"
    ) as client:
        first = await client.post("/api/v1/query", json={"question": "first"})
    assert first.status_code == 200

    receive_calls = 0
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": b'{"question":"second"}', "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/query",
            "raw_path": b"/api/v1/query",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": client_ip,
            "server": ("test", 80),
        },
        receive,
        send,
    )

    assert next(item for item in sent if item["type"] == "http.response.start")["status"] == 429
    assert receive_calls == 0


@pytest.mark.asyncio
async def test_admin_routes_share_per_ip_rate_limit() -> None:
    app, _, _ = _app(
        settings=Settings(
            app_profile="test",
            admin_token="secret",
            query_rate_limit_per_minute=1,
        )
    )
    headers = {"Authorization": "Bearer secret"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post("/api/v1/admin/sources/web", json={"url": "https://example.com"}, headers=headers)
        second = await client.post(
            "/api/v1/admin/sources/github", json={"url": "https://github.com/o/r"}, headers=headers
        )
    assert first.status_code == 200
    assert second.status_code == 429


@pytest.mark.asyncio
async def test_unexpected_error_is_safe_and_every_response_has_request_id(caplog: pytest.LogCaptureFixture) -> None:
    class BrokenSources(FakeSources):
        async def list_sources(self) -> list[SourceRecord]:
            raise RuntimeError("api_key=never-log-this")

    app, _, _ = _app(sources=BrokenSources())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/sources", headers={"X-Request-ID": "safe-request-123"})
    assert response.status_code == 500
    assert response.headers["x-request-id"] == "safe-request-123"
    assert response.json() == {"error": {"code": "RUN_FAILED", "message": "The run could not be completed."}}
    assert "never-log-this" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/v1/query", "/api/v1/query/stream"])
async def test_completed_query_log_correlates_safe_uuid_run_id_without_content(
    path: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="rag_app.http")
    app, _, _ = _app()
    request_id = "safe-correlation-id"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            path,
            json={"question": "private question"},
            headers={"X-Request-ID": request_id},
        )

    if path.endswith("/stream"):
        payloads = [
            json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")
        ]
        run_id = payloads[-1]["run_id"]
    else:
        run_id = response.json()["run_id"]
    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "request_complete" and record.fields["request_id"] == request_id
    )

    assert record.fields["run_id"] == run_id
    UUID(record.fields["run_id"])
    assert "private question" not in caplog.text
    assert "Answer" not in caplog.text


def test_asgi_entrypoint_import_is_network_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_PROFILE", "full")
    monkeypatch.setenv("ADMIN_TOKEN", "import-only-test-token")
    sys.modules.pop("rag_app.main", None)
    module = importlib.import_module("rag_app.main")
    assert module.app.title == "LangGraph Multi-Agent RAG"


@pytest.mark.parametrize("profile", ["full", "test"])
def test_asgi_entrypoint_imports_in_fresh_process_without_runtime_secrets(profile: str, tmp_path: Path) -> None:
    environment = os.environ.copy()
    for name in ("GEMINI_API_KEY", "ADMIN_TOKEN"):
        environment.pop(name, None)
    environment["APP_PROFILE"] = profile

    result = subprocess.run(
        [sys.executable, "-c", "from rag_app.main import app; print(app.title)"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LangGraph Multi-Agent RAG"
