from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory
from a2a.server.tasks import InMemoryTaskStore
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

from rag_app.api.app import AppServices, create_app
from rag_app.config import Settings
from rag_app.contracts import Citation, CitationEvent, FinalEvent, QueryRequest, StatusEvent, StreamEvent, TokenEvent
from rag_app.ingestion.models import SourceRecord


class A2ARunner:
    def __init__(self) -> None:
        self.requests: list[QueryRequest] = []

    async def stream(self, request: QueryRequest, *, cancel_event: object = None) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        citation = Citation(
            source_id=uuid4(),
            chunk_id=uuid4(),
            source_title="Source",
            locator="page 1",
            supporting_excerpt="Evidence",
        )
        yield StatusEvent(message="Retrieving")
        yield TokenEvent(token="Grounded ")
        yield TokenEvent(token="answer")
        yield CitationEvent(citation=citation)
        yield FinalEvent(
            answer="Grounded answer",
            citations=[citation],
            thread_id=request.thread_id,
            run_id=uuid4(),
            used_source_ids=[citation.source_id],
        )


class EmptySources:
    async def list_sources(self) -> list[SourceRecord]:
        return []


def _message(text: str) -> SendMessageRequest:
    return SendMessageRequest(message=Message(message_id=str(uuid4()), role=Role.ROLE_USER, parts=[Part(text=text)]))


@pytest.mark.asyncio
async def test_local_agent_card_advertises_query_sources_and_admin_ingestion() -> None:
    full = create_app(
        settings=Settings(app_profile="test", admin_token="secret", public_base_url="http://test"),
        services=AppServices(runner=A2ARunner(), sources=EmptySources(), a2a_task_store=InMemoryTaskStore()),
    )
    async with full.router.lifespan_context(full):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=full), base_url="http://test") as client:
            full_card = (await client.get("/.well-known/agent-card.json")).json()

    assert full_card["supportedInterfaces"][0] == {
        "url": "http://test/a2a",
        "protocolBinding": "JSONRPC",
        "protocolVersion": "1.0",
    }
    assert {skill["id"] for skill in full_card["skills"]} == {"query", "list_sources", "ingestion"}


@pytest.mark.asyncio
async def test_official_client_send_stream_get_and_list_task() -> None:
    runner = A2ARunner()
    app = create_app(
        settings=Settings(app_profile="test", admin_token="secret", public_base_url="http://test"),
        services=AppServices(runner=runner, sources=EmptySources(), a2a_task_store=InMemoryTaskStore()),
    )
    async with app.router.lifespan_context(app):
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        factory = ClientFactory(
            ClientConfig(httpx_client=http, supported_protocol_bindings=["JSONRPC"], streaming=True)
        )
        client = await factory.create_from_url("http://test")
        try:
            responses = [response async for response in client.send_message(_message("Question"))]
            task_id = next(response.task.id for response in responses if response.HasField("task"))
            final = await client.get_task(GetTaskRequest(id=task_id))
            listing = await client.list_tasks(ListTasksRequest(include_artifacts=True))
        finally:
            await client.close()
            await http.aclose()

    assert final.status.state == TaskState.TASK_STATE_COMPLETED
    assert any(artifact.name == "answer" for artifact in final.artifacts)
    assert task_id in {task.id for task in listing.tasks}
    assert runner.requests[0].thread_id is not None


@pytest.mark.asyncio
async def test_official_client_can_cancel_active_task() -> None:
    class BlockingRunner(A2ARunner):
        async def stream(self, request: QueryRequest, *, cancel_event: object = None) -> AsyncIterator[StreamEvent]:
            self.requests.append(request)
            yield StatusEvent(message="Working")
            assert hasattr(cancel_event, "wait")
            await cancel_event.wait()  # type: ignore[union-attr]

    runner = BlockingRunner()
    app = create_app(
        settings=Settings(app_profile="test", admin_token="secret", public_base_url="http://test"),
        services=AppServices(runner=runner, sources=EmptySources(), a2a_task_store=InMemoryTaskStore()),
    )
    async with app.router.lifespan_context(app):
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        factory = ClientFactory(
            ClientConfig(httpx_client=http, supported_protocol_bindings=["JSONRPC"], streaming=False)
        )
        client = await factory.create_from_url("http://test")
        try:
            request = _message("Question")
            request.configuration.return_immediately = True
            response = await anext(client.send_message(request))
            task_id = response.task.id
            cancelled = await client.cancel_task(CancelTaskRequest(id=task_id))
        finally:
            await client.close()
            await http.aclose()
    assert cancelled.status.state == TaskState.TASK_STATE_CANCELED


@pytest.mark.asyncio
async def test_a2a_ingestion_retries_rate_limit_and_enforces_pdf_bytes() -> None:
    from a2a.server.agent_execution import RequestContext
    from a2a.server.context import ServerCallContext
    from a2a.server.events import EventQueue

    from rag_app.a2a.executor import RagAgentExecutor
    from rag_app.api.limits import QueryCapacity
    from rag_app.errors import AppError, ErrorCode

    class Queue(EventQueue):
        def __init__(self) -> None:
            self.events: list[object] = []

        async def enqueue_event(self, event: object) -> None:
            self.events.append(event)

    class RetrySources(EmptySources):
        def __init__(self) -> None:
            self.calls = 0

        async def ingest_web(self, _url: str) -> object:
            self.calls += 1
            if self.calls == 1:
                raise AppError(ErrorCode.PROVIDER_RATE_LIMITED, "secret provider body")
            return type(
                "Result",
                (),
                {
                    "source": type("Source", (), {"id": uuid4()})(),
                    "job": type("Job", (), {"id": uuid4()})(),
                    "status": "ingested",
                },
            )()

        async def ingest_pdf(self, _filename: str, _data: bytes) -> object:
            raise AssertionError("oversized PDF must not reach ingestion")

    class LocalOnlyAuthorizer:
        def authorize(self, authorization: str | None) -> tuple[bool, int]:
            return (authorization == "Bearer secret", 200 if authorization == "Bearer secret" else 401)

    sources = RetrySources()
    settings = Settings(app_profile="test", provider_max_retries=1, pdf_max_bytes=8)
    executor = RagAgentExecutor(
        runner=A2ARunner(),
        sources=sources,
        settings=settings,
        authorizer=LocalOnlyAuthorizer(),  # type: ignore[arg-type]
        capacity=QueryCapacity(2),
    )

    web_request = _message("ingest")
    web_request.metadata.update({"skill_id": "ingestion", "source_type": "web", "url": "https://example.com"})
    web_context = RequestContext(ServerCallContext(state={"headers": {"authorization": "Bearer secret"}}), web_request)
    web_queue = Queue()
    await executor.execute(web_context, web_queue)
    assert sources.calls == 2
    assert any(
        getattr(getattr(event, "status", None), "state", None) == TaskState.TASK_STATE_COMPLETED
        for event in web_queue.events
    )

    pdf_request = SendMessageRequest(
        message=Message(
            message_id=str(uuid4()),
            role=Role.ROLE_USER,
            parts=[Part(raw=b"%PDF-" + b"x" * 10, filename="large.pdf", media_type="application/pdf")],
        ),
        metadata={"skill_id": "ingestion", "source_type": "pdf"},
    )
    pdf_context = RequestContext(ServerCallContext(state={"headers": {"authorization": "Bearer secret"}}), pdf_request)
    pdf_queue = Queue()
    await executor.execute(pdf_context, pdf_queue)
    assert any(
        getattr(getattr(event, "status", None), "state", None) == TaskState.TASK_STATE_FAILED
        for event in pdf_queue.events
    )


@pytest.mark.asyncio
async def test_a2a_ingestion_cancellation_stops_inflight_source_work() -> None:
    import asyncio

    from a2a.server.agent_execution import RequestContext
    from a2a.server.context import ServerCallContext
    from a2a.server.events import EventQueue

    from rag_app.a2a.executor import RagAgentExecutor
    from rag_app.api.limits import QueryCapacity
    from rag_app.api.security import AdminAuthorizer

    class Queue(EventQueue):
        def __init__(self) -> None:
            self.events: list[object] = []

        async def enqueue_event(self, event: object) -> None:
            self.events.append(event)

    class BlockingSources(EmptySources):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def ingest_web(self, _url: str) -> object:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()

    sources = BlockingSources()
    executor = RagAgentExecutor(
        runner=A2ARunner(),
        sources=sources,
        settings=Settings(app_profile="test"),
        authorizer=AdminAuthorizer("secret"),
        capacity=QueryCapacity(2),
    )
    request = _message("ingest")
    request.metadata.update({"skill_id": "ingestion", "source_type": "web", "url": "https://example.com"})
    context = RequestContext(ServerCallContext(state={"headers": {"authorization": "Bearer secret"}}), request)
    queue = Queue()
    execution = asyncio.create_task(executor.execute(context, queue))
    await sources.started.wait()
    await executor.cancel(context, queue)
    await asyncio.gather(execution, return_exceptions=True)
    assert sources.cancelled.is_set()
    assert any(
        getattr(getattr(event, "status", None), "state", None) == TaskState.TASK_STATE_CANCELED
        for event in queue.events
    )
