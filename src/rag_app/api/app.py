"""FastAPI factory for REST and the co-located official A2A gateway."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, cast
from uuid import UUID

from a2a.server.context import ServerCallContext
from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.types import ListTasksRequest, ListTasksResponse, Task
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncEngine

from rag_app.a2a.server import build_gateway
from rag_app.api.limits import (
    PerIpRateLimitMiddleware,
    QueryCapacity,
    RequestIdMiddleware,
    RequestSizeLimitMiddleware,
    public_error,
)
from rag_app.api.models import DeleteResponse, HealthResponse, IngestionResponse, SourceSummary, SourceUrlRequest
from rag_app.api.protocols import AdminSourceService, Runner, SourceService
from rag_app.api.retry import retry_async
from rag_app.api.security import AdminAuthorizer
from rag_app.config import Settings
from rag_app.contracts import ErrorEvent, FinalEvent, QueryRequest, QueryResponse, StreamEvent
from rag_app.errors import AppError, ErrorCode
from rag_app.ingestion.models import SourceRecord

_STREAM_ADAPTER: TypeAdapter[StreamEvent] = TypeAdapter(StreamEvent)


@dataclass
class AppServices:
    runner: Runner
    sources: SourceService
    a2a_task_store: TaskStore | None = None
    a2a_engine: AsyncEngine | None = None
    readiness: dict[str, bool] | None = None


def _readiness(
    runner: BindableRunner,
    sources: BindableSources,
    task_store: TaskStore,
    *,
    local_resources_ready: bool,
) -> dict[str, bool]:
    return {
        "sqlite": local_resources_ready,
        "chroma": local_resources_ready,
        "checkpoints": local_resources_ready,
        "sources": sources.ready,
        "runner": runner.ready,
        "a2a_store": not isinstance(task_store, BindableTaskStore) or task_store.ready,
    }


class BindableRunner:
    def __init__(self, target: Runner | None = None) -> None:
        self._target = target

    def bind(self, target: Runner | None) -> None:
        self._target = target

    @property
    def ready(self) -> bool:
        return self._target is not None

    async def stream(
        self, request: QueryRequest, *, cancel_event: asyncio.Event | None = None
    ) -> AsyncIterator[StreamEvent]:
        if self._target is None:
            yield ErrorEvent(code=ErrorCode.RUN_FAILED, message="The run could not be completed.")
            return
        async for event in self._target.stream(request, cancel_event=cancel_event):
            yield event


class BindableSources:
    def __init__(self, target: SourceService | None = None) -> None:
        self._target = target

    def bind(self, target: SourceService | None) -> None:
        self._target = target

    @property
    def ready(self) -> bool:
        return self._target is not None

    def _get(self) -> Any:
        if self._target is None:
            raise RuntimeError("Source service is not ready")
        return self._target

    async def list_sources(self) -> list[SourceRecord]:
        return cast(list[SourceRecord], await self._get().list_sources())

    async def ingest_pdf(self, filename: str, data: bytes) -> Any:
        return await self._get().ingest_pdf(filename, data)

    async def ingest_web(self, url: str) -> Any:
        return await self._get().ingest_web(url)

    async def ingest_github(self, url: str) -> Any:
        return await self._get().ingest_github(url)

    async def reindex(self, source_id: UUID) -> Any:
        return await self._get().reindex(source_id)

    async def delete(self, source_id: UUID) -> None:
        await self._get().delete(source_id)


class BindableTaskStore(TaskStore):
    def __init__(self, target: TaskStore | None = None) -> None:
        self._target = target

    def bind(self, target: TaskStore | None) -> None:
        self._target = target

    @property
    def ready(self) -> bool:
        return self._target is not None

    def _get(self) -> TaskStore:
        if self._target is None:
            raise RuntimeError("A2A task store is not ready")
        return self._target

    async def save(self, task: Task, context: ServerCallContext) -> None:
        await self._get().save(task, context)

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        return await self._get().get(task_id, context)

    async def list(self, params: ListTasksRequest, context: ServerCallContext) -> ListTasksResponse:
        return await self._get().list(params, context)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        await self._get().delete(task_id, context)


RuntimeComposer = Callable[[Settings], AbstractAsyncContextManager[AppServices]]


def create_app(
    *,
    settings: Settings | None = None,
    services: AppServices | None = None,
    admin_authorizer: AdminAuthorizer | None = None,
    composer: RuntimeComposer | None = None,
) -> FastAPI:
    """Build an inert-on-import app; all external resources are injected or opened in lifespan."""

    config = settings or Settings()
    runner_proxy = BindableRunner(services.runner if services else None)
    sources_proxy = BindableSources(services.sources if services else None)
    if services is None:
        task_store: TaskStore = BindableTaskStore()
    else:
        task_store = services.a2a_task_store or InMemoryTaskStore()
    configured_services = AppServices(runner=runner_proxy, sources=sources_proxy, a2a_task_store=task_store)
    token = config.admin_token.get_secret_value() if config.admin_token else None
    authorizer = admin_authorizer or AdminAuthorizer(token)
    capacity = QueryCapacity(config.query_concurrency)
    gateway = build_gateway(
        runner=configured_services.runner,
        sources=configured_services.sources,
        settings=config,
        task_store=task_store,
        authorizer=authorizer,
        capacity=capacity,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        runtime_context: AbstractAsyncContextManager[AppServices] | None = None
        runtime_entered = False
        try:
            if services is None:
                config.validate_runtime_credentials()
                if composer is None:
                    from rag_app.composition import compose_runtime

                    runtime_context = compose_runtime(config)
                else:
                    runtime_context = composer(config)
                runtime = await runtime_context.__aenter__()
                runtime_entered = True
                runner_proxy.bind(runtime.runner)
                sources_proxy.bind(runtime.sources)
                cast(BindableTaskStore, task_store).bind(runtime.a2a_task_store or InMemoryTaskStore())
            _app.state.readiness = _readiness(
                runner_proxy,
                sources_proxy,
                task_store,
                local_resources_ready=True,
            )
            yield
        finally:
            try:
                await gateway.aclose()
            finally:
                try:
                    if runtime_context is not None and runtime_entered:
                        await runtime_context.__aexit__(None, None, None)
                finally:
                    _app.state.readiness = {
                        name: False
                        for name in _readiness(
                            runner_proxy,
                            sources_proxy,
                            task_store,
                            local_resources_ready=False,
                        )
                    }
                    if services is None:
                        runner_proxy.bind(None)
                        sources_proxy.bind(None)
                        cast(BindableTaskStore, task_store).bind(None)

    app = FastAPI(title="LangGraph Multi-Agent RAG", version="0.1.0", lifespan=lifespan)
    app.state.settings = config
    app.state.services = configured_services
    app.state.a2a_gateway = gateway
    app.state.query_capacity = capacity
    app.state.readiness = _readiness(
        runner_proxy,
        sources_proxy,
        task_store,
        local_resources_ready=services is not None,
    )
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=config.api_max_request_bytes)
    app.add_middleware(
        PerIpRateLimitMiddleware,
        limit=config.query_rate_limit_per_minute,
        trust_proxy_headers=config.trust_proxy_headers,
    )
    app.add_middleware(RequestIdMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return public_error(422, "INVALID_REQUEST", "The request is invalid.")

    @app.exception_handler(AppError)
    async def application_error(_request: Request, exc: AppError) -> JSONResponse:
        return _app_error_response(exc.code)

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(exc.detail, status_code=exc.status_code)
        return public_error(exc.status_code, "HTTP_ERROR", "The request could not be completed.")

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
        return _app_error_response(ErrorCode.RUN_FAILED)

    @app.get("/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        dependencies = dict(app.state.readiness)
        ready = all(dependencies.values())
        return HealthResponse(
            status="ok" if ready else "degraded", profile=config.app_profile, ready=ready, dependencies=dependencies
        )

    @app.post("/api/v1/query", response_model=QueryResponse)
    async def query(payload: QueryRequest, request: Request) -> QueryResponse | JSONResponse:
        cancel_event = asyncio.Event()
        async with capacity.slot():
            async for event in _bounded_stream(configured_services.runner, payload, cancel_event, config):
                if isinstance(event, ErrorEvent):
                    return _app_error_response(event.code)
                if isinstance(event, FinalEvent):
                    if event.run_id is not None:
                        request.state.run_id = str(event.run_id)
                    return _query_response(event)
        return _app_error_response(ErrorCode.RUN_FAILED)

    @app.post("/api/v1/query/stream")
    async def query_stream(payload: QueryRequest, request: Request) -> StreamingResponse:
        return StreamingResponse(
            _sse_stream(request, configured_services.runner, payload, config, capacity),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/sources", response_model=list[SourceSummary])
    async def list_sources() -> list[SourceSummary]:
        records = await retry_async(
            configured_services.sources.list_sources,
            retries=config.provider_max_retries,
            timeout_seconds=config.dependency_timeout_seconds,
        )
        return [SourceSummary.from_record(source) for source in records]

    async def require_admin(authorization: Annotated[str | None, Header()] = None) -> None:
        allowed, status = authorizer.authorize(authorization)
        if not allowed:
            raise HTTPException(
                status_code=status,
                detail={"error": {"code": "UNAUTHORIZED", "message": "Administrator authorization is required."}},
            )

    admin = cast(AdminSourceService, configured_services.sources)

    @app.post("/api/v1/admin/sources/pdf", response_model=IngestionResponse, dependencies=[Depends(require_admin)])
    async def ingest_pdf(file: Annotated[UploadFile, File()]) -> IngestionResponse:
        data = await _bounded_upload(file, config.pdf_max_bytes)
        result = await retry_async(
            lambda: admin.ingest_pdf(file.filename or "upload.pdf", data),
            retries=config.provider_max_retries,
            timeout_seconds=config.dependency_timeout_seconds,
        )
        return IngestionResponse.from_result(result)

    @app.post("/api/v1/admin/sources/web", response_model=IngestionResponse, dependencies=[Depends(require_admin)])
    async def ingest_web(payload: SourceUrlRequest) -> IngestionResponse:
        result = await retry_async(
            lambda: admin.ingest_web(str(payload.url)),
            retries=config.provider_max_retries,
            timeout_seconds=config.dependency_timeout_seconds,
        )
        return IngestionResponse.from_result(result)

    @app.post("/api/v1/admin/sources/github", response_model=IngestionResponse, dependencies=[Depends(require_admin)])
    async def ingest_github(payload: SourceUrlRequest) -> IngestionResponse:
        result = await retry_async(
            lambda: admin.ingest_github(str(payload.url)),
            retries=config.provider_max_retries,
            timeout_seconds=config.dependency_timeout_seconds,
        )
        return IngestionResponse.from_result(result)

    @app.post(
        "/api/v1/admin/sources/{source_id}/reindex",
        response_model=IngestionResponse,
        dependencies=[Depends(require_admin)],
    )
    async def reindex(source_id: UUID) -> IngestionResponse:
        result = await retry_async(
            lambda: admin.reindex(source_id),
            retries=config.provider_max_retries,
            timeout_seconds=config.dependency_timeout_seconds,
        )
        return IngestionResponse.from_result(result)

    @app.delete(
        "/api/v1/admin/sources/{source_id}",
        response_model=DeleteResponse,
        dependencies=[Depends(require_admin)],
    )
    async def delete(source_id: UUID) -> DeleteResponse:
        await retry_async(
            lambda: admin.delete(source_id),
            retries=config.provider_max_retries,
            timeout_seconds=config.dependency_timeout_seconds,
        )
        return DeleteResponse(source_id=source_id)

    app.router.routes.extend(gateway.routes)
    from rag_app.ui.app import mount_ui

    return mount_ui(
        app,
        settings=config,
        services=configured_services,
        authorizer=authorizer,
        capacity=capacity,
    )


async def _bounded_upload(file: UploadFile, maximum: int) -> bytes:
    body = bytearray()
    while part := await file.read(min(1024 * 1024, maximum + 1)):
        body.extend(part)
        if len(body) > maximum:
            raise HTTPException(
                status_code=413,
                detail={"error": {"code": "REQUEST_TOO_LARGE", "message": "The upload exceeds the byte limit."}},
            )
    return bytes(body)


async def _bounded_stream(
    runner: Runner,
    request: QueryRequest,
    cancellation: asyncio.Event,
    settings: Settings,
) -> AsyncIterator[StreamEvent]:
    try:
        async with asyncio.timeout(settings.query_timeout_seconds):
            async for event in runner.stream(request, cancel_event=cancellation):
                yield event
    except TimeoutError:
        cancellation.set()
        yield ErrorEvent(code=ErrorCode.RUN_FAILED, message="The run could not be completed.")
    except Exception:
        yield ErrorEvent(code=ErrorCode.RUN_FAILED, message="The run could not be completed.")


async def _sse_stream(
    http_request: Request,
    runner: Runner,
    query: QueryRequest,
    settings: Settings,
    capacity: QueryCapacity,
) -> AsyncIterator[bytes]:
    cancellation = asyncio.Event()

    async def watch_disconnect() -> None:
        while not cancellation.is_set():
            if await http_request.is_disconnected():
                cancellation.set()
                return
            await asyncio.sleep(0.05)

    monitor = asyncio.create_task(watch_disconnect())
    terminal = False
    try:
        async with capacity.slot():
            async for event in _bounded_stream(runner, query, cancellation, settings):
                if terminal:
                    break
                if isinstance(event, FinalEvent) and event.run_id is not None:
                    http_request.state.run_id = str(event.run_id)
                yield f"data: {_STREAM_ADAPTER.dump_json(event).decode('utf-8')}\n\n".encode()
                terminal = isinstance(event, FinalEvent | ErrorEvent)
    finally:
        cancellation.set()
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)


def _query_response(event: FinalEvent) -> QueryResponse:
    return QueryResponse(
        answer=event.answer,
        citations=event.citations,
        thread_id=event.thread_id,
        mode=event.mode,
        run_id=event.run_id,
        used_source_ids=event.used_source_ids,
        trace_summary=event.trace_summary,
    )


def _app_error_response(code: ErrorCode) -> JSONResponse:
    payload = AppError(code, "safe").to_payload()
    statuses = {
        ErrorCode.NO_SUPPORTING_EVIDENCE: 422,
        ErrorCode.INVALID_SOURCE: 422,
        ErrorCode.INGESTION_FAILED: 502,
        ErrorCode.PROVIDER_RATE_LIMITED: 429,
        ErrorCode.RUN_CANCELLED: 409,
        ErrorCode.RUN_FAILED: 500,
    }
    return JSONResponse({"error": {"code": payload["code"], "message": payload["message"]}}, status_code=statuses[code])
