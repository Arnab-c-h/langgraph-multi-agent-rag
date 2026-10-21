"""Translate typed LangGraph streams into official A2A task lifecycle events."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Role, Task, TaskState, TaskStatus
from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]
from pydantic import ValidationError

from rag_app.api.limits import QueryCapacity
from rag_app.api.models import SourceSummary
from rag_app.api.protocols import Runner, SourceService
from rag_app.api.retry import retry_async
from rag_app.api.security import AdminAuthorizer
from rag_app.config import Settings
from rag_app.contracts import (
    CitationEvent,
    ErrorEvent,
    FinalEvent,
    QueryRequest,
    ReasoningMode,
    StatusEvent,
    TokenEvent,
)
from rag_app.errors import AppError, ErrorCode


class RagAgentExecutor(AgentExecutor):
    def __init__(
        self,
        *,
        runner: Runner,
        sources: SourceService,
        settings: Settings,
        authorizer: AdminAuthorizer,
        capacity: QueryCapacity,
    ) -> None:
        self._runner = runner
        self._sources = sources
        self._settings = settings
        self._authorizer = authorizer
        self._capacity = capacity
        self._cancellations: dict[str, asyncio.Event] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        if not task_id or not context_id:
            raise ValueError("A2A task and context identifiers are required")
        updater = TaskUpdater(event_queue, task_id, context_id)
        initial = Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            history=[context.message] if context.message else [],
        )
        await event_queue.enqueue_event(initial)
        cancellation = asyncio.Event()
        self._cancellations[task_id] = cancellation
        try:
            await updater.start_work(_agent_message(task_id, context_id, "Working"))
            skill_id = str(_metadata(context).get("skill_id", "query"))
            if skill_id == "list_sources":
                await self._list_sources(updater)
                return
            if skill_id == "ingestion":
                await self._ingest(context, updater, cancellation)
                return
            await self._query(context, updater, cancellation)
        except asyncio.CancelledError:
            raise
        except (AppError, ValidationError) as exc:
            code = exc.code if isinstance(exc, AppError) else ErrorCode.INVALID_SOURCE
            await updater.failed(_agent_message(task_id, context_id, code.value))
        except Exception:
            await updater.failed(_agent_message(task_id, context_id, ErrorCode.RUN_FAILED.value))
        finally:
            self._cancellations.pop(task_id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        cancellation = self._cancellations.get(task_id)
        if cancellation is not None:
            cancellation.set()
        await TaskUpdater(event_queue, task_id, context_id).cancel(
            _agent_message(task_id, context_id, ErrorCode.RUN_CANCELLED.value)
        )

    async def _query(self, context: RequestContext, updater: TaskUpdater, cancellation: asyncio.Event) -> None:
        metadata = _metadata(context)
        query = QueryRequest(
            question=context.get_user_input(),
            reasoning_mode=metadata.get("reasoning_mode", ReasoningMode.STANDARD.value),
            thread_id=_thread_uuid(context.context_id or ""),
            top_k=metadata.get("top_k", 6),
            source_types=metadata.get("source_types"),
            source_ids=metadata.get("source_ids"),
        )
        terminal = False
        first_token = True
        citations: list[dict[str, Any]] = []
        async with self._capacity.slot():
            async with asyncio.timeout(self._settings.query_timeout_seconds):
                async for event in self._runner.stream(query, cancel_event=cancellation):
                    if terminal:
                        break
                    if isinstance(event, StatusEvent):
                        await updater.update_status(
                            TaskState.TASK_STATE_WORKING,
                            _agent_message(updater.task_id, updater.context_id, event.message),
                        )
                    elif isinstance(event, TokenEvent):
                        await updater.add_artifact(
                            [Part(text=event.token)],
                            artifact_id=f"{updater.task_id}-answer",
                            name="answer",
                            append=not first_token,
                            last_chunk=False,
                        )
                        first_token = False
                    elif isinstance(event, CitationEvent):
                        citations.append(event.citation.model_dump(mode="json"))
                    elif isinstance(event, ErrorEvent):
                        terminal = True
                        if event.code is not ErrorCode.RUN_CANCELLED:
                            await updater.failed(_agent_message(updater.task_id, updater.context_id, event.code.value))
                    elif isinstance(event, FinalEvent):
                        terminal = True
                        await updater.add_artifact(
                            [_json_part({"answer": event.answer, "citations": citations})],
                            artifact_id=f"{updater.task_id}-result",
                            name="result",
                            last_chunk=True,
                        )
                        await updater.complete(_agent_message(updater.task_id, updater.context_id, event.answer))

    async def _list_sources(self, updater: TaskUpdater) -> None:
        records = await self._sources.list_sources()
        payload = [SourceSummary.from_record(source).model_dump(mode="json") for source in records]
        await updater.add_artifact(
            [_json_part({"sources": payload})],
            artifact_id=f"{updater.task_id}-sources",
            name="sources",
            last_chunk=True,
        )
        await updater.complete(_agent_message(updater.task_id, updater.context_id, json.dumps(payload)))

    async def _ingest(self, context: RequestContext, updater: TaskUpdater, cancellation: asyncio.Event) -> None:
        headers = context.call_context.state.get("headers", {})
        authorization = headers.get("authorization") if isinstance(headers, dict) else None
        allowed, _status = self._authorizer.authorize(str(authorization) if authorization is not None else None)
        if not allowed:
            await updater.reject(_agent_message(updater.task_id, updater.context_id, "INGESTION_NOT_ALLOWED"))
            return
        metadata = _metadata(context)
        source_type = str(metadata.get("source_type", ""))
        url = str(metadata.get("url", ""))
        service = self._sources
        if source_type == "web" and hasattr(service, "ingest_web"):
            result = await retry_async(
                lambda: service.ingest_web(url),
                retries=self._settings.provider_max_retries,
                timeout_seconds=self._settings.dependency_timeout_seconds,
                cancel_event=cancellation,
            )
        elif source_type == "github" and hasattr(service, "ingest_github"):
            result = await retry_async(
                lambda: service.ingest_github(url),
                retries=self._settings.provider_max_retries,
                timeout_seconds=self._settings.dependency_timeout_seconds,
                cancel_event=cancellation,
            )
        elif source_type == "pdf" and hasattr(service, "ingest_pdf") and context.message:
            raw_part = next((part for part in context.message.parts if part.raw), None)
            if raw_part is None:
                raise AppError(ErrorCode.INVALID_SOURCE, "Missing PDF")
            data = bytes(raw_part.raw)
            if len(data) > self._settings.pdf_max_bytes:
                raise AppError(ErrorCode.INVALID_SOURCE, "PDF exceeds the byte limit")
            result = await retry_async(
                lambda: service.ingest_pdf(raw_part.filename or "upload.pdf", data),
                retries=self._settings.provider_max_retries,
                timeout_seconds=self._settings.dependency_timeout_seconds,
                cancel_event=cancellation,
            )
        else:
            raise AppError(ErrorCode.INVALID_SOURCE, "Unsupported ingestion request")
        payload = {"source_id": str(result.source.id), "job_id": str(result.job.id), "status": result.status}
        await updater.add_artifact(
            [_json_part(payload)],
            artifact_id=f"{updater.task_id}-ingestion",
            name="ingestion",
            last_chunk=True,
        )
        await updater.complete(_agent_message(updater.task_id, updater.context_id, json.dumps(payload)))


def _thread_uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError:
        return uuid5(NAMESPACE_URL, f"a2a-context:{value}")


def _metadata(context: RequestContext) -> dict[str, Any]:
    metadata = context.metadata
    message = context.message
    if message is not None and message.metadata.fields:
        message_metadata = MessageToDict(message.metadata)
        metadata = {**message_metadata, **metadata}
    return metadata


def _agent_message(task_id: str, context_id: str, text: str) -> Message:
    return Message(
        message_id=str(uuid5(NAMESPACE_URL, f"{task_id}:{text}")),
        task_id=task_id,
        context_id=context_id,
        role=Role.ROLE_AGENT,
        parts=[Part(text=text)],
    )


def _json_part(value: object) -> Part:
    return Part(text=json.dumps(value, separators=(",", ":"), sort_keys=True), media_type="application/json")
