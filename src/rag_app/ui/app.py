"""Dependency-injected Gradio UI with profile-enforced administration."""

from __future__ import annotations

import asyncio
import html
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import gradio as gr
from fastapi import FastAPI
from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]

from rag_app.a2a.card import build_agent_card
from rag_app.api.limits import QueryCapacity
from rag_app.api.protocols import AdminSourceService, Runner, SourceService
from rag_app.api.retry import retry_async
from rag_app.api.security import AdminAuthorizer
from rag_app.config import Settings
from rag_app.contracts import (
    Citation,
    CitationEvent,
    ErrorEvent,
    FinalEvent,
    QueryRequest,
    ReasoningMode,
    SourceType,
    StatusEvent,
    TokenEvent,
)
from rag_app.errors import AppError, ErrorCode

_UNSAFE_ATTRIBUTE = re.compile(r"(?i)on(?:error|load)\s*=")
_SOURCE_ID_SEPARATOR = re.compile(r"[,\s]+")


class UiServices(Protocol):
    runner: Runner
    sources: SourceService


class A2ATester(Protocol):
    async def send(self, text: str, mode: ReasoningMode) -> tuple[str, str]: ...

    def stream(self, text: str, mode: ReasoningMode) -> AsyncIterator[tuple[str, str]]: ...

    async def cancel(self, task_id: str) -> str: ...


@dataclass(frozen=True)
class ChatUpdate:
    status: str
    answer: str
    citations_html: str
    thread_id: str
    run_key: str
    error: str


class UiController:
    """Safe callback boundary shared by Gradio events and focused tests."""

    def __init__(
        self,
        *,
        settings: Settings,
        services: UiServices,
        authorizer: AdminAuthorizer,
        a2a_tester: A2ATester,
        readiness_provider: Callable[[], dict[str, bool]] | None = None,
        capacity: QueryCapacity | None = None,
    ) -> None:
        self.settings = settings
        self.services = services
        self._authorizer = authorizer
        self._a2a = a2a_tester
        self._readiness_provider = readiness_provider or (lambda: {})
        self._capacity = capacity or QueryCapacity(settings.query_concurrency)
        self._cancellations: dict[str, asyncio.Event] = {}

    @property
    def _admin_sources(self) -> AdminSourceService:
        return cast(AdminSourceService, self.services.sources)

    async def stream_chat(
        self,
        question: str,
        reasoning_mode: str,
        source_types: list[str] | None,
        source_ids: str,
        thread_id: str,
    ) -> AsyncIterator[ChatUpdate]:
        run_key = str(uuid4())
        cancellation = asyncio.Event()
        self._cancellations[run_key] = cancellation
        answer = ""
        citations: list[Citation] = []
        current_thread = thread_id.strip()
        try:
            request = QueryRequest(
                question=question,
                reasoning_mode=ReasoningMode(reasoning_mode),
                source_types=[SourceType(value) for value in source_types] if source_types else None,
                source_ids=_parse_source_ids(source_ids),
                thread_id=UUID(current_thread) if current_thread else None,
            )
        except (ValueError, TypeError):
            yield ChatUpdate("", "", "", current_thread, run_key, "INVALID_SOURCE")
            self._cancellations.pop(run_key, None)
            return
        try:
            async with self._capacity.slot():
                async with asyncio.timeout(self.settings.query_timeout_seconds):
                    async for event in self.services.runner.stream(request, cancel_event=cancellation):
                        if isinstance(event, StatusEvent):
                            yield ChatUpdate(
                                _safe_text(event.message),
                                answer,
                                _render_citations(citations),
                                current_thread,
                                run_key,
                                "",
                            )
                        elif isinstance(event, TokenEvent):
                            answer += event.token
                            yield ChatUpdate(
                                "Generating", answer, _render_citations(citations), current_thread, run_key, ""
                            )
                        elif isinstance(event, CitationEvent):
                            citations.append(event.citation)
                            yield ChatUpdate(
                                "Validating citations",
                                answer,
                                _render_citations(citations),
                                current_thread,
                                run_key,
                                "",
                            )
                        elif isinstance(event, ErrorEvent):
                            yield ChatUpdate(
                                "", answer, _render_citations(citations), current_thread, run_key, event.code.value
                            )
                            return
                        elif isinstance(event, FinalEvent):
                            current_thread = str(event.thread_id or request.thread_id or "")
                            citations = event.citations
                            yield ChatUpdate(
                                "Complete", event.answer, _render_citations(citations), current_thread, run_key, ""
                            )
                            return
        except TimeoutError:
            cancellation.set()
            yield ChatUpdate(
                "", answer, _render_citations(citations), current_thread, run_key, ErrorCode.RUN_FAILED.value
            )
        except asyncio.CancelledError:
            cancellation.set()
            raise
        except Exception:
            yield ChatUpdate(
                "", answer, _render_citations(citations), current_thread, run_key, ErrorCode.RUN_FAILED.value
            )
        finally:
            self._cancellations.pop(run_key, None)

    def cancel_chat(self, run_key: str) -> str:
        cancellation = self._cancellations.get(run_key)
        if cancellation is None:
            return "NO_ACTIVE_RUN"
        cancellation.set()
        return ErrorCode.RUN_CANCELLED.value

    @staticmethod
    def new_thread() -> str:
        return str(uuid4())

    @staticmethod
    def reset_thread() -> tuple[str, str, str, str, str]:
        return "", "", "", "", ""

    async def list_sources(self) -> list[list[str]]:
        try:
            sources = await self.services.sources.list_sources()
        except Exception:
            return []
        return [
            [
                str(source.id),
                source.source_type.value,
                _safe_text(source.title),
                _safe_text(source.canonical_uri),
                _source_status(source.metadata),
            ]
            for source in sources
        ]

    async def ingest_pdf(self, upload: Any, authorization: str) -> str:
        blocked = self._admin_error(authorization)
        if blocked:
            return blocked
        try:
            raw_path = upload if isinstance(upload, str | Path) else getattr(upload, "name", upload)
            path = Path(str(raw_path))
            if not path.is_file() or path.stat().st_size > self.settings.pdf_max_bytes:
                return ErrorCode.INVALID_SOURCE.value
            data = path.read_bytes()
            result = await self._retry_mutation(lambda: self._admin_sources.ingest_pdf(path.name, data))
            return _ingestion_summary(result)
        except AppError as exc:
            return exc.code.value
        except Exception:
            return ErrorCode.INGESTION_FAILED.value

    async def ingest_web(self, url: str, authorization: str) -> str:
        return await self._ingest_url("web", url, authorization)

    async def ingest_github(self, url: str, authorization: str) -> str:
        return await self._ingest_url("github", url, authorization)

    async def ingest_pdf_progress(self, upload: Any, authorization: str) -> AsyncIterator[str]:
        async for update in self._ingestion_progress(
            "PDF", authorization, lambda: self.ingest_pdf(upload, authorization)
        ):
            yield update

    async def ingest_web_progress(self, url: str, authorization: str) -> AsyncIterator[str]:
        async for update in self._ingestion_progress(
            "web", authorization, lambda: self.ingest_web(url, authorization)
        ):
            yield update

    async def ingest_github_progress(self, url: str, authorization: str) -> AsyncIterator[str]:
        async for update in self._ingestion_progress(
            "GitHub", authorization, lambda: self.ingest_github(url, authorization)
        ):
            yield update

    async def reindex(self, source_id: str, authorization: str) -> str:
        blocked = self._admin_error(authorization)
        if blocked:
            return blocked
        try:
            parsed = UUID(source_id.strip())
            result = await self._retry_mutation(lambda: self._admin_sources.reindex(parsed))
            return _ingestion_summary(result)
        except AppError as exc:
            return exc.code.value
        except (ValueError, TypeError):
            return ErrorCode.INVALID_SOURCE.value
        except Exception:
            return ErrorCode.INGESTION_FAILED.value

    async def delete(self, source_id: str, authorization: str) -> str:
        blocked = self._admin_error(authorization)
        if blocked:
            return blocked
        try:
            parsed = UUID(source_id.strip())
            await self._retry_mutation(lambda: self._admin_sources.delete(parsed))
            return f"deleted:{parsed}"
        except AppError as exc:
            return exc.code.value
        except (ValueError, TypeError):
            return ErrorCode.INVALID_SOURCE.value
        except Exception:
            return ErrorCode.INGESTION_FAILED.value

    async def a2a_send(self, text: str, mode: str) -> tuple[str, str]:
        try:
            task_id, result = await self._a2a.send(text, ReasoningMode(mode))
            return _safe_text(task_id), _safe_text(result)
        except Exception:
            return "", "A2A_FAILED"

    async def a2a_stream(self, text: str, mode: str) -> AsyncIterator[tuple[str, str]]:
        try:
            async for task_id, update in self._a2a.stream(text, ReasoningMode(mode)):
                yield _safe_text(task_id), _safe_text(update)
        except Exception:
            yield "", "A2A_FAILED"

    async def a2a_cancel(self, task_id: str) -> str:
        try:
            return _safe_text(await self._a2a.cancel(task_id.strip()))
        except Exception:
            return "A2A_FAILED"

    def system_summary(self) -> str:
        summary: dict[str, object] = dict(self.settings.safe_summary())
        summary["health_path"] = "/healthz"
        summary["readiness"] = self._readiness_provider()
        return json.dumps(summary, indent=2, sort_keys=True)

    def agent_card_summary(self) -> str:
        return json.dumps(MessageToDict(build_agent_card(self.settings)), indent=2, sort_keys=True)

    def _admin_error(self, authorization: str) -> str:
        header = authorization.strip()
        if header and not header.lower().startswith("bearer "):
            header = f"Bearer {header}"
        allowed, _status = self._authorizer.authorize(header or None)
        if allowed:
            return ""
        return "UNAUTHORIZED"

    async def _ingest_url(self, kind: str, url: str, authorization: str) -> str:
        blocked = self._admin_error(authorization)
        if blocked:
            return blocked
        try:
            if kind == "web":
                result = await self._retry_mutation(lambda: self._admin_sources.ingest_web(url.strip()))
            else:
                result = await self._retry_mutation(lambda: self._admin_sources.ingest_github(url.strip()))
            return _ingestion_summary(result)
        except AppError as exc:
            return exc.code.value
        except Exception:
            return ErrorCode.INGESTION_FAILED.value

    async def _retry_mutation[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        return await retry_async(
            operation,
            retries=self.settings.provider_max_retries,
            timeout_seconds=self.settings.dependency_timeout_seconds,
        )

    async def _ingestion_progress(
        self,
        label: str,
        authorization: str,
        operation: Callable[[], Awaitable[str]],
    ) -> AsyncIterator[str]:
        blocked = self._admin_error(authorization)
        if blocked:
            yield blocked
            return
        yield f"Starting {label} ingestion"
        yield await operation()


class LocalA2ATester:
    """Exercise the co-located official A2A endpoint without external networking."""

    def __init__(self, app_provider: Callable[[], FastAPI], base_url: str) -> None:
        self._app_provider = app_provider
        self._base_url = base_url.rstrip("/")

    async def send(self, text: str, mode: ReasoningMode) -> tuple[str, str]:
        responses = [item async for item in self._send(text, mode, streaming=False)]
        task_id, state = _summarize_a2a_responses(responses)
        return task_id, state

    async def stream(self, text: str, mode: ReasoningMode) -> AsyncIterator[tuple[str, str]]:
        async for response in self._send(text, mode, streaming=True):
            yield _summarize_a2a_responses([response])

    async def cancel(self, task_id: str) -> str:
        import httpx
        from a2a.client import ClientConfig, ClientFactory
        from a2a.types import CancelTaskRequest

        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self._app_provider()), base_url=self._base_url)
        factory = ClientFactory(ClientConfig(httpx_client=http, supported_protocol_bindings=["JSONRPC"]))
        client = await factory.create_from_url(self._base_url)
        try:
            task = await client.cancel_task(CancelTaskRequest(id=task_id))
            return _task_state_name(task.status.state)
        finally:
            await client.close()
            await http.aclose()

    async def _send(self, text: str, mode: ReasoningMode, *, streaming: bool) -> AsyncIterator[Any]:
        import httpx
        from a2a.client import ClientConfig, ClientFactory
        from a2a.types import Message, Part, Role, SendMessageRequest

        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self._app_provider()), base_url=self._base_url)
        factory = ClientFactory(
            ClientConfig(httpx_client=http, supported_protocol_bindings=["JSONRPC"], streaming=streaming)
        )
        client = await factory.create_from_url(self._base_url)
        request = SendMessageRequest(
            message=Message(message_id=str(uuid4()), role=Role.ROLE_USER, parts=[Part(text=text)]),
            metadata={"reasoning_mode": mode.value},
        )
        try:
            async for response in client.send_message(request):
                yield response
        finally:
            await client.close()
            await http.aclose()


def build_ui(controller: UiController) -> gr.Blocks:
    """Construct exactly three tabs without opening external resources."""

    mode_choices = [(mode.value.replace("_", " ").title(), mode.value) for mode in ReasoningMode]
    with gr.Blocks(title="LangGraph Multi-Agent RAG", analytics_enabled=False) as blocks:
        gr.Markdown("# LangGraph Multi-Agent RAG\nEvidence-grounded answers with validated citations.")
        with gr.Tabs():
            with gr.Tab("Chat"):
                question = gr.Textbox(label="Question", lines=3, max_lines=8)
                mode = gr.Dropdown(mode_choices, value=ReasoningMode.STANDARD.value, label="Reasoning mode")
                source_types = gr.CheckboxGroup(
                    [(kind.value.upper(), kind.value) for kind in SourceType], label="Source types"
                )
                source_ids = gr.Textbox(label="Source IDs", placeholder="Comma or space separated UUIDs")
                thread_id = gr.Textbox(label="Thread ID", placeholder="Leave blank for a new conversation")
                with gr.Row():
                    new_thread = gr.Button("New thread")
                    reset_thread = gr.Button("Reset")
                    submit = gr.Button("Ask", variant="primary")
                    cancel = gr.Button("Cancel")
                status = gr.Textbox(label="Status", interactive=False)
                answer = gr.Textbox(label="Answer", lines=8, interactive=False)
                citations = gr.HTML(label="Validated citations")
                error = gr.Textbox(label="Error", interactive=False)
                run_key = gr.Textbox(visible=False)
                query_event = submit.click(
                    _chat_callback(controller),
                    inputs=[question, mode, source_types, source_ids, thread_id],
                    outputs=[status, answer, citations, thread_id, run_key, error],
                )
                cancel.click(controller.cancel_chat, inputs=[run_key], outputs=[error], cancels=[query_event])
                new_thread.click(controller.new_thread, outputs=[thread_id])
                reset_thread.click(
                    controller.reset_thread,
                    outputs=[thread_id, status, answer, citations, error],
                )

            with gr.Tab("Knowledge Base"):
                sources = gr.Dataframe(
                    headers=["Source ID", "Type", "Title", "Canonical URI", "Status"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    label="Indexed sources",
                )
                refresh = gr.Button("Refresh sources")
                refresh.click(controller.list_sources, outputs=[sources])
                admin_token = gr.Textbox(label="Admin token", type="password")
                mutation_result = gr.Textbox(label="Ingestion progress / result", interactive=False)
                upload = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
                upload_button = gr.Button("Ingest PDF")
                upload_button.click(controller.ingest_pdf_progress, [upload, admin_token], [mutation_result])
                web_url = gr.Textbox(label="Ingest web URL")
                gr.Button("Ingest web page").click(
                    controller.ingest_web_progress, [web_url, admin_token], [mutation_result]
                )
                github_url = gr.Textbox(label="Ingest GitHub repository")
                gr.Button("Ingest repository").click(
                    controller.ingest_github_progress, [github_url, admin_token], [mutation_result]
                )
                reindex_id = gr.Textbox(label="Reindex source ID")
                gr.Button("Reindex").click(controller.reindex, [reindex_id, admin_token], [mutation_result])
                delete_id = gr.Textbox(label="Delete source ID")
                gr.Button("Delete").click(controller.delete, [delete_id, admin_token], [mutation_result])

            with gr.Tab("System / A2A"):
                system_status = gr.Code(
                    value=controller.system_summary(), language="json", label="Safe system configuration"
                )
                gr.Button("Refresh system status").click(controller.system_summary, outputs=[system_status])
                gr.Code(value=controller.agent_card_summary(), language="json", label="Agent Card")
                base = str(controller.settings.public_base_url).rstrip("/")
                gr.Markdown(
                    f"Agent Card: `{base}/.well-known/agent-card.json`  \nA2A gateway: `{base}/a2a`"
                )
                a2a_message = gr.Textbox(label="A2A message")
                a2a_mode = gr.Dropdown(mode_choices, value=ReasoningMode.STANDARD.value, label="A2A reasoning mode")
                a2a_task_id = gr.Textbox(label="A2A task ID")
                a2a_result = gr.Textbox(label="A2A result / stream", interactive=False)
                with gr.Row():
                    gr.Button("Send A2A message").click(
                        controller.a2a_send,
                        [a2a_message, a2a_mode],
                        [a2a_task_id, a2a_result],
                    )
                    gr.Button("Stream A2A message").click(
                        controller.a2a_stream,
                        [a2a_message, a2a_mode],
                        [a2a_task_id, a2a_result],
                    )
                    gr.Button("Cancel A2A task").click(controller.a2a_cancel, [a2a_task_id], [a2a_result])
    return cast(gr.Blocks, blocks)


def mount_ui(
    app: FastAPI,
    *,
    settings: Settings,
    services: UiServices,
    authorizer: AdminAuthorizer,
    capacity: QueryCapacity,
) -> FastAPI:
    tester = LocalA2ATester(lambda: app, str(settings.public_base_url))
    controller = UiController(
        settings=settings,
        services=services,
        authorizer=authorizer,
        a2a_tester=tester,
        readiness_provider=lambda: dict(app.state.readiness),
        capacity=capacity,
    )
    app.state.ui_controller = controller
    return cast(FastAPI, gr.mount_gradio_app(app, build_ui(controller), path="/ui", show_error=False))


def _chat_callback(controller: UiController) -> Callable[..., AsyncIterator[tuple[str, str, str, str, str, str]]]:
    async def callback(
        question: str,
        mode: str,
        source_types: list[str] | None,
        source_ids: str,
        thread_id: str,
    ) -> AsyncIterator[tuple[str, str, str, str, str, str]]:
        async for update in controller.stream_chat(question, mode, source_types, source_ids, thread_id):
            yield (
                update.status,
                update.answer,
                update.citations_html,
                update.thread_id,
                update.run_key,
                update.error,
            )

    return callback


def _parse_source_ids(value: str) -> list[UUID] | None:
    tokens = [token for token in _SOURCE_ID_SEPARATOR.split(value.strip()) if token]
    return [UUID(token) for token in tokens] or None


def _safe_text(value: object) -> str:
    scrubbed = _UNSAFE_ATTRIBUTE.sub("blocked=", str(value))
    return html.escape(scrubbed, quote=True)


def _safe_link(value: object) -> str | None:
    raw = str(value)
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return html.escape(raw, quote=True)


def _render_citations(citations: list[Citation]) -> str:
    if not citations:
        return ""
    rendered: list[str] = ["<ol class=\"citations\">"]
    for citation in citations:
        title = _safe_text(citation.source_title)
        link = _safe_link(citation.source_url) if citation.source_url is not None else None
        label = f'<a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a>' if link else title
        locator = _safe_text(citation.locator)
        path = f" — {_safe_text(citation.source_path)}" if citation.source_path else ""
        excerpt = _safe_text(citation.supporting_excerpt)
        rendered.append(f"<li><strong>{label}</strong> ({locator}{path})<blockquote>{excerpt}</blockquote></li>")
    rendered.append("</ol>")
    return "".join(rendered)


def _ingestion_summary(result: Any) -> str:
    return f"{_safe_text(result.status)}:{_safe_text(result.source.id)}:{int(result.chunk_count or 0)} chunks"


def _source_status(metadata: dict[str, Any]) -> str:
    value = metadata.get("status", metadata.get("ingestion_status", ""))
    return _safe_text(value) if isinstance(value, str | int | float | bool) else ""


def _summarize_a2a_responses(responses: list[Any]) -> tuple[str, str]:
    task_id = ""
    state = "TASK_STATE_SUBMITTED"
    content: list[str] = []
    for response in responses:
        try:
            if response.HasField("task"):
                task = response.task
                task_id = str(task.id)
                state = _task_state_name(task.status.state)
                content.extend(_message_text(task.status.message) if task.status.HasField("message") else [])
                for artifact in task.artifacts:
                    content.extend(_parts_text(artifact.parts))
                for message in task.history:
                    if _is_agent_message(message):
                        content.extend(_message_text(message))
            elif response.HasField("message"):
                message = response.message
                task_id = str(message.task_id)
                if _is_agent_message(message):
                    content.extend(_message_text(message))
            elif response.HasField("status_update"):
                update = response.status_update
                task_id = str(update.task_id)
                state = _task_state_name(update.status.state)
                if update.status.HasField("message"):
                    content.extend(_message_text(update.status.message))
            elif response.HasField("artifact_update"):
                update = response.artifact_update
                task_id = str(update.task_id)
                content.extend(_parts_text(update.artifact.parts))
        except (AttributeError, ValueError):
            continue
    summary = "\n".join([state, *_unique_nonempty(content)])
    return task_id, summary


def _task_state_name(value: int) -> str:
    from a2a.types import TaskState

    try:
        return str(TaskState.Name(value))
    except ValueError:
        return "TASK_STATE_UNSPECIFIED"


def _is_agent_message(message: Any) -> bool:
    from a2a.types import Role

    return bool(message.role == Role.ROLE_AGENT)


def _message_text(message: Any) -> list[str]:
    return _parts_text(message.parts)


def _parts_text(parts: Any) -> list[str]:
    return [str(part.text) for part in parts if getattr(part, "text", "")]


def _unique_nonempty(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value.strip()))
