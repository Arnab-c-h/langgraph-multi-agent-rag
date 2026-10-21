"""Typed public stream runner over the compiled LangGraph registry."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any, cast
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig

from rag_app.contracts import (
    Citation,
    CitationEvent,
    ErrorEvent,
    FinalEvent,
    QueryRequest,
    ReasoningMode,
    StatusEvent,
    StreamEvent,
    TokenEvent,
)
from rag_app.errors import AppError, ErrorCode
from rag_app.redaction import redact_inline_secrets
from rag_app.retrieval.models import EvidenceBundle
from rag_app.threads import ThreadStore, ThreadTurn
from rag_app.workflows.registry import WorkflowRegistry
from rag_app.workflows.state import GraphState


class _RunCancelled(Exception):
    """Cooperative cancellation that leaves the stream caller alive."""


class WorkflowRunner:
    def __init__(self, registry: WorkflowRegistry, thread_store: ThreadStore | None = None) -> None:
        self._registry = registry
        self._thread_store = thread_store

    async def stream(
        self,
        request: QueryRequest,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[StreamEvent]:
        if cancel_event is not None and cancel_event.is_set():
            yield _error(ErrorCode.RUN_CANCELLED)
            return
        thread_id = request.thread_id or uuid4()
        run_id = uuid4()
        graph = self._registry.graph_for(request.reasoning_mode)
        config = _run_config(thread_id, request.reasoning_mode, run_id)
        iterator: AsyncIterator[dict[str, Any]] | None = None
        try:
            prior_turns = (
                await self._thread_store.load(thread_id) if self._thread_store is not None and request.thread_id else []
            )
            history = [turn.public_context() for turn in prior_turns]
            state: GraphState = _initial_state(request, run_id, history)
            graph_input: GraphState | None = state
            if request.thread_id is not None and self._registry.has_checkpointer:
                snapshot = await graph.aget_state(config)
                if snapshot.next and snapshot.values.get("request_fingerprint") == state["request_fingerprint"]:
                    state = cast(GraphState, dict(snapshot.values))
                    graph_input = None
                    run_id = UUID(state["run_id"])
                    config = _run_config(thread_id, request.reasoning_mode, run_id)
            iterator = graph.astream(graph_input, config=config, stream_mode="updates").__aiter__()
            while True:
                update = await _next_or_cancel(iterator, cancel_event)
                if update is None:
                    break
                for node_update in update.values():
                    if not isinstance(node_update, dict):
                        continue
                    state.update(cast(GraphState, node_update))
                    status = node_update.get("status")
                    if isinstance(status, str) and status:
                        yield StatusEvent(message=status)
            answer = state.get("answer", "")
            citations = [Citation.model_validate(item) for item in state.get("citations", [])]
            evidence = EvidenceBundle.model_validate(state["evidence"])
            if not state.get("validation_passed", False) or not answer or not citations:
                raise AppError(ErrorCode.NO_SUPPORTING_EVIDENCE, "Incomplete validated answer")
            for token in _tokens(answer):
                yield TokenEvent(token=token)
            for citation in citations:
                yield CitationEvent(citation=citation)
            if self._thread_store is not None:
                await self._thread_store.append(
                    thread_id,
                    ThreadTurn(
                        question=_public_question(request.question),
                        answer=answer,
                        mode=request.reasoning_mode,
                        run_id=run_id,
                    ),
                )
            yield FinalEvent(
                answer=answer,
                citations=citations,
                thread_id=thread_id,
                mode=request.reasoning_mode,
                run_id=run_id,
                used_source_ids=evidence.used_source_ids,
                trace_summary=state.get("trace_summary", {}),
            )
        except _RunCancelled:
            yield _error(ErrorCode.RUN_CANCELLED)
        except asyncio.CancelledError:
            raise
        except AppError as exc:
            yield _error(exc.code)
        except Exception:
            yield _error(ErrorCode.RUN_FAILED)
        finally:
            if iterator is not None:
                await cast(Any, iterator).aclose()


def _public_question(question: str) -> str:
    return redact_inline_secrets(" ".join(question.split()))


def _request_fingerprint(request: QueryRequest) -> str:
    identity = request.model_dump(
        mode="json",
        include={"question", "reasoning_mode", "top_k", "source_types", "source_ids"},
    )
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _initial_state(request: QueryRequest, run_id: UUID, history: list[str] | None = None) -> GraphState:
    public_request = request.model_copy(update={"question": _public_question(request.question)})
    return {
        "request": public_request.model_dump(mode="json"),
        "request_fingerprint": _request_fingerprint(public_request),
        "run_id": str(run_id),
        "conversation_history": list(history or [])[-10:],
        "answer": "",
        "citation_chunk_ids": [],
        "citations": [],
        "public_plan": [],
        "approaches": [],
        "branch_rankings": [],
        "tool_actions": [],
        "action_count": 0,
        "action_complete": False,
        "revision_count": 0,
        "critique_approved": False,
        "candidates": [],
        "candidate_citations": [],
        "subquestions": [],
        "subanswers": [],
        "subquestion_index": 0,
        "recursive_queue": [],
        "recursive_nodes": [],
        "recursive_node_count": 0,
        "validation_passed": False,
        "trace_summary": {},
        "status": "",
    }


def _run_config(thread_id: UUID, mode: ReasoningMode, run_id: UUID) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": _storage_thread_id(thread_id, mode),
            "checkpoint_ns": "",
            "workflow_mode": mode.value,
        },
        "run_id": run_id,
    }


def _storage_thread_id(thread_id: UUID, mode: ReasoningMode) -> str:
    return f"{thread_id}:{mode.value}"


async def _next_or_cancel(
    iterator: AsyncIterator[dict[str, Any]], cancel_event: asyncio.Event | None
) -> dict[str, Any] | None:
    async def get_next() -> dict[str, Any]:
        return await anext(iterator)

    next_task: asyncio.Task[dict[str, Any]] = asyncio.create_task(get_next())
    cancel_task = asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None
    children = (next_task,) if cancel_task is None else (next_task, cancel_task)
    try:
        if cancel_task is None:
            try:
                return await next_task
            except StopAsyncIteration:
                return None
        done, _ = await asyncio.wait(children, return_when=asyncio.FIRST_COMPLETED)
        if cancel_task in done and cancel_event is not None and cancel_event.is_set():
            raise _RunCancelled
        try:
            return next_task.result()
        except StopAsyncIteration:
            return None
    finally:
        for child in children:
            if not child.done():
                child.cancel()
        await asyncio.gather(*children, return_exceptions=True)


def _tokens(answer: str) -> list[str]:
    words = answer.split(" ")
    return [word if index == len(words) - 1 else f"{word} " for index, word in enumerate(words) if word]


def _error(code: ErrorCode) -> ErrorEvent:
    payload = AppError(code, "safe").to_payload()
    return ErrorEvent(code=code, message=str(payload["message"]))
