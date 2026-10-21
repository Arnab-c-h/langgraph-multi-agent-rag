"""Checkpoint-safe state shared by the bounded LangGraph workflows."""

from __future__ import annotations

from typing import NotRequired, TypedDict

JsonObject = dict[str, object]


class RecursiveItem(TypedDict):
    question: str
    depth: int


class GraphState(TypedDict):
    """Only public, serializable workflow artifacts may be stored here."""

    request: JsonObject
    request_fingerprint: str
    run_id: str
    conversation_history: NotRequired[list[str]]
    evidence: NotRequired[JsonObject]
    answer: NotRequired[str]
    citation_chunk_ids: NotRequired[list[str]]
    citations: NotRequired[list[JsonObject]]
    public_plan: NotRequired[list[str]]
    approaches: NotRequired[list[str]]
    branch_rankings: NotRequired[list[str]]
    tool_actions: NotRequired[list[str]]
    action_count: NotRequired[int]
    action_complete: NotRequired[bool]
    revision_count: NotRequired[int]
    critique_approved: NotRequired[bool]
    candidates: NotRequired[list[str]]
    candidate_citations: NotRequired[list[list[str]]]
    subquestions: NotRequired[list[str]]
    subanswers: NotRequired[list[str]]
    subquestion_index: NotRequired[int]
    recursive_queue: NotRequired[list[RecursiveItem]]
    recursive_nodes: NotRequired[list[RecursiveItem]]
    recursive_node_count: NotRequired[int]
    validation_passed: NotRequired[bool]
    trace_summary: NotRequired[dict[str, str | int | float | bool]]
    status: NotRequired[str]
