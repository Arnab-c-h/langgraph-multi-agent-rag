"""Public, typed request, response, and stream contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator

from rag_app.errors import ErrorCode


class ReasoningMode(StrEnum):
    STANDARD = "standard"
    COT = "cot"
    TOT = "tot"
    REACT = "react"
    SELF_REFLECTION = "self_reflection"
    SELF_CONSISTENCY = "self_consistency"
    DECOMPOSED = "decomposed"
    LEAST_TO_MOST = "least_to_most"
    RECURSIVE = "recursive"


class SourceType(StrEnum):
    PDF = "pdf"
    WEB = "web"
    GITHUB = "github"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    reasoning_mode: ReasoningMode = ReasoningMode.STANDARD
    thread_id: UUID | None = None
    top_k: int = Field(default=6, ge=1, le=20)
    source_types: list[SourceType] | None = None
    source_ids: list[UUID] | None = None

    @field_validator("question")
    @classmethod
    def require_non_whitespace_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class Citation(BaseModel):
    source_id: UUID
    chunk_id: UUID
    source_title: str = Field(min_length=1, max_length=512)
    locator: str = Field(min_length=1, max_length=512)
    source_url: HttpUrl | None = None
    source_path: str | None = Field(default=None, max_length=4096)
    supporting_excerpt: str = Field(default="", max_length=4000)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    thread_id: UUID | None = None
    mode: ReasoningMode = ReasoningMode.STANDARD
    run_id: UUID | None = None
    used_source_ids: list[UUID] = Field(default_factory=list)
    trace_summary: dict[str, str | int | float | bool] = Field(default_factory=dict)


class StatusEvent(BaseModel):
    type: Literal["status"] = "status"
    message: str = Field(min_length=1, max_length=512)


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    token: str = Field(min_length=1)


class CitationEvent(BaseModel):
    type: Literal["citation"] = "citation"
    citation: Citation


class FinalEvent(BaseModel):
    type: Literal["final"] = "final"
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    thread_id: UUID | None = None
    mode: ReasoningMode = ReasoningMode.STANDARD
    run_id: UUID | None = None
    used_source_ids: list[UUID] = Field(default_factory=list)
    trace_summary: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: ErrorCode
    message: str = Field(min_length=1, max_length=512)


StreamEvent = Annotated[
    StatusEvent | TokenEvent | CitationEvent | FinalEvent | ErrorEvent,
    Field(discriminator="type"),
]
