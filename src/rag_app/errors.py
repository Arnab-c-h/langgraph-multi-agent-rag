"""Stable application errors that are safe to serialize at public boundaries."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from rag_app.redaction import redact_secrets


class ErrorCode(StrEnum):
    NO_SUPPORTING_EVIDENCE = "NO_SUPPORTING_EVIDENCE"
    INVALID_SOURCE = "INVALID_SOURCE"
    INGESTION_FAILED = "INGESTION_FAILED"
    PROVIDER_RATE_LIMITED = "PROVIDER_RATE_LIMITED"
    RUN_CANCELLED = "RUN_CANCELLED"
    RUN_FAILED = "RUN_FAILED"


_PUBLIC_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.NO_SUPPORTING_EVIDENCE: "No supporting evidence was found.",
    ErrorCode.INVALID_SOURCE: "The source is invalid or unavailable.",
    ErrorCode.INGESTION_FAILED: "The source could not be ingested.",
    ErrorCode.PROVIDER_RATE_LIMITED: "Provider request could not be completed.",
    ErrorCode.RUN_CANCELLED: "The run was cancelled.",
    ErrorCode.RUN_FAILED: "The run could not be completed.",
}


class AppError(Exception):
    """An expected error with a stable public code and secret-safe payload."""

    def __init__(self, code: ErrorCode, message: str, detail: dict[str, Any] | None = None) -> None:
        del message
        super().__init__(_PUBLIC_MESSAGES[code])
        self.code = code
        self.detail = redact_secrets(detail or {})

    def to_payload(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": _PUBLIC_MESSAGES[self.code], "detail": redact_secrets(self.detail)}
