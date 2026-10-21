"""Shared recursive redaction for data crossing public or logging boundaries."""

import re
from typing import Any

_SECRET_FIELD_MARKERS = (
    "api_key",
    "authorization",
    "database_url",
    "password",
    "secret",
    "token",
)

_INLINE_SECRET_PATTERNS = (
    re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[^\s,;]+"),
    re.compile(r"(?i)\bpostgres(?:ql)?://[^\s]+"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"),
    re.compile(
        r"\b(?:AIza[A-Za-z0-9_-]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|sb_(?:secret|publishable)_[A-Za-z0-9_-]{16,})\b"
    ),
    re.compile(
        r"(?i)\b(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|token|password|secret)"
        r"\s*(?:[:=]|\bis\b)\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
    ),
)


def is_secret_field(key: str) -> bool:
    """Return whether a structured field name is likely to hold a credential."""
    normalized = key.lower().replace("-", "_")
    return any(marker in normalized for marker in _SECRET_FIELD_MARKERS)


def redact_inline_secrets(value: str) -> str:
    """Redact common inline credentials without censoring surrounding prose."""
    redacted = value
    for pattern in _INLINE_SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_secrets(value: Any) -> Any:
    """Recursively replace values paired with known secret-bearing field names."""
    if isinstance(value, dict):
        return {key: "[REDACTED]" if is_secret_field(key) else redact_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return redact_inline_secrets(value)
    return value
