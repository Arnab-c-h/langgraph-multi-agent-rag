"""Structured logging helpers that suppress credential values."""

from __future__ import annotations

import json
import logging
from typing import Any

from rag_app.redaction import redact_secrets


class JsonFormatter(logging.Formatter):
    """Encode logs as JSON and include only explicitly structured extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {"level": record.levelname, "message": record.getMessage(), "logger": record.name}
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload["fields"] = redact_secrets(fields)
        return json.dumps(redact_secrets(payload), default=str, sort_keys=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Install JSON logging for the root application logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
