"""Bounded, secret-safe public conversation history persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from rag_app.contracts import ReasoningMode
from rag_app.redaction import redact_inline_secrets
from rag_app.storage.sqlite import SqliteDatabase

MAX_THREAD_TURNS = 10


class ThreadTurn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=16000)
    mode: ReasoningMode
    run_id: UUID

    def safe(self) -> ThreadTurn:
        return self.model_copy(
            update={
                "question": _safe_text(self.question, 4000),
                "answer": _safe_text(self.answer, 16000),
            }
        )

    def public_context(self) -> str:
        return f"Previous user question: {self.question}\nPrevious grounded answer: {self.answer}"


class ThreadStore(Protocol):
    async def load(self, thread_id: UUID, *, limit: int = MAX_THREAD_TURNS) -> list[ThreadTurn]: ...

    async def append(self, thread_id: UUID, turn: ThreadTurn) -> None: ...


class SqliteThreadStore:
    """Store bounded, redacted turns in the local SQLite thread table."""

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    async def load(self, thread_id: UUID, *, limit: int = MAX_THREAD_TURNS) -> list[ThreadTurn]:
        bounded = max(0, min(limit, MAX_THREAD_TURNS))
        if bounded == 0:
            return []
        row = await self._database.fetchone("SELECT metadata FROM threads WHERE id = ?", (str(thread_id),))
        metadata = json.loads(row[0]) if row is not None else {}
        return _parse_turns(metadata)[-bounded:]

    async def append(self, thread_id: UUID, turn: ThreadTurn) -> None:
        safe_turn = turn.safe()

        async def operation(connection: Any) -> None:
            cursor = await connection.execute("SELECT metadata FROM threads WHERE id = ?", (str(thread_id),))
            try:
                row = await cursor.fetchone()
            finally:
                await cursor.close()
            metadata = json.loads(row[0]) if row is not None else {}
            turns = _parse_turns(metadata)[-(MAX_THREAD_TURNS - 1) :]
            turns.append(safe_turn)
            serialized = json.dumps({"turns": [item.model_dump(mode="json") for item in turns]})
            now = datetime.now(UTC).isoformat()
            await connection.execute(
                "INSERT INTO threads (id, title, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET title = excluded.title, metadata = excluded.metadata, "
                "updated_at = excluded.updated_at",
                (str(thread_id), safe_turn.question[:200], serialized, now, now),
            )

        await self._database.write(operation)


def _parse_turns(metadata: object) -> list[ThreadTurn]:
    raw_turns = metadata.get("turns", []) if isinstance(metadata, dict) else []
    turns: list[ThreadTurn] = []
    if not isinstance(raw_turns, list):
        return turns
    for item in raw_turns:
        try:
            turns.append(ThreadTurn.model_validate(item).safe())
        except ValidationError:
            continue
    return turns


def _safe_text(value: str, maximum: int) -> str:
    return redact_inline_secrets(" ".join(value.split()))[:maximum]
