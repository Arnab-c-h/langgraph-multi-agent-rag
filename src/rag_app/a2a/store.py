"""Official A2A task storage over the shared local SQLite database."""

from __future__ import annotations

from pathlib import Path

from a2a.server.tasks import DatabaseTaskStore
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def create_sqlite_task_store(data_dir: Path) -> tuple[DatabaseTaskStore, AsyncEngine]:
    """Build the official A2A task store over the shared local SQLite database."""
    root = data_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    database_path = (root / "rag.sqlite3").as_posix()
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}", connect_args={"timeout": 30})
    return DatabaseTaskStore(engine=engine, create_table=True, table_name="tasks"), engine


async def dispose_task_store_engine(engine: AsyncEngine) -> None:
    """Close SQLAlchemy pools deterministically at application shutdown."""
    await engine.dispose()
