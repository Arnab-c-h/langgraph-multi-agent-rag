"""Bounded retry/timeout helper shared by REST and A2A mutations."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from rag_app.errors import AppError, ErrorCode


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    retries: int,
    timeout_seconds: float,
    cancel_event: asyncio.Event | None = None,
) -> T:
    for attempt in range(retries + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        try:
            async with asyncio.timeout(timeout_seconds):
                return await _cancel_aware(operation(), cancel_event)
        except TimeoutError as exc:
            raise AppError(ErrorCode.RUN_FAILED, "Dependency timeout") from exc
        except AppError as exc:
            if exc.code is not ErrorCode.PROVIDER_RATE_LIMITED or attempt >= retries:
                raise
            await asyncio.sleep(random.SystemRandom().uniform(0.01, 0.05) * (2**attempt))
    raise RuntimeError("unreachable")


async def _cancel_aware[T](operation: Awaitable[T], cancel_event: asyncio.Event | None) -> T:
    operation_task = asyncio.ensure_future(operation)
    if cancel_event is None:
        return await operation_task
    cancellation_task = asyncio.create_task(cancel_event.wait())
    try:
        done, _pending = await asyncio.wait({operation_task, cancellation_task}, return_when=asyncio.FIRST_COMPLETED)
        if cancellation_task in done and cancel_event.is_set():
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            raise asyncio.CancelledError
        return await operation_task
    except asyncio.CancelledError:
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        raise
    finally:
        cancellation_task.cancel()
        await asyncio.gather(cancellation_task, return_exceptions=True)
