"""Small in-process resource guards shared by REST and A2A."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import defaultdict, deque
from typing import cast
from uuid import UUID, uuid4

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_logger = logging.getLogger("rag_app.http")


def public_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status_code)


class RequestSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http" and scope.get("method") in {"POST", "PUT", "PATCH"}:
            headers = {key.lower(): value for key, value in scope.get("headers", [])}
            raw_length = headers.get(b"content-length")
            if raw_length:
                try:
                    too_large = int(raw_length) > self.max_bytes
                except ValueError:
                    too_large = True
                if too_large:
                    response = public_error(413, "REQUEST_TOO_LARGE", "The request exceeds the byte limit.")
                    await response(scope, receive, send)
                    return
            buffered: list[dict[str, object]] = []
            total = 0
            while True:
                message = await receive()
                buffered.append(cast(dict[str, object], message))
                if message["type"] == "http.disconnect":
                    break
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    response = public_error(413, "REQUEST_TOO_LARGE", "The request exceeds the byte limit.")
                    await response(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
            index = 0

            async def replay() -> dict[str, object]:
                nonlocal index
                if index < len(buffered):
                    message = buffered[index]
                    index += 1
                    return message
                return cast(dict[str, object], await receive())

            await self.app(scope, cast(Receive, replay), send)
            return
        await self.app(scope, receive, send)


class PerIpRateLimitMiddleware:
    def __init__(self, app: ASGIApp, limit: int, trust_proxy_headers: bool = False) -> None:
        self.app = app
        self.limit = limit
        self.trust_proxy_headers = trust_proxy_headers
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = str(scope.get("path", ""))
        if scope.get("type") == "http" and (
            path.startswith("/api/v1/query") or path.startswith("/api/v1/admin/") or path == "/a2a"
        ):
            ip = self._client_ip(scope)
            if not await self._allow(ip):
                response = public_error(429, "RATE_LIMITED", "Too many requests.")
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

    async def _allow(self, ip: str) -> bool:
        now = time.monotonic()
        cutoff = now - 60
        async with self._lock:
            entries = self._requests[ip]
            while entries and entries[0] <= cutoff:
                entries.popleft()
            if len(entries) >= self.limit:
                return False
            entries.append(now)
            return True

    def _client_ip(self, scope: Scope) -> str:
        if self.trust_proxy_headers:
            headers = {key.lower(): value for key, value in scope.get("headers", [])}
            forwarded = headers.get(b"x-forwarded-for", b"").decode("latin-1").split(",", 1)[0].strip()
            if forwarded:
                return cast(str, forwarded)
        client = scope.get("client")
        if isinstance(client, tuple) and client:
            host = client[0]
            return host if isinstance(host, str) else str(host)
        return "unknown"


class QueryCapacity:
    def __init__(self, maximum: int) -> None:
        self._semaphore = asyncio.Semaphore(maximum)

    def slot(self) -> asyncio.Semaphore:
        return self._semaphore


class RequestIdMiddleware:
    """Attach a safe correlation ID and log only bounded request metadata."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        supplied = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied if _REQUEST_ID.fullmatch(supplied) else str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        status = 500
        response_started = False

        async def with_id(message: Message) -> None:
            nonlocal response_started, status
            if message["type"] == "http.response.start":
                response_started = True
                status = int(message["status"])
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = response_headers
            await send(message)

        try:
            await self.app(scope, receive, with_id)
        except Exception:
            if not response_started:
                response = public_error(500, "RUN_FAILED", "The run could not be completed.")
                await response(scope, receive, with_id)
        finally:
            fields: dict[str, str | int] = {
                "request_id": request_id,
                "method": str(scope.get("method", "")),
                "path": str(scope.get("path", "")),
                "status": status,
            }
            run_id = _safe_run_id(scope.get("state", {}).get("run_id"))
            if run_id is not None:
                fields["run_id"] = run_id
            _logger.info(
                "request_complete",
                extra={"fields": fields},
            )


def _safe_run_id(value: object) -> str | None:
    try:
        return str(UUID(str(value)))
    except (AttributeError, TypeError, ValueError):
        return None
