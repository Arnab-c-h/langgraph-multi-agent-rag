"""Authentication helpers for server-side administrative mutations."""

from __future__ import annotations

import hmac
from collections.abc import Callable


class AdminAuthorizer:
    """Validate a Bearer credential through a constant-time comparison seam."""

    def __init__(self, expected_token: str | None, comparer: Callable[[str, str], bool] = hmac.compare_digest) -> None:
        self._expected = (expected_token or "").strip()
        self._compare = comparer

    def authorize(self, authorization: str | None) -> tuple[bool, int]:
        received = _bearer_token(authorization)
        matches = self._compare(received, self._expected)
        return bool(self._expected and matches), 200 if matches and self._expected else 401


def _bearer_token(value: str | None) -> str:
    if not value:
        return ""
    scheme, separator, token = value.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return ""
    return token.strip()
