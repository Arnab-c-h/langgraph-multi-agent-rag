"""Guards that prevent smoke tests from using ordinary production variables."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import pytest


def require_opt_in(flag: str) -> None:
    if os.getenv(flag) != "1":
        pytest.skip(f"set {flag}=1 to explicitly enable this live smoke test")


def require_smoke_secret(flag: str, variable: str) -> str:
    require_opt_in(flag)
    value = os.getenv(variable, "").strip()
    if not value:
        pytest.fail(f"{variable} is required after opting in with {flag}=1")
    return value


def require_loopback_a2a_url() -> str:
    url = require_smoke_secret("RAG_RUN_A2A_SMOKE", "RAG_SMOKE_A2A_BASE_URL")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("A2A smoke URL must be HTTP(S) on a loopback host")
    return url.rstrip("/")
