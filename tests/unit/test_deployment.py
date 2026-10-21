from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_dockerfile_builds_one_non_root_full_image_with_persistent_local_data() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert sum(line.startswith("FROM ") for line in dockerfile.splitlines()) == 1
    assert "--extra full" in dockerfile
    assert "uv sync --frozen --no-dev --no-editable --extra full" in dockerfile
    assert "ENV APP_PROFILE=full" in dockerfile
    assert "RAG_DATA_DIR=/data" in dockerfile
    assert 'VOLUME ["/data"]' in dockerfile
    assert "USER app" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    command_line = next(line for line in dockerfile.splitlines() if line.startswith("CMD ["))
    command = json.loads(command_line.removeprefix("CMD "))
    assert command[command.index("--host") + 1] == "0.0.0.0"
    assert command[command.index("--workers") + 1] == "1"


def test_dependencies_keep_http_a2a_and_full_ingestion_without_cloud_database_extras() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "a2a-sdk[http-server]" in pyproject
    assert "hosted = []" not in pyproject
    for forbidden in ("langchain-postgres", "langgraph-checkpoint-postgres", "psycopg", "postgresql"):
        assert forbidden not in pyproject.lower()
    full = pyproject.split("full = [", 1)[1].split("]", 1)[0].lower()
    assert "docling" in full and "pygithub" in full and "trafilatura" in full


def test_example_environment_parses_as_local_full_without_database_or_cloud_values() -> None:
    values = {
        key: value
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for key, value in [line.split("=", 1)]
    }

    assert values["APP_PROFILE"] == "full"
    assert values["RAG_DATA_DIR"] == ""
    assert values["PUBLIC_BASE_URL"] == "http://127.0.0.1:8000"
    assert {"GEMINI_API_KEY", "ADMIN_TOKEN"} <= values.keys()
    assert not any("DATABASE" in name or "SUPABASE" in name or "RENDER" in name for name in values)


def _run_start_script(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    script_dir = tmp_path / "scripts"
    python_dir = tmp_path / ".venv" / "Scripts"
    script_dir.mkdir(parents=True)
    python_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "start.ps1", script_dir / "start.ps1")
    shutil.copy2(Path(sys.executable), python_dir / "python.exe")
    shutil.copy2(Path(sys.prefix) / "pyvenv.cfg", python_dir.parent / "pyvenv.cfg")
    (tmp_path / "uvicorn.py").write_text(
        "import json, os, sys\nprint(json.dumps({'args': sys.argv[1:], 'profile': os.getenv('APP_PROFILE')}))\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_dir / "start.ps1"),
            *arguments,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_start_script_defaults_to_loopback_and_accepts_an_explicit_host(tmp_path: Path) -> None:
    default = _run_start_script(tmp_path / "default")
    assert default.returncode == 0, default.stderr
    default_payload = json.loads(default.stdout.splitlines()[-1])
    assert default_payload["profile"] == "full"
    assert default_payload["args"][default_payload["args"].index("--host") + 1] == "127.0.0.1"

    explicit = _run_start_script(tmp_path / "explicit", "-Host", "0.0.0.0", "-Profile", "test")
    assert explicit.returncode == 0, explicit.stderr
    explicit_payload = json.loads(explicit.stdout.splitlines()[-1])
    assert explicit_payload["profile"] == "test"
    assert explicit_payload["args"][explicit_payload["args"].index("--host") + 1] == "0.0.0.0"


def test_start_script_rejects_removed_hosted_profile(tmp_path: Path) -> None:
    result = _run_start_script(tmp_path, "-Profile", "hosted")
    assert result.returncode != 0


def test_asgi_import_is_inert_for_supported_profiles() -> None:
    code = """
import builtins
original = builtins.__import__
blocked = ('docling', 'github', 'trafilatura')
def guarded(name, *args, **kwargs):
    if name in blocked or name.startswith(tuple(value + '.' for value in blocked)):
        raise AssertionError(f'import touched optional package: {name}')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import rag_app.main
assert rag_app.main.app.state.settings.app_profile in {'full', 'test'}
"""
    environment = os.environ.copy()
    environment["APP_PROFILE"] = "full"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
