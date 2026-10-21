from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.verify_repository import find_repository_secret_hits, find_secret_hits

ROOT = Path(__file__).parents[2]
_DATABASE_URL_KEY = "DATABASE" + "_URL"
_DIRECT_DATABASE_URL_KEY = "DIRECT_" + _DATABASE_URL_KEY
_HOSTED_PROFILE_SETTING = "APP_PROFILE=" + "hosted"


def test_secret_scanner_reports_kind_and_location_without_secret_text(tmp_path: Path) -> None:
    secret = "AIza" + "A" * 28
    candidate = tmp_path / "candidate.txt"
    candidate.write_text(f"value={secret}\n", encoding="utf-8")

    hits = find_secret_hits([candidate])

    assert [(hit.path, hit.line, hit.kind) for hit in hits] == [(candidate, 1, "google-api-key")]
    assert secret not in repr(hits)


def test_secret_scanner_allows_documented_placeholders_and_test_dsns(tmp_path: Path) -> None:
    candidate = tmp_path / "safe.example"
    candidate.write_text(
        f"{_DATABASE_URL_KEY}=postgresql://USER:PASSWORD@SESSION_POOLER_HOST:5432/postgres\n"
        "RAG_TEST_DATABASE_URL=postgresql://user:password@localhost/rag_test\n",
        encoding="utf-8",
    )

    assert find_secret_hits([candidate]) == []


def _run_verifier(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_repository.py"), "all", "--root", str(root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_complete_repository(root: Path) -> None:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "Dockerfile").write_text(
        "\n".join(
            (
                "FROM python:3.12-slim",
                "RUN uv sync --frozen --no-dev --no-editable --extra full",
                "ENV RAG_DATA_DIR=/data",
                'VOLUME ["/data"]',
                "USER app",
                'CMD ["uvicorn", "rag_app.main:app", "--workers", "1"]',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (root / ".env.example").write_text("APP_PROFILE=full\nRAG_DATA_DIR=\n", encoding="utf-8")
    (root / "scripts" / "start.ps1").write_text("$env:APP_PROFILE = 'full'\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "\n".join(
            (
                "[project]",
                'name = "fixture-rag"',
                'version = "0.1.0"',
                'dependencies = ["a2a-sdk[http-server]>=0.2.0", "aiosqlite>=0.20.0"]',
                "",
                "[project.optional-dependencies]",
                'full = ["docling>=2.0.0"]',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        'version = 1\nrevision = 3\nrequires-python = ">=3.12"\n',
        encoding="utf-8",
    )


def test_repository_verifier_accepts_the_current_local_only_repository() -> None:
    result = _run_verifier(ROOT)

    assert result.returncode == 0, result.stdout + result.stderr


def test_repository_verifier_accepts_a_complete_supplied_fixture_root(tmp_path: Path) -> None:
    _write_complete_repository(tmp_path)

    result = _run_verifier(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr


def test_repository_verifier_reports_every_missing_required_artifact(tmp_path: Path) -> None:
    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert result.stdout.splitlines() == [
        "repository invariant failed: missing Dockerfile",
        "repository invariant failed: missing .env.example",
        "repository invariant failed: missing scripts/start.ps1",
        "repository invariant failed: missing pyproject.toml",
    ]


@pytest.mark.parametrize(
    ("relative_path", "content", "expected"),
    [
        (
            "src/rag_app/legacy.py",
            "from langchain_postgres import PGVectorStore\n",
            "forbidden Python import 'langchain_postgres'",
        ),
        (
            "src/rag_app/legacy.py",
            "from rag_app.storage.postgres import SourceRepository\n",
            "forbidden Python import 'rag_app.storage.postgres'",
        ),
        ("src/rag_app/legacy.py", "import supabase\n", "forbidden Python import 'supabase'"),
        ("src/rag_app/legacy.py", "import pgvector\n", "forbidden Python import 'pgvector'"),
        ("scripts/legacy_job.py", "import asyncpg\n", "forbidden Python import 'asyncpg'"),
        (
            "tests/integration/test_legacy.py",
            "import psycopg\n",
            "forbidden Python import 'psycopg'",
        ),
        (
            "migrations/001_legacy.sql",
            "CREATE EXTENSION IF NOT EXISTS vector;\n",
            "forbidden Postgres migration",
        ),
        (
            "pyproject.toml",
            "[project]\nname='x'\nversion='0.1.0'\ndependencies=['asyncpg>=0.30']\n"
            "[project.optional-dependencies]\nfull=['docling>=2']\n",
            "forbidden project dependency 'asyncpg'",
        ),
        (
            "pyproject.toml",
            "[project]\nname='x'\nversion='0.1.0'\ndependencies=['a2a-sdk[http-server,postgresql]>=0.2']\n"
            "[project.optional-dependencies]\nfull=['docling>=2']\n",
            "forbidden A2A extra 'postgresql'",
        ),
        (
            "uv.lock",
            "version=1\nrevision=3\nrequires-python='>=3.12'\n"
            "[[package]]\nname='langgraph-checkpoint-postgres'\nversion='3.1.2'\n",
            "forbidden lock dependency 'langgraph-checkpoint-postgres'",
        ),
        (
            ".env.example",
            f"{_DATABASE_URL_KEY}=\nRAG_DATA_DIR=\n",
            "forbidden database URL configuration",
        ),
        (
            ".env.example",
            f"{_DIRECT_DATABASE_URL_KEY}=\nRAG_DATA_DIR=\n",
            "forbidden database URL configuration",
        ),
        (
            ".env.example",
            f"{_HOSTED_PROFILE_SETTING}\nRAG_DATA_DIR=\n",
            "forbidden hosted application profile",
        ),
        ("render.yaml", "services: []\n", "forbidden Render manifest"),
        ("supabase/config.toml", "project_id = 'legacy'\n", "forbidden Supabase runtime directory"),
    ],
)
def test_repository_verifier_rejects_forbidden_runtime_configuration_in_supplied_root(
    tmp_path: Path, relative_path: str, content: str, expected: str
) -> None:
    _write_complete_repository(tmp_path)
    candidate = tmp_path / relative_path
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(content, encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert expected in result.stdout


@pytest.mark.parametrize(
    ("relative_path", "content", "expected"),
    [
        (
            "pyproject.toml",
            "[project]\nname='x'\nversion='0.1.0'\ndependencies=['supabase>=2']\n"
            "[project.optional-dependencies]\nfull=['docling>=2']\n",
            "forbidden project dependency 'supabase'",
        ),
        (
            "pyproject.toml",
            "[project]\nname='x'\nversion='0.1.0'\ndependencies=['supabase_py>=2']\n"
            "[project.optional-dependencies]\nfull=['docling>=2']\n",
            "forbidden project dependency 'supabase-py'",
        ),
        (
            "pyproject.toml",
            "[project]\nname='x'\nversion='0.1.0'\ndependencies=['postgrest>=1']\n"
            "[project.optional-dependencies]\nfull=['docling>=2']\n",
            "forbidden project dependency 'postgrest'",
        ),
        (
            "uv.lock",
            "version=1\nrevision=3\nrequires-python='>=3.12'\n"
            "[[package]]\nname='supabase'\nversion='2.0.0'\n",
            "forbidden lock dependency 'supabase'",
        ),
        (
            "uv.lock",
            "version=1\nrevision=3\nrequires-python='>=3.12'\n"
            "[[package]]\nname='Supabase_Py'\nversion='2.0.0'\n",
            "forbidden lock dependency 'supabase-py'",
        ),
        (
            "uv.lock",
            "version=1\nrevision=3\nrequires-python='>=3.12'\n"
            "[[package]]\nname='postgrest'\nversion='1.0.0'\n",
            "forbidden lock dependency 'postgrest'",
        ),
    ],
)
def test_repository_verifier_rejects_declared_supabase_distributions(
    tmp_path: Path, relative_path: str, content: str, expected: str
) -> None:
    _write_complete_repository(tmp_path)
    (tmp_path / relative_path).write_text(content, encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert expected in result.stdout


def test_repository_verifier_allows_removed_dependency_names_in_ordinary_prose(tmp_path: Path) -> None:
    _write_complete_repository(tmp_path)
    note = tmp_path / "docs" / "removed-cloud-runtime.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "The supabase-py, supabase, and postgrest dependencies were removed.\n",
        encoding="utf-8",
    )

    result = _run_verifier(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "evidence_name",
    ["review-base..head.diff", "final-review-base..head.diff"],
)
def test_repository_verifier_allows_named_sdd_review_evidence(tmp_path: Path, evidence_name: str) -> None:
    _write_complete_repository(tmp_path)
    evidence = tmp_path / ".superpowers" / "sdd" / "fixture" / evidence_name
    evidence.parent.mkdir(parents=True)
    evidence.write_text(f"-{_HOSTED_PROFILE_SETTING}\n", encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "relative_path",
    ["artifacts/review-base..head.diff", "artifacts/final-review-base..head.diff"],
)
def test_repository_verifier_scans_review_named_files_outside_sdd(
    tmp_path: Path, relative_path: str
) -> None:
    _write_complete_repository(tmp_path)
    active_file = tmp_path / relative_path
    active_file.parent.mkdir(parents=True)
    active_file.write_text(f"{_HOSTED_PROFILE_SETTING}\n", encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert f"forbidden hosted application profile: {Path(relative_path)}:1" in result.stdout


def test_repository_verifier_scans_semantic_imports_in_its_own_supplied_path(tmp_path: Path) -> None:
    _write_complete_repository(tmp_path)
    verifier = tmp_path / "scripts" / "verify_repository.py"
    verifier.write_text("import asyncpg\n", encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert f"forbidden Python import 'asyncpg': {Path('scripts/verify_repository.py')}" in result.stdout


def test_repository_verifier_scans_runtime_text_in_formerly_exempt_active_test(tmp_path: Path) -> None:
    _write_complete_repository(tmp_path)
    active_test = tmp_path / "tests" / "unit" / "test_deployment.py"
    active_test.parent.mkdir(parents=True)
    active_test.write_text(f"{_HOSTED_PROFILE_SETTING}\n", encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert (
        f"forbidden hosted application profile: {Path('tests/unit/test_deployment.py')}:1"
        in result.stdout
    )


def _removed_profile_rejection_source() -> str:
    return (
        "def test_rejects_removed_profile():\n"
        "    with pytest.raises(ValidationError):\n"
        f"        Settings(app_profile={_HOSTED_PROFILE_SETTING.removeprefix('APP_PROFILE=')!r})\n"
    )


def test_repository_verifier_allows_foundation_negative_profile_validation(tmp_path: Path) -> None:
    _write_complete_repository(tmp_path)
    negative_test = tmp_path / "tests" / "unit" / "test_foundation.py"
    negative_test.parent.mkdir(parents=True)
    negative_test.write_text(_removed_profile_rejection_source(), encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "relative_path",
    ["src/rag_app/runtime.py", "tests/unit/test_profile.py"],
)
def test_repository_verifier_rejects_profile_validation_shape_outside_foundation_test(
    tmp_path: Path, relative_path: str
) -> None:
    _write_complete_repository(tmp_path)
    candidate = tmp_path / relative_path
    candidate.parent.mkdir(parents=True)
    candidate.write_text(_removed_profile_rejection_source(), encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert f"forbidden hosted application profile: {Path(relative_path)}:3" in result.stdout


@pytest.mark.parametrize(
    "postgres_sql",
    [
        "CREATE EXTENSION hstore;\n",
        "CREATE TABLE jobs (id uuid DEFAULT gen_random_uuid());\n",
        "CREATE TABLE events (payload jsonb NOT NULL);\n",
        "CREATE TABLE events (created_at timestamptz NOT NULL);\n",
        "CREATE SCHEMA archive;\n",
        "CREATE TABLE jobs (id serial PRIMARY KEY);\n",
        "CREATE TABLE jobs (id bigserial PRIMARY KEY);\n",
    ],
)
def test_repository_verifier_rejects_postgres_dialect_in_any_migration_path(
    tmp_path: Path, postgres_sql: str
) -> None:
    _write_complete_repository(tmp_path)
    migration = tmp_path / "migrations" / "001_legacy.sql"
    migration.parent.mkdir(parents=True)
    migration.write_text(postgres_sql, encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert f"forbidden Postgres migration: {Path('migrations/001_legacy.sql')}" in result.stdout


def test_repository_verifier_allows_sqlite_dialect_migrations(tmp_path: Path) -> None:
    _write_complete_repository(tmp_path)
    migration = tmp_path / "migrations" / "001_local.sql"
    migration.parent.mkdir(parents=True)
    migration.write_text(
        "PRAGMA foreign_keys = ON;\n"
        "CREATE TABLE sources (id INTEGER PRIMARY KEY, title TEXT NOT NULL, content BLOB);\n",
        encoding="utf-8",
    )

    result = _run_verifier(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr


def test_repository_verifier_ignores_only_the_retained_replacement_plan_wording(tmp_path: Path) -> None:
    _write_complete_repository(tmp_path)
    plan = tmp_path / "docs" / "superpowers" / "plans" / "2026-08-18-chroma-sqlite-rag.md"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        "Historical removal targets: asyncpg, pgvector, "
        f"{_DATABASE_URL_KEY}, {_HOSTED_PROFILE_SETTING}, render.yaml.\n",
        encoding="utf-8",
    )

    result = _run_verifier(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr


def test_repository_verifier_does_not_ignore_other_plan_wording(tmp_path: Path) -> None:
    _write_complete_repository(tmp_path)
    plan = tmp_path / "docs" / "superpowers" / "plans" / "other-plan.md"
    plan.parent.mkdir(parents=True)
    plan.write_text(f"Reintroduce {_DATABASE_URL_KEY} configuration.\n", encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert "forbidden database URL configuration" in result.stdout


def test_repository_secret_scan_covers_synthetic_fixture_files_without_whole_file_exclusions() -> None:
    assert find_repository_secret_hits() == []
