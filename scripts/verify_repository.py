"""Behavioral local-deployment, forbidden-runtime, and credential checks."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeGuard

ROOT = Path(__file__).resolve().parents[1]

_SECRET_PATTERNS = {
    "google-api-key": re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    "github-token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "openai-key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "supabase-secret": re.compile(r"\bsb_secret_[A-Za-z0-9_-]{16,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credentialed-postgres-dsn": re.compile(
        r"postgres(?:ql)?://[^\s:/\"']+:([^\s@\"']+)@[^\s\"']+", re.IGNORECASE
    ),
}
_PLACEHOLDER_PASSWORDS = {
    "password",
    "other-password",
    "test-password",
    "secret",
    "PASSWORD",
    "[YOUR-PASSWORD]",
    "<PASSWORD>",
}
_ALLOWED_SYNTHETIC_MATCHES = {
    ("credentialed-postgres-dsn", "postgresql://user:p%40ss@db.example:6543/test?sslmode=require&channel_binding=require"),
    ("credentialed-postgres-dsn", "postgresql://dbuser:dbpassword@db.example.com:5432/postgres"),
    ("credentialed-postgres-dsn", "postgresql://user:pass@host/db"),
    ("google-api-key", "AIzaSyA12345678901234567890123456789012"),
    ("openai-key", "sk-proj-abcdefghijklmnopqrstuvwxyz123456"),
    ("github-token", "ghp_abcdefghijklmnopqrstuvwxyz1234567890"),
    ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue123"),
}

_REQUIRED_ARTIFACTS = ("Dockerfile", ".env.example", "scripts/start.ps1", "pyproject.toml")
_IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
}
_RETAINED_REPLACEMENT_PLAN = Path("docs/superpowers/plans/2026-08-18-chroma-sqlite-rag.md")
_FOUNDATION_PROFILE_NEGATIVE_TEST = Path("tests/unit/test_foundation.py")
_DATABASE_URL_NAME = "DATABASE" + "_URL"
_PROFILE_NAME = "APP_" + "PROFILE"
_HOSTED_VALUE = "host" + "ed"
_LEGACY_TASK_STORE_NAME = "create_database_" + "task_store"

_DATABASE_ENV_PATTERN = re.compile(rf"\b(?:DIRECT_)?{_DATABASE_URL_NAME}\b")
_HOSTED_PROFILE_PATTERN = re.compile(
    rf"(?:{_PROFILE_NAME}\s*=\s*[\"']?{_HOSTED_VALUE}\b|"
    rf"app_profile\s*==\s*[\"']{_HOSTED_VALUE}[\"'])",
    re.IGNORECASE,
)
_LEGACY_IDENTIFIER_PATTERN = re.compile(rf"\b{_LEGACY_TASK_STORE_NAME}\b")
_POSTGRES_MIGRATION_PATTERN = re.compile(
    r"(?:\bCREATE\s+EXTENSION\b"
    r"|\bgen_random_uuid\s*\("
    r"|\bjsonb\b"
    r"|\btimestamptz\b"
    r"|\bCREATE\s+SCHEMA\b"
    r"|\b(?:big)?serial\b"
    r"|\bvector\s*\(\s*\d+\s*\)"
    r"|\bvector_cosine_ops\b)",
    re.IGNORECASE,
)
_REQUIREMENT_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[([^]]+)\])?")
_FORBIDDEN_PACKAGES = {
    "asyncpg",
    "langchain-postgres",
    "langgraph-checkpoint-postgres",
    "pgvector",
    "psycopg",
    "psycopg-binary",
    "psycopg-pool",
    "postgrest",
    "supabase",
    "supabase-py",
}
_FORBIDDEN_IMPORT_PREFIXES = (
    "asyncpg",
    "langchain_postgres",
    "langgraph.checkpoint.postgres",
    "langgraph_checkpoint_postgres",
    "pgvector",
    "psycopg",
    "psycopg_binary",
    "psycopg_pool",
    "rag_app.persistence",
    "rag_app.retrieval.pgvector",
    "rag_app.retrieval.postgres",
    "rag_app.storage.postgres",
    "rag_app.storage.vector",
    "supabase",
)


@dataclass(frozen=True)
class SecretHit:
    path: Path
    line: int
    kind: str
    match: str = field(repr=False, compare=False)


def tracked_files(root: Path = ROOT) -> list[Path]:
    """Return committed/staged files for the credential scanner."""
    if not (root / ".git").exists():
        return repository_files(root)
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def repository_files(root: Path = ROOT) -> list[Path]:
    """Enumerate repository surfaces without relying on Git tracking state."""
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not _IGNORED_DIRECTORY_NAMES.intersection(path.relative_to(root).parts)
    )


def find_secret_hits(paths: list[Path]) -> list[SecretHit]:
    hits: list[SecretHit] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for kind, pattern in _SECRET_PATTERNS.items():
                for match in pattern.finditer(line):
                    if kind == "credentialed-postgres-dsn" and match.group(1) in _PLACEHOLDER_PASSWORDS:
                        continue
                    hits.append(SecretHit(path=path, line=line_number, kind=kind, match=match.group(0)))
    return hits


def find_repository_secret_hits(root: Path = ROOT) -> list[SecretHit]:
    """Scan repository text without suppressing entire ordinary documents."""
    return [
        hit
        for hit in find_secret_hits(tracked_files(root))
        if (hit.kind, hit.match) not in _ALLOWED_SYNTHETIC_MATCHES
    ]


def verify_forbidden_runtime(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    files = repository_files(root)
    relative_files = {path.relative_to(root): path for path in files}

    render_manifests = sorted(
        relative for relative in relative_files if relative.name.casefold() in {"render.yaml", "render.yml"}
    )
    errors.extend(f"forbidden Render manifest: {path}" for path in render_manifests)

    supabase_files = sorted(
        relative for relative in relative_files if "supabase" in {part.casefold() for part in relative.parts}
    )
    if supabase_files:
        errors.append(f"forbidden Supabase runtime directory: {supabase_files[0].parent}")

    for relative, path in relative_files.items():
        if relative == _RETAINED_REPLACEMENT_PLAN:
            continue
        if path.suffix.casefold() == ".py":
            errors.extend(_verify_python_imports(path, relative))
        if path.suffix.casefold() == ".sql":
            errors.extend(_verify_sql_migration(path, relative))
        if relative in {Path("pyproject.toml"), Path("uv.lock")}:
            continue
        if not _is_historical_sdd_text(relative):
            errors.extend(_verify_runtime_text(path, relative))

    pyproject = relative_files.get(Path("pyproject.toml"))
    if pyproject is not None:
        errors.extend(_verify_pyproject_dependencies(pyproject))
    lockfile = relative_files.get(Path("uv.lock"))
    if lockfile is not None:
        errors.extend(_verify_lock_dependencies(lockfile))
    return _deduplicate(errors)


def _is_historical_sdd_text(relative: Path) -> bool:
    if relative.parts[:2] != (".superpowers", "sdd"):
        return False
    name = relative.name
    return bool(re.fullmatch(r"(?:final-)?review-.+\.diff", name)) or bool(
        re.fullmatch(r"task-.+-(?:brief|report)\.md", name)
    )


def _verify_python_imports(path: Path, relative: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
    except (SyntaxError, UnicodeDecodeError):
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.extend(f"{module}.{alias.name}".strip(".") for alias in node.names)
            imports.append(module)
    return [
        f"forbidden Python import '{name}': {relative}"
        for name in imports
        if _is_forbidden_import(name)
    ]


def _is_forbidden_import(name: str) -> bool:
    folded = name.casefold()
    return any(folded == prefix or folded.startswith(prefix + ".") for prefix in _FORBIDDEN_IMPORT_PREFIXES)


def _verify_sql_migration(path: Path, relative: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    if _POSTGRES_MIGRATION_PATTERN.search(text):
        return [f"forbidden Postgres migration: {relative}"]
    return []


def _verify_runtime_text(path: Path, relative: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []
    patterns = (
        ("database URL configuration", _DATABASE_ENV_PATTERN),
        ("hosted application profile", _HOSTED_PROFILE_PATTERN),
        ("Postgres A2A task-store identifier", _LEGACY_IDENTIFIER_PATTERN),
    )
    expected_profile_rejection_lines = _expected_profile_rejection_lines(path, relative)
    errors: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        for name, pattern in patterns:
            if name == "hosted application profile" and line_number in expected_profile_rejection_lines:
                continue
            if pattern.search(line):
                errors.append(f"forbidden {name}: {relative}:{line_number}")
    return errors


def _expected_profile_rejection_lines(path: Path, relative: Path) -> set[int]:
    if relative != _FOUNDATION_PROFILE_NEGATIVE_TEST:
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.With) or not any(
            _is_validation_error_context(item.context_expr) for item in node.items
        ):
            continue
        for statement in node.body:
            for candidate in ast.walk(statement):
                if _is_removed_profile_settings_call(candidate):
                    lines.update(range(candidate.lineno, (candidate.end_lineno or candidate.lineno) + 1))
    return lines


def _is_validation_error_context(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr == "raises"
        and bool(node.args)
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "ValidationError"
    )


def _is_removed_profile_settings_call(node: ast.AST) -> TypeGuard[ast.Call]:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Settings":
        return False
    return any(
        keyword.arg == "app_profile"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == _HOSTED_VALUE
        for keyword in node.keywords
    )


def _verify_pyproject_dependencies(path: Path) -> list[str]:
    data, error = _load_toml(path, "pyproject.toml")
    if error is not None:
        return [error]
    project = data.get("project", {})
    requirements: list[object] = list(project.get("dependencies", [])) if isinstance(project, dict) else []
    optional = project.get("optional-dependencies", {}) if isinstance(project, dict) else {}
    if isinstance(optional, dict):
        for values in optional.values():
            if isinstance(values, list):
                requirements.extend(values)
    return _requirement_errors(requirements, "project")


def _verify_lock_dependencies(path: Path) -> list[str]:
    data, error = _load_toml(path, "uv.lock")
    if error is not None:
        return [error]
    errors: list[str] = []
    packages = data.get("package", [])
    if not isinstance(packages, list):
        return ["repository invariant failed: uv.lock package table is invalid"]
    for package in packages:
        if not isinstance(package, dict):
            continue
        name = _normalize_package(str(package.get("name", "")))
        if name in _FORBIDDEN_PACKAGES:
            errors.append(f"forbidden lock dependency '{name}'")
        metadata = package.get("metadata", {})
        if isinstance(metadata, dict):
            requirements = metadata.get("requires-dist", [])
            if isinstance(requirements, list):
                errors.extend(_requirement_errors(requirements, "lock"))
    return _deduplicate(errors)


def _load_toml(path: Path, label: str) -> tuple[dict[str, Any], str | None]:
    try:
        with path.open("rb") as source:
            return tomllib.load(source), None
    except tomllib.TOMLDecodeError as error:
        return {}, f"repository invariant failed: invalid {label}: {error}"


def _requirement_errors(requirements: Iterable[object], source: str) -> list[str]:
    errors: list[str] = []
    for requirement in requirements:
        parsed = _parse_requirement(requirement)
        if parsed is None:
            continue
        name, extras = parsed
        if name in _FORBIDDEN_PACKAGES:
            errors.append(f"forbidden {source} dependency '{name}'")
        if name == "a2a-sdk" and "postgresql" in extras:
            errors.append("forbidden A2A extra 'postgresql'")
    return errors


def _parse_requirement(requirement: object) -> tuple[str, set[str]] | None:
    if isinstance(requirement, str):
        match = _REQUIREMENT_PATTERN.match(requirement)
        if match is None:
            return None
        extras = {item.strip().casefold().replace("_", "-") for item in (match.group(2) or "").split(",") if item}
        return _normalize_package(match.group(1)), extras
    if isinstance(requirement, dict) and requirement.get("name"):
        extras_value = requirement.get("extras", [])
        extras_items = extras_value if isinstance(extras_value, list) else []
        extras = {
            str(item).strip().casefold().replace("_", "-")
            for item in extras_items
        }
        return _normalize_package(str(requirement["name"])), extras
    return None


def _normalize_package(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().casefold())


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def verify_deployment(root: Path = ROOT) -> list[str]:
    if not root.exists():
        return [f"repository invariant failed: root does not exist: {root}"]
    if not root.is_dir():
        return [f"repository invariant failed: root is not a directory: {root}"]

    errors = [
        f"repository invariant failed: missing {relative}"
        for relative in _REQUIRED_ARTIFACTS
        if not (root / relative).is_file()
    ]

    docker_path = root / "Dockerfile"
    if docker_path.is_file():
        dockerfile = docker_path.read_text(encoding="utf-8")
        checks = {
            "Dockerfile has one image": sum(line.startswith("FROM ") for line in dockerfile.splitlines()) == 1,
            "Dockerfile installs full non-editably": (
                "uv sync --frozen --no-dev --no-editable --extra full" in dockerfile
            ),
            "Dockerfile runs as non-root": "USER app" in dockerfile,
            "Dockerfile persists local data": (
                'RAG_DATA_DIR=/data' in dockerfile and 'VOLUME ["/data"]' in dockerfile
            ),
            "Dockerfile runs one Uvicorn worker": "--workers\", \"1" in dockerfile,
        }
        errors.extend(f"deployment invariant failed: {name}" for name, passed in checks.items() if not passed)

    pyproject_path = root / "pyproject.toml"
    if pyproject_path.is_file():
        data, error = _load_toml(pyproject_path, "pyproject.toml")
        if error is not None:
            errors.append(error)
        else:
            project = data.get("project", {})
            optional = project.get("optional-dependencies", {}) if isinstance(project, dict) else {}
            if not isinstance(optional, dict) or "full" not in optional:
                errors.append("deployment invariant failed: full ingestion extra remains")

    example_path = root / ".env.example"
    if example_path.is_file() and "RAG_DATA_DIR=" not in example_path.read_text(encoding="utf-8"):
        errors.append("deployment invariant failed: environment declares local data root")
    return _deduplicate(errors)


def verify_repository(root: Path = ROOT) -> list[str]:
    return [*verify_forbidden_runtime(root), *verify_deployment(root)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=("all", "deployment", "runtime", "secrets"), nargs="?", default="all")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    errors: list[str] = []
    if args.check in {"all", "runtime"}:
        errors.extend(verify_forbidden_runtime(root))
    if args.check in {"all", "deployment"}:
        errors.extend(verify_deployment(root))
    if args.check in {"all", "secrets"}:
        errors.extend(
            f"potential {hit.kind}: {hit.path.relative_to(root)}:{hit.line}"
            for hit in find_repository_secret_hits(root)
        )
    if errors:
        for error in _deduplicate(errors):
            print(error)
        return 1
    print(f"repository {args.check} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
