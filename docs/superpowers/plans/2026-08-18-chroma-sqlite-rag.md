# Chroma and SQLite RAG Replacement Plan

**Goal:** Replace Supabase/Postgres/pgvector with single-machine Chroma and SQLite persistence while preserving the existing Gemini, LangGraph, REST, Gradio, A2A, ingestion, and strict-grounding behavior.

**Global constraints:** Work on `codex/chroma-sqlite-rag`; use test-first development; keep all public REST/A2A/query contracts stable; remain single-user and single-process; retain Gemini; do not migrate existing Supabase data; remove durable Render claims.

## Task 1: Local configuration and SQLite persistence

- Add failing tests for OS-local `RAG_DATA_DIR` resolution, the reduced `full | test` profiles, runtime credential validation, SQLite schema initialization/versioning, WAL/foreign-key/busy-timeout configuration, source/job/chunk CRUD, bounded thread history, and official A2A task persistence after restart.
- Replace Postgres source/thread/A2A persistence with `aiosqlite` repositories and the official `DatabaseTaskStore` over `sqlite+aiosqlite`.
- Add automatic local schema migrations for `sources`, authoritative `chunks`, `ingestion_jobs`, `threads`, and schema versions.
- Add `AsyncSqliteSaver` lifecycle wiring for mode-isolated LangGraph checkpoints.
- Add direct dependencies for SQLite/checkpoint/platform path support and update focused tests.

## Task 2: Chroma ingestion, retrieval, and runtime composition

- Add failing real-temporary-directory Chroma tests for 768-dimensional storage, cosine ordering, top-k, source filters, restart persistence, version replacement, orphan rejection, authoritative SQLite hydration, deletion, and partial-failure behavior.
- Add `langchain-chroma` with a persistent cosine collection rooted under `RAG_DATA_DIR/chroma`.
- Persist flat Chroma filter metadata only; keep complete content, metadata, and provenance authoritative in SQLite.
- Implement versioned ingestion: Chroma upsert first, SQLite activation second, stale-vector cleanup after commit, and failed-job/orphan handling.
- Replace Postgres runtime composition with SQLite/Chroma/A2A/checkpoint lifecycle management and expanded dependency readiness.
- Preserve strict evidence grading, citations, `NO_SUPPORTING_EVIDENCE`, all nine graphs, cancellation, retries, and public streaming contracts.

## Task 3: Remove Supabase and make deployment honestly local

- Add failing repository/deployment tests proving database URLs, Supabase/Postgres/pgvector imports, migrations, hosted profile, and Render configuration are gone.
- Remove Postgres dependencies, adapters, integration tests, migrations, Supabase smoke paths, and hosted-only behavior.
- Reduce `.env.example` to `APP_PROFILE`, optional `RAG_DATA_DIR`, Gemini/admin secrets, public base URL, and existing safe limits.
- Bind the PowerShell launcher to `127.0.0.1` by default.
- Replace the Dockerfile with one optional local image that requires `/data` persistence; remove `render.yaml`.
- Rewrite README, Windows setup, security/backup, and evaluation documentation for the Supabase-free local workflow.

## Task 4: Acceptance, restart smoke, and branch verification

- Run all nine offline workflow evaluations and exact-citation/refusal/filter/lifecycle/safety thresholds.
- Add a restart-based local smoke test using temporary SQLite and Chroma stores with fake Gemini boundaries; keep the live Gemini smoke explicitly opt-in.
- Verify official A2A send/stream/get/list/cancel behavior and task persistence across application restart.
- Run full pytest, Ruff, strict mypy, repository checks, `uv lock --check`, wheel build/import, and diff hygiene.
- Perform independent per-task and whole-branch review; resolve all load-bearing findings before completion.
