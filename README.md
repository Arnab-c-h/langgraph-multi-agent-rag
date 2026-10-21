# LangGraph Multi-Agent RAG

A single-user, local-first RAG application with nine bounded LangGraph reasoning
modes, Gemini generation and 768-dimensional embeddings, persistent Chroma
retrieval, SQLite metadata/checkpoints/tasks, REST, Gradio, and the official A2A
HTTP server.

The supported runtime is one local process. It binds to `127.0.0.1` by default
and stores durable state outside the repository. The default data root is the OS
local-app-data directory returned by `platformdirs`; on Windows this is normally
`%LOCALAPPDATA%\LangGraphMultiAgentRAG`. This keeps mutable indexes and databases
outside the OneDrive checkout.

## Quick start on Windows

Requirements: Python 3.12, Git, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```powershell
uv sync --frozen --extra full --extra dev
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY`. Generate a unique admin token rather than
inventing a memorable password:

```powershell
$tokenBytes = [byte[]]::new(32)
[Security.Cryptography.RandomNumberGenerator]::Fill($tokenBytes)
[Convert]::ToHexString($tokenBytes).ToLowerInvariant()
```

Put the output in `ADMIN_TOKEN`. Leave `RAG_DATA_DIR=` blank to use the OS
default, or set an absolute path on a local disk. Then start the application:

```powershell
.\scripts\start.ps1
```

Open `http://127.0.0.1:8000/ui/`. To use a different local port or an explicit
bind address:

```powershell
.\scripts\start.ps1 -Port 8080 -Host 127.0.0.1
```

See [docs/windows-setup.md](docs/windows-setup.md) for a complete walkthrough.

## Ingest and query

All knowledge-base mutations require `Authorization: Bearer <ADMIN_TOKEN>`.
The comparison is constant-time. Query, streaming, source, health, UI, and A2A
wire schemas remain stable.

```powershell
$headers = @{ Authorization = "Bearer $env:ADMIN_TOKEN" }

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/admin/sources/web `
  -Headers $headers -ContentType application/json `
  -Body '{"url":"https://example.com/guide"}'

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/query `
  -ContentType application/json `
  -Body '{"question":"What does the indexed guide say?","reasoning_mode":"standard"}'
```

PDF, public web URL, and public GitHub repository ingestion are available in the
Gradio Knowledge Base tab and under `/api/v1/admin/sources/*`. Indexed sources
are listed at `GET /api/v1/sources`.

The official Agent Card is at `/.well-known/agent-card.json`; the JSON-RPC A2A
endpoint is `/a2a`. The card always advertises `query`, `list_sources`, and the
admin-authenticated `ingestion` skill.

## Reasoning modes

`standard`, `cot`, `tot`, `react`, `self_reflection`, `self_consistency`,
`decomposed`, `least_to_most`, and `recursive` all share strict retrieval,
evidence grading, exact citations, stable refusal behavior, bounded retries,
timeouts, cancellation, and public error schemas.

## Verification

```powershell
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev mypy src
uv run --frozen --extra dev python scripts/verify_repository.py all --root .
uv lock --check
uv build --wheel
```

Offline tests use deterministic embeddings/models and open no network
connection. Live Gemini and external A2A smoke tests are explicit opt-ins; see
[docs/evaluation.md](docs/evaluation.md).

## Optional local container

The image installs the full ingestion extra and runs as a non-root user. Its
state root is `/data`, which must be backed by a named volume or bind mount:

```powershell
docker build -t local-rag .
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env `
  -v rag-data:/data local-rag
```

The image is a local packaging option, not a durability or Internet-exposure
service. Read [docs/security-operations.md](docs/security-operations.md) before
copying, backing up, restoring, rotating secrets, or deleting local data.
