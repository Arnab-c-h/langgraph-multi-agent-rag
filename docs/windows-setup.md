# Windows setup

## Prerequisites

Install Git, Python 3.12, and `uv`. Use PowerShell from the repository root:

```powershell
python --version
uv --version
uv sync --frozen --extra full --extra dev
```

The full extra supplies the bounded PDF, public web, and public GitHub loaders.
The base package supplies FastAPI, Gradio, LangGraph, Chroma, SQLite, and the
official A2A HTTP server.

## Configure the local runtime

```powershell
Copy-Item .env.example .env
$tokenBytes = [byte[]]::new(32)
[Security.Cryptography.RandomNumberGenerator]::Fill($tokenBytes)
$adminToken = [Convert]::ToHexString($tokenBytes).ToLowerInvariant()
$adminToken
```

Set `GEMINI_API_KEY` and `ADMIN_TOKEN` in `.env`. Keep `APP_PROFILE=full`.
`RAG_DATA_DIR` is optional: blank resolves through `platformdirs` to the OS
local-app-data directory, normally
`$env:LOCALAPPDATA\LangGraphMultiAgentRAG` on Windows. That directory is outside
this OneDrive repository and contains:

- `rag.sqlite3` for sources, authoritative chunks, ingestion jobs, threads, and
  A2A tasks;
- `checkpoints.sqlite3` for mode-isolated LangGraph checkpoints;
- `chroma\` for the persistent vector index.

To override it, use an absolute local path:

```dotenv
RAG_DATA_DIR=D:\LocalAppData\LangGraphMultiAgentRAG
```

Do not put mutable runtime data inside the repository or a synchronized folder.

## Start and verify

```powershell
.\scripts\start.ps1
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

The launcher accepts only `full` and `test`, defaults to `full`, and binds
`127.0.0.1`. `test` is for injected/offline resources. An explicit bind address
is available when deliberately needed:

```powershell
.\scripts\start.ps1 -Profile full -Host 127.0.0.1 -Port 8080
```

The health payload is ready only after SQLite, Chroma, checkpoints, sources,
workflow runner, and A2A task storage initialize.

## Ingest sources

PowerShell does not automatically load `.env` into `$env:*`. Load the admin
token securely from your normal secret manager, then use a Bearer header:

```powershell
$headers = @{ Authorization = "Bearer $env:ADMIN_TOKEN" }

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/admin/sources/web `
  -Headers $headers -ContentType application/json `
  -Body '{"url":"https://example.com/guide"}'

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/admin/sources/github `
  -Headers $headers -ContentType application/json `
  -Body '{"url":"https://github.com/owner/public-repository"}'

curl.exe -X POST http://127.0.0.1:8000/api/v1/admin/sources/pdf `
  -H "Authorization: Bearer $env:ADMIN_TOKEN" `
  -F "file=@C:\path\guide.pdf;type=application/pdf"
```

The Gradio UI at `/ui/` exposes the same admin-locked mutations.

## Query and A2A

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/query `
  -ContentType application/json `
  -Body '{"question":"What evidence is indexed?","reasoning_mode":"standard"}'

Invoke-RestMethod http://127.0.0.1:8000/.well-known/agent-card.json
```

The Agent Card advertises query, source listing, and admin-authenticated
ingestion. Official A2A clients connect to `http://127.0.0.1:8000/a2a`.

## Tests

```powershell
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev mypy src
uv run --frozen --extra dev python scripts/verify_repository.py all --root .
uv lock --check
```

See [evaluation.md](evaluation.md) for explicitly opted-in live checks and
[security-operations.md](security-operations.md) for backup and restore.
