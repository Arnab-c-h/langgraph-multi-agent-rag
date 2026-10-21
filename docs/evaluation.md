# Evaluation and opt-in smoke tests

## Deterministic offline acceptance

`tests/evaluation/fixtures/corpus.json` is a four-chunk synthetic CC0 corpus:
one PDF-like page segment, one web section, and two GitHub-file records. It has
fixed UUIDs, exact locators/excerpts, and provenance. The harness uses stable
768-dimensional token-hash embeddings, a lexical grader, and a deterministic
workflow model; it opens no network connection.

The PDF fixture describes durable local SQLite metadata and Chroma embeddings.
The evaluation runs the real `StrictRagService`, `StrictCitationValidator`, all
nine compiled LangGraph workflows, and the public `WorkflowRunner` stream. Every
threshold remains 1.0:

| Metric | Meaning |
|---|---|
| refusal accuracy | unsupported question ends only in `NO_SUPPORTING_EVIDENCE` |
| citation validity | every citation UUID, excerpt, and locator matches stored fixture evidence |
| citation completeness | all expected supporting chunks are cited |
| citation precision | every emitted citation is one of the exact expected chunks |
| filter correctness | source-type and source-ID filters admit only allowed records |
| stream lifecycle | all nine modes order status, tokens, citations, final correctly |
| safety | streams contain no injected secret or hidden-reasoning labels |

```powershell
uv run --frozen --extra dev pytest tests/evaluation -q
```

## Provider-free restart acceptance

The local restart acceptance uses a real temporary SQLite database, persistent
Chroma collection, LangGraph SQLite checkpointer, official SQLite A2A task
store, ingestion service, strict RAG service, and all nine compiled workflows.
Only the Gemini embedding, evidence-grading, and workflow-model boundaries are
replaced by deterministic local fakes. It also drives the official A2A client
in process through send, streaming, get, list, and cancellation, then recreates
all durable resources from the same data directory and verifies recovery.

```powershell
uv run --frozen --extra dev pytest tests/acceptance/test_local_restart.py -q
```

## Default-skipped live gates

Live tests never infer permission from normal runtime variables. Each gate needs
its own opt-in and dedicated value, and all skip by default.

Gemini generation plus a 768-dimensional embedding:

```powershell
$env:RAG_RUN_GEMINI_SMOKE = "1"
$env:RAG_SMOKE_GEMINI_API_KEY = "<load securely outside source/history>"
uv run --frozen --extra dev pytest tests/smoke/test_opt_in_smoke.py -k gemini -q
```

External A2A smoke against an already-running local gateway. The guard accepts
only loopback HTTP(S) URLs:

```powershell
$env:RAG_RUN_A2A_SMOKE = "1"
$env:RAG_SMOKE_A2A_BASE_URL = "http://127.0.0.1:8000"
uv run --frozen --extra dev pytest tests/smoke/test_opt_in_smoke.py -k a2a -q
```

```powershell
Remove-Item Env:RAG_RUN_GEMINI_SMOKE,Env:RAG_SMOKE_GEMINI_API_KEY `
  ,Env:RAG_RUN_A2A_SMOKE,Env:RAG_SMOKE_A2A_BASE_URL -ErrorAction SilentlyContinue
```

These opt-ins authorize only the named smoke test. They do not authorize source
mutations, deployment, account changes, or quota spend by other commands.
