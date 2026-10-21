# Local security and operations

## Scope and exposure

This is a single-user, single-process local application. `scripts/start.ps1`
binds `127.0.0.1` by default. Keep it on loopback unless you have separately
designed authentication, TLS, network controls, monitoring, abuse protection,
and recovery. This project makes no security, availability, or durability
guarantee for Internet-facing operation.

All source mutations require a constant-time Bearer comparison against
`ADMIN_TOKEN`. Missing, malformed, and incorrect tokens return a stable 401.
Queries and source metadata are intended for the local user and are not a
multi-user authorization boundary.

## Data sensitivity and permissions

The data root contains extracted document text, URLs, repository content,
embeddings, source provenance, conversation turns, checkpoints, and A2A task
history. Treat the whole directory as sensitive even when original upload files
were temporary.

- Keep the data root outside the repository and synchronized folders.
- Limit directory access to the Windows account running the app. Do not grant
  broad `Users`, network-share, or guest write permissions.
- Protect workstation backups and disk encryption according to the sensitivity
  of the indexed corpus.
- Never commit `.env`, API keys, admin tokens, runtime databases, Chroma files,
  logs containing user content, or copied backups.

## Consistent backup

Stop the application before copying. SQLite and Chroma must be captured as one
consistent unit; copying live files can produce a mismatched or partial backup.

1. Stop `scripts/start.ps1` or the local container and confirm its process has
   exited.
2. Copy the entire `RAG_DATA_DIR` to an access-controlled destination. Include
   `rag.sqlite3`, `checkpoints.sqlite3`, and `chroma\` together.
3. Record the application revision and backup time outside the archive.
4. Restart the app and check `/healthz`.

Do not back up `.env` with the corpus. Store secrets in the normal secret
manager and rotate them independently.

## Restore

1. Stop the application and preserve the current data directory under a new
   name until the restore is verified.
2. Restore the complete backup into a clean directory with restricted
   permissions.
3. Set `RAG_DATA_DIR` to that directory, start once, and wait for automatic local
   schema initialization.
4. Check `/healthz`, list sources, run a known grounded query, and confirm A2A
   task operations before removing the preserved directory.

Never merge individual database or Chroma files from different backups.

## Secret rotation

1. Stop the application.
2. Generate a new random `ADMIN_TOKEN` or obtain a new Gemini key.
3. Update the secret source without committing it and restart.
4. Verify old admin tokens receive 401 and a new token can list/ingest/delete as
   intended; run one grounded Gemini query after provider-key rotation.
5. Revoke the old provider key after the replacement is confirmed.

## Safe deletion

To remove one source, use the authenticated delete endpoint so SQLite and Chroma
are updated together. To erase all local data, stop the application, resolve and
visually verify the exact `RAG_DATA_DIR`, then remove only that directory through
a recoverable mechanism when possible. Never target the repository root, user
profile, drive root, a wildcard, or an unresolved environment variable.

## Ingestion limits and network safety

- PDF uploads are bounded to 25 MiB and 200 pages. Originals are temporary;
  extracted text and provenance persist locally.
- Web ingestion accepts one public HTTP(S) URL, blocks private/link-local/loopback
  destinations and unsafe redirects, caps the response at 5 MiB, and does not
  recursively crawl.
- GitHub ingestion accepts public repositories, bounds files and bytes, and
  rejects symlinks, submodules, binaries, vendor directories, oversized blobs,
  and sensitive configuration names.
- Provider retries, query concurrency, timeouts, request sizes, and per-IP rate
  limits remain bounded. These controls do not turn the app into a hardened
  public service.

Logs and public errors redact known credentials and inline secret patterns, but
operators must still avoid logging source bodies, questions, answers, hidden
reasoning, or raw provider responses.
