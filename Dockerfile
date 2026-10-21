# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
ENV APP_PROFILE=full
ENV RAG_DATA_DIR=/data
ENV PORT=8000

WORKDIR /app
RUN python -m pip install --no-cache-dir uv==0.8.13
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra full --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --extra full

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app app \
    && mkdir -p /data \
    && chown app:app /data

ENV PATH="/app/.venv/bin:$PATH"
VOLUME ["/data"]
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" || exit 1
CMD ["uvicorn", "rag_app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "2", "--timeout-keep-alive", "5", "--timeout-graceful-shutdown", "15"]
