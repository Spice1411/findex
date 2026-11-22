# syntax=docker/dockerfile:1

# --- Stage 1: Build locked virtual environment with uv ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1     UV_LINK_MODE=copy     UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install third-party dependencies first for layer caching
RUN --mount=type=cache,target=/root/.cache/uv     --mount=type=bind,source=uv.lock,target=uv.lock     --mount=type=bind,source=pyproject.toml,target=pyproject.toml     uv sync --locked --no-install-project --no-dev

# Copy application source and install the findex package
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts/build_demo_index.py ./scripts/build_demo_index.py

RUN --mount=type=cache,target=/root/.cache/uv     uv sync --locked --no-dev &&     /app/.venv/bin/python scripts/build_demo_index.py       --corpus /app/data/demo_corpus.jsonl       --out /app/data/demo_index.bin

# --- Stage 2: Minimal non-root runtime image ---
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PATH="/app/.venv/bin:$PATH"     INDEX_PATH="/app/data/demo_index.bin"     LOG_LEVEL="INFO"     PORT="8000"     WORKERS="1"

RUN groupadd --system appuser &&     useradd --system --gid appuser --create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/data /app/data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3   CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get("PORT", "8000")}/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn findex.web:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1} --proxy-headers"]
