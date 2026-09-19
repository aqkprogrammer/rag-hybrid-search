# ---- build stage: resolve and install dependencies with uv ---------------------------------
FROM python:3.12-slim AS builder

# uv from PyPI (pinned); equivalent to `COPY --from=ghcr.io/astral-sh/uv` without a second registry.
RUN pip install --no-cache-dir "uv==0.11.28"

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Optional extras, e.g. `--build-arg EXTRAS=ml` for sentence-transformers + cross-encoder
# (CPU-only torch; adds roughly 1 GB to the image).
ARG EXTRAS=""

WORKDIR /app

# Dependencies first (cached layer), then the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project $([ -n "$EXTRAS" ] && echo "--extra $EXTRAS")

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable $([ -n "$EXTRAS" ] && echo "--extra $EXTRAS")

# Pre-fetch the tiktoken BPE so the container works without outbound network access.
ENV TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# ---- runtime stage --------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/.cache /app/.cache
COPY --chown=app:app sample_docs ./sample_docs

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    HF_HOME=/app/data/hf-cache \
    DATA_DIR=/app/data \
    LOG_FORMAT=json

RUN mkdir -p /app/data && chown app:app /app/data
USER app
VOLUME ["/app/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "hybrid_rag.api.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
