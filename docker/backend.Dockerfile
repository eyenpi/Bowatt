# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home app \
    && mkdir /app/data \
    && chown app:app /app/data

FROM base AS dependencies
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
COPY backend/pyproject.toml backend/uv.lock backend/README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev
# Cache the vocabulary while building, so startup never needs to download it.
RUN python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base")' \
    && chmod -R a+rX /opt/tiktoken-cache

FROM dependencies AS test
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked
COPY backend/app ./app
COPY backend/tests ./tests
COPY backend/examples ./examples
COPY docker/smoke_test.py ./smoke_test.py
ENV PYTEST_ADDOPTS="-p no:cacheprovider"
USER 10001:10001
CMD ["pytest", "-q", "-m", "not live"]

FROM base AS runtime
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=dependencies /opt/tiktoken-cache /opt/tiktoken-cache
COPY backend/app ./app
USER 10001:10001
EXPOSE 8787
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8787"]
