# Research Agent Backend

FastAPI backend for the research-agent technical test. 

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Run locally

```sh
cd backend
cp .env.example .env
uv sync
uv run uvicorn app.main:app --reload --env-file .env
```

The API listens on `http://localhost:8787` by default. Start the supplied frontend in
another terminal with `npm run dev` from `frontend/`.

## API contract

- `GET /health` returns service health and scaffold mode.
- `POST /api/sources` accepts multipart uploads under the repeated `files` field.
- `POST /api/research` accepts `{ "request": "..." }` and streams raw Markdown.

Both POST endpoints accept an optional `X-Workspace-ID` header. Until the frontend
sends one, requests use the configured `local-default` workspace.

## Checks

```sh
uv run pytest
uv run pytest -m integration
uv run ruff check .
```

The integration suite starts Uvicorn on an ephemeral localhost port and verifies the
same CORS, multipart upload, raw Markdown stream, and plain-text error contract used by
the supplied frontend.

