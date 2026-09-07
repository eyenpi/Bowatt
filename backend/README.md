# Research Agent Backend

FastAPI backend for the research-agent technical test. Uploaded sources are chunked,
embedded in bounded batches, and persisted with their vectors in a workspace-scoped
SQLite database. The streamed research response is still a scaffold until the agent
stage is implemented.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Run locally

```sh
cd backend
cp .env.example .env
# Add OPENAI_API_KEY to .env for real uploads.
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

Accepted sources, chunks, and embeddings are committed in one transaction to
`data/research.db`. Re-uploading identical content for the same workspace and embedding
model skips another embedding request. Retrieval is implemented at the application
service layer and will be connected to answer generation in the agent stage.

## Checks

```sh
uv run pytest
uv run pytest -m integration
uv run pytest --cov=app --cov-report=term-missing
uv run ruff check .
```

The integration suite starts Uvicorn on an ephemeral localhost port and verifies the
same CORS, multipart upload, raw Markdown stream, and plain-text error contract used by
the supplied frontend.

The default tests use deterministic fake embeddings and require no credentials or
network access. The live OpenAI embedding smoke test is opt-in:

```sh
BOWATT_RUN_LIVE_TESTS=1 OPENAI_API_KEY=... uv run pytest -m live
```
