# Backend

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Copy `.env.example` to
`.env` if needed and set `OPENAI_API_KEY`. From the repository root:

```sh
cd backend
uv sync --locked
uv run uvicorn app.main:app --reload --port 8787 --env-file .env
```

## API

- `GET /health` — health check.
- `POST /api/sources` — multipart uploads in the `files` field.
- `POST /api/research` — JSON `{ "request": "..." }`; streams Markdown.

## Test

From `backend/`, without an API key:

```sh
uv run pytest -m 'not live'
uv run ruff check .
```
