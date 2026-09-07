# Research Agent Backend

FastAPI backend for the research-agent technical test. Uploaded sources are chunked,
embedded in bounded batches, and persisted with their vectors in a workspace-scoped
SQLite database. Research requests combine relevant uploaded chunks with live web
evidence and stream a citation-grounded Markdown answer.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Run locally

```sh
cd backend
cp .env.example .env
# Add OPENAI_API_KEY to .env for embeddings, web search, and answer generation.
uv sync
# Warm the tokenizer vocabulary cache (requires internet on first use).
uv run python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base")'
uv run uvicorn app.main:app --reload --port 8787 --env-file .env
```

This command serves the API at `http://localhost:8787`. Start the supplied frontend in
another terminal with `npm run dev` from `frontend/`.

## API contract

- `GET /health` returns service health and agent mode.
- `POST /api/sources` accepts multipart uploads under the repeated `files` field.
- `POST /api/research` accepts `{ "request": "..." }` and streams raw Markdown.

Both POST endpoints accept an optional `X-Workspace-ID` header. The supplied frontend
works as-is using the configured `local-default` workspace.

Accepted sources, chunks, and embeddings are committed in one transaction to
`data/research.db`. Re-uploading identical content for the same workspace and embedding
model and indexing configuration skips another embedding request. Re-upload after
changing chunking settings to replace that file's old index atomically.

Chunks preserve heading/paragraph structure and original text positions, with a
token cap that includes heading context. Inspect the example without a model call:

```sh
uv run python -m app.preview_chunks examples/support.md --max-tokens 64
```

See [chunking design and verification](docs/chunking.md) for configuration, migration
behavior, acceptance criteria, example questions, evaluation plans, and limitations.

For research requests, uploaded retrieval and first-pass search planning start in
parallel. Planned web searches run with bounded concurrency and retries. Evidence is
deduplicated and capped before the OpenAI Responses API streams the final answer. The
backend appends the authoritative `[U#]` uploaded-source and `[W#]` web-source mapping,
so every source label has a deterministic target.

The orchestration limits and model are configurable through `.env`; the checked-in
defaults allow at most two planning rounds, three searches per round, three concurrent
searches, and twelve final web sources. Provider input is sent with `store=false`.
Provider streams have a configurable idle timeout and are explicitly closed when the
browser disconnects, including while planning or searching is still in progress.

## Checks

```sh
uv run pytest
uv run pytest -m integration
uv run pytest --cov=app --cov-report=term-missing
uv run ruff check .
```

The integration suite starts Uvicorn on an ephemeral localhost port and verifies the
same CORS, multipart upload, raw Markdown stream, citations, and plain-text error
contract used by the supplied frontend.

The default tests use deterministic fake embeddings, search results, and model streams,
so they require no credentials or provider access. After dependencies and the tokenizer
vocabulary are cached, they run offline. OpenAI embedding and research-provider
smoke tests are opt-in:

```sh
BOWATT_RUN_LIVE_TESTS=1 OPENAI_API_KEY=... uv run pytest -m live
```
