# Bowatt

A research assistant that combines uploaded documents with web sources and streams
a Markdown answer with citations. The supplied React frontend is unchanged.

## Running it

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), Node.js, npm, and an OpenAI API key.

From the repository root:

```sh
cd backend
cp .env.example .env
# Set OPENAI_API_KEY in .env before starting the server.
uv sync --locked
uv run uvicorn app.main:app --reload --port 8787 --env-file .env
```

The first start downloads the tokenizer vocabulary. In another terminal, from the
repository root:

```sh
cd frontend
npm ci
npm run dev
```

Open the address printed by Vite, normally [localhost:5173](http://localhost:5173).

## Design

FastAPI handles uploads and research requests. Files are split around headings,
paragraphs, and sentences within a token limit. Chunks, source positions, and
embeddings live in SQLite, which keeps setup simple for a small document collection.

Each query is embedded for retrieval while the model plans web searches. Searches
run with bounded concurrency and retries, then the evidence is used to stream an
answer. Disconnects cancel ongoing work. Provider interfaces make these paths
testable without paid API calls.

## Examples

Upload [support.md](backend/examples/support.md), then try:

- “Compare Enterprise and Standard response times.” Expected: four business hours
  versus two business days, with uploaded-source citations.
- “Which contact channels does each plan support?” Expected: email and phone for
  Enterprise; email only for Standard.
- “What does the uploaded policy say about refunds?” Expected: it does not specify
  a refund policy. The answer should acknowledge the missing information.

## Testing and evaluation

From `backend/`:

```sh
uv run pytest -m 'not live'
uv run ruff check .
```

The 97 tests cover uploads, retrieval, streaming, cancellation, and index recovery,
including real HTTP requests with fake providers. Live provider checks are still
pending. With the API key in `.env`, run:

```sh
BOWATT_RUN_LIVE_TESTS=1 uv run --env-file .env pytest -m live
```

For answer quality, use questions with known supporting passages. Compare chunking
methods with the same model and context-token budget; check retrieval, citation
accuracy, latency, and token use. Keep some questions aside while tuning.

## Limits and next steps

Uploads support UTF-8 text only. Public deployment would need authentication and
user isolation; the supplied UI currently shares one workspace. Hybrid retrieval
and reranking are future experiments, guided by evaluation. Implementation notes
are in the [backend README](backend/README.md).
