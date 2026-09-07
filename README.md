# Bowatt

A research agent with a React UI and FastAPI backend. It combines uploaded text
with web sources and streams answers with citations. Structure-aware chunks and
embeddings are stored in SQLite.

## Run

Requires Docker Compose 2.24.4+. Copy `backend/.env.example` to `backend/.env` if
needed and set `OPENAI_API_KEY`, then run:

```sh
docker compose up --build -d
```

Open [localhost:5173](http://localhost:5173). Stop with `docker compose down`; uploaded
data is kept. For local development, see [backend](backend/README.md) and
[frontend](frontend/README.md).

## Test

These checks use fake providers and need no API key:

```sh
docker compose run --build --rm --no-deps backend-tests
sh docker/test.sh
```

## Example

Upload [support.md](backend/examples/support.md) and ask “Compare Enterprise and
Standard response times.” Expect four business hours versus two business days,
with source citations. Check answers against known passages to evaluate retrieval
and citation accuracy.
