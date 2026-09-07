# Source chunking: design and verification

The goal is to preserve useful document context with a small, deterministic chunker.
It runs locally and adds no LLM calls. It recognizes Markdown-style ATX headings
(`#` through `######`) in uploaded UTF-8 text and keeps each section separate.
Consecutive headings stay with their following content. Fenced code is excluded
from heading detection.

Within a section, the chunker packs complete paragraphs. Oversized blocks fall back
to lines, sentences, words, and finally Unicode character boundaries. Lists, tables,
and fenced code stay intact when they fit; oversized ones follow the same fallback.
Internal whitespace is preserved. Leading/trailing whitespace at a chunk boundary
may be omitted. At most one complete trailing sentence is repeated, within the
configured overlap budget and the same section.

## Limits and provenance

- `BOWATT_CHUNK_SIZE` remains the hard character limit for the original chunk text
  (default 2,000), preserving the existing setting's unit.
- `BOWATT_CHUNK_OVERLAP` is the maximum character length of the repeated sentence
  (default 200). Zero disables overlap; a longer sentence is not repeated.
- `BOWATT_CHUNK_MAX_TOKENS` caps the actual embedding input, including heading
  context (default 512, supported range 8–8,191).
- Token counts use `cl100k_base`, the encoding specified in the
  [OpenAI embedding guide](https://developers.openai.com/api/docs/guides/embeddings).
  Text resembling special tokens is handled as literal source content.
- Each chunk stores its heading path, original text, and zero-based character
  offsets with an exclusive end. Citations display one-based inclusive character
  positions. Existing chunks without offsets keep their original chunk-only label.
- A heading prefix is added to embedding and answer context, separately from the
  original text. It is bounded to a quarter of the character/token budgets. Tiny
  budgets (under 64 characters or 32 tokens) omit the prefix; metadata is retained.

The index signature includes the chunker version, encoding, size, overlap, token
limit, embedding model, and requested dimensions. An unchanged re-upload skips
embedding. A changed signature triggers reindexing **on the next upload of that
file**. Existing sources remain searchable until then; startup never makes paid
provider calls or rebuilds an index automatically.

Database migration only adds columns, preserving old sources and embeddings.
Replacement embeddings are computed before a transaction replaces the old chunks.
If embedding or persistence fails, the prior index remains intact. Chunking runs
off the event loop, and embedding batches retain bounded concurrency and also
respect the aggregate token budget.

## Reproducible inspection

From `backend/`, run:

```sh
uv run python -m app.preview_chunks examples/support.md --max-tokens 64
```

The command prints source text, heading context, exact offsets, and embedding token
counts. It does not use credentials, send the file to a provider, or alter storage.
The first tokenizer use downloads and caches its vocabulary; subsequent uses can
run offline. For a new offline environment, warm the cache during setup:

```sh
uv run python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base")'
```

For an application demo, upload `examples/support.md` through the supplied UI.
Example questions and expected facts are:

| Question | Expected facts from the uploaded document |
| --- | --- |
| Compare Enterprise and Standard response times. | Four business hours versus two business days, citing the relevant uploaded sections. |
| When is an unanswered critical incident escalated? | After one hour, under the Enterprise section. |
| Which contact channels does each plan support? | Enterprise: email and phone. Standard: email. |
| What does the uploaded policy say about refunds? | It gives no refund policy; the answer must identify that gap. |

Exact generated wording and web evidence vary. These are expectations to check
with real providers, not claims that live answer quality has already been measured.

## Verification plan and acceptance criteria

Run `uv run pytest -m 'not live'` and `uv run ruff check .`.

| Acceptance criterion | Automated evidence |
| --- | --- |
| Original non-whitespace content is preserved; offsets resolve to exact original substrings. | `tests/test_chunking.py`: Unicode, CRLF, special-token text, long headings, long words, and seeded mixed-text cases. |
| Every embedding input fits the token cap; every source chunk fits the character cap. | Chunking limit tests and aggregate embedding-budget test in `tests/test_ingestion.py`. |
| Paragraphs/sentences are preferred; sections do not leak into adjacent sections through overlap. | Paragraph, sentence, overlap, and heading tests in `tests/test_chunking.py`. |
| Fenced code comments are not headings; small code blocks, tables, and lists remain intact. | Fenced-code and structured-content tests in `tests/test_chunking.py`. |
| Context and source positions reach embeddings, persistence, answer input, and citations. | `tests/test_index_migration.py`, `tests/test_openai_research.py`, and `tests/test_research.py`. |
| Legacy databases stay readable; reindexing happens once when settings change. | Legacy migration, restart, and parameterized reindex tests in `tests/test_index_migration.py`. |
| Failed reindexing preserves the prior source and all its chunks/vectors. | Injected embedding and database failures in `tests/test_index_migration.py`. |
| The provided UI's upload/stream/error contract still works. | Live Uvicorn HTTP integration suite in `tests/integration/`. |

These checks verify implementation behavior, not semantic retrieval superiority.
Local verification passed all 97 non-live tests, lint, and Python compilation. The
preview produced five separate sections from the example, each under its 64-token
limit. The frontend has no changes in the Git diff.

To evaluate that separately, label supporting spans in a small fixed corpus and
compare the previous character chunker with this one using real embeddings. Hold
the corpus, embedding model, and retrieved-context token budget fixed. Measure
supporting-span recall, the first relevant result's rank, latency, and context tokens.
Keep some questions aside while tuning, record regressions, and use fixed external
evidence when comparing generated answers. Manually check citation support and
whether unsupported questions are answered with an explicit evidence gap.

## Deliberate limitations

This is a lightweight parser, not a complete Markdown or linguistic parser.
Setext headings, HTML structure, PDF extraction, language-specific sentence rules,
and guaranteed preservation of oversized tables/code are not implemented. A token
limit is a bound, not a claim about the optimal chunk size. Hybrid retrieval,
semantic chunking, and reranking remain follow-up experiments that should be
justified by retrieval measurements before adding more dependencies or services.
