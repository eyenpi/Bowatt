from __future__ import annotations

import asyncio
import math
import os

import pytest

from app.providers.openai_embeddings import OpenAIEmbeddingProvider

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.getenv("BOWATT_RUN_LIVE_TESTS") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Set BOWATT_RUN_LIVE_TESTS=1 and OPENAI_API_KEY to run live provider tests.",
)
def test_openai_embedding_provider_live_smoke() -> None:
    provider = OpenAIEmbeddingProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.getenv("BOWATT_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

    async def run() -> tuple[tuple[float, ...], ...]:
        try:
            vectors = await provider.embed_documents(("research agent", "vector retrieval"))
            return tuple(tuple(vector) for vector in vectors)
        finally:
            await provider.close()

    vectors = asyncio.run(run())

    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert len(vectors[0]) == len(vectors[1])
    assert all(math.isfinite(value) for vector in vectors for value in vector)

