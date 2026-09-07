from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI

from app.errors import EmbeddingProviderError
from app.providers.openai_embeddings import OpenAIEmbeddingProvider


class FakeEmbeddingsResource:
    def __init__(self, data: list[SimpleNamespace]) -> None:
        self._data = data
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(data=self._data)


class FakeOpenAIClient:
    def __init__(self, data: list[SimpleNamespace]) -> None:
        self.embeddings = FakeEmbeddingsResource(data)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_openai_adapter_batches_inputs_and_restores_index_order() -> None:
    client = FakeOpenAIClient(
        [
            SimpleNamespace(index=1, embedding=[0.0, 1.0]),
            SimpleNamespace(index=0, embedding=[1.0, 0.0]),
        ]
    )
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model="embedding-model",
        dimensions=2,
        client=cast(AsyncOpenAI, client),
    )

    embeddings = asyncio.run(provider.embed_documents(("first", "second")))
    asyncio.run(provider.close())

    assert embeddings == ((1.0, 0.0), (0.0, 1.0))
    assert client.embeddings.requests == [
        {
            "input": ["first", "second"],
            "model": "embedding-model",
            "encoding_format": "float",
            "dimensions": 2,
        }
    ]
    assert client.closed is True


def test_openai_adapter_rejects_unexpected_response_indexes() -> None:
    client = FakeOpenAIClient([SimpleNamespace(index=3, embedding=[1.0, 0.0])])
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model="embedding-model",
        client=cast(AsyncOpenAI, client),
    )

    with pytest.raises(EmbeddingProviderError, match="unexpected indexes"):
        asyncio.run(provider.embed_documents(("first",)))


def test_openai_adapter_embeds_a_query_without_optional_dimensions() -> None:
    client = FakeOpenAIClient([SimpleNamespace(index=0, embedding=[0.5, 0.25])])
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        model="embedding-model",
        client=cast(AsyncOpenAI, client),
    )

    embedding = asyncio.run(provider.embed_query("research query"))

    assert embedding == (0.5, 0.25)
    assert client.embeddings.requests == [
        {
            "input": ["research query"],
            "model": "embedding-model",
            "encoding_format": "float",
        }
    ]
