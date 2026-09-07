from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from app.errors import EmbeddingProviderError


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int | None = None,
        timeout_seconds: float = 30.0,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._client = client or AsyncOpenAI(api_key=api_key, timeout=timeout_seconds)

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return await self._embed(texts)

    async def embed_query(self, text: str) -> Sequence[float]:
        embeddings = await self._embed((text,))
        return embeddings[0]

    async def close(self) -> None:
        await self._client.close()

    async def _embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()

        request: dict[str, Any] = {
            "input": list(texts),
            "model": self._model,
            "encoding_format": "float",
        }
        if self._dimensions is not None:
            request["dimensions"] = self._dimensions

        try:
            response = await self._client.embeddings.create(**request)
        except OpenAIError as error:
            raise EmbeddingProviderError("Embedding provider request failed.") from error

        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise EmbeddingProviderError(
                "Embedding provider returned vectors with unexpected indexes."
            )

        return tuple(tuple(float(value) for value in item.embedding) for item in ordered)


class UnavailableEmbeddingProvider:
    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        del texts
        raise EmbeddingProviderError(
            "Embedding provider is not configured. Set OPENAI_API_KEY.",
            status_code=503,
        )

    async def embed_query(self, text: str) -> Sequence[float]:
        del text
        raise EmbeddingProviderError(
            "Embedding provider is not configured. Set OPENAI_API_KEY.",
            status_code=503,
        )

    async def close(self) -> None:
        return None

