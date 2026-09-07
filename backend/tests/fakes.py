from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from hashlib import blake2b

from app.errors import EmbeddingProviderError


class DeterministicEmbeddingProvider:
    """Credential-free test embedding based on stable token hashing."""

    def __init__(
        self,
        *,
        model_name: str = "test-embedding-v1",
        dimensions: int = 64,
        delay_seconds: float = 0,
        fail_on_document_call: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions
        self._delay_seconds = delay_seconds
        self._fail_on_document_call = fail_on_document_call
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []
        self.active_document_calls = 0
        self.peak_document_calls = 0
        self.closed = False

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        call_number = len(self.document_calls) + 1
        self.document_calls.append(tuple(texts))
        self.active_document_calls += 1
        self.peak_document_calls = max(
            self.peak_document_calls,
            self.active_document_calls,
        )
        try:
            if self._delay_seconds:
                await asyncio.sleep(self._delay_seconds)
            if call_number == self._fail_on_document_call:
                raise EmbeddingProviderError("Synthetic embedding failure.")
            return tuple(self._embed(text) for text in texts)
        finally:
            self.active_document_calls -= 1

    async def embed_query(self, text: str) -> Sequence[float]:
        self.query_calls.append(text)
        return self._embed(text)

    async def close(self) -> None:
        self.closed = True

    def _embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            digest = blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest, "little") % self._dimensions
            vector[bucket] += 1.0

        if not tokens:
            vector[0] = 1.0
        return tuple(vector)

