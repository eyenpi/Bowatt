from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from hashlib import blake2b

from app.errors import EmbeddingProviderError, ResearchProviderError
from app.models import RetrievedChunk, SearchPlan, SearchResult


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


class FakeResearchProvider:
    """Deterministic planner, web search, and answer stream for tests."""

    def __init__(
        self,
        *,
        plans: Sequence[SearchPlan] | None = None,
        search_results: dict[str, Sequence[SearchResult]] | None = None,
        failed_queries: frozenset[str] = frozenset(),
        plan_delay_seconds: float = 0,
        search_delay_seconds: float = 0,
        answer_chunk_delay_seconds: float = 0,
        answer_chunks: Sequence[str] = (
            "# Research answer\n\n",
            "The available evidence has been synthesized.",
        ),
    ) -> None:
        self._plans = tuple(plans or (SearchPlan(("background research",), True),))
        self._search_results = search_results or {
            "background research": (
                SearchResult(
                    title="Example evidence",
                    url="https://example.test/evidence",
                    snippet="Deterministic external evidence.",
                ),
            )
        }
        self._failed_queries = failed_queries
        self._plan_delay_seconds = plan_delay_seconds
        self._search_delay_seconds = search_delay_seconds
        self._answer_chunk_delay_seconds = answer_chunk_delay_seconds
        self._answer_chunks = tuple(answer_chunks)
        self.plan_calls: list[tuple[str, tuple[SearchResult, ...], int]] = []
        self.search_calls: list[tuple[str, int]] = []
        self.search_attempts: dict[str, int] = {}
        self.answer_calls: list[
            tuple[str, tuple[RetrievedChunk, ...], tuple[SearchResult, ...]]
        ] = []
        self.active_searches = 0
        self.peak_searches = 0
        self.cancelled_searches = 0
        self.concurrent_searches_started = asyncio.Event()
        self.active_plans = 0
        self.cancelled_plans = 0
        self.plan_started = asyncio.Event()
        self.active_answer_streams = 0
        self.closed_answer_streams = 0
        self.interrupted_answer_streams = 0
        self.closed = False

    async def create_search_plan(
        self,
        request: str,
        prior_web_context: Sequence[SearchResult],
        round_number: int,
    ) -> SearchPlan:
        self.plan_calls.append((request, tuple(prior_web_context), round_number))
        self.active_plans += 1
        self.plan_started.set()
        try:
            if self._plan_delay_seconds:
                await asyncio.sleep(self._plan_delay_seconds)
            index = min(len(self.plan_calls) - 1, len(self._plans) - 1)
            return self._plans[index]
        except asyncio.CancelledError:
            self.cancelled_plans += 1
            raise
        finally:
            self.active_plans -= 1

    async def search(self, query: str, limit: int) -> Sequence[SearchResult]:
        self.search_calls.append((query, limit))
        self.search_attempts[query] = self.search_attempts.get(query, 0) + 1
        self.active_searches += 1
        if self.active_searches > 1:
            self.concurrent_searches_started.set()
        self.peak_searches = max(self.peak_searches, self.active_searches)
        try:
            if self._search_delay_seconds:
                await asyncio.sleep(self._search_delay_seconds)
            if query in self._failed_queries:
                raise ResearchProviderError("Synthetic search failure.")
            return tuple(self._search_results.get(query, ()))[:limit]
        except asyncio.CancelledError:
            self.cancelled_searches += 1
            raise
        finally:
            self.active_searches -= 1

    async def start_answer(
        self,
        request: str,
        uploaded_context: Sequence[RetrievedChunk],
        web_context: Sequence[SearchResult],
    ):
        self.answer_calls.append((request, tuple(uploaded_context), tuple(web_context)))

        async def stream():
            self.active_answer_streams += 1
            completed = False
            try:
                for chunk in self._answer_chunks:
                    if self._answer_chunk_delay_seconds:
                        await asyncio.sleep(self._answer_chunk_delay_seconds)
                    else:
                        await asyncio.sleep(0)
                    yield chunk
                completed = True
            finally:
                self.active_answer_streams -= 1
                self.closed_answer_streams += 1
                if not completed:
                    self.interrupted_answer_streams += 1

        return stream()

    async def close(self) -> None:
        self.closed = True
