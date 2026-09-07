from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import build_container
from app.errors import ResearchProviderError
from app.main import create_app
from app.models import RetrievedChunk, SearchPlan, SearchResult
from app.services.research import ResearchAgentService
from tests.fakes import DeterministicEmbeddingProvider, FakeResearchProvider


def test_planning_and_uploaded_retrieval_start_concurrently(settings: Settings) -> None:
    first_started = asyncio.Event()
    second_started = asyncio.Event()

    class CoordinatedRetriever:
        async def retrieve(
            self, workspace_id: str, query: str, limit: int | None = None
        ) -> Sequence[RetrievedChunk]:
            del workspace_id, query, limit
            first_started.set()
            await asyncio.wait_for(second_started.wait(), timeout=0.2)
            return ()

    class CoordinatedProvider(FakeResearchProvider):
        async def create_search_plan(
            self,
            request: str,
            prior_web_context: Sequence[SearchResult],
            round_number: int,
        ) -> SearchPlan:
            second_started.set()
            await asyncio.wait_for(first_started.wait(), timeout=0.2)
            return await super().create_search_plan(
                request, prior_web_context, round_number
            )

    provider = CoordinatedProvider(plans=(SearchPlan((), True),))
    agent = ResearchAgentService(settings, CoordinatedRetriever(), provider)

    stream = asyncio.run(
        asyncio.wait_for(agent.prepare_answer("workspace", "research this"), timeout=0.5)
    )

    assert stream is not None
    assert first_started.is_set()
    assert second_started.is_set()


def test_searches_are_bounded_and_results_are_capped(settings: Settings) -> None:
    bounded_settings = replace(
        settings,
        research_search_concurrency=2,
        research_max_web_results=3,
    )
    queries = tuple(f"query-{index}" for index in range(5))
    provider = FakeResearchProvider(
        plans=(SearchPlan(queries, True),),
        search_results={
            query: (
                SearchResult(
                    title=query,
                    url=f"https://example.test/{query}",
                    snippet=f"Evidence for {query}",
                ),
            )
            for query in queries
        },
        search_delay_seconds=0.01,
    )
    container = build_container(
        bounded_settings,
        embedding_provider=DeterministicEmbeddingProvider(),
        research_provider=provider,
    )

    with TestClient(
        create_app(settings=bounded_settings, container=container)
    ) as client:
        response = client.post("/api/research", json={"request": "bounded research"})

    assert response.status_code == 200
    assert provider.peak_searches == 2
    assert len(provider.search_calls) == bounded_settings.research_queries_per_round
    assert len(provider.answer_calls[0][2]) == 3


def test_partial_search_failure_is_retried_without_losing_evidence(
    settings: Settings,
) -> None:
    provider = FakeResearchProvider(
        plans=(SearchPlan(("broken", "working"), True),),
        search_results={
            "working": (
                SearchResult("Working source", "https://example.test/ok", "Useful fact"),
            )
        },
        failed_queries=frozenset({"broken"}),
    )
    container = build_container(
        settings,
        embedding_provider=DeterministicEmbeddingProvider(),
        research_provider=provider,
    )

    with TestClient(create_app(settings=settings, container=container)) as client:
        response = client.post("/api/research", json={"request": "resilient research"})

    assert response.status_code == 200
    assert provider.search_attempts["broken"] == settings.research_search_max_attempts
    assert provider.search_attempts["working"] == 1
    assert "[Working source](https://example.test/ok)" in response.text


def test_all_search_failures_without_uploaded_evidence_return_safe_error(
    settings: Settings,
) -> None:
    provider = FakeResearchProvider(
        plans=(SearchPlan(("broken",), True),),
        failed_queries=frozenset({"broken"}),
    )
    container = build_container(
        settings,
        embedding_provider=DeterministicEmbeddingProvider(),
        research_provider=provider,
    )

    with TestClient(create_app(settings=settings, container=container)) as client:
        response = client.post("/api/research", json={"request": "unavailable evidence"})

    assert response.status_code == 502
    assert response.text == "Research sources are temporarily unavailable."


def test_web_results_are_deduplicated_by_canonical_url(settings: Settings) -> None:
    provider = FakeResearchProvider(
        plans=(SearchPlan(("first", "second"), True),),
        search_results={
            "first": (
                SearchResult(
                    "Original",
                    "https://Example.test/report/?utm_source=test#section",
                    "First copy",
                ),
            ),
            "second": (
                SearchResult(
                    "Duplicate",
                    "https://example.test/report",
                    "Second copy",
                ),
            ),
        },
    )
    container = build_container(
        settings,
        embedding_provider=DeterministicEmbeddingProvider(),
        research_provider=provider,
    )

    with TestClient(create_app(settings=settings, container=container)) as client:
        response = client.post("/api/research", json={"request": "deduplicate"})

    assert response.status_code == 200
    assert len(provider.answer_calls[0][2]) == 1
    assert response.text.count("https://Example.test/report/?utm_source=test#section") == 1


def test_agent_never_exceeds_the_configured_search_rounds(settings: Settings) -> None:
    limited_settings = replace(settings, research_max_search_rounds=2)
    provider = FakeResearchProvider(
        plans=(
            SearchPlan(("round-one",), False),
            SearchPlan(("round-two",), False),
            SearchPlan(("must-not-run",), True),
        ),
        search_results={
            "round-one": (
                SearchResult("One", "https://example.test/one", "One"),
            ),
            "round-two": (
                SearchResult("Two", "https://example.test/two", "Two"),
            ),
        },
    )
    container = build_container(
        limited_settings,
        embedding_provider=DeterministicEmbeddingProvider(),
        research_provider=provider,
    )

    with TestClient(
        create_app(settings=limited_settings, container=container)
    ) as client:
        response = client.post("/api/research", json={"request": "multi-round"})

    assert response.status_code == 200
    assert [call[2] for call in provider.plan_calls] == [1, 2]
    assert [call[0] for call in provider.search_calls] == ["round-one", "round-two"]


def test_duplicate_and_oversized_queries_are_normalized(settings: Settings) -> None:
    short_settings = replace(settings, research_max_query_characters=12)
    provider = FakeResearchProvider(
        plans=(
            SearchPlan(
                (
                    "  Same Query  ",
                    "same query",
                    "a very long query that must be cut",
                ),
                True,
            ),
        ),
    )
    container = build_container(
        short_settings,
        embedding_provider=DeterministicEmbeddingProvider(),
        research_provider=provider,
    )

    with TestClient(create_app(settings=short_settings, container=container)) as client:
        response = client.post("/api/research", json={"request": "normalize queries"})

    assert response.status_code == 200
    assert [query for query, _limit in provider.search_calls] == [
        "Same Query",
        "a very long",
    ]


def test_unsafe_web_urls_are_not_passed_to_the_model_or_rendered(
    settings: Settings,
) -> None:
    provider = FakeResearchProvider(
        plans=(SearchPlan(("unsafe",), True),),
        search_results={
            "unsafe": (
                SearchResult("Safe", "https://example.test/safe", "safe"),
                SearchResult("Script", "javascript:alert(1)", "unsafe"),
                SearchResult("Credentials", "https://user@example.test/a", "unsafe"),
                SearchResult("Whitespace", "https://example.test/a\nmalicious", "unsafe"),
            )
        },
    )
    container = build_container(
        settings,
        embedding_provider=DeterministicEmbeddingProvider(),
        research_provider=provider,
    )

    with TestClient(create_app(settings=settings, container=container)) as client:
        response = client.post("/api/research", json={"request": "filter URLs"})

    assert response.status_code == 200
    assert [result.url for result in provider.answer_calls[0][2]] == [
        "https://example.test/safe"
    ]
    assert "javascript:" not in response.text


def test_late_model_failure_is_reported_inside_the_markdown_stream(
    settings: Settings,
) -> None:
    class FailingStreamProvider(FakeResearchProvider):
        async def start_answer(
            self,
            request: str,
            uploaded_context: Sequence[RetrievedChunk],
            web_context: Sequence[SearchResult],
        ):
            self.answer_calls.append(
                (request, tuple(uploaded_context), tuple(web_context))
            )

            async def stream():
                yield "# Partial answer"
                raise ResearchProviderError("Synthetic late failure.")

            return stream()

    provider = FailingStreamProvider()
    container = build_container(
        settings,
        embedding_provider=DeterministicEmbeddingProvider(),
        research_provider=provider,
    )

    with TestClient(create_app(settings=settings, container=container)) as client:
        response = client.post("/api/research", json={"request": "stream safely"})

    assert response.status_code == 200
    assert response.text.startswith("# Partial answer")
    assert "Answer generation stopped because the model provider failed." in response.text
    assert "## Sources" in response.text


def test_cancelling_first_pass_stops_retrieval_and_planning(settings: Settings) -> None:
    retrieval_started = asyncio.Event()
    retrieval_stopped = asyncio.Event()

    class BlockingRetriever:
        async def retrieve(
            self, workspace_id: str, query: str, limit: int | None = None
        ) -> Sequence[RetrievedChunk]:
            del workspace_id, query, limit
            retrieval_started.set()
            try:
                await asyncio.Future()
            finally:
                retrieval_stopped.set()

    provider = FakeResearchProvider(plan_delay_seconds=60)
    agent = ResearchAgentService(settings, BlockingRetriever(), provider)

    async def scenario() -> None:
        task = asyncio.create_task(agent.prepare_answer("workspace", "cancel me"))
        await retrieval_started.wait()
        await provider.plan_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert retrieval_stopped.is_set()
    assert provider.active_plans == 0
    assert provider.cancelled_plans == 1


def test_cancelling_search_stops_all_provider_work(settings: Settings) -> None:
    provider = FakeResearchProvider(
        plans=(SearchPlan(("slow-one", "slow-two"), True),),
        search_delay_seconds=60,
    )

    class EmptyRetriever:
        async def retrieve(
            self, workspace_id: str, query: str, limit: int | None = None
        ) -> Sequence[RetrievedChunk]:
            del workspace_id, query, limit
            return ()

    agent = ResearchAgentService(settings, EmptyRetriever(), provider)

    async def scenario() -> None:
        task = asyncio.create_task(agent.prepare_answer("workspace", "cancel search"))
        await provider.concurrent_searches_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert provider.active_searches == 0
    assert provider.cancelled_searches == 2


def test_stalled_answer_stream_times_out_and_is_closed(settings: Settings) -> None:
    timeout_settings = replace(
        settings,
        research_stream_idle_timeout_seconds=0.01,
        research_stream_close_timeout_seconds=0.1,
    )
    provider = FakeResearchProvider(
        answer_chunk_delay_seconds=60,
    )
    container = build_container(
        timeout_settings,
        embedding_provider=DeterministicEmbeddingProvider(),
        research_provider=provider,
    )

    with TestClient(
        create_app(settings=timeout_settings, container=container)
    ) as client:
        response = client.post("/api/research", json={"request": "stalled stream"})

    assert response.status_code == 200
    assert "Answer generation stopped after the provider stream timed out." in response.text
    assert provider.active_answer_streams == 0
    assert provider.closed_answer_streams == 1
    assert provider.interrupted_answer_streams == 1
