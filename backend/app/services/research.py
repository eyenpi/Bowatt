from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.config import Settings
from app.errors import ApiError, ResearchProviderError
from app.models import RetrievedChunk, SearchPlan, SearchResult
from app.ports import ResearchProvider, SourceRetriever

TRACKING_QUERY_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SearchOutcome:
    results: tuple[SearchResult, ...]
    succeeded: bool


class ResearchAgentService:
    def __init__(
        self,
        settings: Settings,
        source_retriever: SourceRetriever,
        research_provider: ResearchProvider,
    ) -> None:
        self._settings = settings
        self._source_retriever = source_retriever
        self._research_provider = research_provider
        self._search_semaphore = asyncio.Semaphore(
            settings.research_search_concurrency
        )

    async def prepare_answer(
        self, workspace_id: str, request: str
    ) -> AsyncIterator[str]:
        normalized_request = request.strip()
        if not normalized_request:
            raise ApiError(400, "Research request must not be blank.")
        if len(normalized_request) > self._settings.max_request_characters:
            raise ApiError(
                413,
                "Research request exceeds the configured character limit.",
            )

        uploaded_task = asyncio.create_task(
            self._source_retriever.retrieve(workspace_id, normalized_request)
        )
        plan_task = asyncio.create_task(
            self._create_search_plan(normalized_request, (), 1)
        )
        try:
            uploaded_context, plan = await asyncio.gather(
                uploaded_task,
                plan_task,
            )
        except BaseException:
            uploaded_task.cancel()
            plan_task.cancel()
            await asyncio.gather(
                uploaded_task,
                plan_task,
                return_exceptions=True,
            )
            raise

        web_context: list[SearchResult] = []
        seen_queries: set[str] = set()
        seen_urls: set[str] = set()
        attempted_searches = 0
        successful_searches = 0

        for round_number in range(1, self._settings.research_max_search_rounds + 1):
            if round_number > 1:
                plan = await self._create_search_plan(
                    normalized_request,
                    tuple(web_context),
                    round_number,
                )

            queries = self._normalized_queries(plan, seen_queries)
            if not queries:
                break

            outcomes = await asyncio.gather(
                *(self._search_with_retries(query) for query in queries)
            )
            attempted_searches += len(queries)
            successful_searches += sum(outcome.succeeded for outcome in outcomes)
            for outcome in outcomes:
                for result in outcome.results:
                    normalized_result = self._normalized_result(result)
                    if normalized_result is None:
                        continue
                    canonical_url = _canonical_url(normalized_result.url)
                    if canonical_url in seen_urls:
                        continue
                    seen_urls.add(canonical_url)
                    web_context.append(normalized_result)
                    if len(web_context) >= self._settings.research_max_web_results:
                        break
                if len(web_context) >= self._settings.research_max_web_results:
                    break

            if (
                plan.is_complete
                or len(web_context) >= self._settings.research_max_web_results
            ):
                break

        if attempted_searches and not successful_searches and not uploaded_context:
            raise ApiError(502, "Research sources are temporarily unavailable.")

        answer_stream = await self._start_answer(
            normalized_request,
            tuple(uploaded_context),
            tuple(web_context),
        )
        rendered_stream = self._stream_with_sources(
            answer_stream,
            tuple(uploaded_context),
            tuple(web_context),
        )
        return _ManagedAnswerStream(
            rendered_stream,
            answer_stream,
            self._settings.research_stream_close_timeout_seconds,
        )

    async def _create_search_plan(
        self,
        request: str,
        prior_web_context: Sequence[SearchResult],
        round_number: int,
    ) -> SearchPlan:
        try:
            async with asyncio.timeout(
                self._settings.research_provider_timeout_seconds
            ):
                return await self._research_provider.create_search_plan(
                    request,
                    prior_web_context,
                    round_number,
                )
        except TimeoutError as error:
            raise ApiError(504, "Research planning timed out.") from error
        except ResearchProviderError as error:
            raise ApiError(error.status_code, error.public_message) from error

    async def _start_answer(
        self,
        request: str,
        uploaded_context: Sequence[RetrievedChunk],
        web_context: Sequence[SearchResult],
    ) -> AsyncIterator[str]:
        try:
            async with asyncio.timeout(
                self._settings.research_provider_timeout_seconds
            ):
                return await self._research_provider.start_answer(
                    request,
                    uploaded_context,
                    web_context,
                )
        except TimeoutError as error:
            raise ApiError(504, "Answer generation timed out.") from error
        except ResearchProviderError as error:
            raise ApiError(error.status_code, error.public_message) from error

    def _normalized_queries(
        self,
        plan: SearchPlan,
        seen_queries: set[str],
    ) -> tuple[str, ...]:
        queries: list[str] = []
        for raw_query in plan.queries:
            query = " ".join(raw_query.split())[
                : self._settings.research_max_query_characters
            ].strip()
            dedupe_key = query.casefold()
            if not query or dedupe_key in seen_queries:
                continue
            seen_queries.add(dedupe_key)
            queries.append(query)
            if len(queries) >= self._settings.research_queries_per_round:
                break
        return tuple(queries)

    async def _search_with_retries(self, query: str) -> _SearchOutcome:
        for attempt in range(self._settings.research_search_max_attempts):
            try:
                async with self._search_semaphore:
                    async with asyncio.timeout(
                        self._settings.research_search_timeout_seconds
                    ):
                        results = await self._research_provider.search(
                            query,
                            self._settings.research_search_results_per_query,
                        )
                return _SearchOutcome(tuple(results), True)
            except (ResearchProviderError, TimeoutError):
                is_last_attempt = (
                    attempt + 1 >= self._settings.research_search_max_attempts
                )
                if is_last_attempt:
                    return _SearchOutcome((), False)
                delay = self._settings.research_search_retry_delay_seconds * (
                    2**attempt
                )
                if delay:
                    await asyncio.sleep(delay)

        return _SearchOutcome((), False)

    def _normalized_result(self, result: SearchResult) -> SearchResult | None:
        url = result.url.strip()
        if any(character.isspace() or ord(character) < 32 for character in url):
            return None
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return None
        if parsed.username or parsed.password:
            return None
        try:
            _port = parsed.port
        except ValueError:
            return None

        title = " ".join(result.title.split()) or parsed.hostname
        snippet = " ".join(result.snippet.split())[
            : self._settings.research_max_snippet_characters
        ].strip()
        return SearchResult(title=title, url=url, snippet=snippet)

    async def _stream_with_sources(
        self,
        answer_stream: AsyncIterator[str],
        uploaded_context: Sequence[RetrievedChunk],
        web_context: Sequence[SearchResult],
    ) -> AsyncIterator[str]:
        try:
            iterator = answer_stream.__aiter__()
            while True:
                try:
                    async with asyncio.timeout(
                        self._settings.research_stream_idle_timeout_seconds
                    ):
                        chunk = await anext(iterator)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    yield (
                        "\n\n> Answer generation stopped after the provider stream "
                        "timed out.\n"
                    )
                    break
                if chunk:
                    yield chunk
        except ResearchProviderError:
            yield "\n\n> Answer generation stopped because the model provider failed.\n"

        if not uploaded_context and not web_context:
            return

        yield "\n\n## Sources\n\n"
        for index, retrieved in enumerate(uploaded_context, start=1):
            filename = _inline_code(retrieved.chunk.source_name)
            yield (
                f"- [U{index}] Uploaded: `{filename}` "
                f"({retrieved.chunk.location})\n"
            )
        for index, result in enumerate(web_context, start=1):
            title = _markdown_label(result.title)
            link = result.url.replace(" ", "%20").replace(")", "%29")
            yield f"- [W{index}] [{title}]({link})\n"


def _canonical_url(url: str) -> str:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        return url.casefold()
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    netloc = hostname if port is None or default_port else f"{hostname}:{port}"
    path = parsed.path.rstrip("/")
    filtered_query = sorted(
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in TRACKING_QUERY_PARAMETERS
    )
    return urlunsplit((scheme, netloc, path, urlencode(filtered_query), ""))


def _markdown_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _inline_code(value: str) -> str:
    return " ".join(value.split()).replace("`", "'")


class _ManagedAnswerStream:
    def __init__(
        self,
        rendered_stream: AsyncIterator[str],
        provider_stream: AsyncIterator[str],
        close_timeout_seconds: float,
    ) -> None:
        self._rendered_stream = rendered_stream
        self._provider_stream = provider_stream
        self._close_timeout_seconds = close_timeout_seconds
        self._closed = False

    def __aiter__(self) -> _ManagedAnswerStream:
        return self

    async def __anext__(self) -> str:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await anext(self._rendered_stream)
        except StopAsyncIteration:
            await self.aclose()
            raise
        except asyncio.CancelledError:
            await self.aclose()
            raise
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await _close_stream(
            self._rendered_stream,
            self._close_timeout_seconds,
        )
        await _close_stream(
            self._provider_stream,
            self._close_timeout_seconds,
        )


async def _close_stream(stream: object, timeout_seconds: float) -> None:
    close = getattr(stream, "aclose", None)
    if not callable(close):
        return

    try:
        async with asyncio.timeout(timeout_seconds):
            result = close()
            if inspect.isawaitable(result):
                await result
    except TimeoutError:
        logger.warning("Timed out while closing the research provider stream.")
    except Exception:
        logger.exception("Failed to close the research provider stream.")
