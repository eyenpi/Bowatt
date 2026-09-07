from __future__ import annotations

import asyncio
import os

import pytest

from app.providers.openai_research import OpenAIResearchProvider

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.getenv("BOWATT_RUN_LIVE_TESTS") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Set BOWATT_RUN_LIVE_TESTS=1 and OPENAI_API_KEY to run live provider tests.",
)
def test_openai_research_provider_live_smoke() -> None:
    provider = OpenAIResearchProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.getenv("BOWATT_RESEARCH_MODEL", "gpt-5.5"),
        timeout_seconds=60,
        max_output_tokens=300,
        close_timeout_seconds=2,
    )

    async def run() -> tuple[int, str]:
        try:
            results = await provider.search(
                "OpenAI official documentation Responses API web search",
                limit=2,
            )
            stream = await provider.start_answer(
                "In one sentence, what capability does the evidence describe?",
                (),
                results,
            )
            answer = "".join([chunk async for chunk in stream])
            return len(results), answer
        finally:
            await provider.close()

    result_count, answer = asyncio.run(run())

    assert result_count > 0
    assert answer.strip()
