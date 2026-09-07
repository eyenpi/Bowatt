import os

from app.config import Settings
from app.container import build_container
from app.main import create_app
from tests.fakes import DeterministicEmbeddingProvider, FakeResearchProvider

settings = Settings.from_environment()
research_provider = FakeResearchProvider(
    answer_chunks=("# Research answer\n\n",)
    + tuple(f"Stream segment {index}.\n" for index in range(20)),
    answer_chunk_delay_seconds=float(os.getenv("BOWATT_TEST_STREAM_DELAY", "0.01")),
)
container = build_container(
    settings,
    embedding_provider=DeterministicEmbeddingProvider(),
    research_provider=research_provider,
)
app = create_app(settings=settings, container=container)


@app.get("/__test__/provider-state")
async def provider_state() -> dict[str, int]:
    return {
        "active_answer_streams": research_provider.active_answer_streams,
        "closed_answer_streams": research_provider.closed_answer_streams,
        "interrupted_answer_streams": research_provider.interrupted_answer_streams,
    }
