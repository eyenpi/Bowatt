from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.ports import ResearchAgent, SourceIngestor
from app.services.research import ScaffoldResearchAgent
from app.services.source_ingestion import ValidatingSourceIngestor
from app.storage import InMemorySourceRepository


@dataclass(frozen=True, slots=True)
class AppContainer:
    source_ingestor: SourceIngestor
    research_agent: ResearchAgent


def build_container(settings: Settings) -> AppContainer:
    repository = InMemorySourceRepository()
    return AppContainer(
        source_ingestor=ValidatingSourceIngestor(settings, repository),
        research_agent=ScaffoldResearchAgent(settings, repository),
    )

