from app.config import Settings
from app.container import build_container
from app.main import create_app
from tests.fakes import DeterministicEmbeddingProvider

settings = Settings.from_environment()
container = build_container(settings, embedding_provider=DeterministicEmbeddingProvider())
app = create_app(settings=settings, container=container)

