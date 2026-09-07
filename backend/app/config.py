from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _positive_integer(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error

    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")

    return value


def _optional_positive_integer(name: str) -> int | None:
    raw_value = os.getenv(name)
    if not raw_value:
        return None

    return _positive_integer(name, 1)


def _non_negative_integer(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error

    if value < 0:
        raise RuntimeError(f"{name} must be zero or greater")

    return value


def _non_negative_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number") from error

    if value < 0:
        raise RuntimeError(f"{name} must be zero or greater")

    return value


def _positive_float(name: str, default: float) -> float:
    value = _non_negative_float(name, default)
    if value == 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str = "Bowatt Research Agent API"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8787
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    default_workspace_id: str = "local-default"
    max_upload_files: int = 10
    max_upload_bytes: int = 5_000_000
    max_request_characters: int = 10_000
    database_path: Path = Path("data/research.db")
    chunk_size: int = 2_000
    chunk_overlap: int = 200
    chunk_max_tokens: int = 512
    embedding_batch_size: int = 64
    embedding_concurrency: int = 4
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = None
    embedding_timeout_seconds: float = 30.0
    retrieval_limit: int = 5
    research_model: str = "gpt-5.5"
    research_provider_timeout_seconds: float = 60.0
    research_max_output_tokens: int = 3_000
    research_stream_idle_timeout_seconds: float = 30.0
    research_stream_close_timeout_seconds: float = 2.0
    research_max_search_rounds: int = 2
    research_queries_per_round: int = 3
    research_search_results_per_query: int = 5
    research_search_concurrency: int = 3
    research_search_max_attempts: int = 2
    research_search_retry_delay_seconds: float = 0.2
    research_search_timeout_seconds: float = 30.0
    research_max_web_results: int = 12
    research_max_query_characters: int = 500
    research_max_snippet_characters: int = 2_000
    openai_api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        positive_values = {
            "max_upload_files": self.max_upload_files,
            "max_upload_bytes": self.max_upload_bytes,
            "max_request_characters": self.max_request_characters,
            "chunk_size": self.chunk_size,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_concurrency": self.embedding_concurrency,
            "retrieval_limit": self.retrieval_limit,
            "research_max_output_tokens": self.research_max_output_tokens,
            "research_max_search_rounds": self.research_max_search_rounds,
            "research_queries_per_round": self.research_queries_per_round,
            "research_search_results_per_query": self.research_search_results_per_query,
            "research_search_concurrency": self.research_search_concurrency,
            "research_search_max_attempts": self.research_search_max_attempts,
            "research_max_web_results": self.research_max_web_results,
            "research_max_query_characters": self.research_max_query_characters,
            "research_max_snippet_characters": self.research_max_snippet_characters,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be zero or greater")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not 8 <= self.chunk_max_tokens <= 8_191:
            raise ValueError("chunk_max_tokens must be between 8 and 8191")
        if self.embedding_batch_size > 2_048:
            raise ValueError("embedding_batch_size cannot exceed 2048")
        if self.embedding_dimensions is not None and self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be greater than zero")
        if self.embedding_timeout_seconds <= 0:
            raise ValueError("embedding_timeout_seconds must be greater than zero")
        if self.research_provider_timeout_seconds <= 0:
            raise ValueError("research_provider_timeout_seconds must be greater than zero")
        if self.research_stream_idle_timeout_seconds <= 0:
            raise ValueError("research_stream_idle_timeout_seconds must be greater than zero")
        if self.research_stream_close_timeout_seconds <= 0:
            raise ValueError("research_stream_close_timeout_seconds must be greater than zero")
        if self.research_search_timeout_seconds <= 0:
            raise ValueError("research_search_timeout_seconds must be greater than zero")
        if self.research_search_retry_delay_seconds < 0:
            raise ValueError(
                "research_search_retry_delay_seconds must be zero or greater"
            )
        if self.research_max_search_rounds > 5:
            raise ValueError("research_max_search_rounds cannot exceed 5")
        if self.research_queries_per_round > 10:
            raise ValueError("research_queries_per_round cannot exceed 10")
        if self.research_search_max_attempts > 5:
            raise ValueError("research_search_max_attempts cannot exceed 5")
        if self.research_max_web_results > 50:
            raise ValueError("research_max_web_results cannot exceed 50")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be blank")
        if not self.research_model.strip():
            raise ValueError("research_model must not be blank")

    @classmethod
    def from_environment(cls) -> Settings:
        origins = tuple(
            origin.strip()
            for origin in os.getenv("BOWATT_CORS_ORIGINS", "http://localhost:5173").split(",")
            if origin.strip()
        )

        if not origins:
            raise RuntimeError("BOWATT_CORS_ORIGINS must contain at least one origin")

        return cls(
            environment=os.getenv("BOWATT_ENVIRONMENT", "development"),
            host=os.getenv("BOWATT_HOST", "0.0.0.0"),
            port=_positive_integer("BOWATT_PORT", 8787),
            cors_origins=origins,
            default_workspace_id=os.getenv(
                "BOWATT_DEFAULT_WORKSPACE_ID", "local-default"
            ),
            max_upload_files=_positive_integer("BOWATT_MAX_UPLOAD_FILES", 10),
            max_upload_bytes=_positive_integer("BOWATT_MAX_UPLOAD_BYTES", 5_000_000),
            max_request_characters=_positive_integer(
                "BOWATT_MAX_REQUEST_CHARACTERS", 10_000
            ),
            database_path=Path(
                os.getenv("BOWATT_DATABASE_PATH", "data/research.db")
            ).expanduser(),
            chunk_size=_positive_integer("BOWATT_CHUNK_SIZE", 2_000),
            chunk_overlap=_non_negative_integer("BOWATT_CHUNK_OVERLAP", 200),
            chunk_max_tokens=_positive_integer("BOWATT_CHUNK_MAX_TOKENS", 512),
            embedding_batch_size=_positive_integer("BOWATT_EMBEDDING_BATCH_SIZE", 64),
            embedding_concurrency=_positive_integer("BOWATT_EMBEDDING_CONCURRENCY", 4),
            embedding_model=os.getenv(
                "BOWATT_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            embedding_dimensions=_optional_positive_integer(
                "BOWATT_EMBEDDING_DIMENSIONS"
            ),
            embedding_timeout_seconds=_positive_float(
                "BOWATT_EMBEDDING_TIMEOUT_SECONDS", 30.0
            ),
            retrieval_limit=_positive_integer("BOWATT_RETRIEVAL_LIMIT", 5),
            research_model=os.getenv("BOWATT_RESEARCH_MODEL", "gpt-5.5"),
            research_provider_timeout_seconds=_positive_float(
                "BOWATT_RESEARCH_PROVIDER_TIMEOUT_SECONDS", 60.0
            ),
            research_max_output_tokens=_positive_integer(
                "BOWATT_RESEARCH_MAX_OUTPUT_TOKENS", 3_000
            ),
            research_stream_idle_timeout_seconds=_positive_float(
                "BOWATT_RESEARCH_STREAM_IDLE_TIMEOUT_SECONDS", 30.0
            ),
            research_stream_close_timeout_seconds=_positive_float(
                "BOWATT_RESEARCH_STREAM_CLOSE_TIMEOUT_SECONDS", 2.0
            ),
            research_max_search_rounds=_positive_integer(
                "BOWATT_RESEARCH_MAX_SEARCH_ROUNDS", 2
            ),
            research_queries_per_round=_positive_integer(
                "BOWATT_RESEARCH_QUERIES_PER_ROUND", 3
            ),
            research_search_results_per_query=_positive_integer(
                "BOWATT_RESEARCH_SEARCH_RESULTS_PER_QUERY", 5
            ),
            research_search_concurrency=_positive_integer(
                "BOWATT_RESEARCH_SEARCH_CONCURRENCY", 3
            ),
            research_search_max_attempts=_positive_integer(
                "BOWATT_RESEARCH_SEARCH_MAX_ATTEMPTS", 2
            ),
            research_search_retry_delay_seconds=_non_negative_float(
                "BOWATT_RESEARCH_SEARCH_RETRY_DELAY_SECONDS", 0.2
            ),
            research_search_timeout_seconds=_positive_float(
                "BOWATT_RESEARCH_SEARCH_TIMEOUT_SECONDS", 30.0
            ),
            research_max_web_results=_positive_integer(
                "BOWATT_RESEARCH_MAX_WEB_RESULTS", 12
            ),
            research_max_query_characters=_positive_integer(
                "BOWATT_RESEARCH_MAX_QUERY_CHARACTERS", 500
            ),
            research_max_snippet_characters=_positive_integer(
                "BOWATT_RESEARCH_MAX_SNIPPET_CHARACTERS", 2_000
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        )
