from __future__ import annotations

from dataclasses import replace

import pytest

from app.config import Settings


def test_environment_configuration_accepts_zero_chunk_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOWATT_CHUNK_OVERLAP", "0")
    monkeypatch.setenv("BOWATT_CHUNK_MAX_TOKENS", "256")

    settings = Settings.from_environment()

    assert settings.chunk_overlap == 0
    assert settings.chunk_max_tokens == 256


def test_environment_configuration_loads_research_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOWATT_RESEARCH_MODEL", "test-research-model")
    monkeypatch.setenv("BOWATT_RESEARCH_MAX_SEARCH_ROUNDS", "3")
    monkeypatch.setenv("BOWATT_RESEARCH_SEARCH_RETRY_DELAY_SECONDS", "0")
    monkeypatch.setenv("BOWATT_RESEARCH_MAX_WEB_RESULTS", "20")

    settings = Settings.from_environment()

    assert settings.research_model == "test-research-model"
    assert settings.research_max_search_rounds == 3
    assert settings.research_search_retry_delay_seconds == 0
    assert settings.research_max_web_results == 20


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"chunk_size": 10, "chunk_overlap": 10}, "smaller than chunk_size"),
        ({"chunk_max_tokens": 7}, "between 8 and 8191"),
        ({"chunk_max_tokens": 8_192}, "between 8 and 8191"),
        ({"embedding_batch_size": 2_049}, "cannot exceed 2048"),
        ({"embedding_timeout_seconds": 0}, "must be greater than zero"),
        ({"embedding_model": "   "}, "must not be blank"),
    ],
)
def test_settings_reject_invalid_embedding_configuration(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(Settings(), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"research_model": " "}, "must not be blank"),
        ({"research_provider_timeout_seconds": 0}, "must be greater than zero"),
        ({"research_stream_idle_timeout_seconds": 0}, "must be greater than zero"),
        ({"research_stream_close_timeout_seconds": 0}, "must be greater than zero"),
        ({"research_search_timeout_seconds": 0}, "must be greater than zero"),
        ({"research_search_retry_delay_seconds": -1}, "zero or greater"),
        ({"research_max_search_rounds": 6}, "cannot exceed 5"),
        ({"research_queries_per_round": 11}, "cannot exceed 10"),
        ({"research_search_max_attempts": 6}, "cannot exceed 5"),
        ({"research_max_web_results": 51}, "cannot exceed 50"),
    ],
)
def test_settings_reject_invalid_research_configuration(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(Settings(), **changes)


def test_settings_repr_does_not_expose_api_key() -> None:
    settings = Settings(openai_api_key="secret-value")

    assert "secret-value" not in repr(settings)
