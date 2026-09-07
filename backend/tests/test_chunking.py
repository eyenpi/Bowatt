from app.models import StoredSource
from app.services.chunking import TextChunker


def _source(content: str) -> StoredSource:
    return StoredSource(
        workspace_id="workspace-one",
        name="sample.txt",
        size=len(content.encode()),
        media_type="text/plain",
        content=content,
        content_hash="content-hash",
    )


def test_chunker_is_deterministic_and_preserves_metadata() -> None:
    chunker = TextChunker(chunk_size=24, overlap=8)
    source = _source("alpha beta gamma delta epsilon zeta eta theta iota kappa")

    first = chunker.chunk(source)
    second = chunker.chunk(source)

    assert first == second
    assert [chunk.text for chunk in first] == [
        "alpha beta gamma delta",
        "gamma delta epsilon zeta",
        "epsilon zeta eta theta",
        "eta theta iota kappa",
    ]
    assert all(len(chunk.text) <= 24 for chunk in first)
    assert [chunk.index for chunk in first] == [0, 1, 2, 3]
    assert all(chunk.workspace_id == source.workspace_id for chunk in first)
    assert all(chunk.source_name == source.name for chunk in first)
    assert all(chunk.source_hash == source.content_hash for chunk in first)


def test_chunker_normalizes_whitespace_and_discards_empty_content() -> None:
    chunker = TextChunker(chunk_size=40, overlap=5)

    assert chunker.chunk(_source("  alpha\n\n beta\t gamma  "))[0].text == "alpha beta gamma"
    assert chunker.chunk(_source(" \n\t ")) == ()


def test_chunker_rejects_invalid_configuration() -> None:
    try:
        TextChunker(chunk_size=10, overlap=10)
    except ValueError as error:
        assert "smaller than chunk_size" in str(error)
    else:
        raise AssertionError("Expected invalid overlap to be rejected")

