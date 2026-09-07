import random

import pytest

from app.models import SourceChunk, StoredSource
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
    _assert_source_is_preserved(source, first)
    assert all(len(chunk.text) <= 24 for chunk in first)
    assert [chunk.index for chunk in first] == list(range(len(first)))
    assert all(chunk.workspace_id == source.workspace_id for chunk in first)
    assert all(chunk.source_name == source.name for chunk in first)
    assert all(chunk.source_hash == source.content_hash for chunk in first)


def test_chunker_preserves_internal_whitespace_and_discards_empty_content() -> None:
    chunker = TextChunker(chunk_size=40, overlap=5)

    assert chunker.chunk(_source("  alpha\n\n beta\t gamma  "))[0].text == "alpha\n\n beta\t gamma"
    assert chunker.chunk(_source(" \n\t ")) == ()


def test_chunker_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        TextChunker(chunk_size=10, overlap=10)


def _assert_source_is_preserved(source: StoredSource, chunks: tuple[SourceChunk, ...]) -> None:
    covered: set[int] = set()
    previous_start = previous_end = -1
    for chunk in chunks:
        assert chunk.start_offset is not None and chunk.end_offset is not None
        assert chunk.start_offset > previous_start and chunk.end_offset > previous_end
        assert source.content[chunk.start_offset:chunk.end_offset] == chunk.text
        covered.update(range(chunk.start_offset, chunk.end_offset))
        previous_start, previous_end = chunk.start_offset, chunk.end_offset
    assert all(index in covered for index, char in enumerate(source.content) if not char.isspace())


def test_paragraphs_and_sentences_are_preferred_over_size_boundaries() -> None:
    paragraph_one = "Cats are small. They like naps."
    paragraph_two = "Dogs like walks. They chase balls."
    source = _source(paragraph_one + "\n\n" + paragraph_two)
    assert [chunk.text for chunk in TextChunker(40, 0).chunk(source)] == [
        paragraph_one, paragraph_two
    ]
    sentences = _source("Cats purr softly. Dogs bark loudly. Birds sing daily.")
    assert [chunk.text for chunk in TextChunker(30, 0).chunk(sentences)] == [
        "Cats purr softly.", "Dogs bark loudly.", "Birds sing daily."
    ]


def test_only_a_whole_trailing_sentence_is_repeated() -> None:
    source = _source("First sentence. Short. Another sentence. Final sentence.")
    chunks = TextChunker(30, 8).chunk(source)
    assert [chunk.text for chunk in chunks] == [
        "First sentence. Short.", "Short. Another sentence.", "Final sentence."
    ]
    _assert_source_is_preserved(source, chunks)


def test_headings_are_retained_and_overlap_never_crosses_sections() -> None:
    source = _source(
        "# Support\n\n## Enterprise\nFour hours.\n\n"
        "### Weekends\nEight hours.\n\n## Standard\nTwo days."
    )
    chunks = TextChunker(500, 100).chunk(source)
    assert [chunk.heading_path for chunk in chunks] == [
        ("Support", "Enterprise"), ("Support", "Enterprise", "Weekends"),
        ("Support", "Standard"),
    ]
    assert "# Support\n\n## Enterprise\nFour hours." in chunks[0].text
    assert "Four hours." not in chunks[1].text
    assert "Eight hours." not in chunks[2].text
    assert chunks[1].embedding_text.startswith("Section: Support > Enterprise > Weekends\n\n")
    _assert_source_is_preserved(source, chunks)


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_headings_and_blank_lines_in_fenced_code_stay_in_their_section(fence: str) -> None:
    code = f"{fence}python\n# This is a comment\n\nprint('hello')\n{fence}"
    source = _source(f"# Example\n\n{code}\n\n# Next\nOther content.")
    chunks = TextChunker(500, 40).chunk(source)
    assert len(chunks) == 2
    assert chunks[0].heading_path == ("Example",)
    assert code in chunks[0].text
    assert chunks[1].heading_path == ("Next",)
    _assert_source_is_preserved(source, chunks)


def test_lists_and_tables_remain_intact_when_they_fit() -> None:
    source = _source(
        "# Policy\n\n- Email support\n- Phone support\n\n"
        "| Plan | Hours |\n| --- | --- |\n| Enterprise | 4 |\n| Standard | 48 |"
    )
    chunks = TextChunker(500, 40).chunk(source)
    assert len(chunks) == 1
    assert chunks[0].text == source.content


@pytest.mark.parametrize(
    "content",
    [
        "研究資料について詳しく説明します。" * 50,
        "🧑🏽‍💻🚀" * 60,
        "averylongunbrokenword" * 100,
        "# " + "超長標題" * 100 + "\n\n" + "Evidence. " * 100,
        "# Section\r\n\r\nFirst paragraph.\r\n\r\nSecond paragraph.\r\n",
        "<|endoftext|> is literal source text. " * 40,
    ],
)
def test_token_limits_and_exact_offsets_hold_for_difficult_text(content: str) -> None:
    source = _source(content)
    chunker = TextChunker(100, 10, max_tokens=32)
    chunks = chunker.chunk(source)
    assert chunks
    assert all(chunker.count_tokens(chunk.embedding_text) <= 32 for chunk in chunks)
    assert all(0 < len(chunk.text) <= 100 for chunk in chunks)
    assert all("\ufffd" not in chunk.text for chunk in chunks)
    _assert_source_is_preserved(source, chunks)


def test_seeded_mixed_text_preserves_content_with_many_chunk_limits() -> None:
    rng = random.Random(42)
    vocabulary = ["Hello.", "World!", "研究", "😀", "longword" * 20, "\n\n", "\t", "# Topic\n"]
    for _ in range(40):
        source = _source(" ".join(rng.choices(vocabulary, k=40)))
        size = rng.choice([1, 15, 32, 100])
        max_tokens = rng.choice([8, 16, 32])
        chunker = TextChunker(size, size // 4, max_tokens)
        chunks = chunker.chunk(source)
        assert all(chunker.count_tokens(chunk.embedding_text) <= max_tokens for chunk in chunks)
        assert all(len(chunk.text) <= size for chunk in chunks)
        _assert_source_is_preserved(source, chunks)
