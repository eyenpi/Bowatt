from __future__ import annotations

import re
from collections.abc import Iterator

import tiktoken

from app.models import SourceChunk, StoredSource

_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_SENTENCE_END = re.compile(r"[.!?][\"'”’\)\]]*(?=\s|$)|[。！？]")
_SEPARATORS = (
    re.compile(r"\n"),
    re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*"),
    re.compile(r"\s+"),
)


def _fence_after(line: str, fence: str | None) -> str | None:
    match = _FENCE.match(line.rstrip("\r\n"))
    if match:
        marker, suffix = match.groups()
        if fence is None:
            return marker
        if marker[0] == fence[0] and len(marker) >= len(fence) and not suffix.strip():
            return None
    return fence


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sections(text: str) -> Iterator[tuple[int, int, tuple[str, ...]]]:
    start = offset = 0
    headings: list[tuple[int, str]] = []
    fence: str | None = None
    has_body = False
    for line in text.splitlines(keepends=True):
        heading = _HEADING.match(line) if fence is None else None
        if heading:
            if has_body:
                yield start, offset, tuple(title for _, title in headings)
                start = offset
                has_body = False
            level = len(heading[1])
            title = re.sub(r"\s+#+\s*$", "", heading[2]).strip()
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, title))
        elif line.strip():
            has_body = True
        fence = _fence_after(line, fence)
        offset += len(line)
    yield start, len(text), tuple(title for _, title in headings)


def _blocks(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    block_start = offset = start
    fence: str | None = None
    for line in text[start:end].splitlines(keepends=True):
        if not line.strip() and fence is None:
            left, right = _trim(text, block_start, offset)
            if left < right:
                yield left, right
            block_start = offset + len(line)
        fence = _fence_after(line, fence)
        offset += len(line)
    left, right = _trim(text, block_start, end)
    if left < right:
        yield left, right


class TextChunker:
    """Preserve sections and paragraphs, with bounded sentence overlap and token limits."""

    def __init__(self, chunk_size: int, overlap: int, max_tokens: int = 512) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be zero or greater and smaller than chunk_size")
        if not 8 <= max_tokens <= 8_191:
            raise ValueError("max_tokens must be between 8 and 8191")
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._max_tokens = max_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def signature(self) -> str:
        return (
            f"structure-v1:cl100k_base:{self._chunk_size}:"
            f"{self._overlap}:{self._max_tokens}"
        )

    def count_tokens(self, text: str) -> int:
        # Source files may contain strings such as <|endoftext|>; treat these literally.
        return len(self._encoding.encode_ordinary(text))

    def _fits(self, text: str, prefix: str) -> bool:
        return (
            len(text) <= self._chunk_size
            and self.count_tokens(prefix + text) <= self._max_tokens
        )

    def _context_prefix(self, headings: tuple[str, ...]) -> str:
        if not headings or self._chunk_size < 64 or self._max_tokens < 32:
            return ""
        context = ("Section: " + " > ".join(headings))[: self._chunk_size // 4]
        while context and self.count_tokens(context + "\n\n") > self._max_tokens // 4:
            context = context[:-1]
        return context + "\n\n" if context else ""

    def _split_block(
        self, text: str, start: int, end: int, prefix: str, depth: int = 0
    ) -> Iterator[tuple[int, int]]:
        if self._fits(text[start:end], prefix):
            yield start, end
            return
        if depth < len(_SEPARATORS):
            cursor = start
            for match in _SEPARATORS[depth].finditer(text, start, end):
                left, right = _trim(text, cursor, match.end())
                if left < right:
                    yield from self._split_block(text, left, right, prefix, depth + 1)
                cursor = match.end()
            left, right = _trim(text, cursor, end)
            if left < right:
                yield from self._split_block(text, left, right, prefix, depth + 1)
            return

        # Last resort for a single oversized word/line. Slice Unicode characters,
        # never decoded partial tokens, so a boundary cannot corrupt UTF-8 text.
        while start < end:
            right = min(start + self._chunk_size, end)
            while not self._fits(text[start:right], prefix):
                right = start + max(1, (right - start) // 2)
            yield start, right
            start = right

    def _overlap_start(self, text: str, start: int, end: int) -> int | None:
        matches = list(_SENTENCE_END.finditer(text, start, end))
        if not matches or matches[-1].end() != end:
            return None
        left = matches[-2].end() if len(matches) > 1 else start
        left, _ = _trim(text, left, end)
        return left if end - left <= self._overlap else None

    def chunk(self, source: StoredSource) -> tuple[SourceChunk, ...]:
        text = source.content
        chunks: list[SourceChunk] = []
        for start, end, headings in _sections(text):
            prefix = self._context_prefix(headings)
            spans: list[tuple[int, int]] = []
            for left, right in _blocks(text, start, end):
                for unit_start, unit_end in self._split_block(text, left, right, prefix):
                    if not spans:
                        spans.append((unit_start, unit_end))
                    elif self._fits(text[spans[-1][0]:unit_end], prefix):
                        spans[-1] = spans[-1][0], unit_end
                    else:
                        overlap = self._overlap_start(text, *spans[-1])
                        if overlap is not None and self._fits(text[overlap:unit_end], prefix):
                            unit_start = overlap
                        spans.append((unit_start, unit_end))
            for left, right in spans:
                chunks.append(
                    SourceChunk(
                        source_hash=source.content_hash,
                        source_name=source.name,
                        workspace_id=source.workspace_id,
                        index=len(chunks),
                        text=text[left:right],
                        heading_path=headings,
                        start_offset=left,
                        end_offset=right,
                        context_prefix=prefix,
                    )
                )
        return tuple(chunks)
