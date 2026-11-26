from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ragcheck.types import Chunk, Document

Span = tuple[int, int]


def strip_span(text: str, start: int, end: int) -> Span:
    """Shrink ``[start, end)`` so it neither starts nor ends with whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def build_chunks(
    document: Document,
    spans: Iterable[Span],
    metadata: Callable[[int, Span], dict[str, Any]] | None = None,
) -> list[Chunk]:
    """Turn raw spans into :class:`Chunk` objects, dropping whitespace-only spans."""
    out: list[Chunk] = []
    text = document.text
    for raw in spans:
        s, e = strip_span(text, *raw)
        if s >= e:
            continue
        idx = len(out)
        md: dict[str, Any] = dict(document.metadata)
        if metadata is not None:
            md.update(metadata(idx, (s, e)))
        out.append(
            Chunk(
                id=f"{document.id}::{idx}",
                doc_id=document.id,
                text=text[s:e],
                start=s,
                end=e,
                index=idx,
                metadata=md,
            )
        )
    return out


def hard_split(start: int, end: int, size: int) -> list[Span]:
    return [(i, min(i + size, end)) for i in range(start, end, size)]


def merge_spans(
    pieces: Sequence[Span],
    size: int,
    overlap: int,
    length: Callable[[int, int], int],
) -> list[Span]:
    """Greedily merge contiguous pieces into spans of at most ``size`` (by ``length``).

    When a span is emitted, trailing pieces totalling at most ``overlap`` are carried into
    the next span. Each piece must already satisfy ``length(piece) <= size``.
    """
    out: list[Span] = []
    cur: list[Span] = []
    for p in pieces:
        if cur and length(cur[0][0], p[1]) > size:
            out.append((cur[0][0], cur[-1][1]))
            while cur and (
                length(cur[0][0], cur[-1][1]) > overlap or length(cur[0][0], p[1]) > size
            ):
                cur.pop(0)
        cur.append(p)
    if cur:
        out.append((cur[0][0], cur[-1][1]))
    return out


def validate_size(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got {overlap} (size={size})")
