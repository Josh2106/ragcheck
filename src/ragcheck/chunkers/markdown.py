from __future__ import annotations

import re
from typing import Any

from ragcheck.chunkers._base import Span, build_chunks
from ragcheck.chunkers.recursive import RecursiveChunker
from ragcheck.types import Chunk, Document

__all__ = ["MarkdownHeaderChunker"]

_HEADER = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


class MarkdownHeaderChunker:
    """Split Markdown at ATX headers (``#`` .. ``######``), ignoring fenced code blocks.

    Every chunk records its header path in ``metadata["headers"]`` (``{"h1": ..,
    "h2": ..}``) and ``metadata["section"]`` (``"Install > Windows"``). Sections longer
    than ``max_chunk_size`` characters are further split with :class:`RecursiveChunker`.
    """

    def __init__(
        self, max_chunk_size: int = 1000, max_header_level: int = 6, chunk_overlap: int = 0
    ) -> None:
        if not 1 <= max_header_level <= 6:
            raise ValueError("max_header_level must be in 1..6")
        self.max_header_level = max_header_level
        self.max_chunk_size = max_chunk_size
        self._splitter = RecursiveChunker(chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)

    def _sections(self, text: str) -> list[tuple[Span, dict[int, str]]]:
        sections: list[tuple[Span, dict[int, str]]] = []
        stack: dict[int, str] = {}
        sec_start = 0
        sec_headers: dict[int, str] = {}
        fence: str | None = None
        pos = 0
        for line in text.splitlines(keepends=True):
            stripped = line.rstrip("\r\n")
            fm = _FENCE.match(stripped)
            if fm:
                marker = fm.group(1)
                if fence is None:
                    fence = marker[0] * 3
                elif marker.startswith(fence):
                    fence = None
            elif fence is None:
                hm = _HEADER.match(stripped)
                if hm and len(hm.group(1)) <= self.max_header_level:
                    level = len(hm.group(1))
                    if pos > sec_start:
                        sections.append(((sec_start, pos), dict(sec_headers)))
                    stack = {lvl: t for lvl, t in stack.items() if lvl < level}
                    stack[level] = hm.group(2).strip()
                    sec_start = pos
                    sec_headers = dict(stack)
            pos += len(line)
        if pos > sec_start:
            sections.append(((sec_start, pos), dict(sec_headers)))
        return sections

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        spans: list[Span] = []
        span_headers: list[dict[int, str]] = []
        for (s, e), headers in self._sections(text):
            for sub in self._splitter.split_spans(text, s, e):
                spans.append(sub)
                span_headers.append(headers)
        # build_chunks drops whitespace-only spans, so map metadata by (stripped) position
        by_start: dict[int, dict[int, str]] = {}
        for (s, e), h in zip(spans, span_headers, strict=True):
            j = s
            while j < e and text[j].isspace():
                j += 1
            by_start[j] = h

        def meta(_i: int, span: Span) -> dict[str, Any]:
            headers = by_start.get(span[0], {})
            return {
                "chunker": "markdown",
                "headers": {f"h{lvl}": t for lvl, t in sorted(headers.items())},
                "section": " > ".join(t for _, t in sorted(headers.items())),
            }

        return build_chunks(document, spans, meta)
