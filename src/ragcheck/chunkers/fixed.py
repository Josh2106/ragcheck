from __future__ import annotations

from typing import Literal

from ragcheck.chunkers._base import Span, build_chunks, validate_size
from ragcheck.tokenizers import RegexTokenizer, Tokenizer
from ragcheck.types import Chunk, Document

__all__ = ["FixedSizeChunker"]


class FixedSizeChunker:
    """Sliding window of ``size`` characters or tokens with ``overlap``.

    In token mode the window boundaries are token boundaries, so no token is ever cut,
    and a chunk never contains more than ``size`` tokens.
    """

    def __init__(
        self,
        size: int = 512,
        overlap: int = 0,
        unit: Literal["chars", "tokens"] = "chars",
        tokenizer: Tokenizer | None = None,
    ) -> None:
        validate_size(size, overlap)
        if unit not in ("chars", "tokens"):
            raise ValueError(f"unit must be 'chars' or 'tokens', got {unit!r}")
        self.size = size
        self.overlap = overlap
        self.unit = unit
        self.tokenizer: Tokenizer = tokenizer or RegexTokenizer()

    def _char_spans(self, n: int) -> list[Span]:
        step = self.size - self.overlap
        spans: list[Span] = []
        i = 0
        while i < n:
            e = min(i + self.size, n)
            spans.append((i, e))
            if e >= n:
                break
            i += step
        return spans

    def _token_spans(self, text: str) -> list[Span]:
        toks = self.tokenizer.spans(text)
        step = self.size - self.overlap
        spans: list[Span] = []
        i = 0
        while i < len(toks):
            j = min(i + self.size, len(toks))
            spans.append((toks[i][0], toks[j - 1][1]))
            if j >= len(toks):
                break
            i += step
        return spans

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        spans = self._char_spans(len(text)) if self.unit == "chars" else self._token_spans(text)
        return build_chunks(document, spans, lambda _i, _s: {"chunker": "fixed"})
