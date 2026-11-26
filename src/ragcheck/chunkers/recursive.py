from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from ragcheck.chunkers._base import Span, build_chunks, hard_split, merge_spans, validate_size
from ragcheck.tokenizers import RegexTokenizer, Tokenizer
from ragcheck.types import Chunk, Document

__all__ = ["DEFAULT_SEPARATORS", "RecursiveChunker"]

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


class RecursiveChunker:
    """Split on the coarsest separator that works, recursing into oversize pieces.

    Separators are tried in order (paragraphs, lines, sentences, words, characters).
    Separators stay attached to the end of the preceding piece, so the pieces tile the
    original text exactly. Pieces are then merged greedily up to ``chunk_size`` with
    ``chunk_overlap`` carried between neighbouring chunks.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        unit: Literal["chars", "tokens"] = "chars",
        tokenizer: Tokenizer | None = None,
    ) -> None:
        validate_size(chunk_size, chunk_overlap)
        if not separators:
            raise ValueError("separators must not be empty")
        if unit not in ("chars", "tokens"):
            raise ValueError(f"unit must be 'chars' or 'tokens', got {unit!r}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        seps = list(separators)
        if seps[-1] != "":
            seps.append("")  # guarantee termination: final fallback is a hard char split
        self.separators = tuple(seps)
        self.unit = unit
        self.tokenizer: Tokenizer = tokenizer or RegexTokenizer()
        self._text = ""

    def _len(self, s: int, e: int) -> int:
        if self.unit == "chars":
            return e - s
        return self.tokenizer.count(self._text[s:e])

    def _split(self, s: int, e: int, seps: Sequence[str]) -> list[Span]:
        if self._len(s, e) <= self.chunk_size:
            return [(s, e)]
        for i, sep in enumerate(seps):
            if sep == "":
                break
            seg = self._text[s:e]
            if sep not in seg:
                continue
            rest = seps[i + 1 :]
            out: list[Span] = []
            pos = s
            offset = 0
            while (j := seg.find(sep, offset)) >= 0:
                cut = s + j + len(sep)
                out.extend(self._split(pos, cut, rest))
                pos = cut
                offset = j + len(sep)
            if pos < e:
                out.extend(self._split(pos, e, rest))
            return out
        return hard_split(s, e, self.chunk_size)

    def split_spans(self, text: str, start: int = 0, end: int | None = None) -> list[Span]:
        """Split ``text[start:end]`` into spans (offsets relative to ``text``)."""
        self._text = text
        end = len(text) if end is None else end
        if start >= end:
            return []
        pieces = self._split(start, end, self.separators)
        return merge_spans(pieces, self.chunk_size, self.chunk_overlap, self._len)

    def chunk(self, document: Document) -> list[Chunk]:
        spans = self.split_spans(document.text)
        return build_chunks(document, spans, lambda _i, _s: {"chunker": "recursive"})
