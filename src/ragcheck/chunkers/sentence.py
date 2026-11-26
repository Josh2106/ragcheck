from __future__ import annotations

import re

from ragcheck.chunkers._base import Span, build_chunks, hard_split, strip_span
from ragcheck.types import Chunk, Document

__all__ = ["SentenceWindowChunker", "split_sentences"]

# A sentence ends at ., ! or ? (optionally followed by closing quotes/brackets) followed
# by whitespace, or at a blank line. Common abbreviations are protected below.
_BOUNDARY = re.compile(r"(?<=[.!?])[\"'\)\]\u201d\u2019]*\s+|\n\s*\n")
_CLOSERS = "\"')]\u201d\u2019"
_ABBREVIATIONS = frozenset(
    {"e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.", "no.", "fig.", "approx."}
)


def split_sentences(text: str) -> list[Span]:
    """Return stripped sentence spans. Spans plus whitespace gaps tile ``text``."""
    spans: list[Span] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        end = m.start()
        last_word = text[start:end].rsplit(None, 1)[-1].lower() if text[start:end].strip() else ""
        if last_word in _ABBREVIATIONS:
            continue
        # keep closing quotes/brackets with the sentence they close
        stripped_end = m.start() + len(m.group(0)) - len(m.group(0).lstrip(_CLOSERS))
        spans.append((start, stripped_end))
        start = m.end()
    spans.append((start, len(text)))
    out = []
    for s, e in spans:
        s2, e2 = strip_span(text, s, e)
        if s2 < e2:
            out.append((s2, e2))
    return out


class SentenceWindowChunker:
    """Index small units (sentences) but keep surrounding context for generation.

    Each chunk holds ``sentences_per_chunk`` sentences; ``metadata["window"]`` holds the
    text of the chunk plus ``window`` sentences on either side, which a generator can use
    instead of the bare sentence (the "sentence window" pattern popularised by
    LlamaIndex). Sentences longer than ``max_sentence_chars`` are hard-split.
    """

    def __init__(
        self, window: int = 1, sentences_per_chunk: int = 1, max_sentence_chars: int = 1000
    ) -> None:
        if window < 0:
            raise ValueError("window must be >= 0")
        if sentences_per_chunk < 1:
            raise ValueError("sentences_per_chunk must be >= 1")
        if max_sentence_chars < 1:
            raise ValueError("max_sentence_chars must be >= 1")
        self.window = window
        self.sentences_per_chunk = sentences_per_chunk
        self.max_sentence_chars = max_sentence_chars

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        sentences: list[Span] = []
        for s, e in split_sentences(text):
            if e - s > self.max_sentence_chars:
                sentences.extend(hard_split(s, e, self.max_sentence_chars))
            else:
                sentences.append((s, e))
        groups: list[tuple[int, int]] = []  # (first sentence idx, last sentence idx)
        for i in range(0, len(sentences), self.sentences_per_chunk):
            groups.append((i, min(i + self.sentences_per_chunk, len(sentences)) - 1))
        spans = [(sentences[a][0], sentences[b][1]) for a, b in groups]
        windows: dict[Span, Span] = {}
        for (a, b), span in zip(groups, spans, strict=True):
            lo = max(0, a - self.window)
            hi = min(len(sentences) - 1, b + self.window)
            windows[span] = (sentences[lo][0], sentences[hi][1])

        def meta(_i: int, span: Span) -> dict[str, object]:
            ws, we = windows.get(span, span)
            return {
                "chunker": "sentence_window",
                "window": text[ws:we],
                "window_start": ws,
                "window_end": we,
            }

        return build_chunks(document, spans, meta)
