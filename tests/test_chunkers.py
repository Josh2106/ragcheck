from __future__ import annotations

import itertools

import pytest

from ragcheck.chunkers import (
    FixedSizeChunker,
    MarkdownHeaderChunker,
    RecursiveChunker,
    SentenceWindowChunker,
    split_sentences,
)
from ragcheck.chunkers._base import merge_spans
from ragcheck.tokenizers import RegexTokenizer
from ragcheck.types import Document

ALL = [
    FixedSizeChunker(10),
    FixedSizeChunker(4, unit="tokens"),
    RecursiveChunker(20),
    RecursiveChunker(5, unit="tokens"),
    SentenceWindowChunker(),
    MarkdownHeaderChunker(50),
]


def doc(text: str, **md: object) -> Document:
    return Document(id="d", text=text, metadata=dict(md))


@pytest.mark.parametrize("chunker", ALL, ids=lambda c: type(c).__name__)
@pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
def test_empty_and_whitespace_documents_yield_no_chunks(chunker: object, text: str) -> None:
    assert chunker.chunk(doc(text)) == []  # type: ignore[attr-defined]


@pytest.mark.parametrize("chunker", ALL, ids=lambda c: type(c).__name__)
def test_unicode_spans_are_exact(chunker: object) -> None:
    text = "# Título\n\nCafé naïve — 漢字かな交じり文。 Emoji 🚀 ok! Ещё одно предложение."
    chunks = chunker.chunk(doc(text, lang="mixed"))  # type: ignore[attr-defined]
    assert chunks
    for i, c in enumerate(chunks):
        assert c.text == text[c.start : c.end]
        assert c.index == i
        assert c.id == f"d::{i}"
        assert c.metadata["lang"] == "mixed"  # document metadata is inherited


def test_fixed_chars_with_overlap() -> None:
    chunks = FixedSizeChunker(size=4, overlap=2).chunk(doc("abcdefghij"))
    assert [c.text for c in chunks] == ["abcd", "cdef", "efgh", "ghij"]


def test_fixed_tokens_never_cut_tokens() -> None:
    text = "alpha beta, gamma delta epsilon"
    chunks = FixedSizeChunker(size=2, unit="tokens").chunk(doc(text))
    assert [c.text for c in chunks] == ["alpha beta", ", gamma", "delta epsilon"]


@pytest.mark.parametrize(
    ("size", "overlap"), [(0, 0), (-1, 0), (5, 5), (5, -1)], ids=["zero", "neg", "eq", "negov"]
)
def test_invalid_sizes(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match=r"size|overlap"):
        FixedSizeChunker(size, overlap)
    with pytest.raises(ValueError, match=r"size|overlap"):
        RecursiveChunker(size, overlap)


def test_invalid_units_and_separators() -> None:
    with pytest.raises(ValueError, match="unit"):
        FixedSizeChunker(5, unit="words")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unit"):
        RecursiveChunker(5, unit="words")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="separators"):
        RecursiveChunker(5, separators=[])


def test_recursive_prefers_paragraphs() -> None:
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one."
    chunks = RecursiveChunker(chunk_size=30).chunk(doc(text))
    assert [c.text for c in chunks] == [
        "First paragraph here.",
        "Second paragraph here.",
        "Third one.",
    ]


def test_recursive_oversize_token_is_hard_split() -> None:
    blob = "x" * 25
    chunks = RecursiveChunker(chunk_size=10).chunk(doc(f"short {blob} end"))
    assert all(len(c.text) <= 10 for c in chunks)
    assert sum(c.text.count("x") for c in chunks) == len(blob)


def test_recursive_overlap_repeats_trailing_pieces() -> None:
    text = "one two three four five six"
    chunks = RecursiveChunker(chunk_size=14, chunk_overlap=6, separators=[" "]).chunk(doc(text))
    texts = [c.text for c in chunks]
    assert texts[0] == "one two three"
    assert all(len(t) <= 14 for t in texts)
    # consecutive chunks share text because of the overlap
    assert any(a.split()[-1] == b.split()[0] for a, b in itertools.pairwise(texts))


def test_recursive_token_unit_respects_limit() -> None:
    text = "a b c d e f g h i j k l m n o p"
    tok = RegexTokenizer()
    chunks = RecursiveChunker(chunk_size=4, unit="tokens").chunk(doc(text))
    assert all(tok.count(c.text) <= 4 for c in chunks)


def test_recursive_custom_separators_get_fallback() -> None:
    r = RecursiveChunker(5, separators=["|"])
    assert r.separators == ("|", "")
    assert r.split_spans("abc", 2, 2) == []


def test_merge_spans_overlap() -> None:
    pieces = [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert merge_spans(pieces, 6, 0, lambda s, e: e - s) == [(0, 6), (6, 12)]
    assert merge_spans(pieces, 6, 3, lambda s, e: e - s) == [(0, 6), (3, 9), (6, 12)]


def test_split_sentences_handles_abbreviations_and_quotes() -> None:
    text = 'He said "Stop." Then e.g. this continues. Done?  Yes!\n\nNew para'
    sents = [text[s:e] for s, e in split_sentences(text)]
    assert sents == ['He said "Stop."', "Then e.g. this continues.", "Done?", "Yes!", "New para"]


def test_sentence_window_metadata() -> None:
    text = "One. Two. Three. Four."
    chunks = SentenceWindowChunker(window=1).chunk(doc(text))
    assert [c.text for c in chunks] == ["One.", "Two.", "Three.", "Four."]
    assert chunks[0].metadata["window"] == "One. Two."
    assert chunks[1].metadata["window"] == "One. Two. Three."
    assert chunks[3].metadata["window"] == "Three. Four."
    grouped = SentenceWindowChunker(window=0, sentences_per_chunk=2).chunk(doc(text))
    assert [c.text for c in grouped] == ["One. Two.", "Three. Four."]


def test_sentence_window_splits_oversize_sentences() -> None:
    chunks = SentenceWindowChunker(max_sentence_chars=5).chunk(doc("abcdefghijkl."))
    assert all(len(c.text) <= 5 for c in chunks)


@pytest.mark.parametrize(
    "kwargs", [{"window": -1}, {"sentences_per_chunk": 0}, {"max_sentence_chars": 0}]
)
def test_sentence_window_validation(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=">="):
        SentenceWindowChunker(**kwargs)


MD = """Intro text before headers.

# Install

Use brew.

## Windows

Use WSL2.

```bash
# not a header
lark init
```

## macOS ##

Use brew again.

# Configure

Edit the file.
"""


def test_markdown_header_paths_and_code_fences() -> None:
    chunks = MarkdownHeaderChunker(max_chunk_size=500).chunk(doc(MD))
    sections = [c.metadata["section"] for c in chunks]
    assert sections == ["", "Install", "Install > Windows", "Install > macOS", "Configure"]
    windows = chunks[2]
    assert "# not a header" in windows.text
    assert windows.metadata["headers"] == {"h1": "Install", "h2": "Windows"}


def test_markdown_max_header_level_and_oversize_sections() -> None:
    chunks = MarkdownHeaderChunker(max_chunk_size=500, max_header_level=1).chunk(doc(MD))
    assert [c.metadata["section"] for c in chunks] == ["", "Install", "Configure"]
    small = MarkdownHeaderChunker(max_chunk_size=12).chunk(doc(MD))
    assert all(len(c.text) <= 12 for c in small)
    assert {c.metadata["section"] for c in small} >= {"Install > Windows", "Configure"}
    with pytest.raises(ValueError, match="max_header_level"):
        MarkdownHeaderChunker(max_header_level=7)


def test_markdown_tilde_fence() -> None:
    text = "# A\n~~~\n# inside\n~~~\n# B\nbody\n"
    chunks = MarkdownHeaderChunker().chunk(doc(text))
    assert [c.metadata["section"] for c in chunks] == ["A", "B"]
