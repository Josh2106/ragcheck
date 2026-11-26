"""Hypothesis property tests for chunkers and reciprocal rank fusion."""

from __future__ import annotations

import itertools

from hypothesis import given, settings
from hypothesis import strategies as st

from ragcheck.chunkers import (
    FixedSizeChunker,
    MarkdownHeaderChunker,
    RecursiveChunker,
    SentenceWindowChunker,
)
from ragcheck.protocols import Chunker
from ragcheck.retrievers import reciprocal_rank_fusion
from ragcheck.tokenizers import RegexTokenizer
from ragcheck.types import Chunk, Document

_ALPHABET = st.sampled_from(
    [*"abcdefghij ABC xyz", "\n", "\n\n", ". ", "! ", "# ", "## ", "```\n", "é", "漢", "🚀"]
)
texts = st.lists(_ALPHABET, max_size=200).map("".join) | st.text(max_size=300)
TOK = RegexTokenizer()


def _assert_lossless(text: str, chunks: list[Chunk]) -> None:
    covered = [False] * len(text)
    for c in chunks:
        assert c.text == text[c.start : c.end]
        assert c.text
        assert not c.text[0].isspace()
        assert not c.text[-1].isspace()
        for i in range(c.start, c.end):
            covered[i] = True
    # no content is lost: every non-whitespace character lies inside some chunk
    lost = [ch for ch, cov in zip(text, covered, strict=True) if not cov and not ch.isspace()]
    assert not lost


def _check(chunker: Chunker, text: str) -> list[Chunk]:
    chunks = chunker.chunk(Document(id="d", text=text))
    _assert_lossless(text, chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    return chunks


@settings(max_examples=150, deadline=None)
@given(text=texts, size=st.integers(1, 40), data=st.data())
def test_fixed_chars(text: str, size: int, data: st.DataObject) -> None:
    overlap = data.draw(st.integers(0, size - 1))
    for c in _check(FixedSizeChunker(size, overlap), text):
        assert len(c.text) <= size


@settings(max_examples=150, deadline=None)
@given(text=texts, size=st.integers(1, 12), data=st.data())
def test_fixed_tokens(text: str, size: int, data: st.DataObject) -> None:
    overlap = data.draw(st.integers(0, size - 1))
    for c in _check(FixedSizeChunker(size, overlap, unit="tokens"), text):
        assert TOK.count(c.text) <= size


@settings(max_examples=150, deadline=None)
@given(text=texts, size=st.integers(1, 60), data=st.data())
def test_recursive_chars(text: str, size: int, data: st.DataObject) -> None:
    overlap = data.draw(st.integers(0, size - 1))
    chunks = _check(RecursiveChunker(size, overlap), text)
    for c in chunks:
        assert len(c.text) <= size
    if overlap == 0:  # without overlap chunks are disjoint and ordered
        for a, b in itertools.pairwise(chunks):
            assert a.end <= b.start


@settings(max_examples=100, deadline=None)
@given(text=texts, size=st.integers(1, 15))
def test_recursive_tokens(text: str, size: int) -> None:
    for c in _check(RecursiveChunker(size, unit="tokens"), text):
        assert TOK.count(c.text) <= size


@settings(max_examples=100, deadline=None)
@given(text=texts, size=st.integers(1, 80))
def test_sentence_window(text: str, size: int) -> None:
    for c in _check(SentenceWindowChunker(window=1, max_sentence_chars=size), text):
        assert len(c.text) <= size
        assert c.text in c.metadata["window"]


@settings(max_examples=100, deadline=None)
@given(text=texts, size=st.integers(1, 80))
def test_markdown(text: str, size: int) -> None:
    for c in _check(MarkdownHeaderChunker(max_chunk_size=size), text):
        assert len(c.text) <= size


ids = st.sampled_from([f"d{i}" for i in range(12)])
rankings = st.lists(st.lists(ids, max_size=10, unique=True), min_size=1, max_size=5)


@settings(max_examples=200, deadline=None)
@given(rankings=rankings, k=st.floats(1, 100))
def test_rrf_properties(rankings: list[list[str]], k: float) -> None:
    fused = reciprocal_rank_fusion(rankings, k=k)
    fused_ids = [d for d, _ in fused]
    # every input id appears exactly once
    assert sorted(fused_ids) == sorted({d for r in rankings for d in r})
    # scores are non-increasing and match the definition
    scores = [s for _, s in fused]
    assert all(a >= b for a, b in itertools.pairwise(scores))
    for d, s in fused:
        expected = sum(1.0 / (k + r.index(d) + 1) for r in rankings if d in r)
        assert abs(s - expected) < 1e-12
    # the order of the input lists does not matter
    assert reciprocal_rank_fusion(list(reversed(rankings)), k=k) == fused


@given(rankings=rankings)
def test_rrf_unanimous_winner_ranks_first(rankings: list[list[str]]) -> None:
    lists = [["winner", *r] for r in rankings]
    assert reciprocal_rank_fusion(lists)[0][0] == "winner"


@given(ranking=st.lists(ids, min_size=1, max_size=10, unique=True))
def test_rrf_single_list_preserves_order(ranking: list[str]) -> None:
    assert [d for d, _ in reciprocal_rank_fusion([ranking])] == ranking
