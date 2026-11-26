from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from ragcheck.tokenizers import simple_tokens
from ragcheck.types import Chunk, ScoredChunk

__all__ = ["BM25Retriever"]


class BM25Retriever:
    r"""Okapi BM25 (Robertson & Zaragoza, 2009) over an inverted index.

    .. math::

        \text{score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot
            \frac{f(q, D)\,(k_1 + 1)}{f(q, D) + k_1\,(1 - b + b\,|D| / \text{avgdl})}

    with the non-negative ("Lucene") IDF
    :math:`\text{IDF}(q) = \ln\!\left(1 + \frac{N - n(q) + 0.5}{n(q) + 0.5}\right)`.
    Repeated query terms contribute once per occurrence. Only chunks with a positive
    score are returned.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = simple_tokens,
        stopwords: Iterable[str] = (),
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.stopwords = frozenset(stopwords)
        self._chunks: list[Chunk] = []
        self._doc_len: NDArray[np.float64] = np.zeros(0)
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    def _tokens(self, text: str) -> list[str]:
        return [t for t in self.tokenizer(text) if t not in self.stopwords]

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for i, chunk in enumerate(self._chunks):
            toks = self._tokens(chunk.text)
            lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                postings[term].append((i, tf))
        self._postings = dict(postings)
        self._doc_len = np.asarray(lengths, dtype=np.float64)
        n = len(self._chunks)
        self._avgdl = float(self._doc_len.mean()) if n else 0.0
        self._idf = {
            t: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self._postings.items()
        }

    def idf(self, term: str) -> float:
        return self._idf.get(term, 0.0)

    def get_scores(self, query: str) -> NDArray[np.float64]:
        """BM25 score of every indexed chunk for ``query`` (index order)."""
        scores = np.zeros(len(self._chunks), dtype=np.float64)
        if not self._chunks or self._avgdl == 0:
            return scores
        norm = self.k1 * (1.0 - self.b + self.b * self._doc_len / self._avgdl)
        for term in self._tokens(query):
            plist = self._postings.get(term)
            if not plist:
                continue
            idx = np.fromiter((p[0] for p in plist), dtype=np.int64, count=len(plist))
            tf = np.fromiter((p[1] for p in plist), dtype=np.float64, count=len(plist))
            scores[idx] += self._idf[term] * tf * (self.k1 + 1.0) / (tf + norm[idx])
        return scores

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        scores = self.get_scores(query)
        if k <= 0 or scores.size == 0:
            return []
        order = sorted(np.flatnonzero(scores > 0).tolist(), key=lambda i: (-scores[i], i))[:k]
        return [ScoredChunk(chunk=self._chunks[i], score=float(scores[i])) for i in order]
