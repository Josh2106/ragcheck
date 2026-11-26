from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ragcheck.protocols import Reranker, Retriever
from ragcheck.types import Chunk, ScoredChunk

__all__ = ["CallableReranker", "CrossEncoderReranker", "RerankingRetriever"]


def _sorted(candidates: Sequence[ScoredChunk], scores: Sequence[float]) -> list[ScoredChunk]:
    pairs = sorted(
        zip(candidates, scores, strict=True),
        key=lambda p: -float(p[1]),
    )
    return [ScoredChunk(chunk=c.chunk, score=float(s)) for c, s in pairs]


class CallableReranker:
    """Rerank with any ``fn(query, text) -> score`` function."""

    def __init__(self, fn: Callable[[str, str], float]) -> None:
        self.fn = fn

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        return _sorted(candidates, [self.fn(query, c.chunk.text) for c in candidates])


class CrossEncoderReranker:
    """``sentence-transformers`` cross-encoder (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``)."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        scores = self._model.predict([(query, c.chunk.text) for c in candidates])
        return _sorted(candidates, [float(s) for s in scores])


class RerankingRetriever:
    """Fetch ``fetch_k`` candidates from ``base`` then let ``reranker`` pick the top k."""

    def __init__(self, base: Retriever, reranker: Reranker, fetch_k: int = 20) -> None:
        self.base = base
        self.reranker = reranker
        self.fetch_k = fetch_k

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.base.index(chunks)

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        candidates = self.base.retrieve(query, max(k, self.fetch_k))
        return self.reranker.rerank(query, candidates)[:k]
