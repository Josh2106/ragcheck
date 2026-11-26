from __future__ import annotations

import math
from collections.abc import Sequence

from ragcheck.protocols import Retriever
from ragcheck.types import Chunk, ScoredChunk

__all__ = ["HybridRetriever", "reciprocal_rank_fusion"]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: float = 60.0,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    r"""Fuse ranked lists with Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).

    :math:`\text{RRF}(d) = \sum_i w_i / (k + \text{rank}_i(d))` with 1-based ranks. Items
    missing from a list contribute nothing for it; duplicate ids within one list only
    count at their best rank. Ties are broken by best rank, then id, so the output is
    deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")
    contributions: dict[str, list[float]] = {}
    best: dict[str, int] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            contributions.setdefault(item, []).append(w / (k + rank))
            best[item] = min(best.get(item, rank), rank)
    # math.fsum is exactly rounded, so the result is independent of list order
    scores = {item: math.fsum(parts) for item, parts in contributions.items()}
    return sorted(scores.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))


class HybridRetriever:
    """Run several retrievers (typically BM25 + dense) and fuse them with RRF."""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        weights: Sequence[float] | None = None,
        rrf_k: float = 60.0,
        fetch_k: int = 20,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must have one entry per retriever")
        self.retrievers = list(retrievers)
        self.weights = list(weights) if weights is not None else None
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k
        self._chunks: dict[str, Chunk] = {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks.update({c.id: c for c in chunks})
        for r in self.retrievers:
            r.index(chunks)

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        fetch = max(k, self.fetch_k)
        rankings = [[sc.chunk.id for sc in r.retrieve(query, fetch)] for r in self.retrievers]
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=self.weights)[:k]
        return [ScoredChunk(chunk=self._chunks[cid], score=s) for cid, s in fused]
