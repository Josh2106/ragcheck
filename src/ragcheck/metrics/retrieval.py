r"""Rank-aware retrieval metrics with binary relevance at the *document* level.

A chunk is relevant if its ``doc_id`` is in ``EvalCase.relevant_doc_ids``. Retrieved
chunks are collapsed to documents in rank order (first occurrence wins), so retrieving
three chunks of the same relevant document counts once. Let :math:`R` be the relevant set
and :math:`d_1, d_2, \dots` the de-duplicated ranked documents.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

from ragcheck.metrics.base import Metric, MetricKind, Sample

__all__ = [
    "MRR",
    "HitRateAtK",
    "NDCGAtK",
    "PrecisionAtK",
    "RecallAtK",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    r""":math:`|R \cap \{d_1..d_k\}| / |R|`."""
    _check_k(k)
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    r""":math:`|R \cap \{d_1..d_k\}| / k` (the denominator is ``k`` even if fewer were returned)."""
    _check_k(k)
    return len(relevant.intersection(ranked[:k])) / k


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1 if any relevant document is in the top k, else 0."""
    _check_k(k)
    return 1.0 if relevant.intersection(ranked[:k]) else 0.0


def mrr(ranked: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant document (0 if none in the top k)."""
    limit = len(ranked) if k is None else k
    for i, d in enumerate(ranked[:limit], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    r"""Binary nDCG: :math:`\mathrm{DCG}_k / \mathrm{IDCG}_k` with
    :math:`\mathrm{DCG}_k = \sum_{i=1}^{k} \mathbb{1}[d_i \in R] / \log_2(i + 1)`."""
    _check_k(k)
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg


class _RetrievalMetric(Metric):
    kind: ClassVar[MetricKind] = "retrieval"
    label: ClassVar[str] = ""

    def __init__(self, k: int) -> None:
        _check_k(k)
        self.k = k

    @property
    def name(self) -> str:
        return f"{self.label}@{self.k}"

    def _compute(self, ranked: list[str], relevant: set[str]) -> float:
        raise NotImplementedError

    async def score(self, sample: Sample) -> float | None:
        relevant = set(sample.case.relevant_doc_ids)
        if not relevant:
            return None
        return self._compute(sample.context.doc_ids, relevant)


class RecallAtK(_RetrievalMetric):
    label = "recall"

    def _compute(self, ranked: list[str], relevant: set[str]) -> float:
        return recall_at_k(ranked, relevant, self.k)


class PrecisionAtK(_RetrievalMetric):
    label = "precision"

    def _compute(self, ranked: list[str], relevant: set[str]) -> float:
        return precision_at_k(ranked, relevant, self.k)


class HitRateAtK(_RetrievalMetric):
    label = "hit_rate"

    def _compute(self, ranked: list[str], relevant: set[str]) -> float:
        return hit_rate_at_k(ranked, relevant, self.k)


class NDCGAtK(_RetrievalMetric):
    label = "ndcg"

    def _compute(self, ranked: list[str], relevant: set[str]) -> float:
        return ndcg_at_k(ranked, relevant, self.k)


class MRR(Metric):
    kind: ClassVar[MetricKind] = "retrieval"

    def __init__(self, k: int | None = None) -> None:
        if k is not None:
            _check_k(k)
        self.k = k

    @property
    def name(self) -> str:
        return "mrr" if self.k is None else f"mrr@{self.k}"

    async def score(self, sample: Sample) -> float | None:
        relevant = set(sample.case.relevant_doc_ids)
        if not relevant:
            return None
        return mrr(sample.context.doc_ids, relevant, self.k)
