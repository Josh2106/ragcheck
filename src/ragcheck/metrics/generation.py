"""Reference-based answer metrics: exact match, token F1, ROUGE-L, semantic similarity."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from ragcheck.metrics.base import Metric, MetricKind, Sample
from ragcheck.protocols import Embedder
from ragcheck.text import normalize_answer

__all__ = [
    "ExactMatch",
    "RougeL",
    "SemanticSimilarity",
    "TokenF1",
    "exact_match",
    "lcs_length",
    "rouge_l",
    "token_f1",
]


def exact_match(prediction: str, reference: str) -> float:
    """1.0 iff the SQuAD-normalised strings are equal."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD token-level F1 over normalised, whitespace-split tokens (multiset overlap)."""
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    p = common / len(pred)
    r = common / len(ref)
    return 2 * p * r / (p + r)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence, O(len(a)*len(b)) time, O(len(b)) space."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str, beta: float = 1.0) -> float:
    r"""ROUGE-L F-measure (Lin, 2004).

    :math:`P = \mathrm{LCS}/|c|`, :math:`R = \mathrm{LCS}/|r|`,
    :math:`F_\beta = (1+\beta^2) P R / (R + \beta^2 P)`.
    """
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    lcs = lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(pred), lcs / len(ref)
    return (1 + beta**2) * p * r / (r + beta**2 * p)


class _ReferenceMetric(Metric):
    kind: ClassVar[MetricKind] = "generation"

    def _compute(self, prediction: str, reference: str) -> float:
        raise NotImplementedError

    async def score(self, sample: Sample) -> float | None:
        if sample.answer is None or not sample.case.ground_truth:
            return None
        return self._compute(sample.answer.text, sample.case.ground_truth)


class ExactMatch(_ReferenceMetric):
    @property
    def name(self) -> str:
        return "exact_match"

    def _compute(self, prediction: str, reference: str) -> float:
        return exact_match(prediction, reference)


class TokenF1(_ReferenceMetric):
    @property
    def name(self) -> str:
        return "token_f1"

    def _compute(self, prediction: str, reference: str) -> float:
        return token_f1(prediction, reference)


class RougeL(_ReferenceMetric):
    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    @property
    def name(self) -> str:
        return "rouge_l"

    def _compute(self, prediction: str, reference: str) -> float:
        return rouge_l(prediction, reference, self.beta)


class SemanticSimilarity(_ReferenceMetric):
    """Cosine similarity between answer and reference embeddings, clipped to ``[0, 1]``."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def _compute(self, prediction: str, reference: str) -> float:
        if not prediction.strip():
            return 0.0
        vecs = self.embedder.embed([prediction, reference])
        a, b = vecs[0], vecs[1]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return max(0.0, min(1.0, float(a @ b) / denom))
