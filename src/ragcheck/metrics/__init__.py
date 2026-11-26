"""Evaluation metrics and a name-based factory (``"recall@5"`` -> ``RecallAtK(5)``)."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ragcheck.metrics.base import Metric, MetricKind, Sample
from ragcheck.metrics.generation import (
    ExactMatch,
    RougeL,
    SemanticSimilarity,
    TokenF1,
    exact_match,
    lcs_length,
    rouge_l,
    token_f1,
)
from ragcheck.metrics.judge import AnswerRelevance, ContextPrecision, ContextRecall, Faithfulness
from ragcheck.metrics.retrieval import (
    MRR,
    HitRateAtK,
    NDCGAtK,
    PrecisionAtK,
    RecallAtK,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ragcheck.protocols import Embedder, Judge

__all__ = [
    "MRR",
    "AnswerRelevance",
    "ContextPrecision",
    "ContextRecall",
    "ExactMatch",
    "Faithfulness",
    "HitRateAtK",
    "Metric",
    "MetricKind",
    "NDCGAtK",
    "PrecisionAtK",
    "RecallAtK",
    "RougeL",
    "Sample",
    "SemanticSimilarity",
    "TokenF1",
    "build_metric",
    "build_metrics",
    "exact_match",
    "hit_rate_at_k",
    "lcs_length",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "rouge_l",
    "token_f1",
]

_NAME = re.compile(r"^(?P<base>[a-z][a-z0-9_]*)(?:@(?P<k>\d+))?$")
_AT_K = {"recall": RecallAtK, "precision": PrecisionAtK, "ndcg": NDCGAtK, "hit_rate": HitRateAtK}


def build_metric(
    name: str, *, embedder: Embedder | None = None, judge: Judge | None = None
) -> Metric:
    """Construct a metric from its canonical name.

    Retrieval: ``recall@k``, ``precision@k``, ``ndcg@k``, ``hit_rate@k``, ``mrr[@k]``.
    Generation: ``exact_match``, ``token_f1``, ``rouge_l``, ``semantic_similarity``.
    Judge: ``faithfulness``, ``answer_relevance``, ``context_precision[@k]``,
    ``context_recall``.
    """
    m = _NAME.match(name.strip().lower())
    if not m:
        raise ValueError(f"invalid metric name {name!r}")
    base, k_str = m.group("base"), m.group("k")
    k = int(k_str) if k_str is not None else None

    def need_embedder() -> Embedder:
        if embedder is None:
            raise ValueError(f"metric {name!r} needs an embedder")
        return embedder

    def need_judge() -> Judge:
        if judge is None:
            raise ValueError(f"metric {name!r} needs a judge")
        return judge

    if base in _AT_K:
        if k is None:
            raise ValueError(f"metric {base!r} needs a cutoff, e.g. {base}@5")
        return _AT_K[base](k)
    if base == "mrr":
        return MRR(k)
    simple: dict[str, type[Metric]] = {
        "exact_match": ExactMatch,
        "token_f1": TokenF1,
        "rouge_l": RougeL,
    }
    if base in simple and k is None:
        return simple[base]()
    if base == "semantic_similarity" and k is None:
        return SemanticSimilarity(need_embedder())
    if base == "faithfulness" and k is None:
        return Faithfulness(need_judge())
    if base == "answer_relevance" and k is None:
        return AnswerRelevance(need_judge(), need_embedder())
    if base == "context_precision":
        return ContextPrecision(need_judge(), k)
    if base == "context_recall" and k is None:
        return ContextRecall(need_judge())
    raise ValueError(f"unknown metric {name!r}")


def build_metrics(
    names: Iterable[str], *, embedder: Embedder | None = None, judge: Judge | None = None
) -> list[Metric]:
    metrics = [build_metric(n, embedder=embedder, judge=judge) for n in names]
    seen: set[str] = set()
    for metric in metrics:
        if metric.name in seen:
            raise ValueError(f"duplicate metric {metric.name!r}")
        seen.add(metric.name)
    return metrics
