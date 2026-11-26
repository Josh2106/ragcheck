r"""RAGAS-style metrics computed from LLM-judge verdicts.

Definitions follow Es, James, Espinosa-Anke & Schockaert, *RAGAS: Automated Evaluation of
Retrieval Augmented Generation* (EACL 2024 demo track, arXiv:2309.15217) and the
``ragas`` library's documented formulas for context precision/recall.

* **Faithfulness** = :math:`|V| / |S|`: the fraction of claims :math:`S` extracted from
  the answer that the judge verifies (:math:`V`) against the retrieved context (RAGAS §3).
* **Answer relevance** = :math:`\frac{1}{n}\sum_i \cos(E(q), E(q_i))`: the judge writes
  :math:`n` questions :math:`q_i` that the answer would answer; we embed them and the
  original question :math:`q`. Noncommittal answers score 0 (RAGAS §3).
* **Context precision@K** = :math:`\sum_{k=1}^{K} (\mathrm{P@}k \cdot v_k) /
  \sum_{k=1}^{K} v_k`, where :math:`v_k \in \{0,1\}` is the judge's "was chunk *k* useful
  for reaching the reference answer" verdict - i.e. average precision over the ranked
  contexts (ragas docs, "Context Precision").
* **Context recall** = fraction of reference-answer sentences the judge can attribute to
  the retrieved context (ragas docs, "Context Recall").
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

import numpy as np

from ragcheck.metrics.base import Metric, MetricKind, Sample
from ragcheck.protocols import Embedder, Judge

__all__ = ["AnswerRelevance", "ContextPrecision", "ContextRecall", "Faithfulness"]


class _JudgeMetric(Metric):
    kind: ClassVar[MetricKind] = "judge"

    def __init__(self, judge: Judge) -> None:
        self.judge = judge


class Faithfulness(_JudgeMetric):
    @property
    def name(self) -> str:
        return "faithfulness"

    async def score(self, sample: Sample) -> float | None:
        if sample.answer is None or not sample.answer.text.strip():
            return None
        verdicts = await self.judge.verify_claims(sample.answer.text, sample.context.texts)
        if not verdicts:
            return None
        return sum(v.supported for v in verdicts) / len(verdicts)


class AnswerRelevance(_JudgeMetric):
    def __init__(self, judge: Judge, embedder: Embedder, n_questions: int = 3) -> None:
        super().__init__(judge)
        if n_questions < 1:
            raise ValueError("n_questions must be >= 1")
        self.embedder = embedder
        self.n_questions = n_questions

    @property
    def name(self) -> str:
        return "answer_relevance"

    async def score(self, sample: Sample) -> float | None:
        if sample.answer is None:
            return None
        if not sample.answer.text.strip():
            return 0.0
        questions, noncommittal = await self.judge.generate_questions(
            sample.answer.text, self.n_questions
        )
        if noncommittal or not questions:
            return 0.0
        vecs = self.embedder.embed([sample.case.question, *questions]).astype(np.float64)
        norms = np.linalg.norm(vecs, axis=1)
        norms[norms == 0] = 1.0
        vecs = vecs / norms[:, None]
        sims = vecs[1:] @ vecs[0]
        return float(np.clip(sims, 0.0, 1.0).mean())


class ContextPrecision(_JudgeMetric):
    def __init__(self, judge: Judge, k: int | None = None) -> None:
        super().__init__(judge)
        self.k = k

    @property
    def name(self) -> str:
        return "context_precision" if self.k is None else f"context_precision@{self.k}"

    async def score(self, sample: Sample) -> float | None:
        texts = sample.context.texts[: self.k] if self.k else sample.context.texts
        if not texts:
            return 0.0
        verdicts = await asyncio.gather(
            *(
                self.judge.is_context_useful(sample.case.question, sample.case.ground_truth, t)
                for t in texts
            )
        )
        hits = 0
        total = 0.0
        for k, useful in enumerate(verdicts, start=1):
            if useful:
                hits += 1
                total += hits / k
        return total / hits if hits else 0.0


class ContextRecall(_JudgeMetric):
    @property
    def name(self) -> str:
        return "context_recall"

    async def score(self, sample: Sample) -> float | None:
        if not sample.case.ground_truth.strip():
            return None
        attributed = await self.judge.attribute_statements(
            sample.case.ground_truth, sample.context.texts
        )
        if not attributed:
            return None
        return sum(attributed) / len(attributed)
