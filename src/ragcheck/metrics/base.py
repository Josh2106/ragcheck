from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

from ragcheck.types import Answer, EvalCase, RetrievedContext

__all__ = ["Metric", "MetricKind", "Sample"]

MetricKind = Literal["retrieval", "generation", "judge"]


@dataclass(frozen=True)
class Sample:
    """Everything a metric may look at for one evaluation case."""

    case: EvalCase
    context: RetrievedContext
    answer: Answer | None = None


class Metric(ABC):
    """A per-case score in ``[0, 1]``; ``None`` means "not applicable" (e.g. no labels).

    ``None`` values are excluded from aggregates instead of being counted as zero, so a
    case without relevance labels does not drag down Recall@k.
    """

    kind: ClassVar[MetricKind]
    higher_is_better: ClassVar[bool] = True

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def score(self, sample: Sample) -> float | None: ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"
