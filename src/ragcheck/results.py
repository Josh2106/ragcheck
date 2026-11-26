"""Run results: per-case rows, aggregates, (de)serialisation and tabular views."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from rich.table import Table

from ragcheck.types import EvalResult

__all__ = ["MetricSummary", "RunResult", "summarise"]


class MetricSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    count: int = Field(description="Cases with a non-null score.")


def summarise(values: Sequence[float | None]) -> MetricSummary:
    xs = [v for v in values if v is not None and not math.isnan(v)]
    if not xs:
        return MetricSummary(mean=None, std=None, min=None, max=None, count=0)
    return MetricSummary(
        mean=statistics.fmean(xs),
        std=statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        min=min(xs),
        max=max(xs),
        count=len(xs),
    )


class RunResult(BaseModel):
    """The output of :func:`ragcheck.run`. Saved as JSON; the input to ``compare``/``report``."""

    name: str = "run"
    ragcheck_version: str = ""
    created_at: str = ""
    duration_s: float = 0.0
    dataset: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=list)
    rows: list[EvalResult] = Field(default_factory=list)
    aggregates: dict[str, MetricSummary] = Field(default_factory=dict)
    stats: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: Sequence[EvalResult], metrics: Sequence[str], **kw: Any) -> RunResult:
        aggs = {m: summarise([r.scores.get(m) for r in rows]) for m in metrics}
        latencies = [r.latency_ms for r in rows]
        stats = dict(kw.pop("stats", {}))
        if latencies:
            stats.setdefault("latency_ms_mean", statistics.fmean(latencies))
            stats.setdefault("latency_ms_p95", _percentile(latencies, 95))
        stats.setdefault("errors", float(sum(r.error is not None for r in rows)))
        return cls(rows=list(rows), metrics=list(metrics), aggregates=aggs, stats=stats, **kw)

    def mean(self, metric: str) -> float | None:
        agg = self.aggregates.get(metric)
        return None if agg is None else agg.mean

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def summary_table(self, title: str | None = None) -> Table:
        table = Table(title=title or f"ragcheck - {self.name}")
        for col in ("metric", "mean", "std", "min", "max", "n"):
            table.add_column(col, justify="left" if col == "metric" else "right")
        for m in self.metrics:
            a = self.aggregates[m]
            table.add_row(m, _fmt(a.mean), _fmt(a.std), _fmt(a.min), _fmt(a.max), str(a.count))
        return table

    def to_markdown(self) -> str:
        lines = ["| metric | mean | std | n |", "|---|---:|---:|---:|"]
        for m in self.metrics:
            a = self.aggregates[m]
            lines.append(f"| {m} | {_fmt(a.mean)} | {_fmt(a.std)} | {a.count} |")
        return "\n".join(lines)


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def _percentile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
