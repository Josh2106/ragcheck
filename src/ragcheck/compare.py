"""Regression gate: compare two runs against threshold rules.

Rule syntax (one per ``--fail-on``)::

    recall@5<-0.02      fail if current - baseline < -0.02   (absolute delta)
    faithfulness<-5%    fail if (current - baseline) / |baseline| < -5%
    latency_ms_mean>50  fail if the delta is greater than +50 (works on run stats too)
    *<-0.05             apply to every metric present in both runs

Operators: ``<``, ``<=``, ``>``, ``>=``.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ragcheck.results import RunResult

__all__ = ["Comparison", "MetricDelta", "Rule", "RuleError", "compare_runs", "parse_rule"]

_RULE = re.compile(
    r"^\s*(?P<metric>\*|[A-Za-z_][A-Za-z0-9_]*(?:@\d+)?)\s*(?P<op><=|>=|<|>)\s*"
    r"(?P<value>[+-]?(?:\d+\.?\d*|\.\d+))\s*(?P<pct>%)?\s*$"
)
_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


class RuleError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    metric: str
    op: str
    value: float
    relative: bool = False

    def __str__(self) -> str:
        unit = "%" if self.relative else ""
        return f"{self.metric}{self.op}{self.value:g}{unit}"

    def violated(self, delta: float, rel_delta: float | None) -> bool:
        x = rel_delta if self.relative else delta
        if x is None:
            return False
        return _OPS[self.op](x, self.value)


def parse_rule(text: str) -> Rule:
    m = _RULE.match(text)
    if not m:
        raise RuleError(
            f"invalid rule {text!r}; expected e.g. 'recall@5<-0.02' or 'faithfulness<-5%'"
        )
    return Rule(
        metric=m.group("metric").lower(),
        op=m.group("op"),
        value=float(m.group("value")),
        relative=m.group("pct") is not None,
    )


Status = Literal["ok", "fail", "missing"]


@dataclass
class MetricDelta:
    metric: str
    baseline: float | None
    current: float | None
    status: Status = "ok"
    violations: list[str] = field(default_factory=list)

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.current is None:
            return None
        return self.current - self.baseline

    @property
    def rel_delta(self) -> float | None:
        d = self.delta
        if d is None or self.baseline is None or self.baseline == 0:
            return None
        return 100.0 * d / abs(self.baseline)


@dataclass
class Comparison:
    deltas: list[MetricDelta]
    rules: list[Rule]

    @property
    def failed(self) -> bool:
        return any(d.violations for d in self.deltas)

    @property
    def failures(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.violations]


def _values(run: RunResult) -> dict[str, float | None]:
    vals: dict[str, float | None] = {m: run.mean(m) for m in run.metrics}
    for k, v in run.stats.items():
        vals.setdefault(k, v)
    return vals


def compare_runs(baseline: RunResult, current: RunResult, rules: Sequence[Rule]) -> Comparison:
    """Compute per-metric deltas and evaluate every rule.

    A rule naming a metric missing from either run marks it ``missing`` (a failure: a
    silently dropped metric should not pass a gate).
    """
    base, cur = _values(baseline), _values(current)
    names = list(dict.fromkeys([*baseline.metrics, *current.metrics]))
    for r in rules:
        if r.metric != "*" and r.metric not in names:
            names.append(r.metric)
    deltas = {n: MetricDelta(n, base.get(n), cur.get(n)) for n in names}
    for rule in rules:
        targets = (
            [n for n in baseline.metrics if n in current.metrics]
            if rule.metric == "*"
            else [rule.metric]
        )
        for name in targets:
            d = deltas[name]
            if d.delta is None:
                if rule.metric != "*":
                    d.status = "missing"
                    d.violations.append(f"{rule}: metric missing or null in one run")
                continue
            if rule.violated(d.delta, d.rel_delta):
                d.status = "fail"
                d.violations.append(str(rule))
    return Comparison(deltas=list(deltas.values()), rules=list(rules))
