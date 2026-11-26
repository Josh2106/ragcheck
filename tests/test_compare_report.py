from __future__ import annotations

from pathlib import Path

import pytest

from ragcheck.compare import RuleError, compare_runs, parse_rule
from ragcheck.report import render_html, write_html
from ragcheck.results import RunResult
from ragcheck.types import EvalResult


def make_run(scores: dict[str, list[float | None]], name: str = "r", **stats: float) -> RunResult:
    n = len(next(iter(scores.values())))
    rows = [
        EvalResult(
            case_id=f"q{i}",
            question=f"<question {i}>",
            ground_truth="gt",
            answer="<b>answer</b>" if i == 0 else "",
            retrieved_doc_ids=("d1", "d2"),
            relevant_doc_ids=("d1",),
            scores={m: v[i] for m, v in scores.items()},
            latency_ms=1.0 + i,
            error="boom" if i == 1 else None,
        )
        for i in range(n)
    ]
    return RunResult.from_rows(rows, list(scores), name=name, stats=stats)


BASE = make_run({"recall@5": [1.0, 0.8], "faithfulness": [0.9, 0.9], "mrr": [0.5, 0.5]})


def test_parse_rule() -> None:
    r = parse_rule("recall@5<-0.02")
    assert (r.metric, r.op, r.value, r.relative) == ("recall@5", "<", -0.02, False)
    assert str(r) == "recall@5<-0.02"
    p = parse_rule(" Faithfulness <= -5% ")
    assert (p.metric, p.op, p.value, p.relative) == ("faithfulness", "<=", -5.0, True)
    assert parse_rule("*<-.1").metric == "*"
    assert parse_rule("latency_ms_mean>50").op == ">"
    for bad in ("recall@5", "recall@5 << 1", "<-0.1", "x<abc"):
        with pytest.raises(RuleError):
            parse_rule(bad)


def test_compare_detects_regressions() -> None:
    cur = make_run({"recall@5": [0.9, 0.8], "faithfulness": [0.9, 0.9], "mrr": [0.6, 0.6]})
    cmp = compare_runs(BASE, cur, [parse_rule("recall@5<-0.02")])
    assert cmp.failed
    d = {x.metric: x for x in cmp.deltas}
    assert d["recall@5"].delta == pytest.approx(-0.05)
    assert d["recall@5"].rel_delta == pytest.approx(-100 * 0.05 / 0.9)
    assert d["recall@5"].status == "fail"
    assert d["mrr"].status == "ok"
    assert [f.metric for f in cmp.failures] == ["recall@5"]
    ok = compare_runs(BASE, cur, [parse_rule("recall@5<-0.1")])
    assert not ok.failed


def test_compare_relative_wildcard_and_missing() -> None:
    cur = make_run({"recall@5": [0.9, 0.8], "faithfulness": [0.5, 0.5]})
    cmp = compare_runs(BASE, cur, [parse_rule("*<-10%")])
    statuses = {d.metric: d.status for d in cmp.deltas}
    assert statuses == {"recall@5": "ok", "faithfulness": "fail", "mrr": "ok"}
    missing = compare_runs(BASE, cur, [parse_rule("mrr<-0.01")])
    assert missing.failed
    assert {d.metric: d.status for d in missing.deltas}["mrr"] == "missing"
    unknown = compare_runs(BASE, cur, [parse_rule("ghost<0")])
    assert unknown.failed


def test_compare_on_stats_and_zero_baseline() -> None:
    a = make_run({"m": [0.0, 0.0]}, latency_ms_mean=10.0)
    b = make_run({"m": [0.1, 0.1]}, latency_ms_mean=100.0)
    cmp = compare_runs(a, b, [parse_rule("latency_ms_mean>50"), parse_rule("m<-5%")])
    d = {x.metric: x for x in cmp.deltas}
    assert d["latency_ms_mean"].status == "fail"
    assert d["m"].rel_delta is None  # baseline 0: relative delta undefined, rule not violated
    assert d["m"].status == "ok"


def test_render_html_is_self_contained_and_escaped(tmp_path: Path) -> None:
    run = make_run({"recall@5": [1.0, None], "mrr": [0.25, 0.75]})
    html = render_html(run)
    assert html.startswith("<!doctype html>")
    assert "<svg" in html
    assert "&lt;question 0&gt;" in html
    assert "&lt;b&gt;answer&lt;/b&gt;" in html
    assert "<b>answer</b>" not in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "<script" not in html
    assert "1 errors" in html
    assert "<b>d1</b>" in html  # relevant doc highlighted
    diff = render_html(run, baseline=make_run({"recall@5": [0.5, 0.5], "mrr": [1.0, 1.0]}))
    assert "baseline" in diff
    assert "+0.500" in diff
    out = write_html(run, tmp_path / "x" / "r.html")
    assert out.read_text(encoding="utf-8") == html
