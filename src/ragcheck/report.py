"""Self-contained HTML report: inline CSS + inline SVG charts, no JavaScript, no CDN."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path

from ragcheck.results import RunResult

__all__ = ["render_html", "write_html"]

_CSS = """
:root{--bg:#fbfbfa;--fg:#1d1f23;--muted:#6b7079;--card:#ffffff;--line:#e3e4e8;
--accent:#3b6fd8;--base:#a3a9b4;--good:#1f8a4c;--bad:#c2412d;--warn:#b7791f}
@media (prefers-color-scheme:dark){:root{--bg:#15171b;--fg:#e8e9ec;--muted:#9aa0aa;
--card:#1d2026;--line:#2e323a;--accent:#7aa2f7;--base:#5c6370;--good:#4cc38a;--bad:#f07a68;
--warn:#e0b252}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 12px}
.meta{color:var(--muted);font-size:13px}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:16px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.num{text-align:right;white-space:nowrap}.good{color:var(--good)}.bad{color:var(--bad)}
.err{color:var(--bad);font-size:12px}.small{font-size:12px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
.hist h3{font-size:13px;margin:0 0 6px;font-weight:600}
svg text{fill:var(--fg);font-size:12px}svg .muted{fill:var(--muted)}
.legend span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px}
"""


def _f(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _bar_chart(run: RunResult, baseline: RunResult | None) -> str:
    metrics = run.metrics
    row_h = 30 if baseline else 22
    label_w, chart_w, pad = 170, 520, 8
    height = len(metrics) * row_h + 30
    width = label_w + chart_w + 70
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Mean score per metric" xmlns="http://www.w3.org/2000/svg">'
    ]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = label_w + t * chart_w
        parts.append(
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height - 22}" '
            f'stroke="var(--line)" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{height - 6}" text-anchor="middle" class="muted">{t:g}</text>'
        )
    for i, m in enumerate(metrics):
        y = i * row_h + pad / 2
        parts.append(f'<text x="{label_w - 8}" y="{y + 13}" text-anchor="end">{escape(m)}</text>')
        cur = run.mean(m)
        bar_h = (row_h - pad) / (2 if baseline else 1)
        if baseline is not None:
            b = baseline.mean(m)
            if b is not None:
                parts.append(
                    f'<rect x="{label_w}" y="{y:.1f}" width="{max(b, 0) * chart_w:.1f}" '
                    f'height="{bar_h - 1:.1f}" rx="2" fill="var(--base)"><title>baseline '
                    f"{escape(m)}: {b:.3f}</title></rect>"
                )
            y += bar_h
        if cur is not None:
            w = max(min(cur, 1.0), 0.0) * chart_w
            parts.append(
                f'<rect x="{label_w}" y="{y:.1f}" width="{w:.1f}" height="{bar_h - 1:.1f}" '
                f'rx="2" fill="var(--accent)"><title>{escape(m)}: {cur:.3f}</title></rect>'
                f'<text x="{label_w + w + 6:.1f}" y="{y + bar_h - 3:.1f}">{cur:.3f}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _histogram(values: Sequence[float], bins: int = 10) -> str:
    w, h = 220, 70
    counts = [0] * bins
    for v in values:
        counts[min(int(max(v, 0.0) * bins), bins - 1)] += 1
    peak = max(counts) or 1
    bw = w / bins
    parts = [
        f'<svg viewBox="0 0 {w} {h + 16}" width="100%" role="img" '
        f'aria-label="Score distribution" xmlns="http://www.w3.org/2000/svg">'
    ]
    for i, c in enumerate(counts):
        bh = (c / peak) * h
        parts.append(
            f'<rect x="{i * bw + 1:.1f}" y="{h - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" '
            f'rx="1" fill="var(--accent)"><title>[{i / bins:.1f}, {(i + 1) / bins:.1f}): {c}'
            f"</title></rect>"
        )
    parts.append(
        f'<line x1="0" y1="{h}" x2="{w}" y2="{h}" stroke="var(--line)"/>'
        f'<text x="0" y="{h + 14}" class="muted">0</text>'
        f'<text x="{w}" y="{h + 14}" text-anchor="end" class="muted">1</text></svg>'
    )
    return "".join(parts)


def _summary_table(run: RunResult, baseline: RunResult | None) -> str:
    head = "<tr><th>metric</th><th>mean</th><th>std</th><th>min</th><th>max</th><th>n</th>"
    if baseline:
        head += "<th>baseline</th><th>delta</th>"
    rows = [head + "</tr>"]
    for m in run.metrics:
        a = run.aggregates[m]
        cells = (
            f"<td>{escape(m)}</td><td class=num>{_f(a.mean)}</td><td class=num>{_f(a.std)}</td>"
            f"<td class=num>{_f(a.min)}</td><td class=num>{_f(a.max)}</td>"
            f"<td class=num>{a.count}</td>"
        )
        if baseline:
            b = baseline.mean(m)
            d = None if (b is None or a.mean is None) else a.mean - b
            cls = "" if d is None or abs(d) < 1e-9 else ("good" if d > 0 else "bad")
            sign = "" if d is None or d < 0 else "+"
            cells += f"<td class=num>{_f(b)}</td><td class='num {cls}'>{sign}{_f(d)}</td>"
        rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _case_table(run: RunResult) -> str:
    head = "".join(f"<th>{escape(m)}</th>" for m in run.metrics)
    rows = [f"<tr><th>case</th><th>question / answer</th><th>retrieved</th>{head}</tr>"]
    for r in run.rows:
        rel = set(r.relevant_doc_ids)
        retrieved = ", ".join(
            f"<b>{escape(d)}</b>" if d in rel else escape(d) for d in r.retrieved_doc_ids
        )
        scores = "".join(f"<td class=num>{_f(r.scores.get(m), 2)}</td>" for m in run.metrics)
        err = f"<div class=err>{escape(r.error)}</div>" if r.error else ""
        rows.append(
            f"<tr><td>{escape(r.case_id)}</td><td>{escape(r.question)}"
            f"<div class=small>{escape(r.answer) or '<i>(empty)</i>'}</div>{err}</td>"
            f"<td class=small>{retrieved}</td>{scores}</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def render_html(run: RunResult, baseline: RunResult | None = None) -> str:
    """Render ``run`` (optionally against ``baseline``) as one self-contained HTML page."""
    hists = []
    for m in run.metrics:
        vals = [v for v in (r.scores.get(m) for r in run.rows) if v is not None]
        if vals:
            hists.append(f'<div class="card hist"><h3>{escape(m)}</h3>{_histogram(vals)}</div>')
    legend = (
        '<p class="legend small"><span style="background:var(--base)"></span>baseline'
        '<span style="background:var(--accent)"></span>current</p>'
        if baseline
        else ""
    )
    errors = int(run.stats.get("errors", 0))
    meta = (
        f"dataset <b>{escape(run.dataset)}</b> &middot; {len(run.rows)} cases &middot; "
        f"{run.duration_s:.2f}s &middot; ragcheck {escape(run.ragcheck_version)} &middot; "
        f"{escape(run.created_at)}"
        + (f" &middot; <span class=bad>{errors} errors</span>" if errors else "")
    )
    title = escape(f"ragcheck report - {run.name}")
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{_CSS}</style></head><body><main>"
        f"<h1>{escape(run.name)}</h1><div class=meta>{meta}</div>"
        f"<h2>Mean score per metric</h2><div class=card>{legend}"
        f"{_bar_chart(run, baseline)}</div>"
        f"<h2>Summary</h2><div class=card>{_summary_table(run, baseline)}</div>"
        f"<h2>Score distributions</h2><div class=grid>{''.join(hists)}</div>"
        f"<h2>Per-case results</h2><div class=card>{_case_table(run)}</div>"
        "<p class=small>Bold document ids are labelled relevant. Generated by ragcheck.</p>"
        "</main></body></html>"
    )


def write_html(run: RunResult, path: str | Path, baseline: RunResult | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(run, baseline), encoding="utf-8")
    return p
