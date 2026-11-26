"""Command line interface: ``ragcheck run | compare | report | grid | metrics``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ragcheck._version import __version__
from ragcheck.compare import RuleError, compare_runs, parse_rule
from ragcheck.config import ConfigError, build, load_config
from ragcheck.datasets import DatasetError
from ragcheck.experiments import run_grid
from ragcheck.report import write_html
from ragcheck.results import RunResult
from ragcheck.runner import run as run_pipeline

app = typer.Typer(
    name="ragcheck",
    help="Benchmark and regression-test RAG pipelines.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()
err = Console(stderr=True)

EXIT_REGRESSION = 1
EXIT_USAGE = 2


def _version(value: bool) -> None:
    if value:
        console.print(f"ragcheck {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Benchmark and regression-test RAG pipelines."""


def _fail(msg: str) -> typer.Exit:
    err.print(f"[bold red]error:[/] {msg}")
    return typer.Exit(EXIT_USAGE)


def _load_run(path: Path) -> RunResult:
    try:
        return RunResult.load(path)
    except FileNotFoundError as exc:
        raise _fail(f"no such file: {path}") from exc
    except ValueError as exc:
        raise _fail(f"{path} is not a ragcheck results file: {exc}") from exc


@app.command("run")
def run_cmd(
    config: Annotated[Path, typer.Argument(help="YAML run configuration.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Where to write results JSON.")
    ] = None,
    html: Annotated[Path | None, typer.Option(help="Also write an HTML report here.")] = None,
    limit: Annotated[int | None, typer.Option(help="Only evaluate the first N cases.")] = None,
    progress: Annotated[bool, typer.Option(help="Show a progress bar.")] = True,
) -> None:
    """Run an evaluation described by a YAML config."""
    t0 = time.perf_counter()
    try:
        cfg = load_config(config)
        if limit is not None:
            cfg = cfg.model_copy(update={"limit": limit})
        built = build(cfg)
    except (ConfigError, DatasetError) as exc:
        raise _fail(str(exc)) from exc
    result = run_pipeline(
        built.pipeline,
        built.dataset,
        built.metrics,
        concurrency=cfg.concurrency,
        show_progress=progress,
        name=cfg.name,
        config=cfg.model_dump(),
        console=err,
    )
    if built.cache is not None:
        result.stats["cache_hits"] = float(built.cache.hits)
        result.stats["cache_misses"] = float(built.cache.misses)
        built.cache.close()
    console.print(result.summary_table())
    out = output or (Path(cfg.output) if cfg.output else None)
    if out is not None:
        result.save(out)
        console.print(f"results written to [bold]{out}[/]")
    if html is not None:
        write_html(result, html)
        console.print(f"report written to [bold]{html}[/]")
    errors = int(result.stats.get("errors", 0))
    if errors:
        err.print(f"[yellow]{errors} case(s) had errors; see the results file[/]")
    console.print(
        f"{len(result.rows)} cases, {len(built.pipeline.chunks)} chunks, "
        f"{time.perf_counter() - t0:.2f}s total"
    )


@app.command("compare")
def compare_cmd(
    baseline: Annotated[Path, typer.Argument(help="Baseline results JSON.")],
    current: Annotated[Path, typer.Argument(help="Current results JSON.")],
    fail_on: Annotated[
        list[str] | None,
        typer.Option(
            "--fail-on",
            help="Regression rule, e.g. 'recall@5<-0.02' or 'faithfulness<-5%'. Repeatable.",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Compare two runs; exit 1 if any --fail-on rule is violated (the CI gate)."""
    try:
        rules = [parse_rule(r) for r in (fail_on or [])]
    except RuleError as exc:
        raise _fail(str(exc)) from exc
    cmp = compare_runs(_load_run(baseline), _load_run(current), rules)
    if as_json:
        payload = {
            "failed": cmp.failed,
            "rules": [str(r) for r in rules],
            "metrics": [
                {
                    "metric": d.metric,
                    "baseline": d.baseline,
                    "current": d.current,
                    "delta": d.delta,
                    "rel_delta_pct": d.rel_delta,
                    "status": d.status,
                    "violations": d.violations,
                }
                for d in cmp.deltas
            ],
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        table = Table(title=f"{baseline.name} -> {current.name}")
        for col in ("metric", "baseline", "current", "delta", "delta %", "status"):
            table.add_column(col, justify="left" if col in ("metric", "status") else "right")

        def f(x: float | None, sign: bool = False) -> str:
            if x is None:
                return "-"
            return f"{x:+.4f}" if sign else f"{x:.4f}"

        for d in cmp.deltas:
            status = {"ok": "[green]ok[/]", "fail": "[red]FAIL[/]", "missing": "[red]MISSING[/]"}
            pct = "-" if d.rel_delta is None else f"{d.rel_delta:+.1f}%"
            table.add_row(
                d.metric,
                f(d.baseline),
                f(d.current),
                f(d.delta, sign=True),
                pct,
                status[d.status] if d.violations or d.status == "ok" else d.status,
            )
        console.print(table)
        if cmp.failed:
            for d in cmp.failures:
                err.print(f"[red]regression:[/] {d.metric}: {', '.join(d.violations)}")
        elif rules:
            console.print(f"[green]all {len(rules)} rule(s) passed[/]")
    if cmp.failed:
        raise typer.Exit(EXIT_REGRESSION)


@app.command("report")
def report_cmd(
    results: Annotated[Path, typer.Argument(help="Results JSON from 'ragcheck run'.")],
    html: Annotated[Path, typer.Option(help="Output HTML file.")] = Path("report.html"),
    baseline: Annotated[
        Path | None, typer.Option(help="Optional baseline results JSON to diff against.")
    ] = None,
) -> None:
    """Render a self-contained HTML report (inline SVG charts, no CDN)."""
    run = _load_run(results)
    base = _load_run(baseline) if baseline else None
    write_html(run, html, base)
    console.print(f"report written to [bold]{html}[/]")


@app.command("grid")
def grid_cmd(
    config: Annotated[Path, typer.Argument(help="YAML config with a 'grid:' section.")],
    markdown: Annotated[
        Path | None, typer.Option(help="Write the comparison table as Markdown.")
    ] = None,
    metric: Annotated[
        list[str] | None, typer.Option("--metric", "-m", help="Columns to show (default: all).")
    ] = None,
) -> None:
    """Sweep chunk sizes x retrievers and print a comparison table."""
    try:
        cfg = load_config(config)
        if cfg.grid is None:
            raise ConfigError(f"{config} has no 'grid:' section")
        t0 = time.perf_counter()
        grid = run_grid(cfg)
    except (ConfigError, DatasetError) as exc:
        raise _fail(str(exc)) from exc
    cols = metric or grid.metrics
    table = Table(title=f"grid - {cfg.name}")
    for col in ("chunker", "size", "retriever", "chunks", *cols, "time (s)"):
        table.add_column(
            col, justify="left" if col in ("chunker", "retriever") else "right", overflow="fold"
        )
    for row in grid.rows:
        vals = [row.score(m) for m in cols]
        table.add_row(
            row.chunker,
            str(row.chunk_size),
            row.retriever,
            str(row.n_chunks),
            *("-" if v is None else f"{v:.3f}" for v in vals),
            f"{row.seconds:.2f}",
        )
    console.print(table)
    console.print(f"{len(grid.rows)} configurations in {time.perf_counter() - t0:.2f}s")
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(grid.to_markdown(cols) + "\n", encoding="utf-8")
        console.print(f"markdown table written to [bold]{markdown}[/]")


@app.command("metrics")
def metrics_cmd() -> None:
    """List available metric names."""
    rows = [
        ("recall@k", "retrieval", "relevant docs in top-k / all relevant docs"),
        ("precision@k", "retrieval", "relevant docs in top-k / k"),
        ("hit_rate@k", "retrieval", "1 if any relevant doc in top-k"),
        ("mrr[@k]", "retrieval", "1 / rank of first relevant doc"),
        ("ndcg@k", "retrieval", "binary-relevance nDCG"),
        ("exact_match", "generation", "normalised answer == reference"),
        ("token_f1", "generation", "SQuAD token F1"),
        ("rouge_l", "generation", "LCS-based F-measure"),
        ("semantic_similarity", "generation", "cosine(answer, reference) via embedder"),
        ("faithfulness", "judge", "supported claims / claims"),
        ("answer_relevance", "judge", "mean cosine(question, generated questions)"),
        ("context_precision[@k]", "judge", "average precision of useful contexts"),
        ("context_recall", "judge", "attributable reference sentences / sentences"),
    ]
    table = Table(title="metrics")
    for col in ("name", "kind", "definition"):
        table.add_column(col)
    for r in rows:
        table.add_row(*r)
    console.print(table)


def main() -> None:  # pragma: no cover - thin entry point
    app()
