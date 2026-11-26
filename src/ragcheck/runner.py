"""Evaluate a pipeline over a dataset with bounded async concurrency."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from ragcheck._version import __version__
from ragcheck.datasets import Dataset
from ragcheck.metrics.base import Metric, Sample
from ragcheck.protocols import Pipeline
from ragcheck.results import RunResult
from ragcheck.types import EvalCase, EvalResult

__all__ = ["arun", "run"]


async def _eval_case(
    pipeline: Pipeline, case: EvalCase, metrics: Sequence[Metric], sem: asyncio.Semaphore
) -> EvalResult:
    async with sem:
        t0 = time.perf_counter()
        try:
            context, answer = await pipeline.query(case.question, case.id)
        except Exception as exc:
            return EvalResult(
                case_id=case.id,
                question=case.question,
                ground_truth=case.ground_truth,
                answer="",
                retrieved_doc_ids=(),
                relevant_doc_ids=case.relevant_doc_ids,
                scores={m.name: None for m in metrics},
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=f"pipeline: {type(exc).__name__}: {exc}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        sample = Sample(case=case, context=context, answer=answer)
        outcomes = await asyncio.gather(*(m.score(sample) for m in metrics), return_exceptions=True)
        scores: dict[str, float | None] = {}
        errors: list[str] = []
        for metric, outcome in zip(metrics, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                scores[metric.name] = None
                errors.append(f"{metric.name}: {type(outcome).__name__}: {outcome}")
            else:
                scores[metric.name] = outcome
        return EvalResult(
            case_id=case.id,
            question=case.question,
            ground_truth=case.ground_truth,
            answer=answer.text,
            retrieved_doc_ids=tuple(context.doc_ids),
            relevant_doc_ids=case.relevant_doc_ids,
            scores=scores,
            latency_ms=latency_ms,
            error="; ".join(errors) or None,
        )


async def arun(
    pipeline: Pipeline,
    dataset: Dataset,
    metrics: Sequence[Metric],
    *,
    concurrency: int = 8,
    index: bool = True,
    show_progress: bool = False,
    name: str = "run",
    config: dict[str, Any] | None = None,
    console: Console | None = None,
) -> RunResult:
    """Async core of :func:`run`. Rows come back in dataset order regardless of timing."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    names = [m.name for m in metrics]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate metric names in {names}")
    started = time.perf_counter()
    if index:
        pipeline.index(dataset.documents)
    index_s = time.perf_counter() - started
    sem = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(_eval_case(pipeline, c, metrics, sem)) for c in dataset.cases]
    if show_progress:
        with Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console or Console(stderr=True),
            transient=False,
        ) as progress:
            bar = progress.add_task(f"evaluating {dataset.name}", total=len(tasks))
            for fut in asyncio.as_completed(tasks):
                await fut
                progress.advance(bar)
    rows = list(await asyncio.gather(*tasks))
    return RunResult.from_rows(
        rows,
        names,
        name=name,
        ragcheck_version=__version__,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        duration_s=time.perf_counter() - started,
        dataset=dataset.source or dataset.name,
        config=config or {},
        stats={"index_s": index_s, "cases": float(len(rows))},
    )


def run(
    pipeline: Pipeline,
    dataset: Dataset,
    metrics: Sequence[Metric],
    **kwargs: Any,
) -> RunResult:
    """Evaluate ``pipeline`` on ``dataset`` and return a :class:`RunResult`.

    Synchronous wrapper around :func:`arun`; call ``arun`` directly from async code.
    Pipeline and metric failures are recorded per case (``EvalResult.error``) instead of
    aborting the run.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(arun(pipeline, dataset, metrics, **kwargs))
    raise RuntimeError("ragcheck.run() called inside an event loop; use 'await ragcheck.arun()'")
