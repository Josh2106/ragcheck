"""Grid experiments: sweep chunk sizes x retrievers and tabulate the results."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from ragcheck.config import ComponentSpec, RunConfig, build, chunk_size_override
from ragcheck.datasets import Dataset
from ragcheck.results import RunResult
from ragcheck.runner import run

__all__ = ["GridResult", "GridRow", "run_grid"]


@dataclass
class GridRow:
    chunker: str
    chunk_size: int
    retriever: str
    n_chunks: int
    seconds: float
    result: RunResult

    def score(self, metric: str) -> float | None:
        return self.result.mean(metric)


@dataclass
class GridResult:
    rows: list[GridRow] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)

    def best(self, metric: str) -> GridRow:
        scored = [r for r in self.rows if r.score(metric) is not None]
        if not scored:
            raise ValueError(f"no scores for {metric!r}")
        return max(scored, key=lambda r: r.score(metric) or 0.0)

    def to_markdown(self, metrics: Sequence[str] | None = None) -> str:
        cols = list(metrics or self.metrics)
        header = "| chunker | size | retriever | chunks | " + " | ".join(cols) + " | time (s) |"
        sep = "|---|---:|---|---:|" + "---:|" * len(cols) + "---:|"
        lines = [header, sep]
        best = {m: max((r.score(m) or 0.0) for r in self.rows) for m in cols}
        for r in self.rows:
            cells = []
            for m in cols:
                v = r.score(m)
                txt = "-" if v is None else f"{v:.3f}"
                cells.append(f"**{txt}**" if v is not None and v == best[m] and v > 0 else txt)
            lines.append(
                f"| {r.chunker} | {r.chunk_size} | {r.retriever} | {r.n_chunks} | "
                + " | ".join(cells)
                + f" | {r.seconds:.2f} |"
            )
        return "\n".join(lines)


def _label(spec: ComponentSpec) -> str:
    extra = ",".join(f"{k}={v}" for k, v in sorted(spec.params.items()))
    return f"{spec.type}({extra})" if extra else spec.type


def run_grid(
    cfg: RunConfig,
    chunk_sizes: Sequence[int] | None = None,
    retrievers: Sequence[ComponentSpec] | None = None,
    dataset: Dataset | None = None,
) -> GridResult:
    """Run every (chunk size, retriever) combination with everything else from ``cfg``.

    The dataset is loaded once and shared; each cell gets a freshly built pipeline so
    no index state leaks between configurations.
    """
    sizes = list(chunk_sizes or (cfg.grid.chunk_sizes if cfg.grid else []))
    rets = list(retrievers or (cfg.grid.retrievers if cfg.grid else []))
    if not sizes or not rets:
        raise ValueError("grid needs at least one chunk size and one retriever")
    if dataset is None:
        dataset = build(cfg.model_copy(update={"grid": None})).dataset
    out = GridResult(metrics=list(cfg.metrics))
    for size in sizes:
        for rspec in rets:
            cell = cfg.model_copy(
                update={
                    "chunker": chunk_size_override(cfg.chunker, size),
                    "retriever": rspec,
                    "grid": None,
                }
            )
            built = build(cell, dataset=dataset)
            t0 = time.perf_counter()
            result = run(
                built.pipeline,
                built.dataset,
                built.metrics,
                concurrency=cfg.concurrency,
                name=f"{cfg.name}[{cfg.chunker.type}={size},{_label(rspec)}]",
                config=cell.model_dump(),
            )
            out.rows.append(
                GridRow(
                    chunker=cfg.chunker.type,
                    chunk_size=size,
                    retriever=_label(rspec),
                    n_chunks=len(built.pipeline.chunks),
                    seconds=time.perf_counter() - t0,
                    result=result,
                )
            )
    return out
