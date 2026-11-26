"""ragcheck - benchmark and regression-test RAG pipelines.

Quick tour::

    import ragcheck
    from ragcheck.chunkers import RecursiveChunker
    from ragcheck.retrievers import BM25Retriever
    from ragcheck.generators import ExtractiveGenerator
    from ragcheck.metrics import build_metrics

    dataset = ragcheck.load_sample()
    pipeline = ragcheck.RAGPipeline(RecursiveChunker(400), BM25Retriever(), ExtractiveGenerator())
    result = ragcheck.run(pipeline, dataset, build_metrics(["recall@5", "mrr", "token_f1"]))
    print(result.to_markdown())
"""

from ragcheck._version import __version__
from ragcheck.datasets import Dataset, load_dataset, load_jsonl_dataset, load_sample
from ragcheck.pipeline import RAGPipeline
from ragcheck.results import MetricSummary, RunResult
from ragcheck.runner import arun, run
from ragcheck.types import (
    Answer,
    Chunk,
    Document,
    EvalCase,
    EvalResult,
    Query,
    RetrievedContext,
    ScoredChunk,
)

__all__ = [
    "Answer",
    "Chunk",
    "Dataset",
    "Document",
    "EvalCase",
    "EvalResult",
    "MetricSummary",
    "Query",
    "RAGPipeline",
    "RetrievedContext",
    "RunResult",
    "ScoredChunk",
    "__version__",
    "arun",
    "load_dataset",
    "load_jsonl_dataset",
    "load_sample",
    "run",
]
