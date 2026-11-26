from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from rich.console import Console

import ragcheck
from ragcheck import Dataset, RAGPipeline, RunResult
from ragcheck.cache import CachedEmbedder, CachedProvider, SQLiteCache, content_hash
from ragcheck.chunkers import RecursiveChunker
from ragcheck.datasets import DatasetError, load_dataset, load_jsonl_dataset
from ragcheck.embedders import HashingEmbedder
from ragcheck.metrics import Metric, Sample, build_metrics
from ragcheck.providers import FakeProvider, Message
from ragcheck.results import summarise
from ragcheck.retrievers import BM25Retriever
from ragcheck.types import Answer, Document, Query, RetrievedContext

# ------------------------------------------------------------------------------ runner


def test_run_end_to_end(tiny_dataset: Dataset, bm25_pipeline: RAGPipeline) -> None:
    metrics = build_metrics(["recall@1", "mrr", "token_f1", "exact_match"])
    result = ragcheck.run(bm25_pipeline, tiny_dataset, metrics, name="tiny", config={"x": 1})
    assert [r.case_id for r in result.rows] == ["q1", "q2", "q3"]
    assert result.mean("recall@1") == 1.0
    assert result.aggregates["recall@1"].count == 2  # q3 has no labels -> None, excluded
    assert result.rows[2].scores["recall@1"] is None
    assert result.rows[2].scores["token_f1"] is None  # no ground truth
    assert result.rows[0].answer == "Larkspur retries failed tasks five times."
    assert result.mean("exact_match") == 1.0
    assert result.config == {"x": 1}
    assert result.stats["errors"] == 0
    assert result.stats["cases"] == 3
    assert result.ragcheck_version == ragcheck.__version__
    assert "recall@1" in result.to_markdown()
    assert result.mean("nope") is None


def test_run_with_progress_bar(tiny_dataset: Dataset, bm25_pipeline: RAGPipeline) -> None:
    console = Console(record=True, width=100)
    ragcheck.run(
        bm25_pipeline, tiny_dataset, build_metrics(["mrr"]), show_progress=True, console=console
    )
    assert "evaluating tiny" in console.export_text()


class FlakyPipeline:
    def __init__(self) -> None:
        self.indexed = 0
        self.active = 0
        self.peak = 0

    def index(self, documents: Sequence[Document]) -> None:
        self.indexed = len(documents)

    async def query(self, question: str, query_id: str = "") -> tuple[RetrievedContext, Answer]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        if query_id == "q2":
            raise RuntimeError("generator exploded")
        return RetrievedContext(query=Query(id=query_id, text=question)), Answer(text="x")


class BrokenMetric(Metric):
    kind = "generation"

    @property
    def name(self) -> str:
        return "broken"

    async def score(self, sample: Sample) -> float | None:
        raise ZeroDivisionError("bad math")


def test_run_records_errors_and_bounds_concurrency(tiny_dataset: Dataset) -> None:
    cases = [*tiny_dataset.cases] * 4
    ds = Dataset(tiny_dataset.documents, cases, name="many")
    p = FlakyPipeline()
    result = ragcheck.run(p, ds, [*build_metrics(["mrr"]), BrokenMetric()], concurrency=2)
    assert p.indexed == 3
    assert p.peak <= 2
    errs = [r.error for r in result.rows]
    assert sum("generator exploded" in (e or "") for e in errs) == 4
    assert all("bad math" in e for e in errs if e and "exploded" not in e)
    assert result.stats["errors"] == 12
    assert result.aggregates["broken"].count == 0


async def test_arun_and_run_inside_loop(tiny_dataset: Dataset, bm25_pipeline: RAGPipeline) -> None:
    res = await ragcheck.arun(bm25_pipeline, tiny_dataset, build_metrics(["mrr"]), index=True)
    assert len(res.rows) == 3
    with pytest.raises(RuntimeError, match="arun"):
        ragcheck.run(bm25_pipeline, tiny_dataset, [])
    with pytest.raises(ValueError, match="concurrency"):
        await ragcheck.arun(bm25_pipeline, tiny_dataset, [], concurrency=0)
    with pytest.raises(ValueError, match="duplicate"):
        await ragcheck.arun(bm25_pipeline, tiny_dataset, [*build_metrics(["mrr"])] * 2)


def test_retrieval_only_pipeline(tiny_dataset: Dataset) -> None:
    p = RAGPipeline(RecursiveChunker(100), BM25Retriever(), generator=None, top_k=2)
    res = ragcheck.run(p, tiny_dataset, build_metrics(["recall@2", "token_f1"]))
    assert res.rows[0].answer == ""
    assert res.mean("recall@2") == 1.0
    with pytest.raises(ValueError, match="top_k"):
        RAGPipeline(RecursiveChunker(100), BM25Retriever(), top_k=0)


def test_results_roundtrip_and_summary(
    tmp_path: Path, tiny_dataset: Dataset, bm25_pipeline: RAGPipeline
) -> None:
    res = ragcheck.run(bm25_pipeline, tiny_dataset, build_metrics(["mrr", "token_f1"]))
    path = res.save(tmp_path / "sub" / "r.json")
    loaded = RunResult.load(path)
    assert loaded == res
    table = loaded.summary_table()
    assert table.row_count == 2
    s = summarise([0.0, 1.0, None, float("nan")])
    assert (s.mean, s.min, s.max, s.count) == (0.5, 0.0, 1.0, 2)
    assert summarise([]).mean is None
    assert summarise([0.3]).std == 0.0


# ------------------------------------------------------------------------------- cache


class CountingEmbedder:
    def __init__(self) -> None:
        self.inner = HashingEmbedder(dim=32)
        self.seen: list[str] = []

    @property
    def name(self) -> str:
        return self.inner.name

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.seen.extend(texts)
        return self.inner.embed(texts)


def test_content_hash_is_canonical() -> None:
    assert content_hash({"a": 1, "b": [1, 2]}) == content_hash({"b": [1, 2], "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_cached_embedder_only_embeds_misses(tmp_path: Path) -> None:
    cache = SQLiteCache(tmp_path / "c" / "cache.sqlite")
    inner = CountingEmbedder()
    emb = CachedEmbedder(inner, cache)
    first = emb.embed(["a", "b"])
    second = emb.embed(["b", "c", "a"])
    assert inner.seen == ["a", "b", "c"]
    np.testing.assert_array_equal(second[0], first[1])
    assert emb.name == inner.name
    assert emb.embed([]).shape == (0, 0)
    assert len(cache) == 3
    cache.close()
    # persisted across instances
    reopened = SQLiteCache(tmp_path / "c" / "cache.sqlite")
    inner2 = CountingEmbedder()
    CachedEmbedder(inner2, reopened).embed(["a"])
    assert inner2.seen == []
    assert reopened.hits == 1
    reopened.clear()
    assert len(reopened) == 0


async def test_cached_provider() -> None:
    cache = SQLiteCache(":memory:")
    fake = FakeProvider(["one", "two"])
    p = CachedProvider(fake, cache)
    msgs = [Message("user", "hi")]
    assert (await p.complete(msgs)).text == "one"
    assert (await p.complete(msgs)).text == "one"  # cache hit
    assert (await p.complete(msgs, temperature=0.5)).text == "two"  # params are in the key
    assert len(fake.calls) == 2
    assert p.name == "fake:fake"
    assert cache.hits == 1


# ---------------------------------------------------------------------------- datasets


def write_jsonl(path: Path, rows: list[object]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_sample_dataset_is_valid(sample: Dataset) -> None:
    assert len(sample.documents) == 30
    assert len(sample) == 20
    ids = {d.id for d in sample.documents}
    assert all(set(c.relevant_doc_ids) <= ids for c in sample.cases)
    assert load_dataset("builtin:sample").source == "builtin:sample"
    assert len(sample.subset(5)) == 5


def test_load_directory_and_aliases(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "corpus.jsonl", [{"id": "d1", "text": "hello"}])
    (tmp_path / "questions.jsonl").write_text(
        '// comment\n\n{"id": "q", "question": "hi?", "answer": "hello", '
        '"relevant_doc_ids": ["d1"]}\n',
        encoding="utf-8",
    )
    ds = load_dataset(tmp_path)
    assert ds.cases[0].ground_truth == "hello"
    assert ds.name == tmp_path.name


@pytest.mark.parametrize(
    ("corpus", "questions", "match"),
    [
        (["{bad"], [], "invalid JSON"),
        ([[1, 2]], [], "expected a JSON object"),
        ([{"id": "", "text": "x"}], [], "corpus.jsonl:1"),
        ([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}], [], "duplicate"),
        (
            [{"id": "a", "text": "x"}],
            [{"id": "q", "question": "?", "relevant_doc_ids": ["zz"]}],
            "unknown relevant",
        ),
        ([{"id": "a", "text": "x"}], [{"question": "?"}], "questions.jsonl:1"),
    ],
)
def test_dataset_errors(
    tmp_path: Path, corpus: list[object], questions: list[object], match: str
) -> None:
    c = tmp_path / "corpus.jsonl"
    if corpus and corpus[0] == "{bad":
        c.write_text("{bad\n", encoding="utf-8")
    else:
        write_jsonl(c, corpus)
    q = write_jsonl(tmp_path / "questions.jsonl", questions)
    with pytest.raises(DatasetError, match=match):
        load_jsonl_dataset(c, q)


def test_dataset_lenient_and_path_errors(tmp_path: Path) -> None:
    c = write_jsonl(tmp_path / "corpus.jsonl", [{"id": "a", "text": "x"}])
    q = write_jsonl(
        tmp_path / "questions.jsonl", [{"id": "q", "question": "?", "relevant_doc_ids": ["zz"]}]
    )
    assert len(load_jsonl_dataset(c, q, strict=False)) == 1
    with pytest.raises(DatasetError, match="builtin"):
        load_dataset("builtin:nope")
    with pytest.raises(DatasetError, match="not a directory"):
        load_dataset(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetError, match="missing"):
        load_dataset(empty)
