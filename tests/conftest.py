from __future__ import annotations

import pytest

from ragcheck import Dataset, Document, EvalCase, load_sample
from ragcheck.chunkers import RecursiveChunker
from ragcheck.embedders import HashingEmbedder
from ragcheck.generators import ExtractiveGenerator
from ragcheck.pipeline import RAGPipeline
from ragcheck.retrievers import BM25Retriever


@pytest.fixture
def tiny_dataset() -> Dataset:
    docs = [
        Document(id="a", text="Larkspur retries failed tasks five times. Backoff doubles."),
        Document(id="b", text="The dashboard runs on port 7420. It shows queue depth."),
        Document(id="c", text="Webhooks are signed with HMAC-SHA256 using a shared secret."),
    ]
    cases = [
        EvalCase(
            id="q1",
            question="How many times are failed tasks retried?",
            ground_truth="Larkspur retries failed tasks five times.",
            relevant_doc_ids=("a",),
        ),
        EvalCase(
            id="q2",
            question="Which port does the dashboard use?",
            ground_truth="The dashboard runs on port 7420.",
            relevant_doc_ids=("b",),
        ),
        EvalCase(id="q3", question="How are webhooks signed?", ground_truth=""),
    ]
    return Dataset(documents=docs, cases=cases, name="tiny")


@pytest.fixture(scope="session")
def sample() -> Dataset:
    return load_sample()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


@pytest.fixture
def bm25_pipeline() -> RAGPipeline:
    return RAGPipeline(RecursiveChunker(200), BM25Retriever(), ExtractiveGenerator(), top_k=3)
