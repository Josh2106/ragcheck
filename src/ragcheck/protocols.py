"""Structural interfaces for every pluggable component.

These are :class:`typing.Protocol` classes, so any object with the right methods works;
inheritance is never required. All are ``runtime_checkable`` for friendlier errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ragcheck.judges import ClaimVerdict
    from ragcheck.types import Answer, Chunk, Document, RetrievedContext, ScoredChunk

__all__ = ["Chunker", "Embedder", "Generator", "Judge", "Pipeline", "Reranker", "Retriever"]


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into chunks."""

    def chunk(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to dense vectors.

    ``name`` must uniquely identify the model *and* its configuration; it is part of
    the embedding cache key.
    """

    @property
    def name(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


@runtime_checkable
class Retriever(Protocol):
    """Indexes chunks and returns the top-k for a query."""

    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-orders a candidate list, e.g. with a cross-encoder."""

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]: ...


@runtime_checkable
class Generator(Protocol):
    """Produces an answer from a question and retrieved context."""

    async def generate(self, question: str, context: RetrievedContext) -> Answer: ...


@runtime_checkable
class Judge(Protocol):
    """LLM-as-judge primitives used by the RAGAS-style metrics.

    Each method corresponds to one prompt in the RAGAS paper (Es et al., 2023,
    arXiv:2309.15217); the *metric* does the arithmetic, the judge only gives verdicts.
    """

    async def verify_claims(self, answer: str, contexts: Sequence[str]) -> list[ClaimVerdict]:
        """Split ``answer`` into atomic claims and mark each as supported or not."""
        ...

    async def generate_questions(self, answer: str, n: int) -> tuple[list[str], bool]:
        """Return ``n`` questions the answer would answer, plus a noncommittal flag."""
        ...

    async def is_context_useful(self, question: str, ground_truth: str, context: str) -> bool:
        """Was ``context`` useful for arriving at ``ground_truth``?"""
        ...

    async def attribute_statements(self, ground_truth: str, contexts: Sequence[str]) -> list[bool]:
        """For each sentence of ``ground_truth``: can it be attributed to the contexts?"""
        ...


@runtime_checkable
class Pipeline(Protocol):
    """What :func:`ragcheck.run` evaluates."""

    def index(self, documents: Sequence[Document]) -> None: ...

    async def query(self, question: str, query_id: str = "") -> tuple[RetrievedContext, Answer]: ...
