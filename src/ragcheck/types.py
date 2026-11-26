"""Core data model.

Every object that crosses a component boundary (chunker -> retriever -> generator ->
metric -> report) is one of the immutable Pydantic models defined here. Keeping them
frozen makes results reproducible and safe to cache.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "EvalCase",
    "EvalResult",
    "Query",
    "RetrievedContext",
    "ScoredChunk",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Document(_Frozen):
    """A source document in the corpus."""

    id: str = Field(min_length=1)
    text: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(_Frozen):
    """A contiguous span of a :class:`Document`.

    Invariant: ``chunk.text == document.text[chunk.start:chunk.end]``. All built-in
    chunkers preserve it, which is what lets the property tests assert that no content
    is lost.
    """

    id: str = Field(min_length=1)
    doc_id: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    index: int = Field(ge=0, description="Position of the chunk within its document.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_span(self) -> Chunk:
        if self.end < self.start:
            raise ValueError(f"chunk end ({self.end}) < start ({self.start})")
        return self


class Query(_Frozen):
    """A question posed to a pipeline."""

    id: str
    text: str


class ScoredChunk(_Frozen):
    """A chunk together with the retriever's relevance score (higher is better)."""

    chunk: Chunk
    score: float

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


class RetrievedContext(_Frozen):
    """Ranked retrieval output for one query."""

    query: Query
    chunks: tuple[ScoredChunk, ...] = ()

    @property
    def texts(self) -> list[str]:
        return [c.chunk.text for c in self.chunks]

    @property
    def doc_ids(self) -> list[str]:
        """Document ids in rank order, de-duplicated (first occurrence wins)."""
        seen: dict[str, None] = {}
        for c in self.chunks:
            seen.setdefault(c.chunk.doc_id, None)
        return list(seen)


class Answer(_Frozen):
    """A generated answer."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalCase(_Frozen):
    """One labelled evaluation example."""

    id: str = Field(min_length=1)
    question: str
    ground_truth: str = ""
    relevant_doc_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(_Frozen):
    """The per-case row of a run: what the pipeline produced and how it scored."""

    case_id: str
    question: str
    ground_truth: str
    answer: str
    retrieved_doc_ids: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]
    scores: dict[str, float | None]
    latency_ms: float = 0.0
    error: str | None = None
