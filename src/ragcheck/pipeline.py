"""The default chunk -> retrieve -> generate pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from ragcheck.protocols import Chunker, Generator, Retriever
from ragcheck.types import Answer, Chunk, Document, Query, RetrievedContext

__all__ = ["RAGPipeline"]


class RAGPipeline:
    """Compose a chunker, a retriever and (optionally) a generator.

    With ``generator=None`` the pipeline is retrieval-only: answers are empty and
    generation/judge metrics that need an answer report ``None``.
    """

    def __init__(
        self,
        chunker: Chunker,
        retriever: Retriever,
        generator: Generator | None = None,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.chunker = chunker
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.chunks: list[Chunk] = []

    def index(self, documents: Sequence[Document]) -> None:
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.chunker.chunk(doc))
        self.chunks = chunks
        self.retriever.index(chunks)

    def retrieve(self, question: str, query_id: str = "") -> RetrievedContext:
        hits = self.retriever.retrieve(question, self.top_k)
        return RetrievedContext(query=Query(id=query_id, text=question), chunks=tuple(hits))

    async def query(self, question: str, query_id: str = "") -> tuple[RetrievedContext, Answer]:
        context = self.retrieve(question, query_id)
        if self.generator is None:
            return context, Answer(text="", metadata={"generator": None})
        return context, await self.generator.generate(question, context)
