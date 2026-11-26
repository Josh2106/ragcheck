from __future__ import annotations

from collections.abc import Sequence

from ragcheck.protocols import Embedder
from ragcheck.retrievers.index import InMemoryVectorIndex, VectorIndex
from ragcheck.types import Chunk, ScoredChunk

__all__ = ["DenseRetriever"]


class DenseRetriever:
    """Embed chunks once, then rank by cosine similarity to the embedded query."""

    def __init__(
        self, embedder: Embedder, index: VectorIndex | None = None, batch_size: int = 256
    ) -> None:
        self.embedder = embedder
        self.vector_index: VectorIndex = index if index is not None else InMemoryVectorIndex()
        self.batch_size = batch_size
        self._chunks: dict[str, Chunk] = {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            vectors = self.embedder.embed([c.text for c in batch])
            self.vector_index.add([c.id for c in batch], vectors)
            self._chunks.update({c.id: c for c in batch})

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        if not self._chunks or k <= 0:
            return []
        qvec = self.embedder.embed([query])[0]
        return [
            ScoredChunk(chunk=self._chunks[cid], score=score)
            for cid, score in self.vector_index.search(qvec, k)
        ]
