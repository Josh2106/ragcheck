from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["InMemoryVectorIndex", "VectorIndex"]


@runtime_checkable
class VectorIndex(Protocol):
    """Minimal vector store interface; implement it to plug in FAISS, Qdrant, etc."""

    def add(self, ids: Sequence[str], vectors: NDArray[np.float32]) -> None: ...

    def search(self, vector: NDArray[np.float32], k: int) -> list[tuple[str, float]]: ...

    def __len__(self) -> int: ...


class InMemoryVectorIndex:
    """Exact cosine-similarity search over a NumPy matrix.

    Vectors are L2-normalised on insert so search is a single matrix-vector product,
    followed by ``argpartition`` for O(n) top-k selection.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._matrix: NDArray[np.float32] | None = None

    def __len__(self) -> int:
        return len(self._ids)

    @staticmethod
    def _normalise(m: NDArray[np.float32]) -> NDArray[np.float32]:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return (m / norms).astype(np.float32)

    def add(self, ids: Sequence[str], vectors: NDArray[np.float32]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError(
                f"expected a ({len(ids)}, dim) matrix, got shape {tuple(vectors.shape)}"
            )
        if len(ids) == 0:
            return
        if self._matrix is not None and vectors.shape[1] != self._matrix.shape[1]:
            raise ValueError(
                f"dimension mismatch: index has {self._matrix.shape[1]}, got {vectors.shape[1]}"
            )
        normed = self._normalise(vectors)
        self._matrix = normed if self._matrix is None else np.vstack([self._matrix, normed])
        self._ids.extend(ids)

    def search(self, vector: NDArray[np.float32], k: int) -> list[tuple[str, float]]:
        if self._matrix is None or k <= 0:
            return []
        q = self._normalise(np.asarray(vector, dtype=np.float32).reshape(1, -1))[0]
        if q.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"query dimension {q.shape[0]} != index dimension {self._matrix.shape[1]}"
            )
        sims = self._matrix @ q
        k = min(k, len(self._ids))
        top = np.argpartition(-sims, k - 1)[:k]
        # stable tie-break on insertion order for reproducibility
        order = sorted(top.tolist(), key=lambda i: (-float(sims[i]), i))
        return [(self._ids[i], float(sims[i])) for i in order]
