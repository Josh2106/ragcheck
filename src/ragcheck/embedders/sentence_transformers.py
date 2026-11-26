from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["SentenceTransformerEmbedder"]


class SentenceTransformerEmbedder:
    """Adapter for local ``sentence-transformers`` models.

    Requires the optional extra: ``pip install "ragcheck[sentence-transformers]"``.

    The model is loaded lazily on first use so importing ragcheck stays cheap.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: Any = None

    @property
    def name(self) -> str:
        return f"sentence-transformers({self.model_name},norm={self.normalize})"

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError(
                    "SentenceTransformerEmbedder needs the optional extra: "
                    "pip install 'ragcheck[sentence-transformers]'"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        model = self._load()
        vecs = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
