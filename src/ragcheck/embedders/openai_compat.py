from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np
from numpy.typing import NDArray

from ragcheck.providers.base import ProviderError, request_with_retries_sync

__all__ = ["OpenAICompatibleEmbedder"]


class OpenAICompatibleEmbedder:
    """Embedder for any server implementing ``POST {base_url}/embeddings``.

    Works with OpenAI (``https://api.openai.com/v1``), Ollama
    (``http://localhost:11434/v1``) and other OpenAI-compatible gateways. The API key is
    read from ``api_key`` or the environment variable named by ``api_key_env``.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        batch_size: int = 64,
        timeout: float = 60.0,
        max_retries: int = 3,
        dimensions: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.dimensions = dimensions
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.Client(headers=headers, timeout=timeout, transport=transport)

    @property
    def name(self) -> str:
        return f"openai-compat({self.base_url},{self.model},dims={self.dimensions})"

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        resp = request_with_retries_sync(
            self._client, "POST", f"{self.base_url}/embeddings", payload, self.max_retries
        )
        try:
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            vectors = [list(map(float, d["embedding"])) for d in data]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(f"malformed embeddings response: {exc}") from exc
        if len(vectors) != len(texts):
            raise ProviderError(f"expected {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dimensions or 0), dtype=np.float32)
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            rows.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return np.asarray(rows, dtype=np.float32)

    def close(self) -> None:
        self._client.close()
