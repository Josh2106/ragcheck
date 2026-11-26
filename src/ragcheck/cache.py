"""Content-addressed disk cache for embeddings and LLM calls (SQLite).

Keys are SHA-256 hashes of a canonical JSON encoding of everything that determines the
output (model name + inputs + sampling params), so changing any of them is a cache miss,
and re-running an unchanged experiment costs nothing. API keys are never part of a key.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ragcheck.protocols import Embedder
from ragcheck.providers.base import Completion, Message, Provider

__all__ = ["CachedEmbedder", "CachedProvider", "SQLiteCache", "content_hash"]


def content_hash(obj: Any) -> str:
    """SHA-256 of the canonical (sorted-keys, compact) JSON encoding of ``obj``."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class SQLiteCache:
    """A tiny namespaced key/value store. ``path=":memory:"`` keeps it in RAM."""

    def __init__(self, path: str | Path = ".ragcheck_cache/cache.sqlite") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                " ns TEXT NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL,"
                " PRIMARY KEY (ns, key))"
            )
        self.hits = 0
        self.misses = 0

    def get(self, ns: str, key: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM cache WHERE ns = ? AND key = ?", (ns, key)
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return bytes(row[0])

    def set(self, ns: str, key: str, value: bytes) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (ns, key, value) VALUES (?, ?, ?)", (ns, key, value)
            )

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0])

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM cache")

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class CachedEmbedder:
    """Wrap an embedder; only texts never seen before (for this model) are embedded."""

    def __init__(self, embedder: Embedder, cache: SQLiteCache) -> None:
        self.embedder = embedder
        self.cache = cache

    @property
    def name(self) -> str:
        return self.embedder.name

    def _key(self, text: str) -> str:
        return content_hash({"model": self.embedder.name, "text": text})

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        found: dict[int, NDArray[np.float32]] = {}
        missing: list[int] = []
        for i, t in enumerate(texts):
            raw = self.cache.get("embedding", self._key(t))
            if raw is None:
                missing.append(i)
            else:
                found[i] = np.frombuffer(raw, dtype=np.float32)
        if missing:
            computed = np.asarray(
                self.embedder.embed([texts[i] for i in missing]), dtype=np.float32
            )
            for i, vec in zip(missing, computed, strict=True):
                self.cache.set("embedding", self._key(texts[i]), vec.tobytes())
                found[i] = vec
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack([found[i] for i in range(len(texts))]).astype(np.float32)


class CachedProvider:
    """Wrap a provider; identical requests are answered from the cache."""

    def __init__(self, provider: Provider, cache: SQLiteCache) -> None:
        self.provider = provider
        self.cache = cache

    @property
    def name(self) -> str:
        return self.provider.name

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> Completion:
        key = content_hash(
            {
                "provider": self.provider.name,
                "messages": [[m.role, m.content] for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            }
        )
        raw = self.cache.get("llm", key)
        if raw is not None:
            data = json.loads(raw)
            return Completion(text=data["text"], model=data["model"], usage=data["usage"])
        completion = await self.provider.complete(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )
        payload = {"text": completion.text, "model": completion.model, "usage": completion.usage}
        self.cache.set("llm", key, json.dumps(payload).encode("utf-8"))
        return completion
