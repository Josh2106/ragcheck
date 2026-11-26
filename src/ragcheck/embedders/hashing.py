from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ragcheck.tokenizers import simple_tokens

__all__ = ["HashingEmbedder"]


class HashingEmbedder:
    """Deterministic, dependency-free embedder based on the hashing trick.

    Word unigrams, word bigrams and (optionally) character n-grams are hashed with BLAKE2b
    into ``dim`` buckets with a hash-derived sign, then L2-normalised. It is *not* a
    semantic model, but cosine similarity correlates with lexical overlap, which is enough
    to exercise dense/hybrid code paths reproducibly in tests and CI. Unlike Python's
    built-in ``hash`` it is stable across processes and platforms.
    """

    def __init__(self, dim: int = 512, char_ngrams: int = 3, use_bigrams: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if char_ngrams < 0:
            raise ValueError("char_ngrams must be >= 0")
        self.dim = dim
        self.char_ngrams = char_ngrams
        self.use_bigrams = use_bigrams

    @property
    def name(self) -> str:
        return f"hashing(dim={self.dim},char={self.char_ngrams},bigrams={self.use_bigrams})"

    def _features(self, text: str) -> list[tuple[str, float]]:
        toks = simple_tokens(text)
        feats: list[tuple[str, float]] = [("w:" + t, 1.0) for t in toks]
        if self.use_bigrams:
            feats += [(f"b:{a}_{b}", 0.5) for a, b in itertools.pairwise(toks)]
        if self.char_ngrams:
            n = self.char_ngrams
            for t in toks:
                padded = f"<{t}>"
                feats += [("c:" + padded[i : i + n], 0.25) for i in range(len(padded) - n + 1)]
        return feats

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(h, "little")
        return v % self.dim, 1.0 if (v >> 63) & 1 else -1.0

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                idx, sign = self._bucket(feat)
                out[row, idx] += sign * weight
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)
