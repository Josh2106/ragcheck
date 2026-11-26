"""Tokenizers that report character offsets.

Chunkers need token *spans* (not just ids) so chunk boundaries can be mapped back to the
original text. :class:`TiktokenTokenizer` is used when the optional ``tiktoken`` extra is
installed; :class:`RegexTokenizer` is a dependency-free fallback that approximates
sub-word tokenizers by splitting words, numbers and punctuation.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "RegexTokenizer",
    "TiktokenTokenizer",
    "Tokenizer",
    "get_tokenizer",
    "simple_tokens",
]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def simple_tokens(text: str) -> list[str]:
    """Lower-cased word tokens; the shared normaliser for lexical metrics and BM25."""
    return _WORD_RE.findall(text.lower())


@runtime_checkable
class Tokenizer(Protocol):
    name: str

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character ``(start, end)`` offsets of each token, in order."""
        ...

    def count(self, text: str) -> int: ...


class RegexTokenizer:
    """Words and individual punctuation marks are tokens. Whitespace is not."""

    name = "regex"

    def __init__(self, max_token_chars: int = 0) -> None:
        # Optionally split very long "words" (URLs, base64 blobs) into pieces so that a
        # single token can never exceed a sane size.
        self.max_token_chars = max_token_chars

    def spans(self, text: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for m in _TOKEN_RE.finditer(text):
            s, e = m.span()
            if self.max_token_chars > 0 and e - s > self.max_token_chars:
                for i in range(s, e, self.max_token_chars):
                    out.append((i, min(i + self.max_token_chars, e)))
            else:
                out.append((s, e))
        return out

    def count(self, text: str) -> int:
        return len(self.spans(text))


class TiktokenTokenizer:
    """Adapter over ``tiktoken`` that recovers character offsets.

    Leading whitespace that tiktoken attaches to a token (``" hello"``) is kept inside the
    span, so spans tile the text exactly.
    """

    def __init__(self, encoding: str = "cl100k_base", _encoding_obj: Any = None) -> None:
        if _encoding_obj is None:
            import tiktoken

            _encoding_obj = tiktoken.get_encoding(encoding)
        self._enc = _encoding_obj
        self.name = f"tiktoken:{encoding}"

    def spans(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        ids = self._enc.encode(text, disallowed_special=())
        out: list[tuple[int, int]] = []
        pos = 0
        buf = b""
        for tid in ids:
            buf += self._enc.decode_single_token_bytes(tid)
            try:
                piece = buf.decode("utf-8")
            except UnicodeDecodeError:
                continue  # multi-byte character split across tokens; keep accumulating
            buf = b""
            out.append((pos, pos + len(piece)))
            pos += len(piece)
        return out

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


def get_tokenizer(name: str = "auto") -> Tokenizer:
    """Resolve a tokenizer by name.

    ``"auto"`` prefers tiktoken (``cl100k_base``) and falls back to the regex tokenizer if
    tiktoken is missing or its encoding cannot be loaded (e.g. offline CI).
    """
    if name == "regex":
        return RegexTokenizer()
    if name == "auto":
        try:
            return TiktokenTokenizer()
        except Exception:
            return RegexTokenizer()
    if name.startswith("tiktoken:"):
        return TiktokenTokenizer(name.split(":", 1)[1])
    raise ValueError(f"unknown tokenizer {name!r}; use 'auto', 'regex' or 'tiktoken:<encoding>'")
