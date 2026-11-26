"""Text normalisation shared by lexical metrics and the deterministic judge."""

from __future__ import annotations

import re
import string
import unicodedata

from ragcheck.tokenizers import simple_tokens

__all__ = ["STOPWORDS", "content_tokens", "normalize_answer", "token_recall"]

STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with from as is are was were be
    been being it its this that these those there here which who whom whose what when where why
    how do does did doing done can could should would will shall may might must not no yes so
    than too very just also into onto over under about after before between through during per
    via your you we our they their them he she his her i me my mine us any all each some such
    only own same other more most both few nor up down out off again further once has have had
    having""".split()  # noqa: SIM905 - a word list reads better than a list literal
)

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: NFKC, lower-case, strip punctuation/articles/whitespace."""
    text = unicodedata.normalize("NFKC", text).lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def content_tokens(text: str) -> list[str]:
    """Lower-cased word tokens without stopwords."""
    return [t for t in simple_tokens(text) if t not in STOPWORDS]


def token_recall(needle: str, haystack_tokens: set[str]) -> float:
    """Fraction of ``needle``'s content tokens present in ``haystack_tokens`` (1.0 if none)."""
    toks = content_tokens(needle)
    if not toks:
        return 1.0
    return sum(t in haystack_tokens for t in toks) / len(toks)
