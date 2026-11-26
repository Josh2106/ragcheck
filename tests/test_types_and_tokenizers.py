from __future__ import annotations

import sys
import types

import pytest
from pydantic import ValidationError

from ragcheck.tokenizers import (
    RegexTokenizer,
    TiktokenTokenizer,
    get_tokenizer,
    simple_tokens,
)
from ragcheck.types import Chunk, Document, Query, RetrievedContext, ScoredChunk


def _chunk(cid: str, doc: str) -> Chunk:
    return Chunk(id=cid, doc_id=doc, text="x", start=0, end=1, index=0)


def test_models_are_frozen_and_strict() -> None:
    d = Document(id="d", text="hello")
    with pytest.raises(ValidationError):
        d.text = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Document(id="", text="x")
    with pytest.raises(ValidationError):
        Document(id="d", text="x", unknown=1)  # type: ignore[call-arg]


def test_chunk_span_validation() -> None:
    with pytest.raises(ValidationError, match="end"):
        Chunk(id="c", doc_id="d", text="", start=5, end=2, index=0)


def test_retrieved_context_dedupes_doc_ids_in_rank_order() -> None:
    ctx = RetrievedContext(
        query=Query(id="q", text="?"),
        chunks=(
            ScoredChunk(chunk=_chunk("b::0", "b"), score=3),
            ScoredChunk(chunk=_chunk("a::0", "a"), score=2),
            ScoredChunk(chunk=_chunk("b::1", "b"), score=1),
        ),
    )
    assert ctx.doc_ids == ["b", "a"]
    assert ctx.texts == ["x", "x", "x"]
    assert ctx.chunks[0].doc_id == "b"


def test_simple_tokens_unicode() -> None:
    assert simple_tokens("Héllo, WORLD! 漢字 x2") == ["héllo", "world", "漢字", "x2"]


def test_regex_tokenizer_spans_and_long_tokens() -> None:
    tok = RegexTokenizer()
    text = "Hi, there!"
    assert [text[s:e] for s, e in tok.spans(text)] == ["Hi", ",", "there", "!"]
    assert tok.count("") == 0
    long = RegexTokenizer(max_token_chars=4)
    assert [("abcdefghij")[s:e] for s, e in long.spans("abcdefghij")] == ["abcd", "efgh", "ij"]


class _ByteEncoding:
    """Fake tiktoken encoding: one token per UTF-8 byte (exercises multi-byte joins)."""

    def encode(self, text: str, disallowed_special: tuple[str, ...] = ()) -> list[int]:
        return list(text.encode("utf-8"))

    def decode_single_token_bytes(self, tid: int) -> bytes:
        return bytes([tid])


def test_tiktoken_adapter_recovers_char_offsets() -> None:
    tok = TiktokenTokenizer("fake", _encoding_obj=_ByteEncoding())
    text = "aé漢"
    spans = tok.spans(text)
    assert [text[s:e] for s, e in spans] == ["a", "é", "漢"]
    assert tok.count(text) == len(text.encode("utf-8"))
    assert tok.spans("") == []
    assert tok.name == "tiktoken:fake"


def test_get_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(get_tokenizer("regex"), RegexTokenizer)
    with pytest.raises(ValueError, match="unknown tokenizer"):
        get_tokenizer("bogus")
    # auto falls back when tiktoken is unavailable
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    assert isinstance(get_tokenizer("auto"), RegexTokenizer)
    fake = types.ModuleType("tiktoken")
    fake.get_encoding = lambda name: _ByteEncoding()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    assert isinstance(get_tokenizer("auto"), TiktokenTokenizer)
    assert get_tokenizer("tiktoken:o200k_base").name == "tiktoken:o200k_base"
