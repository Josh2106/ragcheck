"""HTTP providers and embedders, exercised only through mocked transports (no network)."""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import httpx
import numpy as np
import pytest
import respx

from ragcheck.embedders import (
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
)
from ragcheck.providers import (
    AnthropicProvider,
    FakeProvider,
    GroqProvider,
    Message,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
)
from ragcheck.providers.base import request_with_retries_sync
from ragcheck.retrievers import CrossEncoderReranker
from ragcheck.types import Chunk, ScoredChunk

MSGS = [Message("system", "be brief"), Message("user", "hi")]


def recorder(responses: list[httpx.Response]) -> tuple[list[httpx.Request], httpx.MockTransport]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses[min(len(seen) - 1, len(responses) - 1)]

    return seen, httpx.MockTransport(handler)


# ----------------------------------------------------------------- hashing embedder


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    e = HashingEmbedder(dim=64)
    a = e.embed(["Retry backoff policy", ""])
    b = HashingEmbedder(dim=64).embed(["Retry backoff policy", ""])
    assert a.dtype == np.float32
    assert a.shape == (2, 64)
    np.testing.assert_array_equal(a, b)
    assert np.linalg.norm(a[0]) == pytest.approx(1.0, abs=1e-5)
    assert not a[1].any()  # empty text -> zero vector
    sim_close = float(e.embed(["retry backoff"])[0] @ a[0])
    sim_far = float(e.embed(["tls certificate rotation"])[0] @ a[0])
    assert sim_close > sim_far
    assert "dim=64" in e.name
    assert HashingEmbedder(char_ngrams=0, use_bigrams=False).embed(["a b"]).shape == (1, 512)


@pytest.mark.parametrize("kwargs", [{"dim": 0}, {"char_ngrams": -1}])
def test_hashing_embedder_validation(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=r"dim|char_ngrams"):
        HashingEmbedder(**kwargs)


# ----------------------------------------------------------- openai-compatible embedder


@respx.mock
def test_openai_embedder_batches_and_sorts_by_index() -> None:
    def reply(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        data = [
            {"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    route = respx.post("http://localhost:11434/v1/embeddings").mock(side_effect=reply)
    e = OpenAICompatibleEmbedder(
        "nomic-embed-text", base_url="http://localhost:11434/v1/", api_key="k", batch_size=2
    )
    out = e.embed(["a", "bb", "ccc"])
    assert route.call_count == 2
    assert out.tolist() == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert route.calls[0].request.headers["authorization"] == "Bearer k"
    assert e.embed([]).shape == (0, 0)
    assert "nomic-embed-text" in e.name
    e.close()


def test_openai_embedder_errors_and_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen, transport = recorder([httpx.Response(200, json={"data": [{"index": 0}]})])
    e = OpenAICompatibleEmbedder("m", dimensions=8, transport=transport)
    with pytest.raises(ProviderError, match="malformed"):
        e.embed(["x"])
    assert json.loads(seen[0].content)["dimensions"] == 8
    assert "authorization" not in seen[0].headers
    _, transport = recorder([httpx.Response(200, json={"data": []})])
    with pytest.raises(ProviderError, match="expected 1"):
        OpenAICompatibleEmbedder("m", transport=transport).embed(["x"])
    _, transport = recorder([httpx.Response(401, text="bad key")])
    with pytest.raises(ProviderError, match="401") as info:
        OpenAICompatibleEmbedder("m", transport=transport).embed(["x"])
    assert info.value.status_code == 401
    with pytest.raises(ValueError, match="batch_size"):
        OpenAICompatibleEmbedder("m", batch_size=0)


def test_sync_retries_then_gives_up() -> None:
    seen, transport = recorder([httpx.Response(503), httpx.Response(200, json={"ok": True})])
    client = httpx.Client(transport=transport)
    resp = request_with_retries_sync(client, "POST", "http://x/y", {}, max_retries=2, backoff=0)
    assert resp.json() == {"ok": True}
    assert len(seen) == 2

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    client = httpx.Client(transport=httpx.MockTransport(boom))
    with pytest.raises(ProviderError, match="after 2 attempts"):
        request_with_retries_sync(client, "POST", "http://x/y", {}, max_retries=1, backoff=0)


# --------------------------------------------------------- sentence-transformers (faked)


def _fake_st_module(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}
    mod = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, name: str, device: str | None = None) -> None:
            calls["model"] = (name, device)

        def encode(self, texts: list[str], **kw: Any) -> np.ndarray:
            calls["encode_kw"] = kw
            return np.array([[len(t), 0.0] for t in texts])

    class CrossEncoder:
        def __init__(self, name: str, device: str | None = None) -> None:
            calls["cross"] = name

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [float(len(p[1])) for p in pairs]

    mod.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    mod.CrossEncoder = CrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return calls


def test_sentence_transformer_adapter_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_st_module(monkeypatch)
    e = SentenceTransformerEmbedder("tiny-model", device="cpu", batch_size=4)
    assert "model" not in calls
    out = e.embed(["ab", "c"])
    assert out.dtype == np.float32
    assert out.tolist() == [[2.0, 0.0], [1.0, 0.0]]
    assert calls["model"] == ("tiny-model", "cpu")
    assert calls["encode_kw"]["normalize_embeddings"] is True
    assert "tiny-model" in e.name


def test_cross_encoder_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_st_module(monkeypatch)
    cands = [
        ScoredChunk(chunk=Chunk(id=i, doc_id=i, text=t, start=0, end=len(t), index=0), score=0)
        for i, t in [("a", "x"), ("b", "xxx"), ("c", "xx")]
    ]
    rr = CrossEncoderReranker("ce")
    assert rr.rerank("q", []) == []
    assert [c.chunk.id for c in rr.rerank("q", cands)] == ["b", "c", "a"]


# ------------------------------------------------------------------- chat providers


async def test_openai_provider_payload_and_parsing() -> None:
    seen, transport = recorder(
        [
            httpx.Response(
                200,
                json={
                    "model": "gpt-x",
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "details": {}},
                },
            )
        ]
    )
    p = OpenAIProvider("gpt-x", api_key="sk-test", transport=transport)
    out = await p.complete(MSGS, temperature=0.2, max_tokens=7, json_mode=True)
    assert out.text == "hello"
    assert out.usage == {"prompt_tokens": 3, "completion_tokens": 1}
    body = json.loads(seen[0].content)
    assert seen[0].url == "https://api.openai.com/v1/chat/completions"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["max_tokens"] == 7
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    assert p.name == "openai:gpt-x"
    await p.aclose()


async def test_groq_provider_uses_groq_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    seen, transport = recorder(
        [httpx.Response(200, json={"choices": [{"message": {"content": None}}]})]
    )
    p = GroqProvider("llama", transport=transport)
    out = await p.complete(MSGS)
    assert out.text == ""
    assert str(seen[0].url) == "https://api.groq.com/openai/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer gsk-test"
    assert p.name == "groq:llama"


async def test_anthropic_provider_system_prompt_and_blocks() -> None:
    seen, transport = recorder(
        [
            httpx.Response(
                200,
                json={
                    "model": "claude-test",
                    "content": [
                        {"type": "text", "text": "Hel"},
                        {"type": "tool_use", "id": "x"},
                        {"type": "text", "text": "lo"},
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                },
            )
        ]
    )
    p = AnthropicProvider("claude-test", api_key="ak", transport=transport)
    out = await p.complete(MSGS, json_mode=True)
    assert out.text == "Hello"
    assert out.usage == {"input_tokens": 5, "output_tokens": 2}
    req = seen[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "ak"
    assert req.headers["anthropic-version"] == AnthropicProvider.API_VERSION
    assert body["system"].startswith("be brief")
    assert "JSON" in body["system"]
    assert body["messages"] == [{"role": "user", "content": "hi"}]


async def test_ollama_provider() -> None:
    seen, transport = recorder(
        [
            httpx.Response(
                200,
                json={"message": {"content": "{}"}, "prompt_eval_count": 4, "eval_count": 2},
            )
        ]
    )
    p = OllamaProvider("llama3.1", transport=transport)
    out = await p.complete([Message("user", "hi")], json_mode=True, max_tokens=9)
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://localhost:11434/api/chat"
    assert body["format"] == "json"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_predict": 9}
    assert out.usage == {"prompt_tokens": 4, "completion_tokens": 2}


@respx.mock
async def test_retry_on_429_honours_retry_after() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "0"}),
            httpx.Response(500),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    p = OpenAIProvider("m", api_key="k", backoff=0)
    assert (await p.complete(MSGS)).text == "ok"
    assert route.call_count == 3


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(400, text="bad request"), "HTTP 400"),
        (httpx.Response(200, text="not json"), "non-JSON"),
        (httpx.Response(200, json=[1, 2]), "unexpected response type"),
        (httpx.Response(200, json={"choices": []}), "malformed"),
    ],
)
async def test_openai_provider_errors(response: httpx.Response, match: str) -> None:
    _, transport = recorder([response])
    p = OpenAIProvider("m", api_key="k", transport=transport, backoff=0)
    with pytest.raises(ProviderError, match=match):
        await p.complete(MSGS)


async def test_malformed_anthropic_and_ollama() -> None:
    _, t1 = recorder([httpx.Response(200, json={"content": "oops"})])
    with pytest.raises(ProviderError, match="malformed"):
        await AnthropicProvider("m", api_key="k", transport=t1).complete(MSGS)
    _, t2 = recorder([httpx.Response(200, json={"nope": 1})])
    with pytest.raises(ProviderError, match="malformed"):
        await OllamaProvider("m", transport=t2).complete(MSGS)


async def test_async_retries_exhausted_on_transport_errors() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    p = OpenAIProvider(
        "m", api_key="k", transport=httpx.MockTransport(boom), max_retries=2, backoff=0
    )
    with pytest.raises(ProviderError, match="after 3 attempts"):
        await p.complete(MSGS)


async def test_retry_after_unparseable_falls_back_to_backoff() -> None:
    _, transport = recorder(
        [
            httpx.Response(503, headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}),
            httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    p = OpenAIProvider("m", api_key="k", transport=transport, backoff=0)
    assert (await p.complete(MSGS)).text == "ok"


def test_provider_requires_model() -> None:
    with pytest.raises(ValueError, match="model"):
        OpenAIProvider("")


async def test_fake_provider() -> None:
    p = FakeProvider(["a", "b"])
    assert [(await p.complete(MSGS)).text for _ in range(3)] == ["a", "b", "a"]
    assert len(p.calls) == 3
    assert p.name == "fake:fake"
    f = FakeProvider(fn=lambda msgs: msgs[-1].content.upper())
    assert (await f.complete(MSGS)).text == "HI"
    with pytest.raises(ValueError, match="responses or fn"):
        FakeProvider()
