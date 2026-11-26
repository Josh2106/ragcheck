"""Concrete HTTP providers: OpenAI(-compatible), Groq, Anthropic and Ollama.

All take an optional ``transport`` so tests can inject ``httpx.MockTransport`` and
never touch the network. API keys come from arguments or environment variables; they
are never logged or included in cache keys.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import httpx

from ragcheck.providers.base import Completion, Message, ProviderError, request_with_retries

__all__ = ["AnthropicProvider", "GroqProvider", "OllamaProvider", "OpenAIProvider"]


class _HTTPProvider:
    provider = "http"

    def __init__(
        self,
        model: str,
        base_url: str,
        headers: dict[str, str],
        timeout: float,
        max_retries: int,
        backoff: float,
        transport: httpx.AsyncBaseTransport | None,
    ) -> None:
        if not model:
            raise ValueError("model must be set")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff = backoff
        self._client = httpx.AsyncClient(
            headers={"Content-Type": "application/json", **headers},
            timeout=timeout,
            transport=transport,
        )

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await request_with_retries(
            self._client, "POST", self.base_url + path, payload, self.max_retries, self.backoff
        )
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(f"non-JSON response from {self.name}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"unexpected response type from {self.name}")
        return data

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAIProvider(_HTTPProvider):
    """OpenAI Chat Completions API, or any compatible server via ``base_url``."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(api_key_env)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        super().__init__(model, base_url, headers, timeout, max_retries, backoff, transport)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = await self._post("/chat/completions", payload)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"malformed chat completion from {self.name}") from exc
        usage = {k: int(v) for k, v in (data.get("usage") or {}).items() if isinstance(v, int)}
        return Completion(text=str(text), model=str(data.get("model", self.model)), usage=usage)


class GroqProvider(OpenAIProvider):
    """Groq's OpenAI-compatible endpoint."""

    provider = "groq"

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        api_key: str | None = None,
        api_key_env: str = "GROQ_API_KEY",
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            model, base_url, api_key, api_key_env, timeout, max_retries, backoff, transport
        )


class AnthropicProvider(_HTTPProvider):
    """Anthropic Messages API (``POST /v1/messages``)."""

    provider = "anthropic"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.anthropic.com/v1",
        api_key: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(api_key_env)
        headers = {"anthropic-version": self.API_VERSION}
        if key:
            headers["x-api-key"] = key
        super().__init__(model, base_url, headers, timeout, max_retries, backoff, transport)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> Completion:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        if json_mode:
            system = (system + "\n\nRespond with a single JSON object and nothing else.").strip()
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages if m.role != "system"
            ],
        }
        if system:
            payload["system"] = system
        data = await self._post("/messages", payload)
        try:
            text = "".join(
                block.get("text", "") for block in data["content"] if block.get("type") == "text"
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(f"malformed messages response from {self.name}") from exc
        usage = {k: int(v) for k, v in (data.get("usage") or {}).items() if isinstance(v, int)}
        return Completion(text=text, model=str(data.get("model", self.model)), usage=usage)


class OllamaProvider(_HTTPProvider):
    """Ollama's native chat API (``POST /api/chat``)."""

    provider = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        max_retries: int = 2,
        backoff: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(model, base_url, {}, timeout, max_retries, backoff, transport)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        data = await self._post("/api/chat", payload)
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"malformed chat response from {self.name}") from exc
        usage = {
            "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
            "completion_tokens": int(data.get("eval_count", 0) or 0),
        }
        return Completion(text=str(text), model=self.model, usage=usage)
