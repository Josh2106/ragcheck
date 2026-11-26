"""LLM provider interface plus shared HTTP plumbing (retries, errors) and a fake."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import httpx

__all__ = [
    "Completion",
    "FakeProvider",
    "Message",
    "Provider",
    "ProviderError",
    "request_with_retries",
    "request_with_retries_sync",
]

Role = Literal["system", "user", "assistant"]
RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Raised for non-retryable HTTP errors or malformed provider responses."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@runtime_checkable
class Provider(Protocol):
    """A chat-completion LLM. ``name`` identifies provider+model for caching."""

    @property
    def name(self) -> str: ...

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> Completion: ...


def _retry_delay(resp: httpx.Response | None, attempt: int, backoff: float) -> float:
    if resp is not None:
        ra = resp.headers.get("retry-after")
        if ra is not None:
            try:
                return min(float(ra), 60.0)
            except ValueError:
                pass
    return float(backoff * (2**attempt))


def _raise_for(resp: httpx.Response) -> None:
    body = resp.text[:500]
    raise ProviderError(
        f"HTTP {resp.status_code} from {resp.request.url}: {body}", resp.status_code
    )


async def request_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    payload: dict[str, Any],
    max_retries: int = 3,
    backoff: float = 0.5,
) -> httpx.Response:
    """Send a JSON request, retrying on 429/5xx/transport errors with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        resp: httpx.Response | None = None
        try:
            resp = await client.request(method, url, json=payload)
        except httpx.TransportError as exc:
            last_exc = exc
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code not in RETRY_STATUS:
                _raise_for(resp)
            last_exc = ProviderError(f"HTTP {resp.status_code}", resp.status_code)
        if attempt < max_retries:
            await asyncio.sleep(_retry_delay(resp, attempt, backoff))
    raise ProviderError(f"request to {url} failed after {max_retries + 1} attempts: {last_exc}")


def request_with_retries_sync(
    client: httpx.Client,
    method: str,
    url: str,
    payload: dict[str, Any],
    max_retries: int = 3,
    backoff: float = 0.5,
) -> httpx.Response:
    """Synchronous twin of :func:`request_with_retries`."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        resp: httpx.Response | None = None
        try:
            resp = client.request(method, url, json=payload)
        except httpx.TransportError as exc:
            last_exc = exc
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code not in RETRY_STATUS:
                _raise_for(resp)
            last_exc = ProviderError(f"HTTP {resp.status_code}", resp.status_code)
        if attempt < max_retries:
            time.sleep(_retry_delay(resp, attempt, backoff))
    raise ProviderError(f"request to {url} failed after {max_retries + 1} attempts: {last_exc}")


class FakeProvider:
    """Offline provider for tests: returns scripted responses or calls a function.

    ``responses`` are returned in order (cycling when exhausted); alternatively
    ``fn(messages) -> str`` computes the reply. Every call is recorded in ``calls``.
    """

    def __init__(
        self,
        responses: Sequence[str] = (),
        fn: Callable[[Sequence[Message]], str] | None = None,
        model: str = "fake",
    ) -> None:
        if not responses and fn is None:
            raise ValueError("FakeProvider needs responses or fn")
        self._responses = list(responses)
        self._fn = fn
        self.model = model
        self.calls: list[list[Message]] = []

    @property
    def name(self) -> str:
        return f"fake:{self.model}"

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,  # noqa: ARG002 - part of the Provider protocol
        max_tokens: int = 512,  # noqa: ARG002
        json_mode: bool = False,  # noqa: ARG002
    ) -> Completion:
        self.calls.append(list(messages))
        if self._fn is not None:
            text = self._fn(messages)
        else:
            text = self._responses[(len(self.calls) - 1) % len(self._responses)]
        return Completion(text=text, model=self.model)
