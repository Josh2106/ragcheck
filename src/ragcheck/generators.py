"""Answer generators: an offline extractive baseline and an LLM-backed generator."""

from __future__ import annotations

import math

from ragcheck.chunkers.sentence import split_sentences
from ragcheck.providers.base import Message, Provider
from ragcheck.text import content_tokens
from ragcheck.types import Answer, RetrievedContext

__all__ = ["DEFAULT_RAG_PROMPT", "ExtractiveGenerator", "LLMGenerator"]

DEFAULT_SYSTEM_PROMPT = (
    "You answer questions using only the provided context. If the context does not contain "
    'the answer, say "I don\'t know". Be concise.'
)
DEFAULT_RAG_PROMPT = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def _context_texts(context: RetrievedContext, use_window: bool) -> list[str]:
    out = []
    for sc in context.chunks:
        window = sc.chunk.metadata.get("window") if use_window else None
        out.append(window if isinstance(window, str) and window else sc.chunk.text)
    return out


class ExtractiveGenerator:
    """Deterministic baseline: return the context sentence(s) that best match the question.

    Sentences are scored by content-token overlap with the question, normalised by
    ``sqrt(len(sentence))`` to avoid favouring long sentences; ties go to the higher
    ranked chunk. No LLM needed, so it is the default for offline runs and CI. It is a
    *baseline*: it measures how often the answer is literally present in what was
    retrieved.
    """

    def __init__(self, max_sentences: int = 1, use_window: bool = True) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        self.max_sentences = max_sentences
        self.use_window = use_window

    async def generate(self, question: str, context: RetrievedContext) -> Answer:
        q = set(content_tokens(question))
        candidates: list[tuple[float, int, int, str]] = []
        order = 0
        for rank, text in enumerate(_context_texts(context, self.use_window)):
            for s, e in split_sentences(text):
                sent = text[s:e]
                toks = content_tokens(sent)
                if not toks:
                    continue
                overlap = len(q.intersection(toks))
                candidates.append((overlap / math.sqrt(len(toks)), rank, order, sent))
                order += 1
        if not candidates or max(c[0] for c in candidates) == 0:
            return Answer(text="", metadata={"generator": "extractive", "abstained": True})
        best = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))[: self.max_sentences]
        best.sort(key=lambda c: (c[1], c[2]))  # restore reading order
        return Answer(
            text=" ".join(c[3] for c in best),
            metadata={"generator": "extractive", "score": best[0][0]},
        )


class LLMGenerator:
    """Prompt an LLM with the numbered retrieved contexts."""

    def __init__(
        self,
        provider: Provider,
        prompt_template: str = DEFAULT_RAG_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_context_chars: int = 12_000,
        use_window: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        if "{context}" not in prompt_template or "{question}" not in prompt_template:
            raise ValueError("prompt_template must contain {context} and {question}")
        self.provider = provider
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.max_context_chars = max_context_chars
        self.use_window = use_window
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_prompt(self, question: str, context: RetrievedContext) -> str:
        parts: list[str] = []
        used = 0
        for i, text in enumerate(_context_texts(context, self.use_window), start=1):
            block = f"[{i}] {text}"
            if used + len(block) > self.max_context_chars and parts:
                break
            parts.append(block[: self.max_context_chars])
            used += len(block)
        return self.prompt_template.format(context="\n\n".join(parts), question=question)

    async def generate(self, question: str, context: RetrievedContext) -> Answer:
        completion = await self.provider.complete(
            [
                Message("system", self.system_prompt),
                Message("user", self.build_prompt(question, context)),
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return Answer(
            text=completion.text.strip(),
            metadata={"generator": self.provider.name, "usage": completion.usage},
        )
