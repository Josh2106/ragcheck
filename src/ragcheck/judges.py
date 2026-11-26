"""LLM-as-judge with structured JSON prompts, a strict parser, and an offline twin.

Prompts follow the decomposition used by RAGAS (Es et al., 2023, "RAGAS: Automated
Evaluation of Retrieval Augmented Generation", arXiv:2309.15217): the judge produces
*verdicts* (claims supported? context useful? statement attributable?) and the metric
code turns them into numbers. This keeps the LLM's job small and the arithmetic
auditable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragcheck.chunkers.sentence import split_sentences
from ragcheck.providers.base import Message, Provider
from ragcheck.text import content_tokens, token_recall

__all__ = [
    "ClaimVerdict",
    "DeterministicJudge",
    "JudgeParseError",
    "LLMJudge",
    "parse_json_response",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ClaimVerdict(_Strict):
    claim: str
    supported: bool
    reason: str = ""


class _ClaimsResponse(_Strict):
    claims: list[ClaimVerdict]


class _QuestionsResponse(_Strict):
    questions: list[str]
    noncommittal: bool


class _UsefulResponse(_Strict):
    useful: bool
    reason: str = ""


class _Attribution(_Strict):
    statement: str
    attributed: bool


class _AttributionResponse(_Strict):
    statements: list[_Attribution] = Field(default_factory=list)


class JudgeParseError(ValueError):
    """The judge's reply was not a JSON object matching the expected schema."""


T = TypeVar("T", bound=BaseModel)
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def parse_json_response(text: str, schema: type[T]) -> T:
    """Parse ``text`` as exactly one JSON object validated against ``schema``.

    Accepted: a bare JSON object, optionally wrapped in a single Markdown code fence.
    Rejected: anything else (prose around the object, arrays, multiple objects, unknown
    keys, wrong types - validation runs in Pydantic strict mode, so ``"true"`` is not a
    bool).
    """
    m = _FENCE.match(text)
    body = (m.group(1) if m else text).strip()
    if not body.startswith("{") or not body.endswith("}"):
        raise JudgeParseError(f"expected a single JSON object, got: {text[:200]!r}")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"invalid JSON: {exc}") from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise JudgeParseError(f"schema mismatch for {schema.__name__}: {exc}") from exc


_SYSTEM = (
    "You are a meticulous, impartial evaluator of retrieval-augmented generation systems. "
    "Judge only from the text you are given; never use outside knowledge. "
    "Reply with exactly one JSON object that matches the requested schema - no prose, "
    "no markdown."
)


def _numbered(contexts: Sequence[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)) or "(no context)"


PROMPTS = {
    "verify_claims": (
        "Break the ANSWER into short, self-contained factual claims. For each claim decide "
        "whether it can be directly inferred from the CONTEXT.\n"
        'Schema: {{"claims": [{{"claim": str, "supported": bool, "reason": str}}]}}\n'
        "If the answer makes no factual claims, return an empty list.\n\n"
        "CONTEXT:\n{contexts}\n\nANSWER:\n{answer}"
    ),
    "generate_questions": (
        "Write {n} distinct questions that the ANSWER below directly answers. Also mark "
        "noncommittal=true if the answer is evasive or vague (e.g. 'I don't know').\n"
        'Schema: {{"questions": [str], "noncommittal": bool}}\n\nANSWER:\n{answer}'
    ),
    "is_context_useful": (
        "Given the QUESTION and its REFERENCE ANSWER, was the CONTEXT useful in arriving at "
        "the reference answer?\n"
        'Schema: {{"useful": bool, "reason": str}}\n\n'
        "QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{ground_truth}\n\nCONTEXT:\n{context}"
    ),
    "attribute_statements": (
        "Split the REFERENCE ANSWER into sentences. For each sentence decide whether it can be "
        "attributed to the CONTEXT.\n"
        'Schema: {{"statements": [{{"statement": str, "attributed": bool}}]}}\n\n'
        "CONTEXT:\n{contexts}\n\nREFERENCE ANSWER:\n{ground_truth}"
    ),
}


class LLMJudge:
    """Judge backed by any :class:`~ragcheck.providers.Provider`.

    On a parse failure the judge re-asks once, showing the model its invalid output and
    the error; a second failure raises :class:`JudgeParseError`, which the runner records
    as an error for that case rather than silently scoring it.
    """

    def __init__(
        self, provider: Provider, max_parse_retries: int = 1, temperature: float = 0.0
    ) -> None:
        self.provider = provider
        self.max_parse_retries = max_parse_retries
        self.temperature = temperature

    async def _ask(self, prompt: str, schema: type[T]) -> T:
        messages = [Message("system", _SYSTEM), Message("user", prompt)]
        last: JudgeParseError | None = None
        for _ in range(self.max_parse_retries + 1):
            completion = await self.provider.complete(
                messages, temperature=self.temperature, max_tokens=1024, json_mode=True
            )
            try:
                return parse_json_response(completion.text, schema)
            except JudgeParseError as exc:
                last = exc
                messages = [
                    *messages,
                    Message("assistant", completion.text),
                    Message(
                        "user",
                        f"That reply was invalid ({exc}). Reply again with only the JSON object.",
                    ),
                ]
        assert last is not None
        raise last

    async def verify_claims(self, answer: str, contexts: Sequence[str]) -> list[ClaimVerdict]:
        prompt = PROMPTS["verify_claims"].format(contexts=_numbered(contexts), answer=answer)
        return (await self._ask(prompt, _ClaimsResponse)).claims

    async def generate_questions(self, answer: str, n: int) -> tuple[list[str], bool]:
        prompt = PROMPTS["generate_questions"].format(n=n, answer=answer)
        resp = await self._ask(prompt, _QuestionsResponse)
        return resp.questions[:n], resp.noncommittal

    async def is_context_useful(self, question: str, ground_truth: str, context: str) -> bool:
        prompt = PROMPTS["is_context_useful"].format(
            question=question, ground_truth=ground_truth, context=context
        )
        return (await self._ask(prompt, _UsefulResponse)).useful

    async def attribute_statements(self, ground_truth: str, contexts: Sequence[str]) -> list[bool]:
        prompt = PROMPTS["attribute_statements"].format(
            contexts=_numbered(contexts), ground_truth=ground_truth
        )
        return [s.attributed for s in (await self._ask(prompt, _AttributionResponse)).statements]


_NONCOMMITTAL = re.compile(
    r"\b(i don'?t know|not sure|cannot answer|can'?t answer|no information|unable to)\b",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in split_sentences(text)]


class DeterministicJudge:
    """Offline judge using lexical overlap; reproducible, free, and good enough for CI.

    A statement counts as supported when at least ``threshold`` of its content tokens
    (stopwords removed) appear in the context. It is a smoke-test stand-in for an LLM
    judge, not a substitute for one.
    """

    def __init__(self, threshold: float = 0.6) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    @staticmethod
    def _vocab(contexts: Sequence[str]) -> set[str]:
        vocab: set[str] = set()
        for c in contexts:
            vocab.update(content_tokens(c))
        return vocab

    async def verify_claims(self, answer: str, contexts: Sequence[str]) -> list[ClaimVerdict]:
        vocab = self._vocab(contexts)
        out = []
        for claim in _sentences(answer):
            r = token_recall(claim, vocab)
            out.append(
                ClaimVerdict(
                    claim=claim, supported=r >= self.threshold, reason=f"token recall {r:.2f}"
                )
            )
        return out

    async def generate_questions(self, answer: str, n: int) -> tuple[list[str], bool]:  # noqa: ARG002
        noncommittal = not answer.strip() or bool(_NONCOMMITTAL.search(answer))
        words = content_tokens(answer)
        if not words:
            return [], noncommittal
        # A single deterministic pseudo-question: the answer's content words. Its embedding
        # similarity to the real question is a lexical proxy for relevance.
        return [" ".join(words) + "?"], noncommittal

    async def is_context_useful(self, question: str, ground_truth: str, context: str) -> bool:
        target = ground_truth or question
        return token_recall(target, self._vocab([context])) >= self.threshold

    async def attribute_statements(self, ground_truth: str, contexts: Sequence[str]) -> list[bool]:
        vocab = self._vocab(contexts)
        return [token_recall(s, vocab) >= self.threshold for s in _sentences(ground_truth)]
