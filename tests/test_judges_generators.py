from __future__ import annotations

import json

import pytest

from ragcheck.generators import ExtractiveGenerator, LLMGenerator
from ragcheck.judges import (
    ClaimVerdict,
    DeterministicJudge,
    JudgeParseError,
    LLMJudge,
    parse_json_response,
)
from ragcheck.providers import FakeProvider, Message
from ragcheck.types import Chunk, Query, RetrievedContext, ScoredChunk


def context(*texts: str, windows: list[str] | None = None) -> RetrievedContext:
    chunks = []
    for i, t in enumerate(texts):
        md = {"window": windows[i]} if windows else {}
        chunks.append(
            ScoredChunk(
                chunk=Chunk(
                    id=f"c{i}", doc_id=f"d{i}", text=t, start=0, end=len(t), index=0, metadata=md
                ),
                score=1.0,
            )
        )
    return RetrievedContext(query=Query(id="q", text="q"), chunks=tuple(chunks))


# ----------------------------------------------------------------------------- parser


def test_parser_accepts_bare_and_fenced_json() -> None:
    raw = '{"claim": "x", "supported": true}'
    assert parse_json_response(raw, ClaimVerdict).supported is True
    fenced = f"```json\n{raw}\n```"
    assert parse_json_response(fenced, ClaimVerdict).claim == "x"
    assert parse_json_response(f"```\n{raw}\n```", ClaimVerdict).claim == "x"


@pytest.mark.parametrize(
    "raw",
    [
        'Sure! {"claim": "x", "supported": true}',  # prose around the object
        '[{"claim": "x", "supported": true}]',  # array, not object
        '{"claim": "x", "supported": "true"}',  # strict: string is not a bool
        '{"claim": "x", "supported": true, "extra": 1}',  # unknown key
        '{"claim": "x"}',  # missing key
        "{not json}",
        "",
    ],
)
def test_parser_is_strict(raw: str) -> None:
    with pytest.raises(JudgeParseError):
        parse_json_response(raw, ClaimVerdict)


# ---------------------------------------------------------------------------- LLMJudge


async def test_llm_judge_verify_claims_prompt_and_parse() -> None:
    reply = json.dumps(
        {
            "claims": [
                {"claim": "a", "supported": True, "reason": "ctx"},
                {"claim": "b", "supported": False},
            ]
        }
    )
    p = FakeProvider([reply])
    judge = LLMJudge(p)
    verdicts = await judge.verify_claims("A and B.", ["ctx one", "ctx two"])
    assert [v.supported for v in verdicts] == [True, False]
    system, user = p.calls[0]
    assert system.role == "system"
    assert "[1] ctx one" in user.content
    assert "[2] ctx two" in user.content
    assert "A and B." in user.content


async def test_llm_judge_other_prompts() -> None:
    p = FakeProvider(
        [
            '{"questions": ["q1", "q2", "q3", "q4"], "noncommittal": false}',
            '{"useful": true, "reason": "has it"}',
            '{"statements": [{"statement": "s", "attributed": true},'
            ' {"statement": "t", "attributed": false}]}',
        ]
    )
    j = LLMJudge(p)
    qs, nc = await j.generate_questions("ans", 3)
    assert qs == ["q1", "q2", "q3"]
    assert nc is False
    assert await j.is_context_useful("q", "gt", "ctx") is True
    assert await j.attribute_statements("gt", []) == [True, False]
    assert "(no context)" in p.calls[2][1].content


async def test_llm_judge_repairs_once_then_raises() -> None:
    p = FakeProvider(["not json", '{"useful": false}'])
    assert await LLMJudge(p).is_context_useful("q", "gt", "c") is False
    repair = p.calls[1]
    assert repair[-2] == Message("assistant", "not json")
    assert "invalid" in repair[-1].content
    bad = FakeProvider(["nope"])
    with pytest.raises(JudgeParseError):
        await LLMJudge(bad, max_parse_retries=1).is_context_useful("q", "gt", "c")
    assert len(bad.calls) == 2


# ---------------------------------------------------------------------- deterministic


async def test_deterministic_judge() -> None:
    j = DeterministicJudge(threshold=0.6)
    v = await j.verify_claims(
        "Tasks retry five times. Bananas are purple.", ["tasks retry five times by default"]
    )
    assert [x.supported for x in v] == [True, False]
    qs, nc = await j.generate_questions("Port 7420 is used.", 3)
    assert qs == ["port 7420 used?"]
    assert nc is False
    assert (await j.generate_questions("I don't know.", 3))[1] is True
    assert await j.generate_questions("", 3) == ([], True)
    assert await j.is_context_useful("which port", "port 7420", "listens on port 7420")
    assert not await j.is_context_useful("which port", "port 7420", "unrelated text")
    assert await j.is_context_useful("which port", "", "port number")  # falls back to question
    assert await j.attribute_statements("Port 7420. Uses COBOL.", ["port 7420"]) == [True, False]
    with pytest.raises(ValueError, match="threshold"):
        DeterministicJudge(threshold=0)


# ------------------------------------------------------------------------- generators


async def test_extractive_generator_picks_best_sentence() -> None:
    g = ExtractiveGenerator()
    ctx = context(
        "Larkspur is a task queue. The server listens on port 7420 by default.",
        "Metrics are on port 9464.",
    )
    ans = await g.generate("Which port does the server listen on?", ctx)
    assert ans.text == "The server listens on port 7420 by default."
    two = await ExtractiveGenerator(max_sentences=2).generate("server port", ctx)
    assert two.text.count(".") == 2
    none = await g.generate("zebra?", context("nothing relevant here."))
    assert none.text == ""
    assert none.metadata["abstained"] is True
    assert (await g.generate("q", context())).text == ""
    with pytest.raises(ValueError, match="max_sentences"):
        ExtractiveGenerator(max_sentences=0)


async def test_extractive_generator_uses_window_metadata() -> None:
    ctx = context("Short.", windows=["Before. Short. The answer is 42."])
    assert (await ExtractiveGenerator().generate("what is the answer", ctx)).text == (
        "The answer is 42."
    )
    no_window = ExtractiveGenerator(use_window=False)
    assert (await no_window.generate("what is the answer", ctx)).text == ""


async def test_llm_generator_prompt_and_truncation() -> None:
    p = FakeProvider(["  7420  "])
    g = LLMGenerator(p, max_context_chars=30)
    ctx = context("a" * 20, "b" * 20)
    ans = await g.generate("port?", ctx)
    assert ans.text == "7420"
    assert ans.metadata["generator"] == "fake:fake"
    user = p.calls[0][1].content
    assert "[1] " + "a" * 20 in user
    assert "b" * 20 not in user  # second context exceeded the budget
    assert user.endswith("Question: port?\nAnswer:")
    with pytest.raises(ValueError, match="prompt_template"):
        LLMGenerator(p, prompt_template="no placeholders")
