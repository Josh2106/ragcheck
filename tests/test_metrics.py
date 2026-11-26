from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from ragcheck.embedders import HashingEmbedder
from ragcheck.judges import ClaimVerdict, DeterministicJudge
from ragcheck.metrics import (
    MRR,
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
    ExactMatch,
    Faithfulness,
    HitRateAtK,
    NDCGAtK,
    PrecisionAtK,
    RecallAtK,
    RougeL,
    Sample,
    SemanticSimilarity,
    TokenF1,
    build_metric,
    build_metrics,
    exact_match,
    hit_rate_at_k,
    lcs_length,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge_l,
    token_f1,
)
from ragcheck.types import Answer, Chunk, EvalCase, Query, RetrievedContext, ScoredChunk


def ctx(*docs: str, texts: Sequence[str] | None = None) -> RetrievedContext:
    texts = texts or [f"text of {d}" for d in docs]
    return RetrievedContext(
        query=Query(id="q", text="q"),
        chunks=tuple(
            ScoredChunk(
                chunk=Chunk(id=f"{d}::{i}", doc_id=d, text=t, start=0, end=len(t), index=i),
                score=1.0 / (i + 1),
            )
            for i, (d, t) in enumerate(zip(docs, texts, strict=True))
        ),
    )


def sample(
    docs: Sequence[str] = (),
    relevant: Sequence[str] = (),
    answer: str | None = "",
    truth: str = "",
    question: str = "q",
    texts: Sequence[str] | None = None,
) -> Sample:
    return Sample(
        case=EvalCase(
            id="c", question=question, ground_truth=truth, relevant_doc_ids=tuple(relevant)
        ),
        context=ctx(*docs, texts=texts),
        answer=None if answer is None else Answer(text=answer),
    )


# --------------------------------------------------------------------------- retrieval

RANKED = ["x", "a", "y", "b", "z"]
REL = {"a", "b", "c"}


def test_retrieval_functions_known_values() -> None:
    assert recall_at_k(RANKED, REL, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, REL, 5) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED, REL, 5) == pytest.approx(2 / 5)
    assert precision_at_k(["a"], REL, 5) == pytest.approx(1 / 5)  # denominator is k
    assert hit_rate_at_k(RANKED, REL, 1) == 0.0
    assert hit_rate_at_k(RANKED, REL, 2) == 1.0
    assert mrr(RANKED, REL) == pytest.approx(0.5)
    assert mrr(RANKED, REL, k=1) == 0.0
    assert mrr(["q"], REL) == 0.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b", "c"], REL, 3) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive"):
        recall_at_k(RANKED, REL, 0)


async def test_retrieval_metric_classes() -> None:
    s = sample(docs=["x", "a", "a", "b"], relevant=["a", "b"])  # duplicate chunk of "a"
    assert await RecallAtK(2).score(s) == pytest.approx(0.5)  # docs deduped: x, a
    assert await RecallAtK(3).score(s) == pytest.approx(1.0)
    assert await PrecisionAtK(3).score(s) == pytest.approx(2 / 3)
    assert await HitRateAtK(1).score(s) == 0.0
    assert await MRR().score(s) == pytest.approx(0.5)
    assert await MRR(1).score(s) == 0.0
    assert await NDCGAtK(3).score(s) is not None
    unlabelled = sample(docs=["x"])
    for m in (RecallAtK(5), PrecisionAtK(5), HitRateAtK(5), NDCGAtK(5), MRR()):
        assert await m.score(unlabelled) is None
    assert [m.name for m in (RecallAtK(5), MRR(), MRR(10), NDCGAtK(3))] == [
        "recall@5",
        "mrr",
        "mrr@10",
        "ndcg@3",
    ]
    assert "RecallAtK" in repr(RecallAtK(5))
    with pytest.raises(ValueError, match="positive"):
        MRR(0)


# -------------------------------------------------------------------------- generation


def test_exact_match_normalisation() -> None:
    assert exact_match("The  Answer!", "answer") == 1.0
    assert exact_match("Café", "cafe") == 0.0
    assert exact_match("", "") == 1.0


def test_token_f1_known_values() -> None:
    # pred: [port, 7420], ref: [server, listens, on, port, 7420] -> p=1, r=0.4
    assert token_f1("port 7420", "the server listens on port 7420") == pytest.approx(
        2 * 1 * 0.4 / 1.4
    )
    assert token_f1("a", "b c") == 0.0
    assert token_f1("", "x") == 0.0
    assert token_f1("", "") == 1.0
    assert token_f1("x x y", "x y y") == pytest.approx(2 / 3)  # multiset overlap


def test_rouge_l_known_values() -> None:
    assert lcs_length("abcbdab", "bdcaba") == 4
    assert lcs_length([], ["a"]) == 0
    # articles are dropped by normalisation: [police killed gunman] vs [police kill gunman]
    # LCS = (police, gunman) = 2, so P = R = F = 2/3
    assert rouge_l("police killed the gunman", "police kill the gunman") == pytest.approx(2 / 3)
    p, r = 2 / 3, 2 / 4
    assert rouge_l("police killed gunman", "police kill gunman man") == pytest.approx(
        2 * p * r / (p + r)
    )
    assert rouge_l("", "") == 1.0
    assert rouge_l("x", "y") == 0.0
    assert rouge_l("x y", "x y z w", beta=2.0) == pytest.approx(
        (1 + 4) * 1.0 * 0.5 / (0.5 + 4 * 1.0)
    )


async def test_reference_metric_classes() -> None:
    s = sample(answer="Port 7420.", truth="port 7420")
    assert await ExactMatch().score(s) == 1.0
    assert await TokenF1().score(s) == 1.0
    assert await RougeL().score(s) == 1.0
    sim = SemanticSimilarity(HashingEmbedder(dim=128))
    assert await sim.score(s) == pytest.approx(1.0, abs=1e-5)
    assert await sim.score(sample(answer="  ", truth="x")) == 0.0
    assert await ExactMatch().score(sample(answer="x", truth="")) is None
    assert await TokenF1().score(sample(answer=None, truth="x")) is None


async def test_semantic_similarity_zero_vectors() -> None:
    sim = SemanticSimilarity(HashingEmbedder(dim=16))
    assert await sim.score(sample(answer="!!!", truth="???")) == 0.0


# ------------------------------------------------------------------------------ judge


class ScriptedJudge:
    """Judge with fixed verdicts, to test the metric arithmetic in isolation."""

    def __init__(
        self,
        claims: list[bool] | None = None,
        questions: tuple[list[str], bool] = (["q"], False),
        useful: list[bool] | None = None,
        attributed: list[bool] | None = None,
    ) -> None:
        self.claims = claims or []
        self.questions = questions
        self.useful = list(useful or [])
        self.attributed = attributed or []

    async def verify_claims(self, answer: str, contexts: Sequence[str]) -> list[ClaimVerdict]:
        return [ClaimVerdict(claim=f"c{i}", supported=s) for i, s in enumerate(self.claims)]

    async def generate_questions(self, answer: str, n: int) -> tuple[list[str], bool]:
        return self.questions

    async def is_context_useful(self, question: str, ground_truth: str, context: str) -> bool:
        return self.useful[int(context)]

    async def attribute_statements(self, ground_truth: str, contexts: Sequence[str]) -> list[bool]:
        return self.attributed


async def test_faithfulness_arithmetic() -> None:
    s = sample(answer="something")
    assert await Faithfulness(ScriptedJudge(claims=[True, True, False, True])).score(s) == 0.75
    assert await Faithfulness(ScriptedJudge(claims=[])).score(s) is None
    assert await Faithfulness(ScriptedJudge(claims=[True])).score(sample(answer="")) is None


async def test_context_precision_is_average_precision() -> None:
    texts = ["0", "1", "2", "3"]
    s = sample(docs=["a", "b", "c", "d"], texts=texts, truth="t")
    # useful at ranks 1 and 3: (1/1 + 2/3) / 2
    j = ScriptedJudge(useful=[True, False, True, False])
    assert await ContextPrecision(j).score(s) == pytest.approx((1 + 2 / 3) / 2)
    assert await ContextPrecision(j, k=2).score(s) == pytest.approx(1.0)
    assert await ContextPrecision(ScriptedJudge(useful=[False] * 4)).score(s) == 0.0
    assert await ContextPrecision(j).score(sample()) == 0.0
    assert ContextPrecision(j, k=3).name == "context_precision@3"


async def test_context_recall_arithmetic() -> None:
    s = sample(truth="A. B. C.")
    assert await ContextRecall(ScriptedJudge(attributed=[True, False, True])).score(
        s
    ) == pytest.approx(2 / 3)
    assert await ContextRecall(ScriptedJudge(attributed=[])).score(s) is None
    assert await ContextRecall(ScriptedJudge()).score(sample(truth=" ")) is None


async def test_answer_relevance() -> None:
    emb = HashingEmbedder(dim=128)
    s = sample(question="what port does the server use", answer="7420")
    perfect = AnswerRelevance(
        ScriptedJudge(questions=(["what port does the server use"], False)), emb
    )
    assert await perfect.score(s) == pytest.approx(1.0, abs=1e-5)
    evasive = AnswerRelevance(ScriptedJudge(questions=(["x"], True)), emb)
    assert await evasive.score(s) == 0.0
    assert await perfect.score(sample(answer="")) == 0.0
    assert await perfect.score(sample(answer=None)) is None
    with pytest.raises(ValueError, match="n_questions"):
        AnswerRelevance(ScriptedJudge(), emb, n_questions=0)


async def test_judge_metrics_with_deterministic_judge() -> None:
    j = DeterministicJudge()
    texts = ["The server listens on port 7420 by default."]
    s = sample(
        docs=["install"],
        texts=texts,
        question="What port does the server listen on?",
        answer="The server listens on port 7420. It is written in COBOL.",
        truth="The server listens on port 7420.",
    )
    assert await Faithfulness(j).score(s) == pytest.approx(0.5)
    assert await ContextRecall(j).score(s) == 1.0
    assert await ContextPrecision(j).score(s) == 1.0
    rel = await AnswerRelevance(j, HashingEmbedder(dim=256)).score(s)
    assert rel is not None
    assert 0 < rel <= 1


# ---------------------------------------------------------------------------- factory


def test_build_metric_names() -> None:
    emb, judge = HashingEmbedder(dim=8), DeterministicJudge()
    names = [
        "recall@5",
        "precision@3",
        "ndcg@10",
        "hit_rate@1",
        "mrr",
        "mrr@3",
        "exact_match",
        "token_f1",
        "rouge_l",
        "semantic_similarity",
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_precision@5",
        "context_recall",
    ]
    built = build_metrics(names, embedder=emb, judge=judge)
    assert [m.name for m in built] == names
    assert {m.kind for m in built} == {"retrieval", "generation", "judge"}
    assert build_metric(" Recall@5 ").name == "recall@5"


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("recall", "cutoff"),
        ("bogus", "unknown"),
        ("token_f1@3", "unknown"),
        ("recall@x", "invalid"),
        ("semantic_similarity", "embedder"),
        ("faithfulness", "judge"),
    ],
)
def test_build_metric_errors(name: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_metric(name)


def test_build_metrics_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_metrics(["mrr", "MRR"])
