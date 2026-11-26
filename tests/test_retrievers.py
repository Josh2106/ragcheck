from __future__ import annotations

import math

import numpy as np
import pytest

from ragcheck.embedders import HashingEmbedder
from ragcheck.retrievers import (
    BM25Retriever,
    CallableReranker,
    DenseRetriever,
    HybridRetriever,
    InMemoryVectorIndex,
    RerankingRetriever,
    reciprocal_rank_fusion,
)
from ragcheck.types import Chunk

DOCS = ["the cat sat", "the dog sat on the mat", "cats and dogs"]


def chunks(texts: list[str] = DOCS) -> list[Chunk]:
    return [
        Chunk(id=f"c{i}", doc_id=f"d{i}", text=t, start=0, end=len(t), index=0)
        for i, t in enumerate(texts)
    ]


class TestBM25:
    def test_scores_match_hand_computed_values(self) -> None:
        r = BM25Retriever(k1=1.5, b=0.75)
        r.index(chunks())
        # N=3, avgdl=4. idf(cat)=ln(1+2.5/1.5), idf(sat)=ln(1+1.5/2.5)
        idf_cat, idf_sat = math.log(1 + 2.5 / 1.5), math.log(1 + 1.5 / 2.5)
        assert r.idf("cat") == pytest.approx(idf_cat)
        assert r.idf("sat") == pytest.approx(idf_sat)
        norm0 = 1.5 * (1 - 0.75 + 0.75 * 3 / 4)
        norm1 = 1.5 * (1 - 0.75 + 0.75 * 6 / 4)
        d0 = (idf_cat + idf_sat) * 2.5 / (1 + norm0)
        d1 = idf_sat * 2.5 / (1 + norm1)
        scores = r.get_scores("cat sat")
        assert scores.tolist() == pytest.approx([d0, d1, 0.0])
        assert scores.tolist() == pytest.approx([1.634741, 0.383676, 0.0], abs=1e-6)

    def test_repeated_query_terms_and_tf_saturation(self) -> None:
        r = BM25Retriever()
        r.index(chunks())
        # "the" occurs twice in d1, but tf saturates and d1 is longer
        assert r.get_scores("the the").tolist() == pytest.approx(
            [1.0591631, 1.1569320, 0.0], abs=1e-6
        )

    def test_retrieve_orders_and_filters_zero_scores(self) -> None:
        r = BM25Retriever()
        r.index(chunks())
        hits = r.retrieve("cat sat", k=10)
        assert [h.chunk.id for h in hits] == ["c0", "c1"]
        assert hits[0].score > hits[1].score
        assert r.retrieve("cat", k=0) == []
        assert r.retrieve("unknownword", k=3) == []

    def test_empty_index_and_stopwords(self) -> None:
        r = BM25Retriever(stopwords={"the"})
        assert r.retrieve("anything", 5) == []
        r.index(chunks())
        assert r.idf("the") == 0.0
        r.index([Chunk(id="e", doc_id="e", text="", start=0, end=0, index=0)])
        assert r.get_scores("x").tolist() == [0.0]

    @pytest.mark.parametrize(("k1", "b"), [(-1, 0.5), (1.2, 1.5)])
    def test_validation(self, k1: float, b: float) -> None:
        with pytest.raises(ValueError, match=r"k1|b"):
            BM25Retriever(k1=k1, b=b)

    def test_b_zero_disables_length_normalisation(self) -> None:
        r = BM25Retriever(b=0.0)
        r.index(chunks(["sat", "sat " + "pad " * 20]))
        s = r.get_scores("sat")
        assert s[0] == pytest.approx(s[1])


class TestVectorIndex:
    def test_cosine_search_and_tie_break(self) -> None:
        idx = InMemoryVectorIndex()
        idx.add(["a", "b", "c"], np.array([[1, 0], [0, 1], [2, 0]], dtype=np.float32))
        assert len(idx) == 3
        res = idx.search(np.array([1, 0.1], dtype=np.float32), 2)
        assert [r[0] for r in res] == ["a", "c"]  # equal cosine: insertion order wins
        assert res[0][1] == pytest.approx(res[1][1])
        assert idx.search(np.array([1, 0], dtype=np.float32), 0) == []
        assert len(idx.search(np.array([1, 0], dtype=np.float32), 99)) == 3

    def test_errors_and_zero_vectors(self) -> None:
        idx = InMemoryVectorIndex()
        assert idx.search(np.ones(2, dtype=np.float32), 3) == []
        idx.add([], np.zeros((0, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="matrix"):
            idx.add(["a"], np.zeros((2, 2), dtype=np.float32))
        idx.add(["z"], np.zeros((1, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="dimension mismatch"):
            idx.add(["y"], np.zeros((1, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="query dimension"):
            idx.search(np.ones(3, dtype=np.float32), 1)
        assert idx.search(np.ones(2, dtype=np.float32), 1) == [("z", 0.0)]


def test_dense_retriever_finds_lexically_similar_chunk() -> None:
    texts = ["retry policy and exponential backoff", "tls certificates", "prometheus metrics"]
    r = DenseRetriever(HashingEmbedder(dim=512), batch_size=2)
    assert r.retrieve("x", 3) == []
    r.index(chunks(texts))
    hits = r.retrieve("what is the retry backoff policy", 2)
    assert hits[0].chunk.id == "c0"
    assert len(hits) == 2
    assert r.retrieve("x", 0) == []


def test_hybrid_fuses_with_rrf() -> None:
    bm25 = BM25Retriever()
    dense = DenseRetriever(HashingEmbedder(dim=256))
    h = HybridRetriever([bm25, dense], rrf_k=60, fetch_k=3)
    h.index(chunks())
    hits = h.retrieve("cat sat", 2)
    expected = reciprocal_rank_fusion(
        [
            [c.chunk.id for c in bm25.retrieve("cat sat", 3)],
            [c.chunk.id for c in dense.retrieve("cat sat", 3)],
        ]
    )[:2]
    assert [(x.chunk.id, x.score) for x in hits] == expected


def test_hybrid_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever([])
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever([BM25Retriever()], weights=[1, 2])
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([["a"]], weights=[1, 2])
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], k=0)


def test_rrf_weights_and_duplicates() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "a"], ["b"]], k=1, weights=[1.0, 3.0])
    assert fused[0][0] == "b"
    assert dict(fused)["a"] == pytest.approx(1 / 2)
    assert dict(fused)["b"] == pytest.approx(1 / 3 + 3 / 2)


def test_reranking_retriever() -> None:
    base = BM25Retriever()
    rr = RerankingRetriever(base, CallableReranker(lambda q, t: -len(t)), fetch_k=5)
    rr.index(chunks())
    hits = rr.retrieve("cat sat", 2)
    assert [h.chunk.id for h in hits] == ["c0", "c1"]
    assert hits[0].score == -len(DOCS[0])
