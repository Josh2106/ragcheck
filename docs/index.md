# ragcheck

**Benchmark and regression-test RAG pipelines.**

ragcheck evaluates a retrieval-augmented generation pipeline against a labelled dataset,
produces per-case and aggregate scores, and compares runs so that CI can fail when a change
makes retrieval or answers worse.

- Swap any component: chunkers, embedders, retrievers, rerankers, generators and judges are
  all `typing.Protocol`s.
- Retrieval metrics (Recall@k, Precision@k, MRR, nDCG@k, hit rate), reference metrics
  (exact match, token F1, ROUGE-L, semantic similarity) and RAGAS-style judge metrics
  (faithfulness, answer relevance, context precision, context recall).
- Fully offline mode for CI: hashing embedder, extractive generator, deterministic judge.
- Disk cache for embeddings and LLM calls, async runner with bounded concurrency.
- `ragcheck compare --fail-on "recall@5<-0.02"` as a CI gate, and a self-contained HTML report.

```bash
pip install -e ".[dev]"
ragcheck run examples/quickstart.yaml --html results/quickstart.html
```

Start with [Getting started](getting-started.md), or read the
[architecture notes](https://github.com/Josh2106/ragcheck/blob/main/ARCHITECTURE.md).
