# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-26

### Added

- Core typed data model (`Document`, `Chunk`, `EvalCase`, `EvalResult`, ...) and protocols
  for chunkers, embedders, retrievers, rerankers, generators, judges and pipelines.
- Chunkers: fixed-size (chars/tokens), recursive, sentence-window, Markdown-header-aware.
- Embedders: deterministic hashing, sentence-transformers adapter, OpenAI-compatible HTTP.
- LLM providers: OpenAI, Groq, Anthropic, Ollama, with retries and backoff.
- Retrievers: Okapi BM25, dense cosine, hybrid reciprocal rank fusion, reranking hook.
- Metrics: Recall@k, Precision@k, MRR, nDCG@k, hit rate, exact match, token F1, ROUGE-L,
  semantic similarity, and RAGAS-style faithfulness, answer relevance, context precision and
  context recall with an LLM judge or a deterministic judge.
- Async runner with bounded concurrency, progress bars and a SQLite content-hash cache.
- CLI: `run`, `compare` (regression gate), `report` (self-contained HTML), `grid`, `metrics`.
- Bundled synthetic sample dataset (30 docs, 20 questions) and example configs.

[Unreleased]: https://github.com/Josh2106/ragcheck/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Josh2106/ragcheck/releases/tag/v0.1.0
