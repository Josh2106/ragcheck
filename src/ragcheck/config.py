"""YAML run configuration and the factory that turns it into live components.

Example::

    name: quickstart
    dataset: builtin:sample          # or a directory with corpus.jsonl + questions.jsonl
    chunker:   {type: recursive, chunk_size: 400, chunk_overlap: 40}
    embedder:  {type: hashing, dim: 512}
    retriever: {type: hybrid}
    generator: {type: extractive}
    judge:     {type: deterministic}
    metrics: [recall@5, mrr, ndcg@5, token_f1, faithfulness]

Relative dataset paths are resolved against the config file's directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragcheck.cache import CachedEmbedder, CachedProvider, SQLiteCache
from ragcheck.chunkers import (
    FixedSizeChunker,
    MarkdownHeaderChunker,
    RecursiveChunker,
    SentenceWindowChunker,
)
from ragcheck.datasets import BUILTIN_PREFIX, Dataset, load_dataset
from ragcheck.embedders import (
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
)
from ragcheck.generators import ExtractiveGenerator, LLMGenerator
from ragcheck.judges import DeterministicJudge, LLMJudge
from ragcheck.metrics import Metric, build_metrics
from ragcheck.pipeline import RAGPipeline
from ragcheck.protocols import Chunker, Embedder, Generator, Judge, Retriever
from ragcheck.providers import (
    AnthropicProvider,
    GroqProvider,
    OllamaProvider,
    OpenAIProvider,
    Provider,
)
from ragcheck.retrievers import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    RerankingRetriever,
)

__all__ = [
    "Built",
    "ComponentSpec",
    "ConfigError",
    "GridSpec",
    "RunConfig",
    "build",
    "load_config",
]

DEFAULT_METRICS = [
    "recall@5",
    "precision@5",
    "mrr",
    "ndcg@5",
    "hit_rate@5",
    "token_f1",
    "rouge_l",
]


class ConfigError(ValueError):
    pass


class ComponentSpec(BaseModel):
    """``{type: <name>, ...params}``; params are passed to the component constructor."""

    model_config = ConfigDict(extra="allow", frozen=True)
    type: str

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})

    def with_params(self, **kw: Any) -> ComponentSpec:
        return ComponentSpec(type=self.type, **{**self.params, **kw})

    @property
    def label(self) -> str:
        return self.type


class GridSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_sizes: list[int] = Field(min_length=1)
    retrievers: list[ComponentSpec] = Field(min_length=1)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "run"
    dataset: str = BUILTIN_PREFIX + "sample"
    chunker: ComponentSpec = ComponentSpec(type="recursive")
    embedder: ComponentSpec | None = ComponentSpec(type="hashing")
    retriever: ComponentSpec = ComponentSpec(type="bm25")
    reranker: ComponentSpec | None = None
    generator: ComponentSpec | None = ComponentSpec(type="extractive")
    judge: ComponentSpec | None = ComponentSpec(type="deterministic")
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_METRICS))
    top_k: int = Field(default=5, gt=0)
    concurrency: int = Field(default=8, gt=0)
    cache: str | None = None
    output: str | None = None
    limit: int | None = Field(default=None, gt=0)
    grid: GridSpec | None = None
    base_dir: str = Field(default=".", exclude=True)


def load_config(path: str | Path) -> RunConfig:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    try:
        cfg = RunConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{p}: {exc}") from exc
    return cfg.model_copy(update={"base_dir": str(p.parent)})


def _params(spec: ComponentSpec, allowed: set[str]) -> dict[str, Any]:
    params = spec.params
    unknown = set(params) - allowed
    if unknown:
        raise ConfigError(
            f"unknown option(s) {sorted(unknown)} for {spec.type!r}; allowed: {sorted(allowed)}"
        )
    return params


_CHUNK_SIZE_PARAM = {
    "fixed": "size",
    "recursive": "chunk_size",
    "markdown": "max_chunk_size",
    "sentence_window": "max_sentence_chars",
}


def chunk_size_override(spec: ComponentSpec, size: int) -> ComponentSpec:
    try:
        key = _CHUNK_SIZE_PARAM[spec.type]
    except KeyError as exc:
        raise ConfigError(f"chunker {spec.type!r} has no size parameter") from exc
    params = spec.params
    # keep overlap valid when sweeping small sizes
    for ov in ("overlap", "chunk_overlap"):
        if ov in params and params[ov] >= size:
            params[ov] = size // 10
    return ComponentSpec(type=spec.type, **{**params, key: size})


def build_chunker(spec: ComponentSpec) -> Chunker:
    t = spec.type
    if t == "fixed":
        return FixedSizeChunker(**_params(spec, {"size", "overlap", "unit"}))
    if t == "recursive":
        return RecursiveChunker(
            **_params(spec, {"chunk_size", "chunk_overlap", "separators", "unit"})
        )
    if t == "sentence_window":
        return SentenceWindowChunker(
            **_params(spec, {"window", "sentences_per_chunk", "max_sentence_chars"})
        )
    if t == "markdown":
        return MarkdownHeaderChunker(
            **_params(spec, {"max_chunk_size", "max_header_level", "chunk_overlap"})
        )
    raise ConfigError(f"unknown chunker type {t!r}")


def build_embedder(spec: ComponentSpec) -> Embedder:
    t = spec.type
    if t == "hashing":
        return HashingEmbedder(**_params(spec, {"dim", "char_ngrams", "use_bigrams"}))
    http_opts = {"model", "base_url", "api_key_env", "batch_size", "timeout", "dimensions"}
    if t == "openai":
        return OpenAICompatibleEmbedder(**_params(spec, http_opts))
    if t == "ollama":
        params = _params(spec, http_opts)
        params.setdefault("base_url", "http://localhost:11434/v1")
        return OpenAICompatibleEmbedder(**params)
    if t == "sentence_transformers":
        return SentenceTransformerEmbedder(
            **_params(spec, {"model_name", "device", "batch_size", "normalize"})
        )
    raise ConfigError(f"unknown embedder type {t!r}")


_PROVIDERS: dict[str, type[OpenAIProvider | AnthropicProvider | OllamaProvider]] = {
    "openai": OpenAIProvider,
    "groq": GroqProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def build_provider(params: dict[str, Any]) -> Provider:
    params = dict(params)
    name = params.pop("provider", None)
    if name not in _PROVIDERS:
        raise ConfigError(f"'provider' must be one of {sorted(_PROVIDERS)}, got {name!r}")
    if "model" not in params:
        raise ConfigError(f"provider {name!r} needs a 'model'")
    try:
        return _PROVIDERS[name](**params)
    except TypeError as exc:
        raise ConfigError(f"bad options for provider {name!r}: {exc}") from exc


def build_retriever(spec: ComponentSpec, embedder: Embedder | None) -> Retriever:
    t = spec.type

    def need_embedder() -> Embedder:
        if embedder is None:
            raise ConfigError(f"retriever {t!r} needs an embedder")
        return embedder

    if t == "bm25":
        return BM25Retriever(**_params(spec, {"k1", "b"}))
    if t == "dense":
        _params(spec, set())
        return DenseRetriever(need_embedder())
    if t == "hybrid":
        params = _params(spec, {"weights", "rrf_k", "fetch_k", "k1", "b"})
        bm25 = BM25Retriever(k1=params.pop("k1", 1.5), b=params.pop("b", 0.75))
        return HybridRetriever([bm25, DenseRetriever(need_embedder())], **params)
    raise ConfigError(f"unknown retriever type {t!r}")


@dataclass
class Built:
    pipeline: RAGPipeline
    metrics: list[Metric]
    dataset: Dataset
    config: RunConfig
    cache: SQLiteCache | None = None


def build(cfg: RunConfig, dataset: Dataset | None = None) -> Built:
    """Instantiate dataset, pipeline and metrics from a :class:`RunConfig`."""
    cache = SQLiteCache(cfg.cache) if cfg.cache else None
    embedder = build_embedder(cfg.embedder) if cfg.embedder else None
    if embedder is not None and cache is not None:
        embedder = CachedEmbedder(embedder, cache)

    def provider_from(spec: ComponentSpec) -> Provider:
        provider = build_provider(spec.params)
        return CachedProvider(provider, cache) if cache is not None else provider

    retriever = build_retriever(cfg.retriever, embedder)
    if cfg.reranker is not None:
        if cfg.reranker.type != "cross_encoder":
            raise ConfigError(f"unknown reranker type {cfg.reranker.type!r}")
        params = _params(cfg.reranker, {"model_name", "device", "fetch_k"})
        fetch_k = params.pop("fetch_k", 20)
        retriever = RerankingRetriever(retriever, CrossEncoderReranker(**params), fetch_k)

    generator: Generator | None = None
    if cfg.generator is not None:
        if cfg.generator.type == "extractive":
            generator = ExtractiveGenerator(
                **_params(cfg.generator, {"max_sentences", "use_window"})
            )
        elif cfg.generator.type == "llm":
            gp = cfg.generator.params
            gen_opts = {k: gp.pop(k) for k in ("max_context_chars", "temperature") if k in gp}
            generator = LLMGenerator(provider_from(ComponentSpec(type="llm", **gp)), **gen_opts)
        else:
            raise ConfigError(f"unknown generator type {cfg.generator.type!r}")

    judge: Judge | None = None
    if cfg.judge is not None:
        if cfg.judge.type == "deterministic":
            judge = DeterministicJudge(**_params(cfg.judge, {"threshold"}))
        elif cfg.judge.type == "llm":
            judge = LLMJudge(provider_from(cfg.judge))
        else:
            raise ConfigError(f"unknown judge type {cfg.judge.type!r}")

    if dataset is None:
        spec = cfg.dataset
        if not spec.startswith(BUILTIN_PREFIX) and not Path(spec).is_absolute():
            spec = str(Path(cfg.base_dir) / spec)
        dataset = load_dataset(spec)
    if cfg.limit:
        dataset = dataset.subset(cfg.limit)
    try:
        metrics = build_metrics(cfg.metrics, embedder=embedder, judge=judge)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    pipeline = RAGPipeline(build_chunker(cfg.chunker), retriever, generator, top_k=cfg.top_k)
    return Built(pipeline=pipeline, metrics=metrics, dataset=dataset, config=cfg, cache=cache)
