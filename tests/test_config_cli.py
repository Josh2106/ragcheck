from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ragcheck.cli import app
from ragcheck.config import (
    ComponentSpec,
    ConfigError,
    RunConfig,
    build,
    build_chunker,
    build_embedder,
    build_provider,
    build_retriever,
    chunk_size_override,
    load_config,
)
from ragcheck.embedders import HashingEmbedder, OpenAICompatibleEmbedder
from ragcheck.experiments import run_grid
from ragcheck.generators import LLMGenerator
from ragcheck.judges import LLMJudge
from ragcheck.providers import AnthropicProvider, GroqProvider, OllamaProvider, OpenAIProvider
from ragcheck.results import RunResult
from ragcheck.retrievers import HybridRetriever, RerankingRetriever

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def write_cfg(tmp_path: Path, **overrides: object) -> Path:
    cfg: dict[str, object] = {
        "name": "t",
        "dataset": "builtin:sample",
        "chunker": {"type": "recursive", "chunk_size": 300},
        "retriever": {"type": "hybrid"},
        "metrics": ["recall@5", "mrr", "token_f1", "faithfulness"],
        "limit": 6,
    }
    cfg.update(overrides)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------ config


def test_load_and_build_defaults(tmp_path: Path) -> None:
    cfg = load_config(write_cfg(tmp_path))
    built = build(cfg)
    assert len(built.dataset) == 6
    assert isinstance(built.pipeline.retriever, HybridRetriever)
    assert [m.name for m in built.metrics] == ["recall@5", "mrr", "token_f1", "faithfulness"]
    assert RunConfig().metrics[0] == "recall@5"


def test_relative_dataset_path_resolves_against_config(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "corpus.jsonl").write_text('{"id": "a", "text": "alpha"}\n', encoding="utf-8")
    (data / "questions.jsonl").write_text(
        '{"id": "q", "question": "alpha?", "relevant_doc_ids": ["a"]}\n', encoding="utf-8"
    )
    built = build(load_config(write_cfg(tmp_path, dataset="data", limit=None)))
    assert built.dataset.documents[0].id == "a"


@pytest.mark.parametrize(
    ("content", "match"),
    [("- a\n- b\n", "mapping"), ("name: [unclosed\n", "cannot read"), ("bogus: 1\n", "bogus")],
)
def test_load_config_errors(tmp_path: Path, content: str, match: str) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_config(p)


def test_component_builders() -> None:
    spec = ComponentSpec
    for t, kw in [
        ("fixed", {"size": 50}),
        ("recursive", {}),
        ("sentence_window", {"window": 2}),
        ("markdown", {"max_chunk_size": 100}),
    ]:
        build_chunker(spec(type=t, **kw))
    with pytest.raises(ConfigError, match="unknown chunker"):
        build_chunker(spec(type="magic"))
    with pytest.raises(ConfigError, match="unknown option"):
        build_chunker(spec(type="fixed", sise=3))
    assert isinstance(build_embedder(spec(type="hashing", dim=8)), HashingEmbedder)
    ollama = build_embedder(spec(type="ollama", model="nomic-embed-text"))
    assert isinstance(ollama, OpenAICompatibleEmbedder)
    assert ollama.base_url == "http://localhost:11434/v1"
    build_embedder(spec(type="openai", model="m", api_key_env="NOPE_KEY"))
    build_embedder(spec(type="sentence_transformers", model_name="x"))
    with pytest.raises(ConfigError, match="unknown embedder"):
        build_embedder(spec(type="magic"))
    with pytest.raises(ConfigError, match="needs an embedder"):
        build_retriever(spec(type="dense"), None)
    with pytest.raises(ConfigError, match="unknown retriever"):
        build_retriever(spec(type="magic"), None)
    assert isinstance(
        build_retriever(spec(type="hybrid", k1=1.2, weights=[1, 2]), HashingEmbedder(8)),
        HybridRetriever,
    )


def test_build_provider() -> None:
    assert isinstance(build_provider({"provider": "openai", "model": "m"}), OpenAIProvider)
    assert isinstance(build_provider({"provider": "groq", "model": "m"}), GroqProvider)
    assert isinstance(build_provider({"provider": "anthropic", "model": "m"}), AnthropicProvider)
    assert isinstance(build_provider({"provider": "ollama", "model": "m"}), OllamaProvider)
    with pytest.raises(ConfigError, match="provider"):
        build_provider({"provider": "skynet", "model": "m"})
    with pytest.raises(ConfigError, match="model"):
        build_provider({"provider": "openai"})
    with pytest.raises(ConfigError, match="bad options"):
        build_provider({"provider": "openai", "model": "m", "colour": "red"})


def test_build_llm_components_and_reranker(tmp_path: Path) -> None:
    cfg = load_config(
        write_cfg(
            tmp_path,
            generator={"type": "llm", "provider": "openai", "model": "m", "temperature": 0.1},
            judge={"type": "llm", "provider": "anthropic", "model": "m"},
            reranker={"type": "cross_encoder", "model_name": "ce", "fetch_k": 10},
            cache=str(tmp_path / "cache.sqlite"),
        )
    )
    built = build(cfg)
    assert isinstance(built.pipeline.generator, LLMGenerator)
    assert built.pipeline.generator.temperature == 0.1
    assert isinstance(built.pipeline.retriever, RerankingRetriever)
    assert built.cache is not None
    judge = built.metrics[-1].judge  # type: ignore[attr-defined]
    assert isinstance(judge, LLMJudge)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"generator": {"type": "magic"}}, "unknown generator"),
        ({"judge": {"type": "magic"}}, "unknown judge"),
        ({"reranker": {"type": "magic"}}, "unknown reranker"),
        ({"judge": None}, "needs a judge"),
    ],
)
def test_build_errors(tmp_path: Path, overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        build(load_config(write_cfg(tmp_path, **overrides)))


def test_chunk_size_override() -> None:
    s = chunk_size_override(ComponentSpec(type="recursive", chunk_overlap=100), 50)
    assert s.params == {"chunk_overlap": 5, "chunk_size": 50}
    assert chunk_size_override(ComponentSpec(type="fixed"), 10).params == {"size": 10}
    with pytest.raises(ConfigError, match="no size"):
        chunk_size_override(ComponentSpec(type="magic"), 10)
    assert ComponentSpec(type="x", a=1).with_params(b=2).params == {"a": 1, "b": 2}
    assert ComponentSpec(type="x").label == "x"


# --------------------------------------------------------------------------------- grid


def test_run_grid(tmp_path: Path) -> None:
    cfg = load_config(
        write_cfg(
            tmp_path,
            grid={
                "chunk_sizes": [200, 600],
                "retrievers": [{"type": "bm25"}, {"type": "hybrid", "rrf_k": 10}],
            },
        )
    )
    grid = run_grid(cfg)
    assert [(r.chunk_size, r.retriever) for r in grid.rows] == [
        (200, "bm25"),
        (200, "hybrid(rrf_k=10)"),
        (600, "bm25"),
        (600, "hybrid(rrf_k=10)"),
    ]
    assert grid.rows[0].n_chunks > grid.rows[2].n_chunks
    md = grid.to_markdown()
    assert md.count("\n") == 5
    assert "**" in md
    assert grid.best("mrr").score("mrr") == max(r.score("mrr") or 0 for r in grid.rows)
    with pytest.raises(ValueError, match="no scores"):
        grid.best("ghost")
    with pytest.raises(ValueError, match="at least one"):
        run_grid(cfg.model_copy(update={"grid": None}))


# ---------------------------------------------------------------------------------- CLI


def test_cli_help_and_version() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("run", "compare", "report", "grid", "metrics"):
        assert cmd in res.output
    assert "ragcheck" in runner.invoke(app, ["--version"]).output
    assert "faithfulness" in runner.invoke(app, ["metrics"]).output


def test_cli_quickstart_config_runs(tmp_path: Path) -> None:
    out, html = tmp_path / "r.json", tmp_path / "r.html"
    res = runner.invoke(
        app,
        [
            "run",
            str(ROOT / "examples" / "quickstart.yaml"),
            "-o",
            str(out),
            "--html",
            str(html),
            "--no-progress",
        ],
    )
    assert res.exit_code == 0, res.output
    result = RunResult.load(out)
    assert len(result.rows) == 20
    assert (result.mean("hit_rate@5") or 0) > 0.8
    assert html.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_cli_run_with_cache_and_limit(tmp_path: Path) -> None:
    cfg = write_cfg(tmp_path, cache=str(tmp_path / "c.sqlite"), output=str(tmp_path / "o.json"))
    for _ in range(2):
        res = runner.invoke(app, ["run", str(cfg), "--limit", "3", "--no-progress"])
        assert res.exit_code == 0, res.output
    result = RunResult.load(tmp_path / "o.json")
    assert len(result.rows) == 3
    assert result.stats["cache_hits"] > 0
    assert result.stats["cache_misses"] == 0  # second run is fully cached


def test_cli_run_reports_case_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ragcheck import pipeline as pipeline_mod

    async def broken(self: object, question: str, query_id: str = "") -> None:
        raise RuntimeError("down")

    monkeypatch.setattr(pipeline_mod.RAGPipeline, "query", broken)
    res = runner.invoke(app, ["run", str(write_cfg(tmp_path)), "--no-progress"])
    assert res.exit_code == 0
    assert "had errors" in res.output


def test_cli_run_bad_config(tmp_path: Path) -> None:
    res = runner.invoke(app, ["run", str(write_cfg(tmp_path, metrics=["bogus"]))])
    assert res.exit_code == 2
    assert "unknown metric" in res.output
    assert runner.invoke(app, ["run", str(tmp_path / "missing.yaml")]).exit_code == 2


def _results(tmp_path: Path, name: str, recall: float) -> Path:
    from ragcheck.types import EvalResult

    row = EvalResult(
        case_id="q",
        question="?",
        ground_truth="",
        answer="",
        retrieved_doc_ids=(),
        relevant_doc_ids=(),
        scores={"recall@5": recall},
    )
    return RunResult.from_rows([row], ["recall@5"], name=name).save(tmp_path / f"{name}.json")


def test_cli_compare_gate(tmp_path: Path) -> None:
    base = _results(tmp_path, "base", 0.80)
    good = _results(tmp_path, "good", 0.79)
    bad = _results(tmp_path, "bad", 0.70)
    ok = runner.invoke(app, ["compare", str(base), str(good), "--fail-on", "recall@5<-0.02"])
    assert ok.exit_code == 0, ok.output
    assert "passed" in ok.output
    fail = runner.invoke(app, ["compare", str(base), str(bad), "--fail-on", "recall@5<-0.02"])
    assert fail.exit_code == 1
    assert "regression" in fail.output
    js = runner.invoke(app, ["compare", str(base), str(bad), "--fail-on", "recall@5<-5%", "--json"])
    assert js.exit_code == 1
    payload = json.loads(js.output)
    assert payload["failed"] is True
    assert payload["metrics"][0]["delta"] == pytest.approx(-0.1)
    missing = runner.invoke(app, ["compare", str(base), str(good), "--fail-on", "mrr<0"])
    assert missing.exit_code == 1
    assert runner.invoke(app, ["compare", str(base), str(good)]).exit_code == 0
    assert runner.invoke(app, ["compare", str(base), str(good), "--fail-on", "??"]).exit_code == 2
    assert runner.invoke(app, ["compare", str(base), str(tmp_path / "nope.json")]).exit_code == 2
    junk = tmp_path / "junk.json"
    junk.write_text("{}", encoding="utf-8")
    junk.write_text('{"rows": 5}', encoding="utf-8")
    assert runner.invoke(app, ["compare", str(base), str(junk)]).exit_code == 2


def test_cli_report(tmp_path: Path) -> None:
    base = _results(tmp_path, "base", 0.8)
    cur = _results(tmp_path, "cur", 0.9)
    out = tmp_path / "rep.html"
    res = runner.invoke(app, ["report", str(cur), "--html", str(out), "--baseline", str(base)])
    assert res.exit_code == 0, res.output
    assert "baseline" in out.read_text(encoding="utf-8")


def test_cli_grid(tmp_path: Path) -> None:
    cfg = write_cfg(tmp_path, grid={"chunk_sizes": [300], "retrievers": [{"type": "bm25"}]})
    md = tmp_path / "g.md"
    res = runner.invoke(app, ["grid", str(cfg), "--markdown", str(md), "-m", "mrr"])
    assert res.exit_code == 0, res.output
    assert md.read_text(encoding="utf-8").startswith("| chunker | size | retriever")
    no_grid = runner.invoke(app, ["grid", str(write_cfg(tmp_path))])
    assert no_grid.exit_code == 2
