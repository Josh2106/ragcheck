# Extending ragcheck

Every component is a `typing.Protocol`, so you never need to inherit from a ragcheck class
(except `Metric`, which is an ABC because it carries shared behaviour).

## Evaluate an existing pipeline

```python
from ragcheck import Answer, Query, RetrievedContext, ScoredChunk, Chunk

class MyPipeline:
    def __init__(self, app):
        self.app = app

    def index(self, documents):
        self.app.ingest([(d.id, d.text) for d in documents])

    async def query(self, question, query_id=""):
        hits, text = await self.app.answer(question)
        chunks = tuple(
            ScoredChunk(
                chunk=Chunk(id=h.id, doc_id=h.doc_id, text=h.text, start=0,
                            end=len(h.text), index=i),
                score=h.score,
            )
            for i, h in enumerate(hits)
        )
        return RetrievedContext(query=Query(id=query_id, text=question), chunks=chunks), Answer(text=text)
```

## A custom metric

```python
from ragcheck.metrics import Metric, Sample

class AnswerLength(Metric):
    kind = "generation"

    @property
    def name(self) -> str:
        return "short_answer"

    async def score(self, sample: Sample) -> float | None:
        if sample.answer is None:
            return None
        return 1.0 if len(sample.answer.text.split()) <= 30 else 0.0
```

Pass instances alongside built-in metrics: `ragcheck.run(p, ds, [*build_metrics([...]), AnswerLength()])`.

## A custom vector store

Implement `add(ids, vectors)`, `search(vector, k) -> [(id, score)]` and `__len__`, then
`DenseRetriever(embedder, index=MyIndex())`.

## A custom LLM provider

Implement `name` and `async complete(messages, *, temperature, max_tokens, json_mode) ->
Completion`. Wrap it in `CachedProvider(provider, SQLiteCache(path))` to cache calls, and
use it with `LLMGenerator` or `LLMJudge`.
