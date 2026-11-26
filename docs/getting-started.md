# Getting started

## Install

```bash
git clone https://github.com/Josh2106/ragcheck && cd ragcheck
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras: `tiktoken` (token-accurate chunking) and `sentence-transformers` (local
embedding models and cross-encoder reranking).

## Run the offline quickstart

```bash
ragcheck run examples/quickstart.yaml --html results/quickstart.html
```

This evaluates a hybrid BM25 + dense pipeline on the bundled synthetic corpus (30 docs about
*Larkspur*, a fictional task queue; 20 questions) with no network access, and writes
`results/quickstart.json` plus an HTML report.

## Bring your own data

Create a directory with two JSONL files:

`corpus.jsonl`

```json
{"id": "install", "title": "Installation", "text": "...", "metadata": {"version": "2.3"}}
```

`questions.jsonl`

```json
{"id": "q01", "question": "What port ...?", "answer": "Port 7420.", "relevant_doc_ids": ["install"]}
```

`answer` (alias `ground_truth`) is optional, but needed for reference and context metrics;
`relevant_doc_ids` is needed for retrieval metrics. Unknown document ids are an error.

Then point a config at it: `dataset: path/to/dir` (relative to the config file).

## Use it from Python

```python
import ragcheck
from ragcheck.chunkers import RecursiveChunker
from ragcheck.retrievers import BM25Retriever
from ragcheck.generators import ExtractiveGenerator
from ragcheck.metrics import build_metrics

dataset = ragcheck.load_dataset("path/to/dir")
pipeline = ragcheck.RAGPipeline(RecursiveChunker(400), BM25Retriever(), ExtractiveGenerator())
result = ragcheck.run(pipeline, dataset, build_metrics(["recall@5", "mrr", "token_f1"]))
result.save("results/bm25.json")
```

Inside async code use `await ragcheck.arun(...)`.
