# Configuration

`ragcheck run` and `ragcheck grid` read a YAML file validated by `ragcheck.config.RunConfig`.
Unknown top-level keys and unknown component options are rejected.

| Key | Default | Notes |
|---|---|---|
| `name` | `run` | Used in reports |
| `dataset` | `builtin:sample` | Directory with `corpus.jsonl` + `questions.jsonl`, relative to the config file |
| `chunker` | `{type: recursive}` | See below |
| `embedder` | `{type: hashing}` | `null` disables dense retrieval and embedding metrics |
| `retriever` | `{type: bm25}` | `bm25`, `dense`, `hybrid` |
| `reranker` | `null` | `{type: cross_encoder, model_name, device, fetch_k}` |
| `generator` | `{type: extractive}` | `extractive`, `llm`, or `null` for retrieval-only |
| `judge` | `{type: deterministic}` | `deterministic` or `llm` |
| `metrics` | retrieval + lexical defaults | Names as in [Metrics](metrics.md) |
| `top_k` | `5` | Chunks passed to the generator |
| `concurrency` | `8` | Max cases in flight |
| `cache` | `null` | SQLite path for embedding and LLM caching |
| `output` | `null` | Results JSON path (overridden by `-o`) |
| `limit` | `null` | Evaluate only the first N cases |
| `grid` | `null` | `{chunk_sizes: [...], retrievers: [{type: ...}, ...]}` for `ragcheck grid` |

## Components

**Chunkers**

- `fixed`: `size`, `overlap`, `unit` (`chars` | `tokens`)
- `recursive`: `chunk_size`, `chunk_overlap`, `separators`, `unit`
- `sentence_window`: `window`, `sentences_per_chunk`, `max_sentence_chars`
- `markdown`: `max_chunk_size`, `max_header_level`, `chunk_overlap`

**Embedders**

- `hashing`: `dim`, `char_ngrams`, `use_bigrams`
- `openai`: `model`, `base_url`, `api_key_env`, `batch_size`, `timeout`, `dimensions`
- `ollama`: same as `openai`, `base_url` defaults to `http://localhost:11434/v1`
- `sentence_transformers`: `model_name`, `device`, `batch_size`, `normalize`

**Retrievers**

- `bm25`: `k1` (1.5), `b` (0.75)
- `dense`: uses the configured embedder
- `hybrid`: BM25 + dense with RRF; `rrf_k` (60), `fetch_k` (20), `weights`, `k1`, `b`

**Generators and judges with an LLM**

```yaml
generator: {type: llm, provider: openai, model: <model-id>, temperature: 0.0}
judge: {type: llm, provider: anthropic, model: <model-id>}
```

`provider` is one of `openai`, `anthropic`, `groq`, `ollama`; other keys (`base_url`,
`api_key_env`, `timeout`, `max_retries`) are passed to the provider. Keys are read from
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GROQ_API_KEY` by default.
