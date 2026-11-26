# Metrics

All metrics return a score in [0, 1] per case, or `None` when the metric does not apply
(for example, no relevance labels or no reference answer). `None` values are excluded from
aggregates; every aggregate reports the number of scored cases `n`.

## Retrieval

Relevance is binary and judged at the document level. Retrieved chunks are mapped to their
`doc_id`s and de-duplicated in rank order. Let $R$ be the set of relevant documents and
$d_1, d_2, \dots$ the ranked, de-duplicated documents.

| Name | Formula |
|---|---|
| `recall@k` | $\lvert R \cap \{d_1..d_k\}\rvert / \lvert R\rvert$ |
| `precision@k` | $\lvert R \cap \{d_1..d_k\}\rvert / k$ |
| `hit_rate@k` | $\mathbb{1}[R \cap \{d_1..d_k\} \neq \emptyset]$ |
| `mrr`, `mrr@k` | $1/\text{rank of first relevant document}$, 0 if none (within $k$) |
| `ndcg@k` | $\text{DCG}_k/\text{IDCG}_k$, $\text{DCG}_k=\sum_{i=1}^{k}\mathbb{1}[d_i\in R]/\log_2(i+1)$ |

Precision@k divides by $k$ even when fewer than $k$ documents were returned, so a retriever
cannot improve precision by returning less.

## Reference-based generation metrics

Answers and references are normalised as in SQuAD (Rajpurkar et al., 2016): Unicode NFKC,
lower-case, punctuation and the articles *a/an/the* removed, whitespace collapsed.

| Name | Formula |
|---|---|
| `exact_match` | $\mathbb{1}[\text{norm}(a) = \text{norm}(r)]$ |
| `token_f1` | $2PR/(P+R)$ over the multiset token overlap |
| `rouge_l` | $F_\beta = (1+\beta^2)PR/(R+\beta^2P)$, $P=\text{LCS}/\lvert a\rvert$, $R=\text{LCS}/\lvert r\rvert$ (Lin, 2004), $\beta=1$ |
| `semantic_similarity` | $\max(0, \cos(E(a), E(r)))$ with the configured embedder |

## Judge metrics (RAGAS-style)

These follow Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*
(EACL 2024 demos, [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)), and the context
precision/recall definitions documented by the `ragas` library. The judge returns verdicts as
JSON; the formulas are computed in Python.

**Faithfulness.** The judge decomposes the answer into atomic claims $S$ and marks each as
supported by the retrieved context or not ($V \subseteq S$).
$\text{faithfulness} = \lvert V\rvert / \lvert S\rvert$. `None` if the answer is empty or
contains no claims.

**Answer relevance.** The judge writes $n$ (default 3) questions $q_i$ that the answer would
answer, and flags noncommittal answers. With embedder $E$ and original question $q$:
$\text{AR} = \frac{1}{n}\sum_i \max(0, \cos(E(q), E(q_i)))$, and 0 for noncommittal or
empty answers.

**Context precision@K.** For each of the top-$K$ retrieved chunks the judge decides whether
it was useful for arriving at the reference answer ($v_k \in \{0,1\}$).
$\text{CP@}K = \sum_{k=1}^{K} (\text{P@}k \cdot v_k) / \sum_{k=1}^{K} v_k$ - the average
precision of the useful chunks; 0 if none is useful. Without `@K`, all retrieved chunks.

**Context recall.** The reference answer is split into sentences; the judge decides for each
whether it can be attributed to the retrieved context.
$\text{CR} = \text{attributed} / \text{sentences}$.

## Judges

- `LLMJudge(provider)` - prompts in `ragcheck.judges.PROMPTS`. Replies must be exactly one
  JSON object (an optional single Markdown fence is allowed) matching a Pydantic schema in
  strict mode. One repair attempt is made; then `JudgeParseError` is recorded on the case.
- `DeterministicJudge(threshold=0.6)` - a statement is "supported" when at least
  `threshold` of its content words (stopwords removed) appear in the context. Reproducible
  and free, intended for CI smoke tests; it cannot recognise paraphrase.
