# Advanced RAG with Evaluation

Production-shaped retrieval for RAG: **hybrid search** (dense embeddings + BM25
fused with Reciprocal Rank Fusion), **cross-encoder reranking**, and a
**deterministic evaluation** that measures retrieval quality so you can prove a
change helped. "I built RAG" is common; "I built RAG and *measured* the retrieval"
is the differentiator.

> **What the eval actually found** (the honest, useful part): on a labeled mix
> that includes keyword-sparse paraphrase queries, **dense leads (hit@1 ≈ 0.88)
> and naive hybrid/RRF does *not* beat it (≈ 0.81)** — BM25's noise on
> synonym-heavy queries drags the fusion down. The lesson isn't "hybrid wins";
> it's *measure your retrieval on your data instead of assuming.*

| Area | What's shown |
|------|--------------|
| **Advanced RAG** | hybrid dense+sparse retrieval, RRF fusion, two-stage retrieve→rerank |
| **RAG evaluation** | labeled queries, hit_rate@k and MRR (no LLM judge needed) |
| **IR fundamentals** | BM25, embeddings, cross-encoder vs bi-encoder trade-off |

## Why each piece

- **Dense** captures paraphrase/semantics; **BM25** nails exact keywords and rare
  terms. **Hybrid (RRF)** fuses both rankings — no score normalization required.
- **Cross-encoder reranking**: the bi-encoder retrieves top-N fast; the
  cross-encoder rescoring the (query, doc) pairs jointly is more accurate, so we
  rerank only those N.
- **Evaluation**: each query has a known-relevant doc, so hit_rate@k and MRR are
  exact — you can A/B retrieval strategies objectively.

## Run the comparison

```bash
docker build -t advanced-rag .    # first build bakes the two models
docker run --rm advanced-rag
```

Prints hit_rate@1 and MRR for **dense-only / BM25-only / hybrid / hybrid+rerank**
over the labeled set (easy queries + adversarial paraphrase/code queries),
reports the best strategy, and gates on the best strategy retrieving well.

## Tests (no models needed)

```bash
pip install -r requirements.txt pytest && python -m pytest
```

Covers BM25 keyword retrieval, the RRF fusion math, and the hit_rate/MRR metrics.

## Layout

```
arag/retrievers.py  DenseRetriever, BM25Retriever, HybridRetriever (RRF)
arag/rerank.py      cross-encoder reranker
arag/embed.py       sentence-transformer embeddings
arag/corpus.py      knowledge base + labeled retrieval eval set
arag/evaluate.py    hit_rate@k + MRR
run_eval.py         compare all four strategies
tests/              BM25 / RRF / metrics tests
```
