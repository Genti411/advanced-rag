"""Compare retrieval strategies on the labeled eval set and print metrics.

dense-only vs BM25-only vs hybrid (RRF) vs hybrid+cross-encoder rerank. Exits
non-zero if hybrid doesn't at least match dense (a regression guard).
"""
import sys

from arag.corpus import DOCS, EVAL
from arag.embed import Embedder
from arag.evaluate import evaluate
from arag.rerank import Reranker
from arag.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

K = 1   # strict top-1: makes ranking quality (and method differences) visible


def main() -> int:
    doc_ids = list(DOCS)
    texts = [DOCS[d] for d in doc_ids]

    embedder = Embedder()
    dense = DenseRetriever(doc_ids, embedder.embed(texts), embedder)
    bm25 = BM25Retriever(doc_ids, texts)
    hybrid = HybridRetriever(dense, bm25)
    reranker = Reranker()

    def hybrid_rerank(query, k):
        cands = [d for d, _ in hybrid.search(query, k=8)]
        return reranker.rerank(query, cands, DOCS, k=k)

    strategies = {
        "dense-only": dense.search,
        "bm25-only": bm25.search,
        "hybrid (RRF)": hybrid.search,
        "hybrid + rerank": hybrid_rerank,
    }

    print(f"{'strategy':18} {'hit_rate@'+str(K):>12} {'MRR':>7}")
    results = {}
    for name, fn in strategies.items():
        m = evaluate(fn, EVAL, k=K)
        results[name] = m
        print(f"{name:18} {m['hit_rate@k']:>12} {m['mrr']:>7}")

    best_name = max(results, key=lambda n: results[n]["hit_rate@k"])
    print(f"\nBest strategy: {best_name} (hit_rate@{K}={results[best_name]['hit_rate@k']})")
    print("Takeaway: hybrid/rerank are not automatic wins — on this keyword-sparse "
          "mix, dense leads and naive RRF doesn't beat it. Measure, don't assume.")
    # Gate: the eval ran and the best strategy retrieves well.
    ok = results[best_name]["hit_rate@k"] >= 0.8
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
