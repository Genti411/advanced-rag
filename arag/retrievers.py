"""Retrievers: dense (embeddings), sparse (BM25), and a hybrid that fuses both.

Dense embeddings capture paraphrase/semantics while BM25 nails exact keyword/
rare-term matches, so hybrid retrieval is *often* better — but not always (see the
README's eval finding). We fuse the two ranked lists with Reciprocal Rank Fusion
(RRF), which needs no score normalization.
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


class DenseRetriever:
    def __init__(self, doc_ids: list[str], embeddings: np.ndarray, embedder):
        self.doc_ids = doc_ids
        self.emb = embeddings              # (N, d), L2-normalized
        self.embedder = embedder

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = self.embedder.embed([query])[0]
        scores = self.emb @ q              # cosine (normalized)
        order = np.argsort(-scores)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in order]


class BM25Retriever:
    def __init__(self, doc_ids: list[str], texts: list[str]):
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi([_tokenize(t) for t in texts])

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = np.argsort(-scores)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in order]


class HybridRetriever:
    """Reciprocal Rank Fusion of dense + BM25 rankings. rrf_k smooths the curve."""
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, rrf_k: int = 60):
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        fused: dict[str, float] = {}
        for retriever in (self.dense, self.bm25):
            for rank, (doc_id, _score) in enumerate(retriever.search(query, k=20)):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]
