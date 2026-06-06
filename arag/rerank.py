"""Cross-encoder reranking.

A bi-encoder (used for first-stage retrieval) embeds query and doc separately;
a cross-encoder scores the (query, doc) PAIR jointly, which is more accurate but
too slow to run over the whole corpus. Standard pattern: retrieve top-N with the
fast retriever, then rerank those N with the cross-encoder.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[str], texts: dict[str, str],
               k: int = 5) -> list[tuple[str, float]]:
        pairs = [(query, texts[doc_id]) for doc_id in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(doc_id, float(s)) for doc_id, s in ranked[:k]]
