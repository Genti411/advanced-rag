"""Retrieval evaluation metrics.

hit_rate@k  — fraction of queries whose relevant doc is in the top-k (recall).
MRR         — mean reciprocal rank of the relevant doc (rewards ranking it high).
These are deterministic (no LLM judge), so retrieval quality is measured exactly.
"""
from __future__ import annotations

from typing import Callable


def evaluate(search_fn: Callable[[str, int], list], eval_set: list[tuple[str, str]],
             k: int = 5) -> dict:
    hits = 0
    rr_sum = 0.0
    for query, relevant in eval_set:
        ids = [doc_id for doc_id, _score in search_fn(query, k)]
        if relevant in ids:
            hits += 1
            rr_sum += 1.0 / (ids.index(relevant) + 1)
    n = len(eval_set)
    return {"hit_rate@k": round(hits / n, 3), "mrr": round(rr_sum / n, 3), "n": n, "k": k}
