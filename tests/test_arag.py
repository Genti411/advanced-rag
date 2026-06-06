import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arag.evaluate import evaluate
from arag.retrievers import BM25Retriever, HybridRetriever, _tokenize


def test_bm25_keyword_retrieval():
    ids = ["a", "b", "c"]
    texts = ["sql injection parameterized queries", "cross site scripting javascript", "ddos traffic flood"]
    bm = BM25Retriever(ids, texts)
    assert bm.search("sql injection", k=1)[0][0] == "a"
    assert bm.search("scripting javascript", k=1)[0][0] == "b"


class _FakeRetriever:
    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, _q, k=20):
        return [(d, 1.0) for d in self.ranking[:k]]


def test_rrf_fusion_rewards_top_finishes():
    dense = _FakeRetriever(["a", "b", "c"])   # a is #1 in dense
    bm25 = _FakeRetriever(["c", "b", "a"])     # c is #1 in bm25
    hybrid = HybridRetriever(dense, bm25, rrf_k=60)
    ids = [d for d, _ in hybrid.search("q", k=3)]
    # a and c each have a #1 finish; b is only ever #2 -> b ends last under RRF
    assert set(ids[:2]) == {"a", "c"}
    assert ids[2] == "b"


def test_evaluate_hit_rate_and_mrr():
    eval_set = [("q1", "a"), ("q2", "x")]

    def sf(q, k):
        return [("a", 1.0), ("b", 0.5)] if q == "q1" else [("y", 1.0), ("z", 0.5)]

    m = evaluate(sf, eval_set, k=2)
    assert m["hit_rate@k"] == 0.5     # q1 hit, q2 miss
    assert m["mrr"] == 0.5            # q1 reciprocal rank 1.0, q2 0.0


def test_tokenize():
    assert _tokenize("SQL-injection, attack!") == ["sql", "injection", "attack"]
