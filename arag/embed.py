"""Sentence-transformer embeddings (all-MiniLM-L6-v2), L2-normalized."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")
