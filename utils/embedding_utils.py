"""
Embedding utility layer used by retrieval/search pipelines.

This module intentionally keeps a small stable interface for existing callsites
(`embed_query`, `embed_batch_cached`) while delegating actual vector generation to
`backend.services.embeddings`, which now supports OpenAI and Ollama providers.
"""

from __future__ import annotations

import numpy as np

from backend.services import embeddings as emb_service


EMBED_MODEL = emb_service.get_embedding_model()


def _norm(v: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D vector matrix/row vector."""
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def embedding_model_version() -> str:
    # Keep a version token for callers that want explicit cache keying semantics.
    return f"{emb_service.get_provider()}:{emb_service.get_embedding_model()}:{emb_service.get_embedding_dims()}"


def embed_query(text: str) -> np.ndarray:
    vec = np.array([emb_service.embed_query(text)], dtype=np.float32)
    return _norm(vec)


def embed_batch_cached(items: list[tuple[str, str]]) -> dict[str, np.ndarray]:
    """Embed text items while returning a dict by requested identifier.

    Input format: list of (id, text). If id is empty, the item is skipped.
    """
    out: dict[str, np.ndarray] = {}
    if not items:
        return out

    ids = [str(i) for i, _ in items if i]
    texts = [t for i, t in items if i]
    if not texts:
        return out

    vecs = emb_service.embed_documents(texts)
    for i, vec in zip(ids, vecs):
        arr = np.array([vec], dtype=np.float32)
        out[i] = _norm(arr)
    return out
