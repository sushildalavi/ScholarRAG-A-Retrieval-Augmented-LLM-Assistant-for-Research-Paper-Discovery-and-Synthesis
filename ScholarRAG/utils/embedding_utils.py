import hashlib
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import numpy as np

from utils.config import get_openai_api_key
from openai import OpenAI


EMBED_MODEL = "text-embedding-3-large"


def _client() -> OpenAI:
    api_key = get_openai_api_key()
    return OpenAI(api_key=api_key)


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


@lru_cache(maxsize=1)
def embedding_model_version() -> str:
    return EMBED_MODEL


def embed_query(text: str) -> np.ndarray:
    resp = _client().embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.array([resp.data[0].embedding], dtype=np.float32)
    return _norm(v)


def embed_batch_cached(items: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
    """
    Embed a batch of texts with a simple in-process cache.

    items: list of (doc_id, text)
    returns: {doc_id: normalized_vector}
    """
    client = _client()
    out: Dict[str, np.ndarray] = {}

    # Simple chunking to respect API limits
    CHUNK = 64
    for i in range(0, len(items), CHUNK):
        chunk = items[i : i + CHUNK]
        inputs = [t for _, t in chunk]
        resp = client.embeddings.create(model=EMBED_MODEL, input=inputs)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        vecs = _norm(vecs)
        for (doc_id, _), vec in zip(chunk, vecs):
            out[doc_id] = vec.reshape(1, -1)
    return out

