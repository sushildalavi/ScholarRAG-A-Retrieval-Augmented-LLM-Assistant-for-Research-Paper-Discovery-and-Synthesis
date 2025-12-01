import os
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils.config import get_openai_api_key
from openai import OpenAI


EMBED_MODEL = "text-embedding-3-small"  # faster/cheaper, good quality
EMBED_CACHE_ENABLED = os.getenv("EMBED_CACHE_ENABLED", "1") != "0"
EMBED_CACHE_PATH = Path(os.getenv("EMBED_CACHE_PATH", "data/embed_cache.sqlite"))
CHUNK_SIZE = 64
MAX_RETRIES = 3


_cache_conn = None


def _client() -> OpenAI:
    api_key = get_openai_api_key()
    return OpenAI(api_key=api_key)


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


@lru_cache(maxsize=1)
def embedding_model_version() -> str:
    return EMBED_MODEL


def _ensure_cache() -> sqlite3.Connection | None:
    global _cache_conn
    if not EMBED_CACHE_ENABLED:
        return None
    if _cache_conn is None:
        EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _cache_conn = sqlite3.connect(EMBED_CACHE_PATH)
        _cache_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
    return _cache_conn


def _cache_key(doc_id: str) -> str:
    return f"{doc_id}:{embedding_model_version()}"


def _cache_fetch(doc_ids: List[str]) -> Dict[str, np.ndarray]:
    conn = _ensure_cache()
    if conn is None or not doc_ids:
        return {}
    placeholders = ",".join(["?"] * len(doc_ids))
    keys = [_cache_key(doc_id) for doc_id in doc_ids]
    rows = conn.execute(
        f"SELECT cache_key, dim, vector FROM embeddings WHERE cache_key IN ({placeholders})",
        keys,
    ).fetchall()
    result: Dict[str, np.ndarray] = {}
    for key, dim, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32).reshape(1, dim)
        doc_id = key.rsplit(":", 1)[0]
        result[doc_id] = vec
    return result


def _cache_store(vectors: Dict[str, np.ndarray]) -> None:
    conn = _ensure_cache()
    if conn is None or not vectors:
        return
    rows = [(_cache_key(doc_id), vec.shape[1], vec.astype(np.float32).tobytes()) for doc_id, vec in vectors.items()]
    conn.executemany("REPLACE INTO embeddings (cache_key, dim, vector) VALUES (?, ?, ?)", rows)
    conn.commit()


def _retry_call(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(min(2 ** attempt, 10))


def embed_query(text: str) -> np.ndarray:
    resp = _retry_call(_client().embeddings.create, model=EMBED_MODEL, input=[text])
    v = np.array([resp.data[0].embedding], dtype=np.float32)
    return _norm(v)


def embed_batch_cached(items: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
    """Embed texts with persistent caching and retries."""
    out: Dict[str, np.ndarray] = {}
    doc_ids = [doc_id for doc_id, _ in items if doc_id]
    cached = _cache_fetch(doc_ids)
    out.update(cached)

    to_embed = [(doc_id, text) for doc_id, text in items if doc_id not in cached]
    if not to_embed:
        return out

    client = _client()
    for i in range(0, len(to_embed), CHUNK_SIZE):
        chunk = to_embed[i : i + CHUNK_SIZE]
        inputs = [text for _, text in chunk]
        if not inputs:
            continue
        resp = _retry_call(client.embeddings.create, model=EMBED_MODEL, input=inputs)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        vecs = _norm(vecs)
        for (doc_id, _), vec in zip(chunk, vecs):
            vec = vec.reshape(1, -1)
            out[doc_id] = vec
        _cache_store({doc_id: vec.reshape(1, -1) for (doc_id, _), vec in zip(chunk, vecs) if doc_id})
    return out
