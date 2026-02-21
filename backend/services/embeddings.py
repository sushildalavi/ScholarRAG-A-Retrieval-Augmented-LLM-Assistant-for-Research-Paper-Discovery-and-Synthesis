"""
Embedding cache helper.
Uses OpenAI `text-embedding-3-large` when OPENAI_API_KEY is set, otherwise raises.
"""
import hashlib
import os
from typing import Dict, List

from backend.services.db import execute, fetchall

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

EMBED_DIM = 1536  # matches text-embedding-3-small (better speed/cost)
EMBED_MODEL = "text-embedding-3-small"


def text_to_hash(text: str) -> str:
    """Stable SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Install with `pip install openai`.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


def _deterministic_stub(text: str) -> List[float]:
    """Deterministic fallback embedding based on text hash."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand to EMBED_DIM floats
    vals = []
    for i in range(EMBED_DIM):
        # simple hash-based pseudo-random in [-1,1]
        byte = h[i % len(h)]
        vals.append((byte / 127.5) - 1.0)
    return vals


def call_embedding_api(text: str) -> List[float]:
    """
    Call OpenAI embedding API. Replace/extend for another provider if desired.
    """
    try:
        client = _client()
        resp = client.embeddings.create(model=EMBED_MODEL, input=[text], timeout=15)
        return resp.data[0].embedding
    except Exception as exc:  # pragma: no cover - fallback for robustness
        # Fallback to deterministic stub so the pipeline continues even if OpenAI fails.
        import logging
        logging.warning("Embedding API failed (%s); using deterministic stub.", exc)
        return _deterministic_stub(text)


def call_embedding_api_batch(texts: List[str]) -> List[List[float]]:
    """
    Batch embedding API call with deterministic fallback.
    """
    if not texts:
        return []
    try:
        client = _client()
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts, timeout=30)
        return [d.embedding for d in resp.data]
    except Exception as exc:  # pragma: no cover - fallback for robustness
        import logging
        logging.warning("Batch embedding API failed (%s); using deterministic stubs.", exc)
        return [_deterministic_stub(text) for text in texts]


def get_embedding(text: str) -> List[float]:
    """Fetch embedding from cache or compute and store."""
    key = text_to_hash(text)
    rows = fetchall("SELECT embedding FROM embedding_cache WHERE text_hash=%s", [key])
    if rows:
        return rows[0]["embedding"]

    emb = call_embedding_api(text)
    execute(
        """
        INSERT INTO embedding_cache (text_hash, dim, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (text_hash) DO NOTHING
        """,
        [key, EMBED_DIM, emb],
    )
    return emb


def _fetch_cached_embeddings(keys: List[str]) -> Dict[str, List[float]]:
    if not keys:
        return {}
    placeholders = ",".join(["%s"] * len(keys))
    rows = fetchall(
        f"SELECT text_hash, embedding FROM embedding_cache WHERE text_hash IN ({placeholders})",
        keys,
    )
    out: Dict[str, List[float]] = {}
    for row in rows:
        out[row["text_hash"]] = row["embedding"]
    return out


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Fetch embeddings for many texts using cache + batched API requests.
    Returns embeddings in the same order as `texts`.
    """
    if not texts:
        return []

    keys = [text_to_hash(text) for text in texts]
    cached = _fetch_cached_embeddings(keys)
    out_by_key: Dict[str, List[float]] = dict(cached)

    missing_keys: List[str] = []
    missing_texts: List[str] = []
    for key, text in zip(keys, texts):
        if key in out_by_key:
            continue
        missing_keys.append(key)
        missing_texts.append(text)

    if missing_texts:
        computed = call_embedding_api_batch(missing_texts)
        for key, emb in zip(missing_keys, computed):
            out_by_key[key] = emb
            execute(
                """
                INSERT INTO embedding_cache (text_hash, dim, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (text_hash) DO NOTHING
                """,
                [key, EMBED_DIM, emb],
            )

    return [out_by_key[key] for key in keys]
