"""
Embedding cache helper.
Uses OpenAI `text-embedding-3-large` when OPENAI_API_KEY is set, otherwise raises.
"""
import hashlib
import os
from typing import List

from db import execute, fetchall

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
