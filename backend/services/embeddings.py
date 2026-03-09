"""
Embedding provider abstraction used across ScholarRAG.

Supports:
- OpenAI embeddings (default)
- Ollama embeddings (`OLLAMA_EMBED_MODEL`)

The module keeps existing cache-backed API so current callers keep working.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from typing import Dict, Iterable, List

from backend.services.db import execute, fetchall

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

from utils.config import get_openai_api_key


EMBED_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBED_DIM = int(os.getenv("OPENAI_EMBEDDING_DIM", "1536") or 1536)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_EMBED_DIM = int(os.getenv("OLLAMA_EMBED_DIM", "768") or 768)
# Primary vector size expected by persisted schemas by default.
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536") or 1536)

OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_EMBED_TIMEOUT", "15") or 15)
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_EMBED_TIMEOUT", "25") or 25)
STUB_LOG_STEPS = int(os.getenv("EMBED_STUB_LOG_STEPS", "1") or 1)


def text_to_hash(text: str) -> str:
    """Stable SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _trim_or_pad_vector(values: Iterable[float], dim: int) -> List[float]:
    vec = [float(v) for v in values]
    if dim <= 0:
        return vec
    if len(vec) == dim:
        return vec
    if len(vec) > dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def _safe_list(value) -> List[float]:
    if value is None:
        return []
    if isinstance(value, list):
        try:
            return [float(v) for v in value]
        except Exception:
            return []
    # pgvector and other drivers can emit custom sequence-like objects.
    try:
        return [float(v) for v in list(value)]
    except Exception:
        return []


def _client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Install with `pip install openai`.")
    return OpenAI(api_key=get_openai_api_key())


def _deterministic_stub(text: str) -> List[float]:
    """Deterministic fallback embedding based on text hash."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    for i in range(max(1, EMBED_DIM)):
        byte = h[i % len(h)]
        vals.append((byte / 127.5) - 1.0)
    return vals


def _call_openai_embedding(text: str) -> List[float]:
    client = _client()
    response = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[text], timeout=OPENAI_TIMEOUT_SECONDS)
    return _safe_list(response.data[0].embedding)


def _call_openai_embedding_batch(texts: List[str]) -> List[List[float]]:
    client = _client()
    response = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts, timeout=max(OPENAI_TIMEOUT_SECONDS, 20))
    out = []
    for item in getattr(response, "data", []) or []:
        out.append(_safe_list(item.embedding))
    if len(out) != len(texts):
        # Defensive fallback if the API returned partial or malformed order.
        return [_safe_list(item.embedding) for item in (response.data if hasattr(response, "data") else [])]
    return out


def _call_ollama_embedding(text: str) -> List[float]:
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "prompt": text,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw or "{}")
    return _safe_list(parsed.get("embedding"))


def _call_ollama_embedding_batch(texts: List[str]) -> List[List[float]]:
    # Ollama server can be reliable but we keep a simple per-text loop to avoid
    # model-specific payload variants and preserve error granularity.
    out = []
    for i, text in enumerate(texts, start=1):
        vec = _retry_call(_call_ollama_embedding, text)
        out.append(vec)
        if STUB_LOG_STEPS > 0 and i % max(1, STUB_LOG_STEPS) == 0:
            # Small throttling hook for high-volume batches.
            time.sleep(0.0)
    return out


def _retry_call(fn, *args, **kwargs):
    attempts = int(os.getenv("EMBEDDING_RETRY_ATTEMPTS", "2") or 2)
    delay = float(os.getenv("EMBEDDING_RETRY_DELAY", "0.4") or 0.4)
    last_err: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - integration defensive path
            last_err = exc
            if attempt >= attempts:
                raise
            time.sleep(delay)
    if last_err is not None:
        raise last_err
    raise RuntimeError("embedding retry failed")


def call_embedding_api(text: str) -> List[float]:
    """Call selected provider for a single embedding vector."""
    model_dim = OPENAI_EMBED_DIM if EMBED_PROVIDER == "openai" else OLLAMA_EMBED_DIM
    dim = EMBED_DIM if EMBED_DIM > 0 else max(1, model_dim)
    try:
        if EMBED_PROVIDER == "ollama":
            return _trim_or_pad_vector(_retry_call(_call_ollama_embedding, text), dim)
        return _trim_or_pad_vector(_retry_call(_call_openai_embedding, text), dim)
    except Exception as exc:  # pragma: no cover - runtime fallback for robustness
        import logging

        logging.warning("Embedding API failed (%s). Using deterministic fallback.", exc)
        return _trim_or_pad_vector(_deterministic_stub(text), dim)


def call_embedding_api_batch(texts: List[str]) -> List[List[float]]:
    """Call selected provider for a batch of embeddings."""
    if not texts:
        return []

    dim = EMBED_DIM if EMBED_DIM > 0 else (OPENAI_EMBED_DIM if EMBED_PROVIDER == "openai" else OLLAMA_EMBED_DIM)
    try:
        if EMBED_PROVIDER == "ollama":
            values = _call_ollama_embedding_batch(texts)
        else:
            values = _call_openai_embedding_batch(texts)
        return [_trim_or_pad_vector(v, dim) for v in values]
    except Exception as exc:  # pragma: no cover - runtime fallback for robustness
        import logging

        logging.warning("Batch embedding API failed (%s). Using deterministic fallbacks.", exc)
        return [_trim_or_pad_vector(_deterministic_stub(t), dim) for t in texts]


def get_embedding(text: str) -> List[float]:
    """Fetch embedding from cache or compute and store."""
    key = text_to_hash(text)
    rows = fetchall("SELECT embedding FROM embedding_cache WHERE text_hash=%s", [key])
    if rows:
        emb = _safe_list(rows[0].get("embedding"))
        return _trim_or_pad_vector(emb, max(1, EMBED_DIM))

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
        text_hash = row.get("text_hash")
        if not text_hash:
            continue
        out[text_hash] = _trim_or_pad_vector(_safe_list(row.get("embedding")), EMBED_DIM)
    return out


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Fetch embeddings for many texts using cache + batch API calls."""
    if not texts:
        return []

    keys = [text_to_hash(text) for text in texts]
    cached = _fetch_cached_embeddings(keys)
    out_by_key: Dict[str, List[float]] = dict(cached)

    missing_texts: List[str] = []
    missing_keys: List[str] = []
    for key, text in zip(keys, texts):
        if key in out_by_key:
            continue
        missing_keys.append(key)
        missing_texts.append(text)

    if missing_texts:
        computed = call_embedding_api_batch(missing_texts)
        for key, emb in zip(missing_keys, computed):
            emb = _trim_or_pad_vector(emb, EMBED_DIM)
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


def get_provider() -> str:
    return EMBED_PROVIDER


def get_embedding_model() -> str:
    if EMBED_PROVIDER == "ollama":
        return OLLAMA_EMBED_MODEL
    return OPENAI_EMBED_MODEL


def get_embedding_dims() -> int:
    return EMBED_DIM
