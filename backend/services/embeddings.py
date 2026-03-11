"""
Centralized embedding service for ScholarRAG.

Production intent:
- use one embedding model/provider contract everywhere
- keep query/document prefixing centralized
- keep storage metadata explicit to avoid mixing vectors from different models
- support local Ollama and remote Ollama-compatible HTTP endpoints
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, Iterable, List

import requests

from backend.services.db import execute, fetchall


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if isinstance(value, str) else default


EMBEDDING_PROVIDER = _env("EMBEDDING_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_EMBED_MODEL = _env("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
EMBEDDING_QUERY_PREFIX = _env(
    "EMBEDDING_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)
EMBEDDING_DOC_PREFIX = _env(
    "EMBEDDING_DOC_PREFIX",
    "Represent this document for retrieval: ",
)
EMBEDDING_BATCH_SIZE = int(_env("EMBEDDING_BATCH_SIZE", "16") or 16)
EMBEDDING_TIMEOUT_SECONDS = float(_env("EMBEDDING_TIMEOUT_SECONDS", _env("OLLAMA_EMBED_TIMEOUT", "30")) or 30)
EMBEDDING_RETRY_ATTEMPTS = int(_env("EMBEDDING_RETRY_ATTEMPTS", "3") or 3)
EMBEDDING_RETRY_DELAY = float(_env("EMBEDDING_RETRY_DELAY", "0.5") or 0.5)
EMBEDDING_VERSION = _env("EMBEDDING_VERSION", "mxbai-embed-large-v1")
EMBEDDING_MAX_QUERY_WORDS = int(_env("EMBEDDING_MAX_QUERY_WORDS", "128") or 128)
EMBEDDING_MAX_DOC_WORDS = int(_env("EMBEDDING_MAX_DOC_WORDS", "256") or 256)

# Keep store dimension configurable for backward-compatible pgvector schemas.
# mxbai-embed-large is 1024-d, but some existing DB schemas still use vector(1536).
EMBEDDING_RAW_DIM = int(_env("EMBEDDING_RAW_DIM", "1024") or 1024)
VECTOR_STORE_DIM = int(_env("VECTOR_STORE_DIM", "1536") or 1536)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_key(text: str, kind: str) -> str:
    material = "::".join(
        [
            EMBEDDING_PROVIDER,
            get_embedding_model(),
            get_embedding_version(),
            kind,
            text,
        ]
    )
    return _hash_text(material)


def _trim_or_pad(values: Iterable[float], dim: int) -> List[float]:
    if isinstance(values, str):
        raw = values.strip()
        if not raw:
            vec = []
        else:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    values = parsed
                else:
                    values = [x for x in raw.strip("[]").split(",") if x.strip()]
            except Exception:
                values = [x for x in raw.strip("[]").split(",") if x.strip()]
    vec = [float(v) for v in values]
    if len(vec) == dim:
        return vec
    if len(vec) > dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def _prepare_text(text: str, kind: str) -> str:
    cleaned = " ".join((text or "").split())
    max_words = EMBEDDING_MAX_QUERY_WORDS if kind == "query" else EMBEDDING_MAX_DOC_WORDS
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words])


def _validate_embedding_payload(payload: dict) -> List[float]:
    values = payload.get("embedding")
    if values is None and isinstance(payload.get("embeddings"), list):
        embeddings = payload.get("embeddings") or []
        if embeddings and isinstance(embeddings[0], list):
            values = embeddings[0]
    if not isinstance(values, list) or not values:
        raise RuntimeError("Embedding response missing `embedding` list.")
    try:
        out = [float(v) for v in values]
    except Exception as exc:
        raise RuntimeError("Embedding response contains non-numeric values.") from exc
    if len(out) < 128:
        raise RuntimeError(f"Embedding response too short ({len(out)} dims).")
    return out


def _extract_ollama_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            msg = payload.get("error") or payload.get("message") or payload.get("detail")
            if msg:
                return str(msg)
    except Exception:
        pass
    text = (response.text or "").strip()
    return text or f"HTTP {response.status_code}"


def _is_context_length_error(message: str) -> bool:
    msg = (message or "").lower()
    return "context length" in msg or "input length exceeds" in msg


def _post_ollama_embedding(text: str) -> List[float]:
    attempts = [
        (f"{OLLAMA_BASE_URL}/api/embed", {"model": OLLAMA_EMBED_MODEL, "input": [text]}),
        (f"{OLLAMA_BASE_URL}/api/embeddings", {"model": OLLAMA_EMBED_MODEL, "prompt": text}),
    ]
    last_error: Exception | None = None
    for url, body in attempts:
        try:
            response = requests.post(url, json=body, timeout=EMBEDDING_TIMEOUT_SECONDS)
            if response.status_code == 404:
                err = _extract_ollama_error(response)
                if "model" in err.lower() and "not found" in err.lower():
                    raise RuntimeError(
                        f"Ollama model `{OLLAMA_EMBED_MODEL}` is not installed on {OLLAMA_BASE_URL}. "
                        f"Run `ollama pull {OLLAMA_EMBED_MODEL}` on that host."
                    )
                last_error = RuntimeError(f"Ollama 404 from {url}: {err}")
                continue
            if response.status_code in (400, 413, 422, 500):
                err = _extract_ollama_error(response)
                runtime_err = RuntimeError(err)
                if _is_context_length_error(err):
                    raise runtime_err
                last_error = runtime_err
                continue
            response.raise_for_status()
            payload = response.json()
            return _validate_embedding_payload(payload)
        except RuntimeError as exc:
            if _is_context_length_error(str(exc)):
                raise
            last_error = exc
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Ollama embedding call failed without response.")


def _retry(fn, *args, **kwargs):
    last_err: Exception | None = None
    for attempt in range(1, max(1, EMBEDDING_RETRY_ATTEMPTS) + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_err = exc
            if attempt >= EMBEDDING_RETRY_ATTEMPTS:
                break
            time.sleep(EMBEDDING_RETRY_DELAY * attempt)
    if last_err is not None:
        raise RuntimeError(f"Embedding call failed: {last_err}") from last_err
    raise RuntimeError("Embedding call failed without exception.")


def _embed_single(text: str, kind: str) -> List[float]:
    if EMBEDDING_PROVIDER != "ollama":
        raise RuntimeError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")
    prefix = EMBEDDING_QUERY_PREFIX if kind == "query" else EMBEDDING_DOC_PREFIX
    prepared = _prepare_text(text, kind)
    words = prepared.split()
    limits = []
    initial = len(words)
    if initial:
        limits = [initial, min(initial, max(96, initial // 2)), min(initial, 128), min(initial, 96), min(initial, 64)]
    else:
        limits = [0]
    deduped_limits = []
    seen = set()
    for limit in limits:
        if limit not in seen:
            deduped_limits.append(limit)
            seen.add(limit)

    last_err: Exception | None = None
    for limit in deduped_limits:
        attempt_text = " ".join(words[:limit]) if limit > 0 else prepared
        try:
            raw = _retry(_post_ollama_embedding, f"{prefix}{attempt_text}")
            return _trim_or_pad(raw, VECTOR_STORE_DIM)
        except Exception as exc:
            last_err = exc
            if not _is_context_length_error(str(exc)):
                raise
            continue
    if last_err is not None:
        raise RuntimeError(
            f"Embedding failed after adaptive truncation for model {OLLAMA_EMBED_MODEL}: {last_err}"
        ) from last_err
    raise RuntimeError("Embedding failed without exception.")


def _fetch_cached(keys: List[str]) -> Dict[str, List[float]]:
    if not keys:
        return {}
    rows = fetchall(
        f"SELECT text_hash, embedding FROM embedding_cache WHERE text_hash IN ({','.join(['%s'] * len(keys))})",
        keys,
    )
    out: Dict[str, List[float]] = {}
    for row in rows:
        key = row.get("text_hash")
        if not key:
            continue
        out[key] = _trim_or_pad(row.get("embedding") or [], VECTOR_STORE_DIM)
    return out


def _store_cached(cache_key: str, embedding: List[float]) -> None:
    execute(
        """
        INSERT INTO embedding_cache (text_hash, dim, embedding, provider, model, embedding_version)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (text_hash) DO NOTHING
        """,
        [cache_key, VECTOR_STORE_DIM, embedding, get_provider(), get_embedding_model(), get_embedding_version()],
    )


def embed_query(text: str) -> List[float]:
    if not (text or "").strip():
        raise RuntimeError("embed_query requires non-empty text.")
    key = _cache_key(text, "query")
    cached = _fetch_cached([key]).get(key)
    if cached:
        return cached
    embedding = _embed_single(text, "query")
    _store_cached(key, embedding)
    return embedding


def embed_documents(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    results: List[List[float]] = []
    for start in range(0, len(texts), max(1, EMBEDDING_BATCH_SIZE)):
        batch = texts[start:start + max(1, EMBEDDING_BATCH_SIZE)]
        keys = [_cache_key(text, "document") for text in batch]
        cached = _fetch_cached(keys)
        for key, text in zip(keys, batch):
            emb = cached.get(key)
            if emb is None:
                emb = _embed_single(text, "document")
                _store_cached(key, emb)
            results.append(emb)
    return results


def healthcheck_embeddings() -> dict:
    diagnostics = {
        "provider": get_provider(),
        "model": get_embedding_model(),
        "embedding_version": get_embedding_version(),
        "raw_dim": EMBEDDING_RAW_DIM,
        "vector_store_dim": VECTOR_STORE_DIM,
        "base_url": OLLAMA_BASE_URL if get_provider() == "ollama" else None,
    }
    probe = embed_query("embedding healthcheck")
    diagnostics["ok"] = True
    diagnostics["returned_dim"] = len(probe)
    diagnostics["max_query_words"] = EMBEDDING_MAX_QUERY_WORDS
    diagnostics["max_doc_words"] = EMBEDDING_MAX_DOC_WORDS
    return diagnostics


# Backward-compatible wrappers for existing callsites.
def get_embedding(text: str) -> List[float]:
    return embed_query(text)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    return embed_documents(texts)


def get_provider() -> str:
    return EMBEDDING_PROVIDER


def get_embedding_model() -> str:
    return OLLAMA_EMBED_MODEL if EMBEDDING_PROVIDER == "ollama" else _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")


def get_embedding_version() -> str:
    return EMBEDDING_VERSION


def get_embedding_dims() -> int:
    return VECTOR_STORE_DIM


def get_raw_embedding_dims() -> int:
    return EMBEDDING_RAW_DIM
