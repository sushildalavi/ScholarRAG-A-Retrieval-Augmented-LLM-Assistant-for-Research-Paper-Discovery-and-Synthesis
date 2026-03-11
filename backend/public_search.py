import numpy as np
import os
import re
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.embedding_utils import embed_query, embed_batch_cached
from utils.openalex_utils import fetch_candidates_from_openalex
from utils.arxiv_utils import fetch_arxiv_candidates
from utils.crossref_utils import fetch_from_crossref
from utils.semanticscholar_utils import fetch_from_s2
from utils.springer_utils import fetch_from_springer
from utils.elsevier_utils import fetch_from_elsevier
from utils.ieee_utils import fetch_from_ieee

OPENALEX_LIMIT = int(os.getenv("PUBLIC_OPENALEX_LIMIT", "15")) or 15
ARXIV_LIMIT = int(os.getenv("PUBLIC_ARXIV_LIMIT", "15")) or 15
CROSSREF_LIMIT = int(os.getenv("PUBLIC_CROSSREF_LIMIT", "10")) or 10
S2_LIMIT = int(os.getenv("PUBLIC_S2_LIMIT", "10")) or 10
SPRINGER_LIMIT = int(os.getenv("PUBLIC_SPRINGER_LIMIT", "10")) or 10
ELSEVIER_LIMIT = int(os.getenv("PUBLIC_ELSEVIER_LIMIT", "10")) or 10
IEEE_LIMIT = int(os.getenv("PUBLIC_IEEE_LIMIT", "10")) or 10
PUBLIC_SPARSE_WEIGHT = float(os.getenv("PUBLIC_SPARSE_WEIGHT", "0.25"))
logger = logging.getLogger(__name__)
_DISABLED_PROVIDERS: set[str] = set()
_PUBLIC_SEARCH_CACHE: dict[tuple[str, str, int], tuple[float, dict]] = {}
PUBLIC_SEARCH_CACHE_TTL_SECONDS = int(os.getenv("PUBLIC_SEARCH_CACHE_TTL_SECONDS", "300")) or 300
PUBLIC_PROVIDER_MAX_WORKERS = int(os.getenv("PUBLIC_PROVIDER_MAX_WORKERS", "4")) or 4


def _normalize_public_query(query: str) -> str:
    """
    Normalize chatty user prompts into search-friendly keyword queries.
    """
    q = (query or "").strip().lower()
    if not q:
        return ""

    # Remove common prompt wrappers/noise.
    noise_phrases = (
        "give me",
        "fetch",
        "please",
        "can you",
        "i want",
        "show me",
        "find me",
        "relevant",
        "research papers",
        "research paper",
        "papers",
        "paper",
        "from ieee",
        "from springer",
        "from elsevier",
        "from arxiv",
        "from openalex",
        "from semantic scholar",
        "only",
    )
    for p in noise_phrases:
        q = q.replace(p, " ")

    q = re.sub(r"[^a-z0-9\s-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        return ""

    stop = {
        "about", "info", "information", "the", "and", "for", "with", "that", "this",
        "what", "who", "where", "when", "why", "how", "into", "using", "use",
    }
    toks = [t for t in q.split() if len(t) > 2 and t not in stop]
    if not toks:
        return q
    # Keep query focused but not too short.
    return " ".join(toks[:14])


def _query_variants(query: str) -> list[str]:
    core = _normalize_public_query(query)
    variants = []
    if core:
        variants.append(core)
    raw = (query or "").strip()
    if raw and raw not in variants:
        variants.append(raw)
    # Keep bounded to avoid over-calling providers.
    return variants[:2] if variants else []


def _tokenize_for_sparse(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2]


def _sparse_overlap_score(query: str, text: str) -> float:
    q = _tokenize_for_sparse(query)
    if not q:
        return 0.0
    t = set(_tokenize_for_sparse(text))
    if not t:
        return 0.0
    return len({x for x in q if x in t}) / max(1, len(set(q)))


def _fetch_provider(provider: str, query: str, limit: int) -> list[dict]:
    if provider in _DISABLED_PROVIDERS:
        return []
    if provider == "openalex" and OPENALEX_LIMIT > 0:
        return fetch_candidates_from_openalex(query, limit=min(limit, OPENALEX_LIMIT))
    if provider == "arxiv" and ARXIV_LIMIT > 0:
        return fetch_arxiv_candidates(query, limit=min(limit, ARXIV_LIMIT))
    if provider == "crossref" and CROSSREF_LIMIT > 0:
        return fetch_from_crossref(query, limit=min(limit, CROSSREF_LIMIT))
    if provider == "semanticscholar" and S2_LIMIT > 0:
        return fetch_from_s2(query, limit=min(limit, S2_LIMIT))
    if provider == "springer" and SPRINGER_LIMIT > 0:
        return fetch_from_springer(query, limit=min(limit, SPRINGER_LIMIT))
    if provider == "elsevier" and ELSEVIER_LIMIT > 0:
        return fetch_from_elsevier(query, limit=min(limit, ELSEVIER_LIMIT))
    if provider == "ieee" and IEEE_LIMIT > 0:
        return fetch_from_ieee(query, limit=min(limit, IEEE_LIMIT))
    return []


def public_live_search(query: str, k: int = 8, source_only: str | None = None, return_metadata: bool = False):
    """
    Fetch fresh candidates from external sources and rerank with embeddings + sparse overlap.
    """
    # trivial chatty queries: skip external search
    qnorm = (query or "").strip().lower()
    if not qnorm or len(qnorm) < 3 or qnorm in {"hi", "hello", "hey", "thanks", "thank you"}:
        return {"results": [], "provider_status": {}} if return_metadata else []

    provider = (source_only or "").strip().lower()
    query_variants = _query_variants(query)
    if not query_variants:
        return {"results": [], "provider_status": {}} if return_metadata else []
    primary_query = query_variants[0]
    cache_key = (primary_query, provider, int(k))
    cached = _PUBLIC_SEARCH_CACHE.get(cache_key)
    if cached and (time.time() - cached[0] <= PUBLIC_SEARCH_CACHE_TTL_SECONDS):
        payload = cached[1]
        if return_metadata:
            return {
                "results": list(payload.get("results", [])),
                "provider_status": dict(payload.get("provider_status", {})),
            }
        return list(payload.get("results", []))

    candidates = []
    provider_status: dict[str, dict] = {}
    if provider:
        for qv in query_variants:
            rows = _fetch_provider(provider, qv, limit=max(k * 3, 12))
            provider_status[provider] = {"queried": True, "variant": qv, "fetched": len(rows)}
            candidates += rows
            if len(candidates) >= max(k * 5, 20):
                break
    else:
        providers = ("openalex", "arxiv", "crossref", "semanticscholar", "springer", "elsevier", "ieee")
        limit = max(k * 2, 10)
        with ThreadPoolExecutor(max_workers=min(PUBLIC_PROVIDER_MAX_WORKERS, len(providers))) as executor:
            future_map = {
                executor.submit(_fetch_provider, p, primary_query, limit): p
                for p in providers
            }
            for future in as_completed(future_map):
                p = future_map[future]
                try:
                    rows = future.result() or []
                except Exception:
                    rows = []
                provider_status[p] = {"queried": True, "variant": primary_query, "fetched": len(rows)}
                candidates += rows

    # dedupe by DOI/id/title
    seen = set()
    uniq = []
    for c in candidates:
        doi = (c.get("doi") or "").strip().lower()
        cid = str(c.get("id") or "").strip().lower()
        title = (c.get("title") or "").strip().lower()
        key = doi or cid or title
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    if not uniq:
        return {"results": [], "provider_status": provider_status} if return_metadata else []

    texts = []
    ids = []
    sparse_vals = []
    for i, c in enumerate(uniq):
        text = f"{c.get('title', '')}\n{c.get('abstract') or c.get('summary') or ''}"
        texts.append(text)
        ids.append(i)
        sparse_vals.append(_sparse_overlap_score(query_variants[0], text))

    emb_map = embed_batch_cached(list(zip([str(i) for i in ids], texts)))
    qv = embed_query(query_variants[0])
    scored = []
    for i, c in enumerate(uniq):
        vec = emb_map.get(str(i))
        if vec is None:
            continue
        sim = float(np.dot(qv, vec.T)[0][0])
        sparse = float(sparse_vals[i])
        c["_sim"] = sim
        c["_sparse"] = sparse
        c["_hybrid"] = round((1.0 - PUBLIC_SPARSE_WEIGHT) * sim + PUBLIC_SPARSE_WEIGHT * sparse, 6)
        if not c.get("source"):
            c["source"] = "unknown_public"
        scored.append(c)

    scored.sort(key=lambda x: x.get("_hybrid", x.get("_sim", 0.0)), reverse=True)
    final_results = scored[:k]
    final_counts: dict[str, int] = {}
    for row in final_results:
        src = (row.get("source") or "unknown_public").lower()
        final_counts[src] = final_counts.get(src, 0) + 1
    for provider_name, meta in provider_status.items():
        meta["selected"] = final_counts.get(provider_name, 0)
        meta["contributed"] = meta["selected"] > 0
    logger.info(
        "public_search provider status query=%r status=%s",
        query_variants[0] if query_variants else query,
        provider_status,
    )
    _PUBLIC_SEARCH_CACHE[cache_key] = (
        time.time(),
        {"results": list(final_results), "provider_status": dict(provider_status)},
    )
    if return_metadata:
        return {"results": final_results, "provider_status": provider_status}
    return final_results
