import numpy as np
import os

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


def public_live_search(query: str, k: int = 8, source_only: str | None = None):
    """
    Fetch fresh candidates from external sources and rerank with embeddings (no FAISS).
    """
    # trivial chatty queries: skip external search
    qnorm = (query or "").strip().lower()
    if not qnorm or len(qnorm) < 3 or qnorm in {"hi", "hello", "hey", "thanks", "thank you"}:
        return []

    provider = (source_only or "").strip().lower()
    candidates = []
    if provider:
        if provider == "openalex" and OPENALEX_LIMIT > 0:
            candidates += fetch_candidates_from_openalex(query, limit=OPENALEX_LIMIT)
        elif provider == "arxiv" and ARXIV_LIMIT > 0:
            candidates += fetch_arxiv_candidates(query, limit=ARXIV_LIMIT)
        elif provider == "crossref" and CROSSREF_LIMIT > 0:
            candidates += fetch_from_crossref(query, limit=CROSSREF_LIMIT)
        elif provider == "semanticscholar" and S2_LIMIT > 0:
            candidates += fetch_from_s2(query, limit=S2_LIMIT)
        elif provider == "springer" and SPRINGER_LIMIT > 0:
            candidates += fetch_from_springer(query, limit=SPRINGER_LIMIT)
        elif provider == "elsevier" and ELSEVIER_LIMIT > 0:
            candidates += fetch_from_elsevier(query, limit=ELSEVIER_LIMIT)
        elif provider == "ieee" and IEEE_LIMIT > 0:
            candidates += fetch_from_ieee(query, limit=IEEE_LIMIT)
    else:
        if OPENALEX_LIMIT > 0:
            candidates += fetch_candidates_from_openalex(query, limit=OPENALEX_LIMIT)
        if ARXIV_LIMIT > 0:
            candidates += fetch_arxiv_candidates(query, limit=ARXIV_LIMIT)
        if CROSSREF_LIMIT > 0:
            candidates += fetch_from_crossref(query, limit=CROSSREF_LIMIT)
        if S2_LIMIT > 0:
            candidates += fetch_from_s2(query, limit=S2_LIMIT)
        if SPRINGER_LIMIT > 0:
            candidates += fetch_from_springer(query, limit=SPRINGER_LIMIT)
        if ELSEVIER_LIMIT > 0:
            candidates += fetch_from_elsevier(query, limit=ELSEVIER_LIMIT)
        if IEEE_LIMIT > 0:
            candidates += fetch_from_ieee(query, limit=IEEE_LIMIT)

    # dedupe by title
    seen = set()
    uniq = []
    for c in candidates:
        title = (c.get("title") or "").strip().lower()
        if not title or title in seen:
            continue
        seen.add(title)
        uniq.append(c)

    if not uniq:
        return []

    texts = []
    ids = []
    for i, c in enumerate(uniq):
        title = c.get("title") or ""
        abs_ = c.get("abstract") or c.get("summary") or ""
        texts.append(f"{title}\n{abs_}")
        ids.append(i)

    emb_map = embed_batch_cached(list(zip([str(i) for i in ids], texts)))
    qv = embed_query(query)
    scored = []
    for i, c in enumerate(uniq):
        vec = emb_map.get(str(i))
        if vec is None:
            continue
        sim = float(np.dot(qv, vec.T)[0][0])
        c["_sim"] = sim
        if not c.get("source"):
            c["source"] = "unknown_public"
        scored.append(c)
    scored.sort(key=lambda x: x.get("_sim", 0), reverse=True)
    return scored[:k]
