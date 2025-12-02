import numpy as np

from utils.embedding_utils import embed_query, embed_batch_cached
from utils.openalex_utils import fetch_candidates_from_openalex
from utils.arxiv_utils import fetch_arxiv_candidates
from utils.crossref_utils import fetch_from_crossref


def public_live_search(query: str, k: int = 8):
    """
    Fetch fresh candidates from external sources and rerank with embeddings (no FAISS).
    """
    candidates = []
    candidates += fetch_candidates_from_openalex(query, limit=15)
    candidates += fetch_arxiv_candidates(query, limit=15)
    candidates += fetch_from_crossref(query, limit=10)

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
        scored.append(c)
    scored.sort(key=lambda x: x.get("_sim", 0), reverse=True)
    return scored[:k]
