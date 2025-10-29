from typing import Dict, List, Tuple, Optional

import faiss
import numpy as np

from utils.embedding_utils import embed_query, embed_batch_cached
from utils.openalex_utils import fetch_candidates_from_openalex


class Retriever:
    def __init__(self, index: faiss.Index, metadata: List[Dict]):
        self.index = index
        self.meta = metadata

    def search_faiss(self, query: str, topk: int = 50) -> List[Tuple[int, float]]:
        qv = embed_query(query)
        D, I = self.index.search(qv, topk)
        # FAISS returns distances; if index is IP with normalized vecs, higher is better
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]

    def _pool_from_faiss(self, hits: List[Tuple[int, float]]) -> List[Dict]:
        pool = []
        for idx, score in hits:
            if 0 <= idx < len(self.meta):
                m = dict(self.meta[idx])
                m["_faiss_score"] = score
                m["_local_id"] = idx
                pool.append(m)
        return pool

    def retrieve(self, query: str, k: int = 10, min_pool: int = 20, year_from: Optional[int] = None, year_to: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        hits = self.search_faiss(query, topk=max(50, k * 5))
        pool = self._pool_from_faiss(hits)
        fallback_used = False

        if len(pool) < min_pool or (hits and hits[0][1] < 0.25):
            fallback_used = True
            extra = fetch_candidates_from_openalex(query, limit=300, year_from=year_from, year_to=year_to)
            # Merge by a simple id/title key to avoid duplicates
            seen = { (p.get("id"), p.get("title")) for p in pool }
            for doc in extra:
                key = (doc.get("id"), doc.get("title"))
                if key not in seen:
                    pool.append(doc)
                    seen.add(key)

        # Semantic rerank
        texts: List[Tuple[str, str]] = []
        for i, doc in enumerate(pool):
            doc_id = str(doc.get("id") or doc.get("_local_id") or i)
            text = (doc.get("abstract") or doc.get("summary") or doc.get("title") or "")
            texts.append((doc_id, text))

        emb_map = embed_batch_cached(texts) if texts else {}
        qv = embed_query(query)

        scored = []
        for (doc_id, _), doc in zip(texts, pool):
            vec = emb_map.get(doc_id)
            if vec is None:
                continue
            sim = float(np.dot(qv, vec.T)[0][0])
            doc["_sim"] = sim
            scored.append(doc)

        scored.sort(key=lambda d: d.get("_sim", 0.0), reverse=True)
        topk_docs = scored[:k]

        stats = {
            "fallback_used": fallback_used,
            "candidate_counts": {"pool": len(pool), "scored": len(scored)},
        }
        return topk_docs, stats

