import logging
import os
import re
from typing import Dict, List, Tuple, Optional

import faiss
import numpy as np

# Optional cross-encoder reranking
USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "false").lower() == "true"
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_TOPN = int(os.getenv("CROSS_ENCODER_TOPN", "30")) or 30

# Lightweight lexical blend to boost exact matches (names, years, keywords)
USE_LEXICAL_BLEND = os.getenv("USE_LEXICAL_BLEND", "true").lower() == "true"
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", "0.12"))

if USE_CROSS_ENCODER:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # noqa
        logging.warning("Cross-encoder requested but transformers/torch not available: %s", exc)
        USE_CROSS_ENCODER = False

from utils.embedding_utils import embed_query, embed_batch_cached
from utils.openalex_utils import fetch_candidates_from_openalex
from utils.arxiv_utils import fetch_arxiv_candidates
from utils.crossref_utils import fetch_from_crossref
from utils.semanticscholar_utils import fetch_from_s2
from utils.springer_utils import fetch_from_springer
from utils.elsevier_utils import fetch_from_elsevier
from utils.ieee_utils import fetch_from_ieee

# Tighter defaults to improve latency and avoid slow/blocked sources by default.
OPENALEX_FALLBACK_LIMIT = int(os.getenv("OPENALEX_FALLBACK_LIMIT", "20")) or 20
ARXIV_FALLBACK_LIMIT = int(os.getenv("ARXIV_FALLBACK_LIMIT", "20")) or 20
S2_FALLBACK_LIMIT = int(os.getenv("S2_FALLBACK_LIMIT", "0")) or 0
CROSSREF_FALLBACK_LIMIT = int(os.getenv("CROSSREF_FALLBACK_LIMIT", "20")) or 20
SPRINGER_FALLBACK_LIMIT = int(os.getenv("SPRINGER_FALLBACK_LIMIT", "10")) or 10
ELSEVIER_FALLBACK_LIMIT = int(os.getenv("ELSEVIER_FALLBACK_LIMIT", "0")) or 0
IEEE_FALLBACK_LIMIT = int(os.getenv("IEEE_FALLBACK_LIMIT", "0")) or 0
MULTI_HOP_MAX = int(os.getenv("MULTI_HOP_MAX", "2"))
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, index: faiss.Index, metadata: List[Dict]):
        self.index = index
        self.meta = metadata
        self.cross_encoder = None
        self.cross_tokenizer = None
        if USE_CROSS_ENCODER:
            try:
                self.cross_tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
                self.cross_encoder = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL)
                self.cross_encoder.eval()
                logger.info("Loaded cross-encoder model: %s", CROSS_ENCODER_MODEL)
            except Exception as exc:  # noqa
                logger.warning("Failed to load cross-encoder %s: %s", CROSS_ENCODER_MODEL, exc)
                self.cross_encoder = None
                self.cross_tokenizer = None

    @staticmethod
    def _year_within(doc: Dict, year_from: Optional[int], year_to: Optional[int]) -> bool:
        if year_from is None and year_to is None:
            return True
        year = doc.get("year") or doc.get("publication_year")
        if year is None:
            return True
        if year_from is not None and int(year) < int(year_from):
            return False
        if year_to is not None and int(year) > int(year_to):
            return False
        return True

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

    def _dedup_merge(self, pool: List[Dict], extra: List[Dict]) -> int:
        added = 0
        seen = {(p.get("id"), p.get("title")) for p in pool}
        seen.update({(p.get("doi"), p.get("title")) for p in pool if p.get("doi")})
        for doc in extra:
            key = (doc.get("id") or doc.get("doi"), doc.get("title"))
            if key not in seen:
                pool.append(doc)
                seen.add(key)
                added += 1
        return added

    def _semantic_rerank(self, query: str, pool: List[Dict], k: int) -> Tuple[List[Dict], Dict]:
        texts: List[Tuple[str, str]] = []
        for i, doc in enumerate(pool):
            doc_id = str(doc.get("id") or doc.get("_local_id") or i)
            text = (doc.get("abstract") or doc.get("summary") or doc.get("title") or "")
            texts.append((doc_id, text))

        emb_map = embed_batch_cached(texts) if texts else {}
        qv = embed_query(query)

        def lexical_overlap_score(q: str, t: str) -> float:
            if not USE_LEXICAL_BLEND:
                return 0.0
            qtoks = {w for w in re.findall(r"[a-zA-Z0-9]+", q.lower()) if len(w) > 2}
            ttoks = {w for w in re.findall(r"[a-zA-Z0-9]+", t.lower()) if len(w) > 2}
            if not qtoks or not ttoks:
                return 0.0
            return len(qtoks & ttoks) / max(len(qtoks), 1)

        scored = []
        for (doc_id, _), doc in zip(texts, pool):
            vec = emb_map.get(doc_id)
            if vec is None:
                continue
            sim = float(np.dot(qv, vec.T)[0][0])
            lex = lexical_overlap_score(query, doc.get("abstract") or doc.get("summary") or doc.get("title") or "")
            doc["_sim"] = sim
            doc["_lex"] = lex
            doc["_score"] = sim + (LEXICAL_WEIGHT * lex)
            scored.append(doc)

        scored.sort(key=lambda d: d.get("_score", d.get("_sim", 0.0)), reverse=True)
        return scored[:k], {"pool": len(pool), "scored": len(scored)}

    def _cross_encoder_rerank(self, query: str, pool: List[Dict], k: int) -> Tuple[List[Dict], Dict]:
        """Use a cross-encoder to rerank the pool. Falls back to semantic if CE unavailable."""
        if self.cross_encoder is None or self.cross_tokenizer is None:
            return self._semantic_rerank(query, pool, k)

        topn = min(CROSS_ENCODER_TOPN, len(pool))
        candidates = pool[:topn]

        pairs = []
        for i, doc in enumerate(candidates):
            doc_id = str(doc.get("id") or doc.get("_local_id") or i)
            text = (doc.get("title") or "") + "\n" + (doc.get("abstract") or doc.get("summary") or "")
            pairs.append((doc_id, text, doc))

        if not pairs:
            return [], {"pool": 0, "scored": 0}

        scores = []
        batch_size = 16
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            enc = self.cross_tokenizer(
                [query] * len(chunk),
                [t[1] for t in chunk],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.cross_encoder(**enc).logits.squeeze(-1)
                batch_scores = logits.cpu().numpy().tolist()
            for (doc_id, _, doc), s in zip(chunk, batch_scores):
                doc["_ce"] = float(s)
                scores.append(doc)

        scores.sort(key=lambda d: d.get("_ce", 0.0), reverse=True)
        return scores[:k], {"pool": len(pool), "scored": len(scores)}

    def _single_hop(self, query: str, k: int, min_pool: int, year_from: Optional[int], year_to: Optional[int]) -> Tuple[List[Dict], Dict]:
        hits = self.search_faiss(query, topk=max(50, k * 5))
        pool = [doc for doc in self._pool_from_faiss(hits) if self._year_within(doc, year_from, year_to)]
        fallback_used = False
        added_from_openalex = 0
        added_from_arxiv = 0
        added_from_crossref = 0
        added_from_s2 = 0
        added_from_springer = 0
        added_from_elsevier = 0
        added_from_ieee = 0

        if len(pool) < min_pool or (hits and hits[0][1] < 0.25):
            fallback_used = True
            if OPENALEX_FALLBACK_LIMIT > 0:
                extra_openalex = fetch_candidates_from_openalex(query, limit=OPENALEX_FALLBACK_LIMIT, year_from=year_from, year_to=year_to)
                added_from_openalex = self._dedup_merge(pool, [d for d in extra_openalex if self._year_within(d, year_from, year_to)])

            if ARXIV_FALLBACK_LIMIT > 0:
                extra_arxiv = fetch_arxiv_candidates(query, limit=ARXIV_FALLBACK_LIMIT, year_from=year_from, year_to=year_to)
                added_from_arxiv = self._dedup_merge(pool, [d for d in extra_arxiv if self._year_within(d, year_from, year_to)])

            if CROSSREF_FALLBACK_LIMIT > 0:
                extra_crossref = fetch_from_crossref(query, limit=CROSSREF_FALLBACK_LIMIT, year_from=year_from, year_to=year_to)
                added_from_crossref = self._dedup_merge(pool, [d for d in extra_crossref if self._year_within(d, year_from, year_to)])

            if S2_FALLBACK_LIMIT > 0:
                extra_s2 = fetch_from_s2(query, limit=S2_FALLBACK_LIMIT, year_from=year_from, year_to=year_to)
                added_from_s2 = self._dedup_merge(pool, [d for d in extra_s2 if self._year_within(d, year_from, year_to)])

            if SPRINGER_FALLBACK_LIMIT > 0:
                extra_springer = fetch_from_springer(query, limit=SPRINGER_FALLBACK_LIMIT, year_from=year_from, year_to=year_to)
                added_from_springer = self._dedup_merge(pool, [d for d in extra_springer if self._year_within(d, year_from, year_to)])

            if ELSEVIER_FALLBACK_LIMIT > 0:
                extra_elsevier = fetch_from_elsevier(query, limit=ELSEVIER_FALLBACK_LIMIT, year_from=year_from, year_to=year_to)
                added_from_elsevier = self._dedup_merge(pool, [d for d in extra_elsevier if self._year_within(d, year_from, year_to)])

            if IEEE_FALLBACK_LIMIT > 0:
                extra_ieee = fetch_from_ieee(query, limit=IEEE_FALLBACK_LIMIT, year_from=year_from, year_to=year_to)
                added_from_ieee = self._dedup_merge(pool, [d for d in extra_ieee if self._year_within(d, year_from, year_to)])

        topk_docs, counters = self._cross_encoder_rerank(query, pool, k)

        stats = {
            "fallback_used": fallback_used,
            "candidate_counts": {
                **counters,
                "openalex": added_from_openalex,
                "arxiv": added_from_arxiv,
                "crossref": added_from_crossref,
                "semanticscholar": added_from_s2,
                "springer": added_from_springer,
                "elsevier": added_from_elsevier,
                "ieee": added_from_ieee,
            },
        }
        return topk_docs, stats

    def retrieve(
        self,
        query: str,
        k: int = 10,
        min_pool: int = 20,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        multi_hop: bool = True,
    ) -> Tuple[List[Dict], Dict]:
        if not multi_hop:
            return self._single_hop(query, k, min_pool, year_from, year_to)

        # Heuristic: split complex queries into sub-queries on conjunctions/commas
        pieces = [p.strip() for p in query.replace("?", ".").split(".") if p.strip()]
        if len(pieces) == 0:
            pieces = [query]
        hops = pieces[: max(MULTI_HOP_MAX, 1)]

        merged_pool: List[Dict] = []
        hop_stats: List[Dict] = []
        for hop_q in hops:
            hop_docs, stats = self._single_hop(hop_q, k=max(k * 2, 20), min_pool=min_pool, year_from=year_from, year_to=year_to)
            hop_stats.append({"query": hop_q, **stats})
            self._dedup_merge(merged_pool, hop_docs)

        top_docs, counters = self._cross_encoder_rerank(query, merged_pool, k)
        any_fallback = any(s.get("fallback_used") for s in hop_stats)
        stats = {
            "fallback_used": any_fallback,
            "candidate_counts": {
                "pool": counters.get("pool"),
                "scored": counters.get("scored"),
                "openalex": sum(s["candidate_counts"].get("openalex", 0) for s in hop_stats),
                "arxiv": sum(s["candidate_counts"].get("arxiv", 0) for s in hop_stats),
                "crossref": sum(s["candidate_counts"].get("crossref", 0) for s in hop_stats),
                "semanticscholar": sum(s["candidate_counts"].get("semanticscholar", 0) for s in hop_stats),
                "springer": sum(s["candidate_counts"].get("springer", 0) for s in hop_stats),
                "elsevier": sum(s["candidate_counts"].get("elsevier", 0) for s in hop_stats),
                "ieee": sum(s["candidate_counts"].get("ieee", 0) for s in hop_stats),
            },
            "hops": hop_stats,
        }
        return top_docs, stats
