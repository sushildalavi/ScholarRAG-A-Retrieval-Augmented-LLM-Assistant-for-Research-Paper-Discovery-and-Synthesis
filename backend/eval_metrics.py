from __future__ import annotations

import math
from typing import Dict, List, Optional


def _dcg(rels: List[float], k: int) -> float:
    out = 0.0
    for i, r in enumerate(rels[:k], start=1):
        out += (2**r - 1) / math.log2(i + 1)
    return out


def recall_at_k(pred_doc_ids: List[int], gold_doc_id: Optional[int], k: int) -> float:
    if gold_doc_id is None:
        return 0.0
    return 1.0 if gold_doc_id in pred_doc_ids[:k] else 0.0


def mrr(pred_doc_ids: List[int], gold_doc_id: Optional[int]) -> float:
    if gold_doc_id is None:
        return 0.0
    for i, did in enumerate(pred_doc_ids, start=1):
        if did == gold_doc_id:
            return 1.0 / i
    return 0.0


def ndcg_at_k(pred_doc_ids: List[int], gold_doc_id: Optional[int], k: int) -> float:
    if gold_doc_id is None:
        return 0.0
    rels = [1.0 if did == gold_doc_id else 0.0 for did in pred_doc_ids]
    dcg = _dcg(rels, k)
    idcg = _dcg([1.0], min(1, k))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def aggregate_metrics(rows: List[Dict]) -> Dict:
    if not rows:
        return {
            "count": 0,
            "recall_at": {"1": 0.0, "3": 0.0, "5": 0.0, "10": 0.0},
            "mrr": 0.0,
            "ndcg_at": {"3": 0.0, "5": 0.0, "10": 0.0},
        }

    n = len(rows)
    r1 = r3 = r5 = r10 = 0.0
    mrr_sum = 0.0
    n3 = n5 = n10 = 0.0

    for row in rows:
        pred = row.get("pred_doc_ids", [])
        gold = row.get("gold_doc_id")
        r1 += recall_at_k(pred, gold, 1)
        r3 += recall_at_k(pred, gold, 3)
        r5 += recall_at_k(pred, gold, 5)
        r10 += recall_at_k(pred, gold, 10)
        mrr_sum += mrr(pred, gold)
        n3 += ndcg_at_k(pred, gold, 3)
        n5 += ndcg_at_k(pred, gold, 5)
        n10 += ndcg_at_k(pred, gold, 10)

    return {
        "count": n,
        "recall_at": {
            "1": round(r1 / n, 3),
            "3": round(r3 / n, 3),
            "5": round(r5 / n, 3),
            "10": round(r10 / n, 3),
        },
        "mrr": round(mrr_sum / n, 3),
        "ndcg_at": {
            "3": round(n3 / n, 3),
            "5": round(n5 / n, 3),
            "10": round(n10 / n, 3),
        },
    }
