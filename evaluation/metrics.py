"""
Lightweight evaluation helpers: Recall@k, nDCG, and MRR for retrieval runs.
Expected input: a list of queries with ground truth ids and ranked result ids.
"""
from typing import Dict, List, Sequence
import math


def recall_at_k(truth: Sequence[str], preds: Sequence[str], k: int) -> float:
    topk = set(preds[:k])
    if not truth:
        return 0.0
    return sum(1 for t in truth if t in topk) / len(truth)


def dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(truth: Sequence[str], preds: Sequence[str], k: int) -> float:
    if not truth:
        return 0.0
    rels = [1 if p in truth else 0 for p in preds[:k]]
    ideal = sorted(rels, reverse=True)
    denom = dcg(ideal) or 1e-9
    return dcg(rels) / denom


def mrr(truth: Sequence[str], preds: Sequence[str]) -> float:
    for idx, p in enumerate(preds):
        if p in truth:
            return 1.0 / (idx + 1)
    return 0.0


def evaluate(run: List[Dict], k: int = 10) -> Dict[str, float]:
    recalls, ndcgs, mrrs = [], [], []
    for row in run:
        truth = row.get("truth_ids", [])
        preds = row.get("predicted_ids", [])
        recalls.append(recall_at_k(truth, preds, k))
        ndcgs.append(ndcg_at_k(truth, preds, k))
        mrrs.append(mrr(truth, preds))
    n = len(run) or 1
    return {
        f"recall@{k}": sum(recalls) / n,
        f"ndcg@{k}": sum(ndcgs) / n,
        "mrr": sum(mrrs) / n,
    }
