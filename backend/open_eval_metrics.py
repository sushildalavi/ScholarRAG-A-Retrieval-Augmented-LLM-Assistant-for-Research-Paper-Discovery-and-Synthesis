from __future__ import annotations

import math
from typing import Any


_GAIN_BY_LABEL = {
    "relevant": 2.0,
    "partially_relevant": 1.0,
    "partial": 1.0,
    "not_relevant": 0.0,
    "irrelevant": 0.0,
    "unsupported": 0.0,
    "": 0.0,
}


def relevance_gain(label: object) -> float:
    normalized = str(label or "").strip().lower()
    return _GAIN_BY_LABEL.get(normalized, 0.0)


def relevance_binary(label: object) -> float:
    return 1.0 if relevance_gain(label) > 0.0 else 0.0


def _dcg(gains: list[float], k: int) -> float:
    score = 0.0
    for rank, gain in enumerate(gains[:k], start=1):
        score += (2**gain - 1.0) / math.log2(rank + 1.0)
    return score


def ranked_doc_ids(query_row: dict[str, Any]) -> list[int]:
    retrieved_docs = query_row.get("retrieved_docs")
    if isinstance(retrieved_docs, list) and retrieved_docs:
        ordered = sorted(
            [row for row in retrieved_docs if isinstance(row, dict) and row.get("doc_id") is not None],
            key=lambda row: int(row.get("rank") or 0),
        )
        return [int(row["doc_id"]) for row in ordered]

    seen: set[int] = set()
    ordered: list[int] = []
    for row in query_row.get("retrieved") or []:
        if not isinstance(row, dict) or row.get("doc_id") is None:
            continue
        doc_id = int(row["doc_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ordered.append(doc_id)
    return ordered


def relevant_doc_gains(query_row: dict[str, Any]) -> dict[int, float]:
    gains: dict[int, float] = {}
    corpus_docs = query_row.get("corpus_docs") or []
    for row in corpus_docs:
        if not isinstance(row, dict) or row.get("doc_id") is None:
            continue
        gain = relevance_gain(row.get("relevance_label"))
        if gain > 0:
            gains[int(row["doc_id"])] = gain

    if gains:
        return gains

    for row in query_row.get("retrieved_docs") or []:
        if not isinstance(row, dict) or row.get("doc_id") is None:
            continue
        gain = relevance_gain(row.get("relevance_label"))
        if gain > 0:
            gains[int(row["doc_id"])] = gain
    return gains


def recall_at_k(pred_doc_ids: list[int], gold_gains: dict[int, float], k: int) -> float:
    if not gold_gains:
        return 0.0
    positives = {doc_id for doc_id, gain in gold_gains.items() if gain > 0.0}
    if not positives:
        return 0.0
    hits = sum(1 for doc_id in pred_doc_ids[:k] if doc_id in positives)
    return hits / len(positives)


def mrr(pred_doc_ids: list[int], gold_gains: dict[int, float]) -> float:
    positives = {doc_id for doc_id, gain in gold_gains.items() if gain > 0.0}
    if not positives:
        return 0.0
    for rank, doc_id in enumerate(pred_doc_ids, start=1):
        if doc_id in positives:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(pred_doc_ids: list[int], gold_gains: dict[int, float], k: int) -> float:
    if not gold_gains:
        return 0.0
    gains = [gold_gains.get(doc_id, 0.0) for doc_id in pred_doc_ids[:k]]
    dcg = _dcg(gains, k)
    ideal = sorted(gold_gains.values(), reverse=True)
    idcg = _dcg(ideal, k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def aggregate_query_metrics(query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not query_rows:
        return {
            "count": 0,
            "recall_at": {"1": 0.0, "3": 0.0, "5": 0.0, "10": 0.0},
            "mrr": 0.0,
            "ndcg_at": {"10": 0.0},
        }

    totals = {
        "r1": 0.0,
        "r3": 0.0,
        "r5": 0.0,
        "r10": 0.0,
        "mrr": 0.0,
        "ndcg10": 0.0,
    }
    per_query: list[dict[str, Any]] = []

    for query_row in query_rows:
        pred_doc_ids = ranked_doc_ids(query_row)
        gains = relevant_doc_gains(query_row)
        recall_1 = recall_at_k(pred_doc_ids, gains, 1)
        recall_3 = recall_at_k(pred_doc_ids, gains, 3)
        recall_5 = recall_at_k(pred_doc_ids, gains, 5)
        recall_10 = recall_at_k(pred_doc_ids, gains, 10)
        mrr_score = mrr(pred_doc_ids, gains)
        ndcg_10 = ndcg_at_k(pred_doc_ids, gains, 10)
        row_metrics = {
            "query_id": query_row.get("query_id"),
            "query": query_row.get("query"),
            "pred_doc_ids": pred_doc_ids,
            "gold_doc_ids": sorted(gains.keys()),
            "recall_at": {
                "1": round(recall_1, 4),
                "3": round(recall_3, 4),
                "5": round(recall_5, 4),
                "10": round(recall_10, 4),
            },
            "mrr": round(mrr_score, 4),
            "ndcg_at": {
                "10": round(ndcg_10, 4),
            },
        }
        per_query.append(row_metrics)
        totals["r1"] += recall_1
        totals["r3"] += recall_3
        totals["r5"] += recall_5
        totals["r10"] += recall_10
        totals["mrr"] += mrr_score
        totals["ndcg10"] += ndcg_10

    n = len(query_rows)
    return {
        "count": n,
        "label_mapping": {
            "binary_relevant": ["relevant", "partially_relevant"],
            "binary_not_relevant": ["not_relevant"],
            "ndcg_gain": {"relevant": 2.0, "partially_relevant": 1.0, "not_relevant": 0.0},
        },
        "recall_at": {
            "1": round(totals["r1"] / n, 4),
            "3": round(totals["r3"] / n, 4),
            "5": round(totals["r5"] / n, 4),
            "10": round(totals["r10"] / n, 4),
        },
        "mrr": round(totals["mrr"] / n, 4),
        "ndcg_at": {
            "10": round(totals["ndcg10"] / n, 4),
        },
        "per_query": per_query,
    }
