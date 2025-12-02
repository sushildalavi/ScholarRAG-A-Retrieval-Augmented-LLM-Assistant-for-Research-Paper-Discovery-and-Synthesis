"""
Lightweight retrieval evaluation runner.

Usage:
  PYTHONPATH=. python evaluation/run_eval.py

You can customize `EVAL_SAMPLES` with your labeled queries and relevant IDs.
Metrics: Recall@k, nDCG@k, MRR.
"""

from typing import List, Dict

from search import search_papers  # uses pgvector/embeddings for public index if available


# Example eval set; replace with your own labeled queries.
EVAL_SAMPLES: List[Dict] = [
    {"query": "transformer models for summarization", "relevant_ids": ["paper-1", "paper-2"]},
    {"query": "graph neural networks in chemistry", "relevant_ids": ["paper-3", "paper-4"]},
]


def recall_at_k(results: List[str], relevant_ids: List[str], k: int) -> float:
    topk = set(results[:k])
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    return len(topk & rel) / len(rel)


def dcg_at_k(results: List[str], relevant_ids: List[str], k: int) -> float:
    rel = set(relevant_ids)
    dcg = 0.0
    for i, pid in enumerate(results[:k]):
        gain = 1.0 if pid in rel else 0.0
        dcg += gain / (i + 1)
    return dcg


def ndcg_at_k(results: List[str], relevant_ids: List[str], k: int) -> float:
    ideal = dcg_at_k(relevant_ids, relevant_ids, min(k, len(relevant_ids)))
    if ideal == 0:
        return 0.0
    return dcg_at_k(results, relevant_ids, k) / ideal


def mrr(results: List[str], relevant_ids: List[str]) -> float:
    rel = set(relevant_ids)
    for i, pid in enumerate(results, start=1):
        if pid in rel:
            return 1.0 / i
    return 0.0


def run_eval(k: int = 10):
    recalls, ndcgs, mrrs = [], [], []
    for sample in EVAL_SAMPLES:
        q = sample["query"]
        rel = sample["relevant_ids"]
        res = search_papers(q, k=k)
        ids = [r.get("paper_id") or r.get("id") for r in res]
        recalls.append(recall_at_k(ids, rel, k))
        ndcgs.append(ndcg_at_k(ids, rel, k))
        mrrs.append(mrr(ids, rel))
        print(f"[{q}] Recall@{k}={recalls[-1]:.3f} nDCG@{k}={ndcgs[-1]:.3f} MRR={mrrs[-1]:.3f}")

    print("Averages:",
          f"Recall@{k}={sum(recalls)/len(recalls):.3f}",
          f"nDCG@{k}={sum(ndcgs)/len(ndcgs):.3f}",
          f"MRR={sum(mrrs)/len(mrrs):.3f}")


if __name__ == "__main__":
    run_eval(k=10)
