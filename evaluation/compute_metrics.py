#!/usr/bin/env python3
"""Compute retrieval metrics (Recall@k, nDCG, MRR) from batch run outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def id_matches(source: Dict, relevant_ids: List[str]) -> bool:
    identifiers = [source.get("openalex_id"), source.get("doi")]
    identifiers = [i for i in identifiers if i]
    rel = set(str(r) for r in relevant_ids)
    return any(str(identifier) in rel for identifier in identifiers)


def dcg(relevances: List[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg(relevances: List[int], ideal_relevances: List[int]) -> float:
    best = dcg(ideal_relevances)
    if best == 0:
        return 0.0
    return dcg(relevances) / best


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute retrieval metrics from /ask batch output.")
    parser.add_argument("--input", required=True, type=Path, help="runs/ask_results.jsonl from evaluation/run_batch.py")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for Recall@k and nDCG@k")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if not records:
        raise SystemExit("Input file is empty")

    per_query = []
    mrr_values = []

    for record in records:
        relevant_ids = record.get("relevant_ids") or []
        retrieved = record.get("retrieved") or []
        if not retrieved:
            continue
        flags = [1 if id_matches(src, relevant_ids) else 0 for src in retrieved]
        topk_flags = flags[: args.k]

        if relevant_ids:
            recall = sum(topk_flags) / len(relevant_ids)
        else:
            recall = float("nan")
        precision = sum(topk_flags) / len(topk_flags) if topk_flags else float("nan")

        first_hit = next((idx for idx, flag in enumerate(flags, start=1) if flag), None)
        if first_hit:
            mrr_values.append(1 / first_hit)

        ideal = sorted(flags, reverse=True)
        per_query.append(
            {
                "query": record.get("query"),
                "recall_at_k": recall,
                "precision_at_k": precision,
                "ndcg_at_k": ndcg(topk_flags, ideal[: args.k]) if relevant_ids else float("nan"),
            }
        )

    df = pd.DataFrame(per_query)
    print("=== Retrieval Metrics ===")
    print(df)
    print()
    print(
        "Averages => Recall@k: {recall:.3f}, Precision@k: {precision:.3f}, nDCG@k: {ndcg:.3f}, MRR: {mrr:.3f}".format(
            recall=df["recall_at_k"].mean(skipna=True),
            precision=df["precision_at_k"].mean(skipna=True),
            ndcg=df["ndcg_at_k"].mean(skipna=True),
            mrr=sum(mrr_values) / len(mrr_values) if mrr_values else 0.0,
        )
    )
if __name__ == "__main__":
    main()
