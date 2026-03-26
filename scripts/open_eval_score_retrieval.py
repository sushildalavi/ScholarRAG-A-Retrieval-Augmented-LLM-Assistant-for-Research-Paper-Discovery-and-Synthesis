#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import dump_json, load_json, utc_now_iso
from backend.open_eval_metrics import aggregate_query_metrics
from backend.open_eval_spreadsheet import load_retrieval_annotations_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Recall@K, MRR, and nDCG from annotated open-corpus retrieval exports."
    )
    parser.add_argument("--annotations", required=True, help="Path to annotated retrieval JSON")
    parser.add_argument(
        "--corpus-docs",
        default=None,
        help="Optional corpus-doc relevance CSV. When provided with a retrieval CSV, enables stricter Recall/MRR denominators.",
    )
    parser.add_argument("-o", "--output", required=True, help="Output JSON path for computed metrics")
    args = parser.parse_args()

    input_path = Path(args.annotations)
    if input_path.suffix.lower() == ".csv":
        queries = load_retrieval_annotations_csv(input_path, corpus_csv_path=args.corpus_docs)
        if not queries:
            raise SystemExit("Annotated retrieval CSV must contain at least one labeled row.")
    else:
        payload = load_json(args.annotations)
        queries = payload.get("queries") if isinstance(payload, dict) else None
        if not isinstance(queries, list) or not queries:
            raise SystemExit("Annotated retrieval file must contain a non-empty `queries` list.")

    metrics = aggregate_query_metrics(queries)
    report = {
        "mode": "open_corpus_retrieval_metrics",
        "created_at": utc_now_iso(),
        "source_file": args.annotations,
        "corpus_doc_file": args.corpus_docs,
        "source_format": input_path.suffix.lower().lstrip(".") or "json",
        "summary": {
            "Recall@1": metrics["recall_at"]["1"],
            "Recall@3": metrics["recall_at"]["3"],
            "Recall@5": metrics["recall_at"]["5"],
            "Recall@10": metrics["recall_at"]["10"],
            "MRR": metrics["mrr"],
            "nDCG@10": metrics["ndcg_at"]["10"],
        },
        **metrics,
    }
    dump_json(report, args.output)
    print(f"Wrote retrieval metrics to {args.output}")


if __name__ == "__main__":
    main()
