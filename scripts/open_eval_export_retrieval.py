#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import dump_json, export_retrieval_for_query, load_query_set, ready_documents, utc_now_iso


def _annotation_payload(base: dict) -> dict:
    return {
        **base,
        "annotation_scheme": {
            "retrieved.relevance_label": ["relevant", "partially_relevant", "not_relevant"],
            "retrieved_docs.relevance_label": ["relevant", "partially_relevant", "not_relevant"],
            "corpus_docs.relevance_label": ["relevant", "partially_relevant", "not_relevant"],
            "binary_conversion": {
                "relevant": 1,
                "partially_relevant": 1,
                "not_relevant": 0,
            },
            "ndcg_gain": {
                "relevant": 2,
                "partially_relevant": 1,
                "not_relevant": 0,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval over uploaded documents for an open-corpus query set and export annotation-ready JSON."
    )
    parser.add_argument("--queries", required=True, help="Path to open-corpus query JSON")
    parser.add_argument("--k", type=int, default=10, help="Top-k chunks to export per query")
    parser.add_argument("--out", required=True, help="Output JSON path for retrieval results")
    parser.add_argument(
        "--annotation-out",
        default=None,
        help="Optional output JSON path for retrieval annotation template (defaults to --out content)",
    )
    args = parser.parse_args()

    query_set = load_query_set(args.queries)
    docs = ready_documents()
    if not docs:
        raise SystemExit("No ready uploaded documents found.")

    exported_queries = [
        export_retrieval_for_query(query_entry, k=max(1, args.k), all_docs=docs)
        for query_entry in query_set["queries"]
    ]

    payload = {
        "mode": "open_corpus_retrieval_export",
        "created_at": utc_now_iso(),
        "k": max(1, args.k),
        "queries": exported_queries,
    }
    dump_json(payload, args.out)

    if args.annotation_out:
        dump_json(_annotation_payload(payload), args.annotation_out)
        print(f"Wrote retrieval export to {args.out} and annotation template to {args.annotation_out}")
    else:
        print(f"Wrote retrieval export to {args.out}")


if __name__ == "__main__":
    main()
