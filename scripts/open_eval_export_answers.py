#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import (
    build_claim_annotation_entry,
    dump_json,
    export_answer_for_query,
    load_query_set,
    ready_documents,
    utc_now_iso,
)


def _claim_annotation_payload(base_queries: list[dict]) -> dict:
    return {
        "mode": "open_corpus_claim_annotations",
        "created_at": utc_now_iso(),
        "annotation_scheme": {
            "claims.label": ["supported", "unsupported"],
            "claims.citation_correct": [True, False, None],
        },
        "queries": [build_claim_annotation_entry(query_row) for query_row in base_queries],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ScholarRAG answers and claim-level annotation files for an open-corpus query set."
    )
    parser.add_argument("--queries", required=True, help="Path to open-corpus query JSON")
    parser.add_argument("--k", type=int, default=8, help="Top-k evidence chunks to use per query")
    parser.add_argument("--out", required=True, help="Output JSON path for answer export")
    parser.add_argument(
        "--claim-annotation-out",
        default=None,
        help="Optional output JSON path for claim-support annotation template",
    )
    parser.add_argument(
        "--compute-msa",
        action="store_true",
        help="Request per-citation M/S/A features during answer export (more expensive).",
    )
    parser.add_argument(
        "--run-judge-llm",
        action="store_true",
        help="When --compute-msa is set, also enable the LLM judge in assistant_answer.",
    )
    args = parser.parse_args()

    query_set = load_query_set(args.queries)
    docs = ready_documents()
    if not docs:
        raise SystemExit("No ready uploaded documents found.")
    exported_queries: list[dict] = []
    errors: list[dict] = []

    for query_entry in query_set["queries"]:
        try:
            exported_queries.append(
                export_answer_for_query(
                    query_entry,
                    k=max(1, args.k),
                    compute_msa=bool(args.compute_msa),
                    run_judge_llm=bool(args.run_judge_llm),
                    all_docs=docs,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "query_id": query_entry.get("query_id"),
                    "query": query_entry.get("query"),
                    "error": str(exc),
                }
            )

    payload = {
        "mode": "open_corpus_answer_export",
        "created_at": utc_now_iso(),
        "k": max(1, args.k),
        "compute_msa": bool(args.compute_msa),
        "run_judge_llm": bool(args.run_judge_llm),
        "queries": exported_queries,
        "errors": errors,
    }
    dump_json(payload, args.out)

    if args.claim_annotation_out:
        dump_json(_claim_annotation_payload(exported_queries), args.claim_annotation_out)
        print(f"Wrote answer export to {args.out} and claim annotation template to {args.claim_annotation_out}")
    else:
        print(f"Wrote answer export to {args.out}")

    if errors:
        print(f"Skipped {len(errors)} queries due to errors; see `errors` in {args.out}")


if __name__ == "__main__":
    main()
