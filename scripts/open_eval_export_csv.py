#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import dump_json, export_answer_for_query, export_retrieval_for_query, load_query_set, ready_documents
from backend.open_eval_spreadsheet import (
    CLAIM_ANNOTATION_FIELDS,
    CORPUS_DOC_FIELDS,
    QUERY_SUMMARY_FIELDS,
    RETRIEVAL_ANNOTATION_FIELDS,
    build_claim_annotation_rows,
    build_corpus_doc_rows,
    build_query_summary_rows,
    build_retrieval_annotation_rows,
    dump_csv_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Excel-friendly CSV annotation files for open-corpus evaluation over uploaded documents."
    )
    parser.add_argument("--queries", required=True, help="Path to open-corpus query JSON")
    parser.add_argument("--out-dir", required=True, help="Directory for CSV outputs")
    parser.add_argument("--retrieval-k", type=int, default=10, help="Top-k retrieval chunks to export per query")
    parser.add_argument("--answer-k", type=int, default=8, help="Top-k evidence chunks to use per answer")
    parser.add_argument(
        "--compute-msa",
        action="store_true",
        help="Request per-claim M/S/A features during answer export.",
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieval_rows: list[dict] = []
    answer_rows: list[dict] = []
    errors: list[dict] = []

    for query_entry in query_set["queries"]:
        query_id = query_entry.get("query_id")
        try:
            retrieval_rows.append(
                export_retrieval_for_query(
                    query_entry,
                    k=max(1, int(args.retrieval_k)),
                    all_docs=docs,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "stage": "retrieval",
                    "query_id": query_id,
                    "query": query_entry.get("query"),
                    "error": str(exc),
                }
            )
            continue

        try:
            answer_rows.append(
                export_answer_for_query(
                    query_entry,
                    k=max(1, int(args.answer_k)),
                    compute_msa=bool(args.compute_msa),
                    run_judge_llm=bool(args.run_judge_llm),
                    all_docs=docs,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "stage": "answer",
                    "query_id": query_id,
                    "query": query_entry.get("query"),
                    "error": str(exc),
                }
            )

    query_summary_path = out_dir / "query_summary.csv"
    retrieval_path = out_dir / "retrieval_annotations.csv"
    claim_path = out_dir / "claim_annotations.csv"
    corpus_path = out_dir / "corpus_doc_relevance.csv"

    dump_csv_rows(query_summary_path, QUERY_SUMMARY_FIELDS, build_query_summary_rows(answer_rows))
    dump_csv_rows(retrieval_path, RETRIEVAL_ANNOTATION_FIELDS, build_retrieval_annotation_rows(retrieval_rows))
    dump_csv_rows(claim_path, CLAIM_ANNOTATION_FIELDS, build_claim_annotation_rows(answer_rows))
    dump_csv_rows(corpus_path, CORPUS_DOC_FIELDS, build_corpus_doc_rows(retrieval_rows))

    manifest = {
        "mode": "open_corpus_csv_export",
        "queries_requested": len(query_set["queries"]),
        "retrieval_exports": len(retrieval_rows),
        "answer_exports": len(answer_rows),
        "compute_msa": bool(args.compute_msa),
        "run_judge_llm": bool(args.run_judge_llm),
        "files": {
            "query_summary": str(query_summary_path),
            "retrieval_annotations": str(retrieval_path),
            "claim_annotations": str(claim_path),
            "corpus_doc_relevance": str(corpus_path),
        },
        "errors": errors,
    }
    dump_json(manifest, out_dir / "export_manifest.json")

    print(f"Wrote query summary CSV to {query_summary_path}")
    print(f"Wrote retrieval annotations CSV to {retrieval_path}")
    print(f"Wrote claim annotations CSV to {claim_path}")
    print(f"Wrote corpus doc relevance CSV to {corpus_path}")
    if errors:
        print(f"Completed with {len(errors)} errors; see {out_dir / 'export_manifest.json'}")


if __name__ == "__main__":
    main()
