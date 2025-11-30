#!/usr/bin/env python3
"""
Batch evaluator for ScholarRAG.

Reads a JSONL file of queries, calls the running FastAPI `/ask` endpoint,
and writes a JSONL with the full responses for downstream analysis.

Input JSONL schema (one object per line):
{
  "query": "What are advances in few-shot learning?",
  "k": 10,                        # optional override
  "year_from": 2018,              # optional
  "year_to": 2024,                # optional
  "relevant_ids": ["W123", ...]   # optional ground-truth ids for metrics
}

Example usage:
    python evaluation/run_batch.py \
        --input evaluation/samples/queries.jsonl \
        --output runs/ask_results.jsonl \
        --backend http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests


def load_queries(path: Path) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            queries.append(json.loads(line))
    return queries


def call_backend(
    backend_url: str,
    query: str,
    k: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
) -> Dict[str, Any]:
    payload = {"query": query, "k": k}
    if year_from is not None:
        payload["year_from"] = year_from
    if year_to is not None:
        payload["year_to"] = year_to

    resp = requests.post(f"{backend_url.rstrip('/')}/ask", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def format_record(
    query_obj: Dict[str, Any],
    backend_resp: Dict[str, Any],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "query": query_obj.get("query"),
        "k": query_obj.get("k", 10),
        "year_from": query_obj.get("year_from"),
        "year_to": query_obj.get("year_to"),
        "relevant_ids": query_obj.get("relevant_ids", []),
        "answer": backend_resp.get("answer"),
        "sources": backend_resp.get("sources", []),
        "candidate_counts": backend_resp.get("candidate_counts", {}),
        "fallback_used": backend_resp.get("fallback_used", False),
        "metrics": backend_resp.get("metrics", {}),
    }

    # Flatten core retrieval info for metrics
    retrieved = []
    for idx, source in enumerate(record["sources"], start=1):
        retrieved.append(
            {
                "rank": idx,
                "title": source.get("title"),
                "year": source.get("year"),
                "doi": source.get("doi"),
                "openalex_id": source.get("openalex_id"),
                "similarity": source.get("similarity"),
            }
        )
    record["retrieved"] = retrieved
    return record


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in records:
            f.write(json.dumps(item) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run ScholarRAG /ask for evaluation.")
    parser.add_argument("--input", required=True, type=Path, help="JSONL file containing queries.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL to write responses.")
    parser.add_argument("--backend", default="http://127.0.0.1:8000", help="Base URL for FastAPI backend.")
    args = parser.parse_args()

    queries = load_queries(args.input)
    if not queries:
        raise SystemExit("No queries loaded. Check the input file.")

    results: List[Dict[str, Any]] = []
    for idx, query_obj in enumerate(queries, start=1):
        q = query_obj.get("query")
        if not q:
            continue
        print(f"[{idx}/{len(queries)}] Asking: {q!r}")
        try:
            resp = call_backend(
                backend_url=args.backend,
                query=q,
                k=int(query_obj.get("k", 10)),
                year_from=query_obj.get("year_from"),
                year_to=query_obj.get("year_to"),
            )
        except requests.RequestException as exc:
            print(f"  ! Request failed: {exc}")
            results.append({**query_obj, "error": str(exc)})
            continue
        results.append(format_record(query_obj, resp))

    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
