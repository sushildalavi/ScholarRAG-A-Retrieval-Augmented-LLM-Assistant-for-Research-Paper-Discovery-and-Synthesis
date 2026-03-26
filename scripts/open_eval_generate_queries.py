#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import build_generated_query_set, dump_json, ready_documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an open-corpus query set for currently uploaded documents."
    )
    parser.add_argument("--doc-ids", nargs="*", type=int, default=None, help="Optional subset of ready uploaded doc ids")
    parser.add_argument("--per-doc", type=int, default=4, help="Queries to generate per document")
    parser.add_argument("--cross-doc", type=int, default=6, help="Cross-document queries to generate")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    docs = ready_documents(args.doc_ids)
    if not docs:
        raise SystemExit("No ready uploaded documents found for query generation.")

    payload = build_generated_query_set(
        docs,
        per_doc=max(1, args.per_doc),
        cross_doc=max(0, args.cross_doc),
    )
    dump_json(payload, args.output)
    print(f"Wrote {len(payload['queries'])} queries to {args.output}")


if __name__ == "__main__":
    main()
