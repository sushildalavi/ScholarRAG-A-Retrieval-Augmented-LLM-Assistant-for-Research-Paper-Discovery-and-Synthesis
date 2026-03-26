#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import dump_json, dump_jsonl, load_json, utc_now_iso
from backend.open_eval_spreadsheet import build_calibration_records_from_claim_csv


def _build_records(queries: list[dict]) -> list[dict]:
    records: list[dict] = []
    for query_row in queries:
        query_id = query_row.get("query_id")
        for claim in query_row.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            label = str(claim.get("label") or "").strip().lower()
            if label not in {"supported", "unsupported"}:
                continue
            sentence = str(claim.get("text") or "").strip()
            evidence_text = str(claim.get("evidence_text") or "").strip()
            if not sentence or not evidence_text:
                continue

            record = {
                "query_id": query_id,
                "claim_id": claim.get("claim_id"),
                "sentence": sentence,
                "evidence_text": evidence_text,
                "label": label,
            }
            msa = claim.get("msa")
            if isinstance(msa, dict) and all(key in msa for key in ("M", "S", "A")):
                record["msa"] = {
                    "M": float(msa.get("M", 0.0)),
                    "S": float(msa.get("S", 0.0)),
                    "A": float(msa.get("A", 0.0)),
                }
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build /confidence/calibrate records from manually annotated open-corpus claim-support files."
    )
    parser.add_argument("--claims", required=True, help="Path to annotated claim-support JSON or CSV")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    parser.add_argument("--model-name", default="msa_open_corpus_manual", help="Calibration model_name field")
    parser.add_argument("--label", default="open_corpus_manual", help="Calibration label field")
    parser.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default=None,
        help="Output format. Defaults to json unless --output ends with .jsonl",
    )
    args = parser.parse_args()

    claim_path = Path(args.claims)
    if claim_path.suffix.lower() == ".csv":
        records = build_calibration_records_from_claim_csv(claim_path)
        if not records:
            raise SystemExit("Claim annotation CSV must contain at least one supported/unsupported row with evidence_text.")
    else:
        payload = load_json(args.claims)
        queries = payload.get("queries") if isinstance(payload, dict) else None
        if not isinstance(queries, list) or not queries:
            raise SystemExit("Claim annotation file must contain a non-empty `queries` list.")
        records = _build_records(queries)
    out = {
        "mode": "open_corpus_calibration_records",
        "created_at": utc_now_iso(),
        "model_name": args.model_name,
        "label": args.label,
        "records": records,
    }
    output_format = args.format or ("jsonl" if str(args.output).lower().endswith(".jsonl") else "json")
    if output_format == "jsonl":
        rows = []
        for record in records:
            row = dict(record)
            row["model_name"] = args.model_name
            row["calibration_label"] = args.label
            rows.append(row)
        dump_jsonl(rows, args.output)
    else:
        dump_json(out, args.output)
    print(f"Wrote {len(records)} calibration records to {args.output} ({output_format})")


if __name__ == "__main__":
    main()
