#!/usr/bin/env python3
"""
export_msa_records.py

Convert a /eval/judge response JSON into calibration records for
POST /confidence/calibrate.

Usage
-----
# From a saved response file:
python scripts/export_msa_records.py judge_response.json -o records.jsonl

# From stdin (pipe curl output directly):
curl -s -X POST http://localhost:8000/eval/judge \
     -H "Content-Type: application/json" \
     -d @eval_data/judge_eval_cases_with_ids.json | \
python scripts/export_msa_records.py - -o records.jsonl

# Dry-run: print to stdout only
python scripts/export_msa_records.py judge_response.json

Input format (the full /eval/judge response body)
--------------------------------------------------
{
  "run_id": 7,
  "details": [
    {
      "query": "...",
      "answer": "...",
      "citations": [
        {"id": "chunk_42", "text": "...", "doc_id": 3, ...}
      ],
      "faithfulness": {
        "overall_score": 0.83,
        "citation_coverage": 0.75,
        "supported_count": 5,
        "unsupported_count": 1,
        "sentence_count": 6,
        "claims": [
          {
            "sentence_id": "S1",
            "sentence": "RAG combines parametric and non-parametric memory.",
            "supported": true,
            "evidence_ids": ["chunk_42"],
            "reason": "Supported by the retrieved passage."
          },
          ...
        ],
        "unsupported": [...],
        "method": "llm"
      },
      "doc_id": 3,
      "doc_ids": null,
      "scope": "uploaded"
    },
    ...
  ]
}

Output format (one JSON object per line — JSONL)
------------------------------------------------
Each output record matches the _build_msa_records() input schema in app.py:

{
  "sentence": "RAG combines parametric and non-parametric memory.",
  "evidence_text": "RAG models combine parametric memory (the LM itself) ...",
  "S": null,
  "A": null,
  "label": "supported"
}

Fields
------
- sentence      : claim sentence from faithfulness.claims[].sentence
- evidence_text : joined text of all cited chunks (looked up from citations by evidence_ids)
- S             : source-quality score — null, let the API compute it from metadata
- A             : answer-fluency score — null, let the API compute it from metadata
- label         : "supported" if claims[].supported == true, else "unsupported"

The API will compute M automatically via NLI (entailment_prob) when sentence +
evidence_text are present and msa dict is absent.

Notes
-----
- Records where evidence_ids is empty or no matching citation text is found are
  skipped (M cannot be computed without evidence).
- "method: heuristic" claims are included; annotators should review them.
- To pre-supply M/S/A values instead of letting the API compute them, add an
  "msa": {"M": 0.9, "S": 0.7, "A": 0.8} field to each record before posting.
"""

import argparse
import json
import sys
from pathlib import Path


def _build_citation_index(citations: list) -> dict[str, str]:
    """Return {chunk_id -> text} from a citations list."""
    index: dict[str, str] = {}
    if not isinstance(citations, list):
        return index
    for c in citations:
        if not isinstance(c, dict):
            continue
        # Citations may use "id", "chunk_id", or "source_id" as the key field.
        chunk_id = (
            c.get("id")
            or c.get("chunk_id")
            or c.get("source_id")
        )
        text = c.get("text") or c.get("content") or c.get("snippet") or ""
        if chunk_id is not None and text:
            index[str(chunk_id)] = text.strip()
    return index


def _export_detail(detail: dict) -> list[dict]:
    """Extract calibration records from one detail entry."""
    records = []
    citations = detail.get("citations") or []
    citation_index = _build_citation_index(citations)

    faithfulness = detail.get("faithfulness") or {}
    claims = faithfulness.get("claims") or []

    for claim in claims:
        if not isinstance(claim, dict):
            continue

        sentence = (claim.get("sentence") or "").strip()
        if not sentence:
            continue

        supported: bool = bool(claim.get("supported", False))
        evidence_ids: list = claim.get("evidence_ids") or []

        # Collect evidence text from cited chunks.
        evidence_parts = []
        for eid in evidence_ids:
            text = citation_index.get(str(eid))
            if text:
                evidence_parts.append(text)

        if not evidence_parts:
            # No resolvable evidence — skip (M cannot be computed).
            continue

        evidence_text = " ".join(evidence_parts)
        label = "supported" if supported else "unsupported"

        records.append(
            {
                "sentence": sentence,
                "evidence_text": evidence_text,
                "S": None,   # annotator or API fills in
                "A": None,   # annotator or API fills in
                "label": label,
            }
        )

    return records


def export(response: dict) -> list[dict]:
    """Convert a full /eval/judge response dict into a flat list of records."""
    details = response.get("details") or []
    all_records: list[dict] = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        all_records.extend(_export_detail(detail))
    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert /eval/judge response JSON to calibration JSONL for POST /confidence/calibrate"
    )
    parser.add_argument(
        "input",
        help="Path to judge response JSON file, or '-' to read from stdin",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSONL file path (default: print to stdout)",
    )
    parser.add_argument(
        "--include-heuristic",
        action="store_true",
        default=True,
        help="Include records from heuristic (non-LLM) judge runs (default: true)",
    )
    parser.add_argument(
        "--min-evidence-chars",
        type=int,
        default=20,
        help="Skip records whose evidence_text is shorter than N chars (default: 20)",
    )
    args = parser.parse_args()

    # Load input
    if args.input == "-":
        try:
            response = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON from stdin: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        path = Path(args.input)
        if not path.exists():
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path) as f:
            try:
                response = json.load(f)
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse JSON from {path}: {e}", file=sys.stderr)
                sys.exit(1)

    records = export(response)

    # Apply filters
    if args.min_evidence_chars > 0:
        before = len(records)
        records = [
            r for r in records
            if len(r.get("evidence_text") or "") >= args.min_evidence_chars
        ]
        skipped = before - len(records)
        if skipped:
            print(f"Skipped {skipped} records with short evidence_text (<{args.min_evidence_chars} chars)", file=sys.stderr)

    if not records:
        print("WARNING: No records produced. Check that the input has faithfulness.claims with evidence_ids.", file=sys.stderr)
        sys.exit(0)

    # Count label distribution
    supported_n = sum(1 for r in records if r.get("label") == "supported")
    unsupported_n = len(records) - supported_n
    print(
        f"Exported {len(records)} records: {supported_n} supported, {unsupported_n} unsupported",
        file=sys.stderr,
    )

    # Write output
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Written to {out_path}", file=sys.stderr)
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
