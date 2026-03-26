#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_payload(path: str, model_name: str | None, label: str | None) -> dict:
    input_path = Path(path)
    if not input_path.exists():
        raise SystemExit(f"File not found: {input_path}")

    if input_path.suffix.lower() == ".jsonl":
        rows = []
        with input_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if not rows:
            raise SystemExit("Calibration JSONL file contains no records.")
        inferred_model_name = model_name or str(rows[0].get("model_name") or "msa_open_corpus_manual")
        inferred_label = label or str(rows[0].get("calibration_label") or "open_corpus_manual")
        records = []
        for row in rows:
            record = dict(row)
            record.pop("model_name", None)
            record.pop("calibration_label", None)
            records.append(record)
        return {
            "model_name": inferred_model_name,
            "label": inferred_label,
            "records": records,
        }

    payload = json.loads(input_path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit("Calibration JSON payload must be an object.")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise SystemExit("Calibration JSON payload must contain a non-empty `records` list.")
    if model_name:
        payload["model_name"] = model_name
    if label:
        payload["label"] = label
    payload.setdefault("model_name", "msa_open_corpus_manual")
    payload.setdefault("label", "open_corpus_manual")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post open-corpus calibration records into the existing /confidence/calibrate endpoint."
    )
    parser.add_argument("--input", required=True, help="Calibration records .json or .jsonl file")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Backend base URL for ScholarRAG",
    )
    parser.add_argument("--model-name", default=None, help="Optional override for calibration model_name")
    parser.add_argument("--label", default=None, help="Optional override for calibration label")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the response JSON",
    )
    args = parser.parse_args()

    payload = _load_payload(args.input, args.model_name, args.label)
    response = requests.post(
        f"{args.base_url.rstrip('/')}/confidence/calibrate",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    if args.output:
        Path(args.output).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote calibration response to {args.output}")
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
