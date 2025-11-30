#!/usr/bin/env python3
"""
LLM-as-judge evaluator for ScholarRAG answers.

Usage:
    python evaluation/llm_judge.py --input runs/latest.jsonl --output runs/latest_scored.jsonl

Input format (JSONL):
{
  "query": "What are transformer improvements?",
  "answer": "...",
  "sources": [
      {
          "title": "...",
          "year": 2023,
          "doi": "10.1234/abc",
          "snippet": "...",
          "why_relevant": "...",
          "similarity": 0.83
      }
  ]
}

Output: JSONL with added fields score (0-5) and justification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from openai import OpenAI

from utils.config import get_openai_api_key

DEFAULT_MODEL = "gpt-4o"

SYSTEM_PROMPT = (
    "You are an expert reviewer evaluating answers produced by a retrieval-augmented system. "
    "You will receive a user's question, the system's answer, and the list of sources the system cited. "
    "Assess factual accuracy, citation faithfulness, and coverage. "
    "Return a strict JSON object with keys: "
    '{"score": int (0-5), "verdict": "pass" or "fail", "justification": str, "issues": [str]}. '
    "Score rubric: 5=excellent answer, correct and fully supported; "
    "4=good answer with minor omissions; "
    "3=partially correct but missing key details; "
    "2=significant issues or unsupported claims; "
    "1=mostly incorrect; 0=off-topic or hallucinated. "
    "Use pass for scores >=4, fail otherwise."
)


def load_samples(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def build_user_prompt(sample: Dict[str, Any]) -> str:
    query = sample.get("query") or sample.get("question") or ""
    answer = sample.get("answer") or sample.get("response") or ""
    sources = sample.get("sources") or []

    parts = [
        f"Question:\n{query}",
        "",
        "System answer:\n" + answer,
        "",
        "Sources:",
    ]

    if not sources:
        parts.append(" - (none provided)")
    else:
        for idx, src in enumerate(sources, start=1):
            title = src.get("title", "Unknown title")
            year = src.get("year", "n.d.")
            doi = src.get("doi") or ""
            snippet = src.get("snippet") or ""
            why = src.get("why_relevant") or ""
            parts.append(
                f" - [{idx}] {title} ({year})"
                + (f" DOI: {doi}" if doi else "")
                + (f"\n   Snippet: {snippet}" if snippet else "")
                + (f"\n   Why relevant: {why}" if why else "")
            )

    parts.append("")
    parts.append("Evaluate the answer. Respond ONLY with the JSON object.")
    return "\n".join(parts)


def evaluate_sample(client: OpenAI, sample: Dict[str, Any], model: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(sample)},
    ]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    content = response.choices[0].message.content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "score": None,
            "verdict": "fail",
            "justification": f"Could not parse judge output: {content}",
            "issues": ["invalid_json"],
        }
    result["_raw"] = content
    result["_usage"] = response.usage.model_dump() if hasattr(response, "usage") else {}
    return result


def aggregate(results: Iterable[Dict[str, Any]]) -> Tuple[float, float]:
    scores = [res.get("score") for res in results if isinstance(res.get("score"), (int, float))]
    passes = [res for res in results if res.get("verdict") == "pass"]
    avg = sum(scores) / len(scores) if scores else 0.0
    pass_rate = len(passes) / len(results) if results else 0.0
    return avg, pass_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluator for ScholarRAG.")
    parser.add_argument("--input", required=True, type=Path, help="Path to JSONL file with samples.")
    parser.add_argument("--output", type=Path, help="Where to write scored JSONL (defaults to stdout).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Judge model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--limit", type=int, help="Optional cap on number of samples.")
    args = parser.parse_args()

    samples = load_samples(args.input, args.limit)
    if not samples:
        raise SystemExit("No samples loaded. Check the input file.")

    client = OpenAI(api_key=get_openai_api_key())

    scored: List[Dict[str, Any]] = []
    for sample in samples:
        evaluation = evaluate_sample(client, sample, args.model)
        scored.append({**sample, "evaluation": evaluation})

    avg_score, pass_rate = aggregate([s["evaluation"] for s in scored])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for item in scored:
                f.write(json.dumps(item) + "\n")
    else:
        for item in scored:
            print(json.dumps(item))

    print(f"Average score: {avg_score:.2f}")
    print(f"Pass rate: {pass_rate*100:.1f}%")


if __name__ == "__main__":
    main()
