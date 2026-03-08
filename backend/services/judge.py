from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List

from openai import OpenAI

from utils.config import get_openai_api_key


JUDGE_MODEL = "gpt-4o-mini"
JUDGE_TIMEOUT_SECONDS = 15


def _client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    out = re.split(r"(?<=[\.!?])\s+", (text or "").strip())
    return [s.strip() for s in out if s.strip()]


def _safe_int(x: object, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_judge_json(raw: str) -> Dict:
    try:
        return json.loads((raw or "").strip())
    except Exception:
        # try extract first object if assistant adds formatting
        m = re.search(r"\{[\s\S]*\}", raw or "")
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}


def _fallback_report(sentences: List[str], citations: List[Dict]) -> Dict:
    cited_count = 0
    supported = 0
    unsupported = []
    for idx, s in enumerate(sentences):
        ids = re.findall(r"\[S(\d+)\]", s)
        if ids:
            cited_count += 1
            supported += 1
        else:
            unsupported.append({"sentence_id": idx + 1, "sentence": s, "supported": False, "reason": "missing_citation"})

    if not sentences:
        coverage = 0.0
    else:
        coverage = cited_count / len(sentences)
    cited_ids = set()
    for sentence in sentences:
        for m in re.finditer(r"\[S(\d+)\]", sentence):
            cited_ids.add(str(m.group(1)))
    coverage_by_id = {}
    for idx, c in enumerate(citations or [], start=1):
        sid = str((c.get("id") or idx))
        coverage_by_id[sid] = sid in cited_ids
    return {
        "overall_score": float(supported) / len(sentences) if sentences else 0.0,
        "citation_coverage": coverage,
        "unsupported_count": len(unsupported),
        "supported_count": supported,
        "sentence_count": len(sentences),
        "unsupported": unsupported,
        "method": "heuristic",
        "evidence_coverage_by_id": coverage_by_id,
    }

def evaluate_faithfulness(query: str, answer: str, citations: List[Dict], use_llm: bool = True) -> Dict:
    sentences = _split_sentences(answer)
    if not use_llm:
        return _fallback_report(sentences, citations)

    # Build compact evidence block for judge.
    evidences = []
    for i, c in enumerate(citations or [], start=1):
        evidences.append(
            {
                "id": c.get("id", i),
                "source": c.get("source") or "uploaded",
                "title": c.get("title") or "",
                "snippet": c.get("snippet", ""),
                "metadata": {
                    "doc_id": c.get("doc_id"),
                    "chunk_id": c.get("chunk_id"),
                    "page": c.get("page"),
                },
            }
        )

    prompt = (
        "You are a strict evaluator for evidence-grounded scientific QA.\n"
        "Return strict JSON only with this shape:\n"
        "{\n"
        '  "overall_score": 0.0,\n'
        '  "citation_coverage": 0.0,\n'
        '  "supported_count": 0,\n'
        '  "unsupported_count": 0,\n'
        '  "sentence_count": 0,\n'
        '  "claims": [\n'
        '    {"sentence_id":1, "sentence":"...", "supported":true, "evidence_ids":["S1"], "reason":"..."}\n'
        '  ]\n'
        "}\n\n"
        f"Query: {query}\n"
        f"Answer: {answer}\n\n"
        f"Evidence items: {json.dumps(evidences)[:12000]}"
    )

    try:
        completion = _client().chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a strict scientific claim evaluator."},
                {"role": "user", "content": prompt},
            ],
            timeout=JUDGE_TIMEOUT_SECONDS,
        )
        payload = _parse_judge_json(completion.choices[0].message.content or "")
        claims = payload.get("claims") if isinstance(payload.get("claims"), list) else []

        cleaned_claims: List[Dict] = []
        unsupported = []
        for c in claims:
            if not isinstance(c, dict):
                continue
            sentence_id = _safe_int(c.get("sentence_id"), 0)
            sentence = (c.get("sentence") or "").strip()
            supported = bool(c.get("supported", False))
            ids = c.get("evidence_ids")
            if not isinstance(ids, list):
                ids = []
            reason = (c.get("reason") or "").strip()
            item = {
                "sentence_id": sentence_id,
                "sentence": sentence,
                "supported": supported,
                "evidence_ids": ids,
                "reason": reason,
            }
            cleaned_claims.append(item)
            if not supported:
                unsupported.append(item)

        total = max(1, _safe_int(payload.get("sentence_count"), len(sentences)))
        return {
            "overall_score": _safe_float(payload.get("overall_score"), len(cleaned_claims) / total if sentences else 0.0),
            "citation_coverage": _safe_float(payload.get("citation_coverage"), 0.0),
            "supported_count": _safe_int(payload.get("supported_count"), max(0, total - len(unsupported))),
            "unsupported_count": _safe_int(payload.get("unsupported_count"), len(unsupported)),
            "sentence_count": total,
            "claims": cleaned_claims,
            "unsupported": unsupported,
            "method": "llm", 
        }
    except Exception:
        return _fallback_report(sentences, citations)


def aggregate_judge_report(reports: Iterable[Dict]) -> Dict:
    rows = list(reports)
    if not rows:
        return {
            "mean_overall_score": 0.0,
            "mean_coverage": 0.0,
            "unsupported_total": 0,
            "count": 0,
        }
    n = len(rows)
    mean_overall = sum(float(r.get("overall_score", 0.0) or 0.0) for r in rows) / n
    mean_cov = sum(float(r.get("citation_coverage", 0.0) or 0.0) for r in rows) / n
    unsupported = sum(int(r.get("unsupported_count", 0) or 0) for r in rows)
    return {
        "mean_overall_score": round(mean_overall, 4),
        "mean_coverage": round(mean_cov, 4),
        "unsupported_total": unsupported,
        "count": n,
    }
