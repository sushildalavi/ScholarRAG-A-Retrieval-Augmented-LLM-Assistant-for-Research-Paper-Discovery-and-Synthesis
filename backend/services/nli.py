from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from typing import Dict, Tuple

from openai import OpenAI

from utils.config import get_openai_api_key


NLI_MODEL = "gpt-4o-mini"
NLI_TIMEOUT_SECONDS = 12


def _client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


def _heuristic_entailment_prob(hypothesis: str, premise: str) -> float:
    h = _tokens(hypothesis)
    p = _tokens(premise)
    if not h or not p:
        return 0.0
    overlap = len(h & p) / max(1, len(h))
    return max(0.0, min(1.0, overlap))


def _parse_prob_text(text: str) -> Tuple[float, float, float]:
    """Parse entailment / neutral / contradiction probabilities from freeform text."""
    lower = (text or "").lower()
    m = re.search(r"entailment\s*[:=]\s*([01]?(?:\.\d+)?|1(?:\.0+)?|0(?:\.\d+)?)", lower)
    n = re.search(r"neutral\s*[:=]\s*([01]?(?:\.\d+)?|1(?:\.0+)?|0(?:\.\d+)?)", lower)
    c = re.search(r"contradiction\s*[:=]\s*([01]?(?:\.\d+)?|1(?:\.0+)?|0(?:\.\d+)?)", lower)

    def _v(match):
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:
            return None

    p_entail = _v(m)
    p_neutral = _v(n)
    p_contra = _v(c)

    if p_entail is not None:
        p_neutral = 0.0 if p_neutral is None else p_neutral
        p_contra = 0.0 if p_contra is None else p_contra
        total = p_entail + p_neutral + p_contra
        if total <= 0:
            return _heuristic_entailment_prob(text, ""), 0.0, 0.0
        return p_entail / total, p_neutral / total, p_contra / total

    nums = re.findall(r"([01]?(?:\.\d+)?|1(?:\.0+)?|0(?:\.\d+)?)", lower)
    if nums:
        v = float(nums[0])
        v = max(0.0, min(1.0, v))
        return v, 1.0 - v, 0.0

    return 0.5, 0.5, 0.0


def _response_json(text: str) -> Dict:
    try:
        return json.loads((text or "").strip())
    except Exception:
        # try extract first JSON object
        m = re.search(r"\{[\s\S]*\}", text or "")
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}


def _cache_key(hypothesis: str, premise: str) -> str:
    raw = f"{(hypothesis or '').strip().lower()}::{(premise or '').strip().lower()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


@lru_cache(maxsize=2000)
def _cached_entailment(hypothesis: str, premise: str) -> float:
    """Bounded cache for entailment probability."""
    return _call_entailment_model(hypothesis, premise)


@lru_cache(maxsize=2000)
def _cached_entailment_meta(hypothesis: str, premise: str) -> Dict[str, float]:
    """Bounded cache for full entailment payload."""
    return _call_entailment_model_meta(hypothesis, premise)


def _call_entailment_model(hypothesis: str, premise: str) -> float:
    if not hypothesis or not premise:
        return 0.0

    payload = _call_entailment_model_meta(hypothesis, premise)
    return float(payload.get("entailment", 0.0))


def _call_entailment_model_meta(hypothesis: str, premise: str) -> Dict[str, float]:
    if not hypothesis or not premise:
        return {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}

    prompt = (
        "You are a strict NLI classifier. Given a hypothesis and premise, return probabilities that "
        "sum to 1 for entailment, neutral, contradiction."
        "\n\n"
        f"HYPOTHESIS:\n{hypothesis}\n\nPREMISE:\n{premise}"
    )

    try:
        completion = _client().chat.completions.create(
            model=NLI_MODEL,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an NLI evaluator. Output strict JSON only: "
                        "{\"entailment\": 0.xx, \"neutral\": 0.xx, \"contradiction\": 0.xx}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            timeout=NLI_TIMEOUT_SECONDS,
        )
        text = (completion.choices[0].message.content or "").strip()
        p_entail, p_neutral, p_contra = _parse_prob_text(text)
        return {
            "entailment": p_entail,
            "neutral": p_neutral,
            "contradiction": p_contra,
        }
    except Exception:
        score = _heuristic_entailment_prob(hypothesis, premise)
        return {
            "entailment": score,
            "neutral": max(0.0, 1.0 - score * 1.2) * 0.6,
            "contradiction": 1.0 - score,
        }


def entailment_prob(hypothesis: str, premise: str) -> float:
    """Return entailment probability in [0,1]. Uses cache + fallback heuristics."""
    try:
        return _cached_entailment(hypothesis, premise)
    except Exception:
        return _heuristic_entailment_prob(hypothesis, premise)


def entailment_meta(hypothesis: str, premise: str) -> Dict[str, float]:
    """Returns full entailment score set."""
    raw = _cache_key(hypothesis, premise)
    payload = _cached_entailment_meta(hypothesis, premise)
    p_entail = float(payload.get("entailment", 0.0))
    p_neutral = float(payload.get("neutral", 0.0))
    p_contra = float(payload.get("contradiction", 0.0))
    total = p_entail + p_neutral + p_contra
    if total <= 0:
        p_entail = _heuristic_entailment_prob(hypothesis, premise)
        p_neutral = max(0.0, 1.0 - p_entail)
        p_contra = 1.0 - p_entail - p_neutral if p_entail + p_neutral <= 1.0 else 0.0
    else:
        p_entail = p_entail / total
        p_neutral = p_neutral / total
        p_contra = p_contra / total

    return {
        "entailment": round(max(0.0, min(1.0, p_entail)), 4),
        "neutral": round(max(0.0, min(1.0, p_neutral)), 4),
        "contradiction": round(max(0.0, min(1.0, p_contra)), 4),
        "text_len": float(len((hypothesis or "") + "\n" + (premise or ""))),
        "hash": raw,
    }
