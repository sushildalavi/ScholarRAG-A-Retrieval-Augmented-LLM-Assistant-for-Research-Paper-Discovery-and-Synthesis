from __future__ import annotations

import math
from typing import Dict, Optional


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def confidence_label(score: float) -> str:
    s = clamp01(score)
    if s >= 0.75:
        return "High"
    if s >= 0.5:
        return "Med"
    return "Low"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_msa_score(m: float, s: float, a: float, weights: Optional[Dict[str, float]] = None) -> float:
    """Compute calibrated confidence from M/S/A in [0,1] using logistic combination."""
    m = clamp01(m)
    s = clamp01(s)
    a = clamp01(a)
    w = weights or {}
    base = float(w.get("b", 0.0)) + float(w.get("w1", 0.58)) * m + float(w.get("w2", 0.22)) * s + float(w.get("w3", 0.20)) * a
    return clamp01(sigmoid(base))


def build_confidence(
    *,
    top_sim: float,
    top_rerank_norm: float,
    citation_coverage: float,
    evidence_margin: float,
    ambiguity_penalty: float,
    insufficiency_penalty: float,
    scope_penalty: float = 0.0,
    needs_clarification: bool = False,
    msa: Optional[Dict[str, float]] = None,
) -> Dict:
    sim = clamp01(top_sim)
    rerank = clamp01(top_rerank_norm)
    coverage = clamp01(citation_coverage)
    margin = clamp01(evidence_margin)
    amb_pen = clamp01(ambiguity_penalty)
    ins_pen = clamp01(insufficiency_penalty)
    scope_pen = clamp01(scope_penalty)

    base = clamp01((0.35 * sim) + (0.25 * rerank) + (0.25 * coverage) + (0.15 * margin))
    score = clamp01(base - (0.45 * amb_pen) - (0.4 * ins_pen) - (0.5 * scope_pen))

    msa_score = None
    if isinstance(msa, dict):
        msa_score = compute_msa_score(msa.get("M", 0.0), msa.get("S", 0.0), msa.get("A", 0.0), msa.get("weights"))
        # Blend retrieval-score with MSA score to stay backward-compatible yet evidence-aware.
        score = clamp01(0.62 * score + 0.38 * msa_score)

    if needs_clarification:
        score = min(score, 0.25)

    explanation = (
        "Confidence reflects evidence strength (cosine/rerank), citation coverage, and evidence separation; "
        "it is reduced by ambiguity, insufficiency, and scope limitations. "
        "Per-citation calibration uses M/S/A where M = entailment probability, "
        "S = retrieval stability rate, and A = multi-source agreement."
    )

    factors = {
        "top_sim": round(sim, 4),
        "top_rerank_norm": round(rerank, 4),
        "citation_coverage": round(coverage, 4),
        "evidence_margin": round(margin, 4),
        "ambiguity_penalty": round(amb_pen, 4),
        "insufficiency_penalty": round(ins_pen, 4),
        "scope_penalty": round(scope_pen, 4),
    }
    if msa is not None:
        factors["msa"] = {
            "M": round(clamp01(msa.get("M", 0.0)), 4),
            "S": round(clamp01(msa.get("S", 0.0)), 4),
            "A": round(clamp01(msa.get("A", 0.0)), 4),
            "msa_score": round(msa_score, 4) if msa_score is not None else 0.0,
            "weights": msa.get("weights", {}),
        }

    return {
        "score": round(score, 4),
        "label": confidence_label(score),
        "needs_clarification": bool(needs_clarification),
        "factors": factors,
        "explanation": explanation,
    }


def score_percent(probability: float) -> float:
    return round(clamp01(probability) * 100.0, 2)
