from __future__ import annotations

from typing import Dict


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def confidence_label(score: float) -> str:
    s = clamp01(score)
    if s >= 0.75:
        return "High"
    if s >= 0.5:
        return "Med"
    return "Low"


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

    if needs_clarification:
        score = min(score, 0.25)

    explanation = (
        f"Confidence reflects evidence strength (cosine/rerank), citation coverage, and evidence separation; "
        f"it is reduced by ambiguity ({amb_pen:.2f}), insufficiency ({ins_pen:.2f}), and scope limitations ({scope_pen:.2f})."
    )

    return {
        "score": round(score, 4),
        "label": confidence_label(score),
        "needs_clarification": bool(needs_clarification),
        "factors": {
            "top_sim": round(sim, 4),
            "top_rerank_norm": round(rerank, 4),
            "citation_coverage": round(coverage, 4),
            "evidence_margin": round(margin, 4),
            "ambiguity_penalty": round(amb_pen, 4),
            "insufficiency_penalty": round(ins_pen, 4),
            "scope_penalty": round(scope_pen, 4),
        },
        "explanation": explanation,
    }
