from __future__ import annotations

import re
from typing import Dict, List, Tuple

AMBIGUOUS_TERMS = {
    "transformer": ["ML Transformer models", "Electrical power transformers"],
    "python": ["Python programming language", "Python snake"],
    "apple": ["Apple Inc.", "apple fruit"],
    "jaguar": ["Jaguar animal", "Jaguar car brand"],
    "java": ["Java programming language", "Java island/coffee"],
    "rust": ["Rust programming language", "rust corrosion"],
    "spark": ["Apache Spark", "electric spark"],
    "shell": ["Unix shell", "shell (physical)"],
    "bert": ["BERT language model", "person/entity named Bert"],
    "git": ["Git version control", "git as noun/other"],
    "linux": ["Linux operating system", "Linux distribution ecosystem"],
    "stream": ["data stream/computing", "natural stream"],
    "node": ["Node.js runtime", "graph/node concept"],
    "react": ["React framework", "react verb/chemistry"],
}

SENSE_KEYWORDS = {
    "ML Transformer models": ("transformer", "attention", "llm", "nlp", "bert", "gpt", "encoder", "decoder"),
    "Electrical power transformers": ("electrical", "power", "voltage", "substation", "thermal", "condition monitoring"),
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _detect_term(query: str) -> str | None:
    q = _tokens(query)
    for t in AMBIGUOUS_TERMS.keys():
        if t in q or (t + "s") in q:
            return t
    return None


def _score_sense(sense: str, snippets: List[str]) -> float:
    keys = SENSE_KEYWORDS.get(sense, ())
    if not keys:
        return 0.0
    joined = " ".join(snippets).lower()
    return float(sum(1 for k in keys if k in joined))


def resolve_sense(query: str, top_chunks: List[Dict], chosen_sense: str | None = None) -> Dict:
    term = _detect_term(query)
    if not term:
        return {
            "is_ambiguous": False,
            "term": None,
            "options": [],
            "rationale": "No curated ambiguous term detected.",
            "recommended_option": None,
        }

    options = AMBIGUOUS_TERMS.get(term, [])
    snippets = [f"{c.get('title','')} {c.get('snippet','')}" for c in (top_chunks or [])[:8]]

    sense_scores = [(opt, _score_sense(opt, snippets)) for opt in options]
    sense_scores.sort(key=lambda x: x[1], reverse=True)

    if chosen_sense and chosen_sense in options:
        return {
            "is_ambiguous": False,
            "term": term,
            "options": options,
            "rationale": f"User selected sense: {chosen_sense}.",
            "recommended_option": chosen_sense,
        }

    if len(options) < 2:
        return {
            "is_ambiguous": False,
            "term": term,
            "options": options,
            "rationale": "Only one sense configured.",
            "recommended_option": options[0] if options else None,
        }

    nonzero = [sc for _, sc in sense_scores if sc > 0]
    # If two+ senses are present in evidence, ask clarification.
    if len(nonzero) >= 2:
        is_ambiguous = True
    else:
        # If evidence strongly supports one sense we can proceed, otherwise ask.
        top = sense_scores[0][1]
        second = sense_scores[1][1] if len(sense_scores) > 1 else 0.0
        is_ambiguous = (top - second) <= 1.0

    return {
        "is_ambiguous": bool(is_ambiguous),
        "term": term,
        "options": options,
        "rationale": f"Detected ambiguous term '{term}' with close sense evidence scores.",
        "recommended_option": sense_scores[0][0] if sense_scores else None,
        "sense_scores": [{"sense": s, "score": float(sc)} for s, sc in sense_scores],
    }


def filter_citations_by_sense(citations: List[Dict], sense: str | None) -> List[Dict]:
    if not sense:
        return citations
    keys = SENSE_KEYWORDS.get(sense, ())
    if not keys:
        return citations
    out = []
    for c in citations:
        hay = f"{c.get('title','')} {c.get('snippet','')}".lower()
        if any(k in hay for k in keys):
            out.append(c)
    return out or citations
