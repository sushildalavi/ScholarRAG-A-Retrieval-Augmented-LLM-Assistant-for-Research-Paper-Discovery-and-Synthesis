from __future__ import annotations

import difflib
import hashlib
import re
from pathlib import Path

from backend.confidence import build_confidence
from backend.pdf_ingest import search_chunks as search_uploaded_chunks
from backend.public_search import public_live_search
from backend.services.db import execute, fetchall, fetchone
from backend.services.nli import entailment_prob


def _load_latest_calibration_weights() -> dict:
    row = fetchone(
        """
        SELECT weights
        FROM confidence_calibration
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    if not row:
        return {"w1": 0.58, "w2": 0.22, "w3": 0.20, "b": 0.0}
    weights = row.get("weights") or {}
    if not isinstance(weights, dict):
        return {"w1": 0.58, "w2": 0.22, "w3": 0.20, "b": 0.0}
    return {
        "w1": float(weights.get("w1", 0.58)),
        "w2": float(weights.get("w2", 0.22)),
        "w3": float(weights.get("w3", 0.20)),
        "b": float(weights.get("b", 0.0)),
    }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _normalize_inverse(value: float, min_v: float, max_v: float) -> float:
    span = max(1e-6, max_v - min_v)
    return _clamp01((max_v - value) / span)


def _normalize_forward(value: float, min_v: float, max_v: float) -> float:
    span = max(1e-6, max_v - min_v)
    return _clamp01((value - min_v) / span)


def _base_confidence(match_strength: float, rank: int, total: int, agreement: float) -> float:
    rank_stability = 1.0 if total <= 1 else 1.0 - ((rank - 1) / (total - 1))
    raw = _clamp01(0.65 * match_strength + 0.2 * rank_stability + 0.15 * agreement)
    calibrated = 0.28 + 0.58 * raw
    return round(_clamp01(calibrated), 3)


def _confidence_breakdown(match_strength: float, rank: int, total: int, agreement: float) -> dict:
    rank_stability = 1.0 if total <= 1 else 1.0 - ((rank - 1) / (total - 1))
    raw = _clamp01(0.65 * match_strength + 0.2 * rank_stability + 0.15 * agreement)
    calibrated = 0.28 + 0.58 * raw
    return {
        "match_strength": round(match_strength, 3),
        "rank_stability": round(rank_stability, 3),
        "agreement": round(agreement, 3),
        "raw": round(raw, 3),
        "calibrated": round(_clamp01(calibrated), 3),
    }


def _normalize_inline_citations(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return text
    text = re.sub(r"\[(?:S)?(\d+)\]", r"[S\1]", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])\s*\[S(\d+)\]\s*([.,;:!?])", r"\1 [S\2]", text)
    text = re.sub(r"\[S(\d+)\]\s*([.,;:!?])", r"\2 [S\1]", text)
    return text


def _humanize_answer_text(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return text
    replacements = [
        (r"\bInsufficient evidence is available\b", "I only found limited evidence in your uploaded sources"),
        (r"\bInsufficient evidence exists\b", "I only found limited evidence in your uploaded sources"),
        (r"\bInsufficient evidence\b", "I only found limited evidence in your uploaded sources"),
        (r"\bBased on the provided context\b", "From what I found in your documents"),
        (r"\bBased on your uploaded documents\b", "From your uploaded documents"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def _citation_coverage_stats(answer: str) -> tuple[float, int, int]:
    parts = [p.strip() for p in re.split(r"\n{2,}", (answer or "").strip()) if p.strip()]
    if not parts:
        return 0.0, 0, 0
    cited = 0
    for p in parts:
        if re.search(r"\[S\d+\]", p):
            cited += 1
    coverage = cited / max(1, len(parts))
    unsupported = max(0, len(parts) - cited)
    return coverage, unsupported, len(parts)


def _apply_usage_boost(citations: list[dict], answer: str) -> list[dict]:
    if not citations:
        return citations
    tags = re.findall(r"\[S(\d+)\]", answer or "")
    if not tags:
        return citations
    counts = {}
    for t in tags:
        sid = int(t)
        counts[sid] = counts.get(sid, 0) + 1
    max_count = max(counts.values()) if counts else 1
    for c in citations:
        sid = int(c.get("id", 0) or 0)
        used = counts.get(sid, 0)
        usage = used / max_count if max_count > 0 else 0.0
        base = float(c.get("confidence", 0.5))
        boosted = _clamp01(0.8 * base + 0.2 * usage)
        c["base_confidence"] = round(base, 3)
        c["usage_boost"] = round(usage, 3)
        c["confidence"] = round(min(0.92, max(0.2, boosted)), 3)
        c["used_in_answer"] = bool(used)
    return citations


def _is_doc_visibility_query(qnorm: str) -> bool:
    doc_terms = ("doc", "docs", "document", "documents", "uploaded", "attach", "attached", "file", "files")
    visibility_terms = ("see", "access", "read", "view", "visible")
    has_doc = any(t in qnorm for t in doc_terms)
    has_visibility = any(t in qnorm for t in visibility_terms)
    is_question = qnorm.startswith(("can ", "do ", "are ", "is ", "did ", "have "))
    return has_doc and has_visibility and is_question


def _is_doc_intent_query(qnorm: str) -> bool:
    doc_terms = (
        "doc", "docs", "document", "documents", "uploaded", "attach", "attached", "file", "files",
        "pdf", "page", "chunk", "source", "citation", "cite", "resume", "assignment", "lecture",
    )
    return any(t in qnorm for t in doc_terms)


def _scope_evidence_label(scope: str) -> str:
    return "uploaded documents" if scope == "uploaded" else "public sources"


def _normalize_source_url(value: str | None) -> str | None:
    v = (value or "").strip()
    if not v:
        return None
    if v.startswith("http://") or v.startswith("https://"):
        return v
    if v.startswith("10."):
        return f"https://doi.org/{v}"
    if v.lower().startswith("doi.org/"):
        return f"https://{v}"
    return None


def _build_public_evidence_fallback(query: str, citations: list[dict]) -> str:
    if not citations:
        return "I couldn’t find enough reliable public source evidence for this query."
    lines = []
    paper_intent = bool(re.search(r"\b(papers?|research papers?|studies|survey|surveys|references?)\b", query or "", flags=re.I))
    for i, c in enumerate(citations[:5 if paper_intent else 3], start=1):
        title = c.get("title") or f"Source {i}"
        year = c.get("year")
        snippet = (c.get("snippet") or "").strip()
        snippet = re.sub(r"\s+", " ", snippet)[:220]
        header = f"{title} ({year})" if year else title
        url = _normalize_source_url(c.get("url"))
        if snippet:
            suffix = f" Link: {url}" if url else ""
            lines.append(f"- {header}: {snippet} [S{i}]{suffix}")
        else:
            suffix = f" Link: {url}" if url else ""
            lines.append(f"- {header} [S{i}]{suffix}")
    return (
        ("I found relevant public research papers for your query. Here are the strongest matches:\n"
         if paper_intent
         else "I found relevant public research sources for your query. Here are the strongest matches from the retrieved evidence:\n")
        + "\n".join(lines)
    )


def _build_public_source_listing_answer(citations: list[dict]) -> str:
    if not citations:
        return "I couldn’t find relevant scholarly sources for that query."
    lines = ["Here are the most relevant sources:"]
    for i, c in enumerate(citations[:6], start=1):
        title = (c.get("title") or f"Source {i}").strip()
        year = c.get("year")
        source = (c.get("source") or "source").strip()
        snippet = re.sub(r"\s+", " ", (c.get("snippet") or "").strip())
        snippet = re.split(r"(?<=[.!?;])\s+", snippet)[0][:180].strip(" -:")
        reason = snippet or "Relevant to the query based on semantic and lexical match."
        header = f"{i}. {title}"
        meta = f" ({year}, {source})" if year else f" ({source})"
        lines.append(f"{header}{meta}\n   Relevance: {reason} [S{i}]")
    return "\n".join(lines)


def _build_public_synthesis_fallback(citations: list[dict]) -> str:
    if not citations:
        return "I couldn’t find enough reliable scholarly evidence to synthesize an answer."
    bullets = []
    for i, c in enumerate(citations[:4], start=1):
        snippet = re.sub(r"\s+", " ", (c.get("snippet") or "").strip())
        snippet = re.split(r"(?<=[.!?;])\s+", snippet)[0][:190].strip(" -:")
        if snippet:
            bullets.append(f"- {snippet} [S{i}]")
    if not bullets:
        return _build_public_source_listing_answer(citations)
    return "Here is a research-backed synthesis based on the strongest retrieved sources:\n" + "\n".join(bullets)


def _append_public_source_links(answer: str, citations: list[dict]) -> str:
    if not answer or not citations:
        return answer
    public = [c for c in citations if (c.get("source") or "").lower() != "uploaded"]
    if not public:
        return answer
    chosen = [c for c in public if c.get("used_in_answer") and _normalize_source_url(c.get("url"))] or [
        c for c in public if _normalize_source_url(c.get("url"))
    ]
    if not chosen:
        return answer
    lines = []
    seen = set()
    for c in chosen[:5]:
        title = (c.get("title") or "Source").strip()
        url = _normalize_source_url(c.get("url"))
        key = (title.lower(), url)
        if not url or key in seen:
            continue
        seen.add(key)
        lines.append(f"- {title}: {url}")
    if not lines:
        return answer
    if "Source links:" in answer:
        return answer
    return answer.rstrip() + "\n\nSource links:\n" + "\n".join(lines)


def _build_uploaded_related_work_fallback(citations: list[dict]) -> str:
    if not citations:
        return "I couldn’t find related-work evidence in your uploaded paper."
    lines = []
    for i, c in enumerate(citations[:5], start=1):
        title = c.get("title") or f"Document {c.get('doc_id', '?')}"
        page = c.get("page")
        snippet = re.sub(r"\s+", " ", (c.get("snippet") or "").strip())[:220]
        header = f"{title} (p.{page})" if page is not None else title
        if snippet:
            lines.append(f"- {header}: {snippet} [S{i}]")
        else:
            lines.append(f"- {header} [S{i}]")
    return (
        "I found related/prior-work evidence in your uploaded paper. "
        "Here are the most relevant excerpts:\n"
        + "\n".join(lines)
    )


def _is_uploaded_doc_summary_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    cues = (
        "attached research paper",
        "attached paper",
        "uploaded paper",
        "uploaded research paper",
        "this paper",
        "that paper",
        "my paper",
        "summarize the paper",
        "summary of the paper",
        "tell me about the attached",
        "tell me about this document",
        "tell me about the uploaded doc",
        "tell me about the uploaded document",
        "tell me about my document",
        "tell me about my doc",
        "summarize this resume",
        "what skills are listed in this resume",
        "summarize this document",
    )
    return any(c in q for c in cues)


def _build_uploaded_evidence_fallback(query: str, citations: list[dict]) -> str:
    if not citations:
        return "I couldn’t find enough evidence in your uploaded documents for that query."

    unique_docs = {
        (c.get("title") or f"Document {c.get('doc_id', '?')}")
        for c in citations
        if c.get("title") or c.get("doc_id") is not None
    }
    multi_doc = len(unique_docs) > 1
    lead = "Here is a grounded summary from the selected uploaded evidence"
    if _is_uploaded_doc_summary_query(query):
        lead = "Here is a grounded summary from the uploaded documents" if multi_doc else "Here is a grounded summary from the uploaded document"
    grouped: dict[str, list[tuple[int, str]]] = {}
    for i, c in enumerate(citations[:6], start=1):
        title = c.get("title") or f"Document {c.get('doc_id', '?')}"
        page = c.get("page")
        snippet = re.sub(r"\s+", " ", (c.get("snippet") or "").strip())
        snippet = re.split(r"(?:\s*[|•]\s*|\s{2,})", snippet)[0]
        snippet = re.split(r"(?<=[.!?;])\s+", snippet)[0]
        snippet = snippet[:180].strip(" -:")
        header = f"{title} (p.{page})" if page is not None else title
        grouped.setdefault(header, [])
        if snippet:
            grouped[header].append((i, snippet))

    sections = []
    for header, items in grouped.items():
        joined = " ".join(f"{snippet} [S{idx}]" for idx, snippet in items[:2]).strip()
        if joined:
            prefix = f"{header}: " if multi_doc else "- "
            sections.append(f"{prefix}{joined}" if multi_doc else f"- {header}: {joined}")
        else:
            sections.append(header if multi_doc else f"- {header}")

    return f"{lead}:\n" + "\n".join(sections)


def _rebalance_uploaded_multi_doc_citations(citations: list[dict], doc_ids: list[int] | None, k: int) -> list[dict]:
    if not citations or not doc_ids or len(doc_ids) <= 1:
        return citations

    by_doc: dict[int, list[dict]] = {}
    for c in citations:
        did = c.get("doc_id")
        if did is None:
            continue
        did = int(did)
        by_doc.setdefault(did, []).append(c)

    if len(by_doc) <= 1:
        return citations

    for did in by_doc:
        by_doc[did] = sorted(
            by_doc[did],
            key=lambda item: (
                -float(item.get("rerank_norm", item.get("rerank_raw", item.get("sim_score", 0.0))) or 0.0),
                -float(item.get("sim_score", 0.0) or 0.0),
            ),
        )

    balanced: list[dict] = []
    seen: set[tuple[int | None, int | None]] = set()
    max_rounds = max(len(rows) for rows in by_doc.values())
    for idx in range(max_rounds):
        for did in doc_ids:
            rows = by_doc.get(int(did), [])
            if idx >= len(rows):
                continue
            row = rows[idx]
            key = (row.get("doc_id"), row.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            balanced.append(row)
            if len(balanced) >= int(k):
                return balanced

    return balanced or citations


def _build_multi_doc_uploaded_summary(citations: list[dict], doc_ids: list[int] | None) -> str:
    if not citations or not doc_ids or len(doc_ids) <= 1:
        return _build_uploaded_evidence_fallback("summary", citations)

    grouped: dict[int, list[dict]] = {}
    for c in citations:
        did = c.get("doc_id")
        if did is None:
            continue
        grouped.setdefault(int(did), []).append(c)

    sections: list[str] = []
    for did in doc_ids:
        rows = grouped.get(int(did), [])
        if not rows:
            continue
        title = rows[0].get("title") or f"Document {did}"
        bullets: list[str] = []
        for row in rows[:2]:
            sid = row.get("id")
            page = row.get("page")
            snippet = re.sub(r"\s+", " ", (row.get("snippet") or "").strip())
            snippet = re.split(r"(?:\s*[|•]\s*|\s{2,})", snippet)[0]
            snippet = re.split(r"(?<=[.!?;])\s+", snippet)[0]
            snippet = snippet[:170].strip(" -:")
            if not snippet:
                continue
            cite = f" [S{sid}]" if sid else ""
            page_prefix = f"(p.{page}) " if page is not None else ""
            bullets.append(f"- {page_prefix}{snippet}{cite}")
        if bullets:
            sections.append(f"{title}\n" + "\n".join(bullets))

    if not sections:
        return _build_uploaded_evidence_fallback("summary", citations)

    combined_takeaways: list[str] = []
    for did in doc_ids:
        rows = grouped.get(int(did), [])
        if not rows:
            continue
        title = rows[0].get("title") or f"Document {did}"
        first = rows[0]
        sid = first.get("id")
        cite = f" [S{sid}]" if sid else ""
        snippet = re.sub(r"\s+", " ", (first.get("snippet") or "").strip())
        snippet = re.split(r"(?:\s*[|•]\s*|\s{2,})", snippet)[0]
        snippet = re.split(r"(?<=[.!?;])\s+", snippet)[0]
        snippet = snippet[:120].strip(" -:")
        if snippet:
            combined_takeaways.append(f"- {title}: {snippet}{cite}")

    parts = ["Here is a grounded cross-document summary:"]
    parts.extend(sections)
    if combined_takeaways:
        parts.append("Combined takeaways")
        parts.extend(combined_takeaways)
    return "\n\n".join(parts)


def _is_explicit_uploaded_summary_request(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    markers = [
        "summarize the selected uploaded document",
        "summarize the selected uploaded documents",
        "organize the response by document",
        "combined takeaways",
        "extract the key skills",
        "extract the key skills, topics, or main points from each selected uploaded document",
        "what evidence best supports the main claims in each selected uploaded document",
    ]
    return any(marker in q for marker in markers)


def _source_breakdown(citations: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in citations or []:
        src = (c.get("source") or "unknown").lower()
        counts[src] = counts.get(src, 0) + 1
    return counts


def _uploaded_evidence_strength(citations: list[dict]) -> float:
    uploaded = [c for c in citations if (c.get("source") or "").lower() == "uploaded"]
    if not uploaded:
        return 0.0
    avg_conf = sum(float(c.get("confidence", 0.0) or 0.0) for c in uploaded) / max(1, len(uploaded))
    unique_docs = len({c.get("doc_id") for c in uploaded if c.get("doc_id") is not None})
    doc_coverage = _clamp01(unique_docs / 2.0)
    hit_factor = _clamp01(len(uploaded) / 6.0)
    return round(_clamp01(0.55 * avg_conf + 0.25 * hit_factor + 0.2 * doc_coverage), 3)


def _normalize_tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "what", "about", "tell", "into",
        "your", "have", "does", "is", "are", "was", "were", "can", "could", "would", "should",
        "any", "all", "how", "why", "when", "where", "who", "whom", "which", "whose",
    }
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in toks if len(t) > 2 and t not in stop}


def _query_anchor_terms(query: str) -> set[str]:
    generic = {
        "company", "general", "overview", "background", "about", "tell",
        "what", "who", "where", "when", "which", "please", "info", "information",
    }
    toks = _normalize_tokens(query)
    anchors = {t for t in toks if t not in generic}
    if anchors:
        return anchors
    return toks


def _primary_anchor_term(query: str) -> str | None:
    generic = {
        "company", "general", "overview", "background", "about", "tell",
        "what", "who", "where", "when", "which", "please", "info", "information",
        "in", "on", "for", "with", "the", "a", "an",
        "need", "needs", "know", "kinda", "kind", "type", "is", "this", "that",
        "want", "wanna", "would", "like", "need", "about", "me", "you", "i",
    }
    qlow = (query or "").lower()
    ordered = re.findall(r"[a-z0-9]+", qlow)
    m = re.search(r"(?:about|on|for)\s+([a-z0-9]+)", qlow)
    if m:
        cand = m.group(1)
        if len(cand) > 2 and cand not in generic:
            return cand
    for t in ordered:
        if len(t) <= 2 or t in generic:
            continue
        return t
    return None


def _has_anchor_match(query: str, citation: dict) -> bool:
    anchors = _query_anchor_terms(query)
    if not anchors:
        return True
    hay = f"{citation.get('title','')} {citation.get('snippet','')}".lower()
    primary = _primary_anchor_term(query)
    if primary and primary not in hay:
        return False
    return True


def _query_has_disambiguator(query: str) -> bool:
    q = (query or "").lower()
    hints = (
        "nlp", "llm", "language model", "bert", "gpt", "attention", "machine learning",
        "computer vision", "vision", "image",
        "electrical", "power", "grid", "voltage", "substation",
    )
    return any(h in q for h in hints)


def _infer_domain(citation: dict) -> str:
    hay = f"{citation.get('title','')} {citation.get('snippet','')}".lower()
    domain_rules = {
        "nlp_ai": ("nlp", "language model", "llm", "gpt", "bert", "token", "text"),
        "vision_ai": ("computer vision", "image", "segmentation", "detection"),
        "power_electrical": ("electrical", "power system", "transformer condition", "voltage", "thermal", "substation"),
    }
    best_domain = "other"
    best_hits = 0
    for d, keys in domain_rules.items():
        hits = sum(1 for k in keys if k in hay)
        if hits > best_hits:
            best_hits = hits
            best_domain = d
    return best_domain


def _ambiguous_domain_mix(query: str, citations: list[dict]) -> tuple[bool, list[str]]:
    if not citations:
        return False, []
    if _query_has_disambiguator(query):
        return False, []
    counts = {}
    for c in citations[:6]:
        d = _infer_domain(c)
        counts[d] = counts.get(d, 0) + 1
    counts.pop("other", None)
    if len(counts) <= 1:
        return False, []
    total = sum(counts.values())
    if total <= 0:
        return False, []
    dominant = max(counts.values()) / total
    if dominant < 0.72:
        labels = []
        if "nlp_ai" in counts:
            labels.append("NLP/LLM transformers")
        if "vision_ai" in counts:
            labels.append("computer vision transformers")
        if "power_electrical" in counts:
            labels.append("electrical power transformers")
        return True, labels
    return False, []


def _query_overlap_strength(query: str, citations: list[dict]) -> float:
    q = _normalize_tokens(query)
    if not q or not citations:
        return 0.0
    best = 0.0
    for c in citations[:6]:
        s = _normalize_tokens(c.get("snippet", ""))
        if not s:
            continue
        overlap = len(q & s) / max(1, len(q))
        best = max(best, overlap)
    return round(best, 3)


def _prune_public_citations(query: str, citations: list[dict]) -> list[dict]:
    if not citations:
        return citations
    q_tokens = _normalize_tokens(query)
    kept = []
    for c in citations:
        if not _has_anchor_match(query, c):
            continue
        ov = _chunk_query_overlap(query, c)
        hay = f"{c.get('title','')} {c.get('snippet','')}".lower()
        has_exact_query_token = any(t in hay for t in q_tokens) if q_tokens else False
        if ov >= 0.12 or has_exact_query_token or _definition_relevance_boost(query, c) > 0.0:
            kept.append(c)
    return kept


def _chunk_query_overlap(query: str, citation: dict) -> float:
    q = _normalize_tokens(query)
    if not q:
        return 0.0
    hay = f"{citation.get('title','')} {citation.get('snippet','')}"
    s = _normalize_tokens(hay)
    if not s:
        return 0.0
    return len(q & s) / max(1, len(q))


def _prune_uploaded_citations(query: str, citations: list[dict]) -> list[dict]:
    if len(citations) <= 2:
        return citations

    scored = []
    for c in citations:
        ov = _chunk_query_overlap(query, c)
        scored.append((ov, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_overlap = scored[0][0]
    best_doc = scored[0][1].get("doc_id")

    keep = []
    threshold = max(0.12, best_overlap - 0.16)
    for ov, c in scored:
        same_doc_as_best = c.get("doc_id") == best_doc
        conf = float(c.get("confidence", 0.0) or 0.0)
        if ov >= threshold or (same_doc_as_best and conf >= 0.45):
            keep.append(c)

    if not keep:
        keep = [c for _, c in scored[:2]]
    return keep


def _source_scope(citation: dict) -> str:
    hay = f"{citation.get('title','')} {citation.get('snippet','')}".lower()
    if any(k in hay for k in ("resume", "curriculum vitae", "experience", "co-op", "intern")):
        return "personal_profile"
    if any(k in hay for k in ("assignment", "lecture", "coursework", "homework")):
        return "course_material"
    if citation.get("source") == "uploaded":
        return "uploaded_document"
    return "public_reference"


def _is_definition_style_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if _is_doc_intent_query(q):
        return False
    starters = ("what is", "who is", "tell me about", "explain", "define")
    return q.startswith(starters) or " company" in q


def _is_profile_context_query(query: str) -> bool:
    q = (query or "").lower()
    profile_cues = (
        "resume",
        "cv",
        "profile",
        "experience",
        "worked",
        "intern",
        "project",
        "role",
        "gaurav",
        "my docs",
    )
    return any(c in q for c in profile_cues)


def _is_general_knowledge_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    cues = (
        "in general",
        "generally",
        "what is",
        "who is",
        "tell me about",
        "company",
        "overview",
        "background",
    )
    return any(c in q for c in cues)


def _is_source_listing_query(query: str) -> bool:
    q = (query or "").strip().lower()
    cues = (
        "show me papers",
        "list papers",
        "give me papers",
        "give me sources",
        "give me references",
        "relevant papers",
        "relevant sources",
        "papers only",
        "sources only",
        "references only",
        "bibliography",
        "citation list",
        "evidence only",
    )
    return any(c in q for c in cues)


def _is_research_synthesis_query(query: str) -> bool:
    q = (query or "").strip().lower()
    cues = (
        "what do papers say",
        "in the literature",
        "based on recent papers",
        "summarize research",
        "summarize the literature",
        "recent findings",
        "recent research",
        "what evidence supports",
        "findings about",
        "compare",
        "limitations",
    )
    return any(c in q for c in cues)


def _classify_answer_mode(query: str) -> str:
    if _is_source_listing_query(query):
        return "source_listing"
    if _is_research_synthesis_query(query) or _is_related_work_query(query):
        return "research_synthesis"
    return "explanatory"


def _build_generation_prompt(
    query: str,
    context: str,
    answer_mode: str,
    allow_general_background: bool,
    compare_instruction: str = "",
) -> str:
    background_rule = (
        "You may draw on well-established general knowledge to provide context, but every specific claim must be grounded in the provided sources."
        if allow_general_background
        else "You must rely ONLY on the provided sources. Do not introduce claims from general knowledge not present in the evidence."
    )

    system_block = f"""\
You are ScholarRAG, a PhD-level research assistant with deep expertise in analyzing and synthesizing academic literature. Your answers are authoritative, precise, and analytically rigorous — exceeding the quality of a generic language model by grounding every claim in the provided evidence.

CORE PRINCIPLES:
- Answer the question directly and completely. Lead with your answer, not with a description of what sources you found.
- Write in clear, scholarly prose. Use markdown formatting: ## headings for major sections, **bold** for key terms, bullet or numbered lists for multi-part findings, and `code` for technical identifiers.
- Every substantive factual claim must carry an inline citation in the form [S#] where # is the source number. Place citations immediately after the relevant sentence or clause, before the period.
- Never open with "I found sources", "Based on my search", "Here are the papers", or similar retrieval-reporting phrases.
- {background_rule}
- Do not fabricate citations, invent studies, or extrapolate beyond what the evidence supports.
- Do not repeat the question back to the user.
- The evidence panel handles source listing separately — your job is synthesis and explanation, not source dumping.
{("- " + compare_instruction.strip()) if compare_instruction and compare_instruction.strip() else ""}"""

    if answer_mode == "source_listing":
        mode_block = """\
RESPONSE FORMAT — source_listing:
The user is requesting a curated list of sources. Provide:
1. A single opening sentence summarizing the landscape (no citation needed here).
2. A numbered list of the most relevant sources. For each:
   - **Title** [S#]
   - One sentence on why it is relevant to the query
   - Key contribution or finding in 1–2 sentences
3. Do not write a long essay. Keep each entry concise."""

    elif answer_mode == "research_synthesis":
        mode_block = """\
RESPONSE FORMAT — research_synthesis:
Synthesize across multiple sources like a literature review section:
1. **Opening synthesis** (2–3 sentences): State the key consensus, dominant finding, or core tension in the literature on this topic. Use [S#] citations.
2. **Thematic sections** (use ## headings): Group findings by theme, methodology, or debate. Each section must contain at least one [S#] citation.
3. **Gaps & limitations** (if supported): Note disagreements, limitations, or open questions mentioned in the sources.
4. **Conclusion** (1 sentence): Summarize the take-home message.
Write at PhD-thesis quality. Avoid vague summaries — be analytically specific."""

    else:  # explanatory
        mode_block = """\
RESPONSE FORMAT — explanatory:
Answer like a brilliant research mentor explaining to a graduate student:
1. **Direct answer** (1–2 sentences): State the core answer immediately with a citation [S#] if applicable.
2. **Key explanation** (1–3 paragraphs): Explain the concept, mechanism, or finding in depth. Use bold for key terms, cite every substantive claim [S#].
3. **Supporting detail or examples** (optional, use bullets or numbered list): If the question calls for it, provide concrete examples, equations, or comparisons drawn directly from the sources.
4. **Nuances or caveats** (optional): Note limitations, assumptions, or conditions where the answer may differ.
Keep the answer focused. Do not pad with generic statements."""

    return (
        f"{system_block}\n\n"
        f"{mode_block}\n\n"
        f"---\n"
        f"QUESTION: {query}\n\n"
        f"EVIDENCE (numbered sources):\n{context}\n"
        f"---\n\n"
        f"Now write a complete, citation-grounded answer:"
    )


def _definition_relevance_boost(query: str, citation: dict) -> float:
    if not _is_definition_style_query(query):
        return 0.0
    hay = f"{citation.get('title','')} {citation.get('snippet','')} {(citation.get('source') or '')}".lower()
    boost = 0.0
    if any(term in hay for term in ("survey", "overview", "introduction", "intro", "tutorial")):
        boost += 0.18
    if any(term in hay for term in ("we define", "is a", "refers to", "designed for", "used for", "architecture")):
        boost += 0.12
    return boost


def _is_related_work_query(query: str) -> bool:
    q = (query or "").lower()
    cues = (
        "related work",
        "similar work",
        "similar papers",
        "relevant work",
        "prior work",
        "literature review",
        "baseline papers",
        "closest papers",
        "papers similar",
    )
    return any(c in q for c in cues)


def _is_company_intent_query(query: str) -> bool:
    q = (query or "").lower()
    if _is_doc_intent_query(q):
        return False
    company_cues = (
        " inc", " llc", " ltd", " corp", " company", " co.", " corporation",
        " technologies", " systems", " holdings", " group", " enterprises",
    )
    business_intent_cues = (
        "company overview", "about the company", "what company", "what does",
        "headquartered", "ticker", "founded", "market cap", "industry",
    )
    return any(c in q for c in company_cues) or any(c in q for c in business_intent_cues)


def _requested_public_source(query: str) -> str | None:
    q = (query or "").lower()
    mapping = (
        ("ieee", "ieee"),
        ("springer", "springer"),
        ("spirnger", "springer"),
        ("srpinger", "springer"),
        ("elsevier", "elsevier"),
        ("semantic scholar", "semanticscholar"),
        ("semanticscholar", "semanticscholar"),
        ("semantic", "semanticscholar"),
        ("openalex", "openalex"),
        ("arxiv", "arxiv"),
        ("crossref", "crossref"),
    )
    for token, source in mapping:
        if token in q:
            return source

    normalized_tokens = re.findall(r"[a-z]+", q)
    provider_names = ["ieee", "springer", "elsevier", "semanticscholar", "openalex", "arxiv", "crossref"]
    for t in normalized_tokens:
        match = difflib.get_close_matches(t, provider_names, n=1, cutoff=0.78)
        if match:
            return match[0]
    return None


def _is_entity_level_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if _is_doc_intent_query(q):
        return False
    patterns = (
        r"^tell me about\s+[a-z0-9 .,&-]+$",
        r"^what is\s+[a-z0-9 .,&-]+\??$",
        r"^[a-z0-9 .,&-]+\s+company$",
        r"^[a-z0-9 .,&-]+\s+irvine$",
    )
    has_pattern = any(re.match(p, q) for p in patterns)
    tokens = re.findall(r"[a-z0-9]+", q)
    short_entity_like = 1 <= len(tokens) <= 3
    role_terms = {"worked", "experience", "did", "role", "intern", "resume", "cv", "my"}
    research_terms = {"paper", "research", "study", "method", "results", "abstract", "dataset", "uploaded", "attached", "document", "docs"}
    has_role_intent = any(t in tokens for t in role_terms)
    has_research_intent = any(t in tokens for t in research_terms)
    return (has_pattern or short_entity_like or _is_company_intent_query(q)) and not has_role_intent and not has_research_intent


def _resolve_effective_doc_id(doc_id: int | None, scope: str, query: str) -> int | None:
    if scope != "uploaded" or doc_id is not None:
        return doc_id

    rows = fetchall(
        """
        SELECT id, title
        FROM documents
        WHERE status='ready'
        ORDER BY created_at DESC
        LIMIT 20
        """
    )
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0].get("id")

    q = (query or "").lower()
    for r in rows:
        title = (r.get("title") or "").lower()
        if title and (title in q or Path(title).stem in q):
            return r.get("id")
    return None


def _needs_scope_limited_answer(query: str, citations: list[dict]) -> bool:
    if not citations:
        return False
    if not (_is_definition_style_query(query) or _is_company_intent_query(query)):
        return False
    if _is_profile_context_query(query):
        return False
    has_public = any((c.get("scope") == "public_reference") for c in citations)
    if has_public:
        return False
    has_profile_or_course = any(
        (c.get("scope") in {"personal_profile", "course_material"}) for c in citations
    )
    return has_profile_or_course


def _has_official_company_docs() -> bool:
    row = fetchone(
        """
        SELECT COUNT(*) AS c
        FROM documents
        WHERE status='ready' AND doc_type IN ('official_doc', 'research_paper')
        """
    )
    return bool(row and int(row.get("c", 0) or 0) > 0)


def _scope_limited_answer(query: str, citations: list[dict]) -> str:
    first = citations[0] if citations else {}
    title = first.get("title") or "your uploaded source"
    sid = first.get("id", 1)
    q = (query or "").lower()
    q = re.sub(r"^(what is|who is|tell me about|explain|define|what company is)\s+", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(please|pls|kindly|about|the|a|an)\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip(" ?.")
    topic = q.title() if q else "this topic"
    return (
        f"I only found `{topic}` mentioned in profile/course context in your uploaded files "
        f"(for example, `{title}`), not as a general reference source. "
        f"I don’t have enough reliable evidence here to give a broad definition. [S{sid}]"
    )


def _rank_and_trim_citations(query: str, citations: list[dict], k: int, prefer_public: bool = False) -> list[dict]:
    if not citations:
        return citations
    ranked = []
    source_prior = {
        "semanticscholar": 0.18,
        "openalex": 0.16,
        "ieee": 0.16,
        "springer": 0.15,
        "elsevier": 0.15,
        "arxiv": 0.14,
        "web": 0.08,
        "crossref": -0.08,
        "unknown_public": 0.0,
        "uploaded": 0.0,
    }
    for idx, c in enumerate(citations, start=1):
        ov = _chunk_query_overlap(query, c)
        conf = float(c.get("confidence", 0.0) or 0.0)
        rel = (0.65 * ov) + (0.35 * conf)
        src = (c.get("source") or "").lower()
        rel += source_prior.get(src, 0.0)
        rel += _definition_relevance_boost(query, c)
        if prefer_public:
            if (c.get("source") or "").lower() != "uploaded":
                rel += 0.18
            else:
                rel -= 0.08
        cc = dict(c)
        cc["initial_rank"] = idx
        cc["rerank_raw"] = round(ov, 4)
        cc["rerank_norm"] = round(ov, 4)
        cc["reranker_type"] = "lexical_overlap"
        cc["_rel"] = rel
        ranked.append(cc)
    ranked.sort(key=lambda x: x.get("_rel", 0.0), reverse=True)
    top = ranked[0].get("_rel", 0.0)
    q = (query or "").lower()
    list_intent = any(t in q for t in ("papers", "paper", "studies", "references", "sources", "literature", "survey"))
    threshold = max(0.02, top - 0.45) if list_intent else max(0.10, top - 0.25)
    min_keep = min(max(1, k), 8 if list_intent else 3)
    kept = [c for c in ranked if c.get("_rel", 0.0) >= threshold][: max(1, k)]
    if len(kept) < min_keep:
        kept = ranked[:min_keep]
    if not kept:
        kept = ranked[: max(1, k)]

    if prefer_public:
        has_public = any((c.get("source") or "").lower() != "uploaded" for c in kept)
        if not has_public:
            public_candidates = [c for c in ranked if (c.get("source") or "").lower() != "uploaded"]
            if public_candidates:
                kept = [public_candidates[0]] + kept[:-1]

    for c in kept:
        c.pop("_rel", None)
    return kept


def _build_evidence_id(citation: dict) -> str:
    if (citation.get("source") or "") == "uploaded":
        doc_id = citation.get("doc_id")
        chunk_id = citation.get("chunk_id")
        page = citation.get("page")
        return f"uploaded:{doc_id}:{chunk_id}:{page}"

    source = (citation.get("source") or "public").lower()
    doi = (citation.get("url") or "").replace("https://doi.org/", "").replace("http://doi.org/", "")
    title = (citation.get("title") or "").lower()
    year = citation.get("year")
    base = doi or title or source
    return f"{source}:{base}:{year or ''}"


def _split_answer_sentences(answer: str) -> list[str]:
    if not answer:
        return []
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_sentence_citation_ids(sentence: str) -> list[int]:
    ids = []
    for m in re.finditer(r"\[S(\d+)\]", sentence):
        try:
            ids.append(int(m.group(1)))
        except Exception:
            continue
    return ids


def _stability_lookup_uploaded(q: str, k: int, doc_id: int | None, perturb: bool = False) -> set[str]:
    query = q if q else ""
    if perturb:
        query = (query + " methods overview").strip()
    results = search_uploaded_chunks(query, k=k, doc_id=doc_id).get("results", []) if query else []
    out = set()
    for r in results:
        c = {
            "source": "uploaded",
            "doc_id": r.get("document_id"),
            "chunk_id": r.get("id"),
            "page": r.get("page_no"),
        }
        out.add(_build_evidence_id(c))
    return out


def _stability_lookup_public(q: str, k: int, source_only: str | None = None, perturb: bool = False) -> set[str]:
    query = q if q else ""
    if perturb:
        query = (query + " methods").strip() if query else ""
        query = query.strip()
    papers = public_live_search(query, k=k, source_only=source_only) if query else []
    out = set()
    for p in papers:
        citation = {
            "source": p.get("source") or p.get("venue") or "public",
            "title": p.get("title"),
            "year": p.get("year"),
            "url": _normalize_source_url(p.get("url") or p.get("doi")),
        }
        out.add(_build_evidence_id(citation))
    return out


def _compute_stability_scores(query: str, k: int, scope: str, doc_id: int | None = None, source_only: str | None = None) -> dict[str, float]:
    if not query:
        return {}

    run_sets: list[set[str]] = []
    if scope == "uploaded":
        run_sets.append(_stability_lookup_uploaded(query, k, doc_id, perturb=False))
        run_sets.append(_stability_lookup_uploaded((query + " " + "related"), k, doc_id, perturb=True))
        run_sets.append(_stability_lookup_uploaded((query + " " + "overview"), k, doc_id, perturb=True))
        if doc_id is not None:
            run_sets.append(_stability_lookup_uploaded(query, k, None, perturb=False))
    else:
        run_sets.append(_stability_lookup_public(query, k, source_only=source_only, perturb=False))

    runs = max(1, len(run_sets))
    seen: dict[str, int] = {}
    for ids in run_sets:
        for evidence_id in ids:
            seen[evidence_id] = seen.get(evidence_id, 0) + 1

    return {eid: count / float(runs) for eid, count in seen.items()}


def _compute_agreement_score(sentence: str, context_map: dict[int, dict], evidence_id: str) -> float:
    if not sentence:
        return 0.0
    if not context_map:
        return 0.0

    candidates = []
    for c in context_map.values():
        text = c.get("snippet") or c.get("title") or ""
        if not text:
            continue
        src = c.get("doc_id") or c.get("source") or c.get("title") or c.get("chunk_id")
        candidates.append((src, text))

    if not candidates:
        return 0.0

    by_source = []
    seen = set()
    for src, text in candidates:
        if src in seen:
            continue
        seen.add(src)
        by_source.append(text)
        if len(by_source) >= 6:
            break

    support = 0
    total = max(1, len(by_source))
    for text in by_source:
        if entailment_prob(sentence, text) >= 0.70:
            support += 1

    return round(support / total, 4)


def _compute_citation_msa(
    query: str,
    answer: str,
    citations: list[dict],
    scope: str,
    k: int,
    doc_id: int | None = None,
    source_only: str | None = None,
) -> tuple[dict[int, dict], int]:
    stability = _compute_stability_scores(query, k=max(k, 8), scope=scope, doc_id=doc_id, source_only=source_only)

    context_by_id: dict[int, dict] = {}
    for idx, c in enumerate(citations, start=1):
        context_by_id[idx] = {
            "evidence_id": c.get("evidence_id"),
            "snippet": c.get("snippet", ""),
            "source": c.get("source"),
            "doc_id": c.get("doc_id"),
            "chunk_id": c.get("chunk_id"),
        }

    sentence_rows: list[dict] = []
    unsupported = 0
    for sidx, sentence in enumerate(_split_answer_sentences(answer), start=1):
        cleaned_sentence = re.sub(r"\[(?:S)?(\d+)\]", "", sentence).strip()
        if not cleaned_sentence:
            continue
        cited = _extract_sentence_citation_ids(sentence)
        if not cited:
            unsupported += 1
            continue
        for cidx in cited:
            cmeta = context_by_id.get(cidx)
            if not cmeta:
                unsupported += 1
                continue
            evidence_id = cmeta.get("evidence_id") or f"sentence-{sidx}-citation-{cidx}"
            m = round(entailment_prob(cleaned_sentence, cmeta.get("snippet", "")), 4)
            s = round(float(stability.get(evidence_id, 0.0)), 4)
            a = round(_compute_agreement_score(sentence, context_by_id, evidence_id), 4)

            sentence_rows.append(
                {
                    "sentence_id": sidx,
                    "citation_id": cidx,
                    "evidence_id": evidence_id,
                    "M": m,
                    "S": s,
                    "A": a,
                    "msa_score": build_confidence(
                        top_sim=0.0,
                        top_rerank_norm=0.0,
                        citation_coverage=0.0,
                        evidence_margin=0.0,
                        ambiguity_penalty=0.0,
                        insufficiency_penalty=0.0,
                        msa={"M": m, "S": s, "A": a, "weights": _load_latest_calibration_weights()},
                    )["factors"].get("msa", {}).get("msa_score", 0.0),
                }
            )

    request_id = hashlib.md5((query + answer).encode("utf-8")).hexdigest()
    for row in sentence_rows:
        try:
            execute(
                """
                INSERT INTO evidence_scores (request_id, sentence_id, citation_id, evidence_id, m_score, s_score, a_score, score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [request_id, row.get("sentence_id"), row.get("citation_id"), row.get("evidence_id"), row.get("M"), row.get("S"), row.get("A"), row.get("msa_score")],
            )
        except Exception:
            pass

    return {int(r.get("citation_id", 0)): {"M": r.get("M"), "S": r.get("S"), "A": r.get("A"), "msa_score": r.get("msa_score")} for r in sentence_rows}, unsupported
