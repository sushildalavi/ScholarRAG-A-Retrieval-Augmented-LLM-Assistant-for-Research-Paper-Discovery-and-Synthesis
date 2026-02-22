# ------------------------------
# app.py — ScholarRAG Backend API
# ------------------------------

import faiss
import json
import numpy as np
import os
import time
import re
import difflib
from datetime import datetime
from pathlib import Path
from typing import Any
from fastapi import Body
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from utils.config import get_openai_api_key
from backend.retriever import Retriever
from backend.generator import synthesize_answer
from backend import auth
from backend import memory
from backend import agents
from backend import pdf_ingest
from backend.user_store import get_user_index
from backend import chat
from fastapi.responses import Response
from backend.pdf_ingest import search_chunks as search_uploaded_chunks
from utils.embedding_utils import embed_query, embed_batch_cached
from utils.logging_utils import setup_file_logger, log_json
from backend.public_search import public_live_search
from backend.public_web import public_web_search
from backend.services.db import fetchall, fetchone, execute
from backend.confidence import build_confidence
from backend.eval_metrics import aggregate_metrics
from backend.sense_resolver import resolve_sense, filter_citations_by_sense

# Initialize FastAPI app
app = FastAPI(title="ScholarRAG API", version="1.0")

# Allow frontend access
# CORS: allow local dev frontend
# CORS: allow local dev frontend explicitly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(memory.router)
app.include_router(agents.router)
app.include_router(pdf_ingest.router)
app.include_router(chat.router)

RESEARCH_CHAT_MODEL = os.getenv("RESEARCH_CHAT_MODEL", "gpt-4o-mini")
ENABLE_WEB_FALLBACK = os.getenv("ENABLE_WEB_FALLBACK", "false").strip().lower() == "true"


def _ensure_eval_schema() -> None:
    execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            id SERIAL PRIMARY KEY,
            name TEXT,
            scope TEXT DEFAULT 'uploaded',
            k INT DEFAULT 10,
            case_count INT DEFAULT 0,
            metrics_retrieval_only JSONB,
            metrics_retrieval_rerank JSONB,
            latency_breakdown JSONB,
            details JSONB,
            created_at TIMESTAMP DEFAULT now()
        )
        """
    )


_ensure_eval_schema()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _normalize_inverse(value: float, min_v: float, max_v: float) -> float:
    # Lower value is better (e.g., vector distance).
    span = max(1e-6, max_v - min_v)
    return _clamp01((max_v - value) / span)


def _normalize_forward(value: float, min_v: float, max_v: float) -> float:
    # Higher value is better (e.g., cosine similarity).
    span = max(1e-6, max_v - min_v)
    return _clamp01((value - min_v) / span)


def _base_confidence(match_strength: float, rank: int, total: int, agreement: float) -> float:
    rank_stability = 1.0 if total <= 1 else 1.0 - ((rank - 1) / (total - 1))
    raw = _clamp01(0.65 * match_strength + 0.2 * rank_stability + 0.15 * agreement)
    # Calibrate to avoid extreme/overconfident values.
    calibrated = 0.28 + 0.58 * raw  # roughly [0.28, 0.86]
    return round(_clamp01(calibrated), 3)


def _confidence_breakdown(match_strength: float, rank: int, total: int, agreement: float) -> dict:
    rank_stability = 1.0 if total <= 1 else 1.0 - ((rank - 1) / (total - 1))
    raw = _clamp01(0.65 * match_strength + 0.2 * rank_stability + 0.15 * agreement)
    # Calibrate to avoid extreme/overconfident values.
    calibrated = 0.28 + 0.58 * raw  # roughly [0.28, 0.86]
    return {
        "match_strength": round(match_strength, 3),
        "rank_stability": round(rank_stability, 3),
        "agreement": round(agreement, 3),
        "raw": round(raw, 3),
        "calibrated": round(_clamp01(calibrated), 3),
    }


def _normalize_inline_citations(answer: str) -> str:
    """
    Normalize inline citation formatting without forcing citations into every sentence.
    """
    text = (answer or "").strip()
    if not text:
        return text
    # Normalize [S12] -> [S12] (already canonical) and collapse spacing/punctuation around citations.
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
    """
    Adjust confidence by how often each source is actually cited in the answer text.
    """
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
        # Bias toward retrieval confidence, then reward true usage in the answer.
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


def _chat_answer(query: str) -> str:
    if client is None:
        return "I can help with questions, but the language model is not configured right now."
    completion = client.chat.completions.create(
        model=RESEARCH_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise assistant. Answer naturally in plain language. "
                    "Do not fabricate citations or claim to read hidden files."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.4,
    )
    return (completion.choices[0].message.content or "").strip()


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
    """
    Deterministic fallback for public mode when strict claim-coverage check blocks LLM output.
    Keeps output grounded to retrieved public sources and explicit citations.
    """
    if not citations:
        return "I couldn’t find enough reliable public source evidence for this query."
    lines = []
    for i, c in enumerate(citations[:3], start=1):
        title = c.get("title") or f"Source {i}"
        year = c.get("year")
        snippet = (c.get("snippet") or "").strip()
        snippet = re.sub(r"\s+", " ", snippet)[:220]
        header = f"{title} ({year})" if year else title
        if snippet:
            lines.append(f"- {header}: {snippet} [S{i}]")
        else:
            lines.append(f"- {header} [S{i}]")
    return (
        "I found relevant public research sources for your query. "
        "Here are the strongest matches from the retrieved evidence:\n"
        + "\n".join(lines)
    )


def _source_breakdown(citations: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in citations or []:
        src = (c.get("source") or "unknown").lower()
        counts[src] = counts.get(src, 0) + 1
    return counts


def _uploaded_evidence_strength(citations: list[dict]) -> float:
    """
    Estimate how strong uploaded-only evidence is for a query.
    """
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
    """
    Extract anchor terms (entity/topic tokens) from query to avoid generic matches.
    Example: 'skyworks the company' -> {'skyworks'}.
    """
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
    """
    Pick the main entity/topic anchor from left to right to avoid matching on
    context modifiers like location words (e.g., 'in irvine').
    """
    generic = {
        "company", "general", "overview", "background", "about", "tell",
        "what", "who", "where", "when", "which", "please", "info", "information",
        "in", "on", "for", "with", "the", "a", "an",
        "need", "needs", "know", "kinda", "kind", "type", "is", "this", "that",
        "want", "wanna", "would", "like", "need", "about", "me", "you", "i",
    }
    qlow = (query or "").lower()
    ordered = re.findall(r"[a-z0-9]+", qlow)
    # Prefer token immediately after query intent cues.
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
    # Keep secondary anchors permissive after primary is satisfied.
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
    # If no single meaning dominates, ask for clarification.
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
    """
    Lexical sanity check: if query terms don't appear in retrieved snippets,
    uploaded evidence is probably off-topic even if vector ranks exist.
    """
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
    """
    Filter obviously irrelevant public citations for the current query.
    """
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
        # Keep only query-relevant public evidence to avoid unrelated "high confidence" bleed-through.
        if ov >= 0.12 or has_exact_query_token:
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
    """
    Remove weak off-topic uploaded chunks so answers are grounded in the right doc.
    """
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
    """
    Coarse source scope classification to prevent overgeneralization.
    """
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


def _is_company_intent_query(query: str) -> bool:
    q = (query or "").lower()
    company_cues = (" inc", " llc", " ltd", " corp", " company", " solutions", " technologies", " systems")
    question_cues = ("what is", "tell me", "about", "overview", "background")
    return any(c in q for c in company_cues) or any(c in q for c in question_cues)


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

    # Fuzzy fallback for misspelled provider names.
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
    patterns = (
        r"^tell me about\s+[a-z0-9 .,&-]+$",
        r"^what is\s+[a-z0-9 .,&-]+\??$",
        r"^[a-z0-9 .,&-]+\s+company$",
        r"^[a-z0-9 .,&-]+\s+irvine$",
    )
    has_pattern = any(re.match(p, q) for p in patterns)
    tokens = re.findall(r"[a-z0-9]+", q)
    short_entity_like = 1 <= len(tokens) <= 6
    role_terms = {"worked", "experience", "did", "role", "intern", "resume", "cv", "my"}
    has_role_intent = any(t in tokens for t in role_terms)
    return (has_pattern or short_entity_like or _is_company_intent_query(q)) and not has_role_intent


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
    """
    Generic relevance ranking across mixed sources.
    """
    if not citations:
        return citations
    ranked = []
    # Source-quality prior: prioritize scholarly provider APIs over generic metadata noise.
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
        # lexical relevance is primary; confidence secondary
        rel = (0.65 * ov) + (0.35 * conf)
        src = (c.get("source") or "").lower()
        rel += source_prior.get(src, 0.0)
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
    threshold = max(0.10, top - 0.25)
    kept = [c for c in ranked if c.get("_rel", 0.0) >= threshold][: max(1, k)]
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


@app.get("/favicon.ico")
def favicon():
    """Avoid 404 spam from browsers requesting favicon."""
    return Response(status_code=204)


# ------------------------------
# Metrics endpoint (stub / cached)
# ------------------------------


@app.get("/metrics")
def metrics():
    """
    Return simple evaluation/ops metrics for the UI.

    Replace the stub values with real numbers from your eval pipeline or a
    metrics table when available.
    """
    now = datetime.utcnow().isoformat() + "Z"
    return {
        "updated_at": now,
        "retrieval": {
            "recall_at_5": 0.73,
            "ndcg_at_10": 0.61,
            "mrr": 0.55,
        },
        "latency_ms": {
            "p50": 420,
            "p95": 980,
            "p99": 1600,
        },
        "token": {
            "avg_prompt": 520,
            "avg_completion": 180,
        },
        "sources": {
            "uploaded": 42,
            "arxiv": 28,
            "openalex": 18,
            "springer": 7,
            "ieee": 5,
        },
    }


# ------------------------------
# Assistant endpoint (RAG + GPT-4o-mini)
# ------------------------------


@app.post("/assistant/answer")
def assistant_answer(
    payload: dict = Body(
        ...,
        example={
            "query": "What does the paper really address?",
            "scope": "uploaded",
            "doc_id": None,
            "k": 10,
        },
    )
):
    """
    Unified QA endpoint for uploaded docs (chunk RAG) or public papers (FAISS/external).
    Returns answer plus lightweight citations.
    """
    started = time.time()
    t0 = time.perf_counter()
    query = payload.get("query") or ""
    scope = payload.get("scope") or "uploaded"
    doc_id = payload.get("doc_id")
    k = int(payload.get("k") or 10)
    multi_hop = bool(payload.get("multi_hop"))
    debug_confidence = bool(payload.get("debug_confidence"))
    allow_general_background = bool(payload.get("allow_general_background"))
    chosen_sense = payload.get("sense")
    compare_senses = bool(payload.get("compare_senses"))

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    qnorm = query.strip().lower()
    requested_public_source = _requested_public_source(query)
    # Heuristic routing: simple chat vs research
    small_talk_triggers = {"hi", "hello", "hey", "heyy", "sup", "ssup", "thanks", "thank you", "yo", "help", "how are you"}
    ui_help_actions = {"where", "how", "click", "button", "panel", "screen", "ui", "app"}
    ui_help_targets = {"upload", "uploaded", "document", "documents", "doc", "docs"}
    research_cues = {
        "paper",
        "study",
        "research",
        "citation",
        "doi",
        "arxiv",
        "openalex",
        "ieee",
        "springer",
        "journal",
        "conference",
        "dataset",
        "method",
        "results",
        "conclusion",
        "abstract",
        "experiment",
    }
    is_small_talk = len(qnorm.split()) <= 4 and any(t in qnorm for t in small_talk_triggers)
    # Only trigger canned UI guidance when question clearly asks "how/where to use the UI".
    # This avoids hijacking actual document questions like "can you see my attached docs?".
    is_ui_help = (
        any(t in qnorm for t in ui_help_actions)
        and any(t in qnorm for t in ui_help_targets)
        and not any(t in qnorm for t in research_cues)
    )
    is_research = any(t in qnorm for t in research_cues) or len(qnorm.split()) >= 6

    if is_ui_help:
        answer = (
            "Yes, upload files in the left `Upload & Query Docs` panel using `+ Upload Source` "
            "or drag-and-drop. Wait until the file status changes from `Processing` to `Processed`, "
            "then ask in `Ask about my docs...` for doc-grounded answers. "
            "Use the right `AI Assistant` box for general/public questions."
        )
        return {"answer": answer, "citations": []}

    if scope == "uploaded" and _is_doc_visibility_query(qnorm):
        rows = fetchall(
            "SELECT title, status FROM documents ORDER BY created_at DESC LIMIT 8"
        )
        ready = [r for r in rows if (r.get("status") or "").lower() == "ready"]
        if ready:
            preview = ", ".join((r.get("title") or "Untitled") for r in ready[:3])
            more = f" (+{len(ready) - 3} more)" if len(ready) > 3 else ""
            answer = (
                f"Yes. I can use your uploaded documents for retrieval. "
                f"Currently processed: {preview}{more}. "
                "Ask a content question (for example: 'Summarize DES key schedule from CSCI531_Lec6.pdf')."
            )
        else:
            answer = (
                "I can’t use uploaded docs yet because none are in `Processed` state. "
                "Wait for processing to finish, then ask your question again."
            )
        return {"answer": answer, "citations": []}

    # Keep uploaded-doc questions in retrieval mode even when short/colloquial.
    if not is_research and scope != "uploaded":
        try:
            answer = _chat_answer(query)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc

        log_json(
            REQUEST_LOG,
            {
                "ts": time.time(),
                "event": "assistant_answer_chat",
                "query": query,
                "scope": "chat",
                "citations": 0,
                "latency_ms": int((time.time() - started) * 1000),
            },
        )
        return {"answer": answer, "citations": []}

    if scope == "uploaded" and is_small_talk and not _is_doc_intent_query(qnorm):
        try:
            answer = _chat_answer(query)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc
        return {
            "answer": answer,
            "citations": [],
            "why_answer": {"rerank_changed_order": False, "top_chunks": []},
            "latency_breakdown_ms": {"retrieve": 0.0, "rerank": 0.0, "generate": 0.0, "total": int((time.time() - started) * 1000)},
            "retrieval_policy": {"mode": "chat-bypass", "uploaded_hits": 0, "public_hits": 0, "uploaded_strength": 0.0, "uploaded_overlap": 0.0, "used_public_fallback": False},
        }

    retrieval_ms = 0.0
    rerank_ms = 0.0
    generate_ms = 0.0

    def fetch_context(q: str, mode: str):
        local_citations = []
        if mode == "uploaded":
            results = search_uploaded_chunks(q, k=k, doc_id=doc_id)["results"]
            distances = [float(r.get("distance", 1.0) or 1.0) for r in results] or [1.0]
            cosines = [max(-1.0, min(1.0, 1.0 - d)) for d in distances] or [0.0]
            min_s, max_s = min(cosines), max(cosines)
            doc_counts = {}
            for r in results:
                did = r.get("document_id")
                doc_counts[did] = doc_counts.get(did, 0) + 1
            total = max(1, len(results))
            for rank, r in enumerate(results, start=1):
                dist = float(r.get("distance", 1.0) or 1.0)
                cosine = max(-1.0, min(1.0, 1.0 - dist))
                match_strength = _normalize_forward(cosine, min_s, max_s)
                support = _clamp01((doc_counts.get(r.get("document_id"), 1) - 1) / 3.0)
                conf = _base_confidence(match_strength, rank, total, support)
                conf_meta = _confidence_breakdown(match_strength, rank, total, support)
                local_citations.append(
                    {
                        "title": r.get("title") or f"Document {r.get('document_id')}",
                        "source": "uploaded",
                        "doc_id": r.get("document_id"),
                        "doc_type": r.get("doc_type") or "other",
                        "chunk_id": r.get("id"),
                        "page": r.get("page_no"),
                        "distance": r.get("distance"),
                        "sim_score": match_strength,
                        "sim_raw": round(cosine, 4),
                        "confidence": conf,
                        "_confidence_meta": conf_meta,
                        "snippet": r.get("text", ""),
                    }
                )
            local_citations = _prune_uploaded_citations(q, local_citations)
        elif mode == "public":
            docs = public_live_search(q, k=min(k, 8), source_only=requested_public_source)
            source_count = len({(d.get("source") or d.get("venue") or "public").lower() for d in docs})
            sims = [float(d.get("_sim", 0.0) or 0.0) for d in docs] or [0.0]
            min_s, max_s = min(sims), max(sims)
            total = max(1, len(docs))
            for rank, d in enumerate(docs, start=1):
                sim = float(d.get("_sim", 0.0) or 0.0)
                match_strength = _normalize_forward(sim, min_s, max_s)
                agreement = _clamp01(source_count / 3.0)
                conf = _base_confidence(match_strength, rank, total, agreement)
                conf_meta = _confidence_breakdown(match_strength, rank, total, agreement)
                local_citations.append(
                    {
                        "title": d.get("title"),
                        "year": d.get("year"),
                        "source": d.get("source") or d.get("venue"),
                        "url": _normalize_source_url(d.get("url") or d.get("doi")),
                        "similarity": d.get("_sim"),
                        "sim_score": match_strength,
                        "sim_raw": round(sim, 4),
                        "confidence": conf,
                        "_confidence_meta": conf_meta,
                        "snippet": d.get("abstract") or d.get("summary") or "",
                    }
                )
            if requested_public_source:
                local_citations = [
                    c for c in local_citations if (c.get("source") or "").lower() == requested_public_source
                ]
        elif mode == "web":
            docs = public_web_search(q, k=min(k, 8))
            sims = [float(d.get("_sim", 0.0) or 0.0) for d in docs] or [0.0]
            min_s, max_s = min(sims), max(sims)
            total = max(1, len(docs))
            for rank, d in enumerate(docs, start=1):
                sim = float(d.get("_sim", 0.0) or 0.0)
                match_strength = _normalize_forward(sim, min_s, max_s)
                agreement = _clamp01(1.0)
                conf = _base_confidence(match_strength, rank, total, agreement)
                conf_meta = _confidence_breakdown(match_strength, rank, total, agreement)
                local_citations.append(
                    {
                        "title": d.get("title"),
                        "year": None,
                        "source": d.get("source") or "web",
                        "url": _normalize_source_url(d.get("url")),
                        "similarity": d.get("_sim"),
                        "sim_score": match_strength,
                        "sim_raw": round(sim, 4),
                        "confidence": conf,
                        "_confidence_meta": conf_meta,
                        "snippet": d.get("snippet") or "",
                    }
                )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown retrieval mode: {mode}")
        return local_citations

    citations: list[dict] = []
    used_public_fallback = False
    uploaded_hits = 0
    public_hits = 0
    uploaded_strength = 0.0
    uploaded_overlap = 0.0

    if scope == "uploaded":
        if multi_hop and (" and " in query or ";" in query or "," in query):
            subqs = [q.strip() for q in re.split(r"and|;|,", query) if q.strip()]
            for sq in subqs:
                citations.extend(fetch_context(sq, "uploaded"))
        else:
            citations.extend(fetch_context(query, "uploaded"))

        if allow_general_background and (_is_general_knowledge_query(query) or _is_company_intent_query(query)):
            public_citations = fetch_context(query, "public")
            public_citations = _prune_public_citations(query, public_citations)
            if not public_citations and ENABLE_WEB_FALLBACK:
                public_citations = fetch_context(query, "web")
            if public_citations:
                used_public_fallback = True
                public_hits = len(public_citations)
                # In general-background mode, prefer public/web evidence as primary context.
                citations = public_citations + citations

        uploaded_hits = len(citations)
        uploaded_strength = _uploaded_evidence_strength(citations)
        uploaded_overlap = _query_overlap_strength(query, citations)
        strong_uploaded_match = uploaded_overlap >= 0.6 and uploaded_hits >= 1
        weak_uploaded = (
            not strong_uploaded_match
            and (
                uploaded_overlap < 0.22
                or (uploaded_hits < 3 and uploaded_strength < 0.52)
            )
        )
        if weak_uploaded and allow_general_background:
            used_public_fallback = True
            public_citations = fetch_context(query, "public")
            public_citations = _prune_public_citations(query, public_citations)
            public_hits = len(public_citations)
            if public_hits > 0:
                # Keep uploaded citations first in uploaded mode.
                citations = citations + public_citations
    else:
        if multi_hop and (" and " in query or ";" in query or "," in query):
            subqs = [q.strip() for q in re.split(r"and|;|,", query) if q.strip()]
            for sq in subqs:
                citations.extend(fetch_context(sq, "public"))
        else:
            citations.extend(fetch_context(query, "public"))
        public_hits = len(citations)

    retrieval_ms = (time.perf_counter() - t0) * 1000

    if not citations:
        # No retrieval: return a lightweight response without calling LLM on junk
        evidence_label = _scope_evidence_label(scope)
        if requested_public_source and scope != "uploaded":
            provider = requested_public_source.upper()
            return {
                "answer": (
                    f"I couldn't find relevant material from {provider} for that query. "
                    "Try a more specific topic (keywords, year range, or exact paper title)."
                ),
                "citations": [],
            }
        return {
            "answer": (
                f"I couldn't find relevant material in the selected {evidence_label} for that query. "
                "Try a more specific research question."
            ),
            "citations": [],
        }

    entity_query = _is_entity_level_query(query)
    all_personal_resume = all(
        ((c.get("doc_type") in {"resume"}) or (_source_scope(c) == "personal_profile"))
        for c in citations
    )
    no_official_docs = not _has_official_company_docs()
    if entity_query and all_personal_resume and no_official_docs:
        entity = _primary_anchor_term(query) or "this entity"
        return {
            "answer": (
                f"I only found references to {entity} within a personal document (for example a resume). "
                "I do not have broader company-level documentation in your uploaded sources. "
                "Would you like a summary of the resume context or allow general background knowledge?"
            ),
            "citations": [],
            "needs_clarification": True,
            "clarification": {
                "question": "Choose answer scope:",
                "options": ["Resume context summary", "Public company overview"],
                "recommended_option": "Public company overview" if allow_general_background else "Resume context summary",
                "rationale": "Entity-level query with only personal-document evidence.",
                "term": entity,
            },
            "confidence": {
                "score": 0.18,
                "label": "Context-limited",
                "needs_clarification": True,
                "factors": {
                    "top_sim": 0.0,
                    "top_rerank_norm": 0.0,
                    "citation_coverage": 0.0,
                    "evidence_margin": 0.0,
                    "ambiguity_penalty": 0.0,
                    "insufficiency_penalty": 0.35,
                    "scope_penalty": 1.0,
                },
                "explanation": "Only personal-document context is available for an entity-level/company query.",
            },
            "answer_scope": "personal_document_context",
            "unsupported_claims": 0,
            "why_answer": {"rerank_changed_order": False, "top_chunks": []},
            "latency_breakdown_ms": {
                "retrieve": round(retrieval_ms, 2),
                "rerank": 0.0,
                "generate": 0.0,
                "total": int((time.time() - started) * 1000),
            },
            "retrieval_policy": {
                "mode": "entity-scope-guard",
                "uploaded_hits": uploaded_hits,
                "public_hits": public_hits,
                "uploaded_strength": uploaded_strength,
                "uploaded_overlap": uploaded_overlap,
                "used_public_fallback": used_public_fallback,
            },
        }

    if scope == "uploaded" and allow_general_background and _is_general_knowledge_query(query):
        primary = _primary_anchor_term(query)
        has_public_primary = any(
            (c.get("source") or "").lower() != "uploaded"
            and (primary in f"{c.get('title','')} {c.get('snippet','')}".lower() if primary else True)
            for c in citations
        )
        if not has_public_primary:
            return {
                "answer": (
                    "I couldn’t find reliable public/web evidence for that specific entity/topic. "
                    "Please refine the query (for example include official company name/ticker) or provide a trusted source."
                ),
                "citations": [],
                "needs_clarification": True,
                "clarification": {
                    "question": "Can you provide a more specific public identifier (official name, ticker, or domain)?",
                    "options": [],
                    "recommended_option": None,
                    "rationale": "No public/web evidence matched the primary entity anchor.",
                    "term": primary,
                },
                "confidence": {
                    "score": 0.1,
                    "label": "Low",
                    "needs_clarification": True,
                    "factors": {
                        "top_sim": 0.0,
                        "top_rerank_norm": 0.0,
                        "citation_coverage": 0.0,
                        "evidence_margin": 0.0,
                        "ambiguity_penalty": 0.0,
                        "insufficiency_penalty": 1.0,
                    },
                    "explanation": "No public/web source matched the primary entity anchor.",
                },
                "why_answer": {"rerank_changed_order": False, "top_chunks": []},
                "latency_breakdown_ms": {
                    "retrieve": round(retrieval_ms, 2),
                    "rerank": 0.0,
                    "generate": 0.0,
                    "total": int((time.time() - started) * 1000),
                },
                "retrieval_policy": {
                    "mode": "general-background-anchor-guard",
                    "uploaded_hits": uploaded_hits,
                    "public_hits": public_hits,
                    "uploaded_strength": uploaded_strength,
                    "uploaded_overlap": uploaded_overlap,
                    "used_public_fallback": used_public_fallback,
                },
            }

    rerank_start = time.perf_counter()
    prefer_public = scope == "uploaded" and allow_general_background and (
        _is_general_knowledge_query(query) or _is_company_intent_query(query)
    )
    citations = _rank_and_trim_citations(query, citations, k, prefer_public=prefer_public)
    if prefer_public:
        public_only = [c for c in citations if (c.get("source") or "").lower() != "uploaded"]
        if public_only:
            citations = public_only[:k]
    # For definition-style questions, don't get stuck on resume/course-only evidence:
    # automatically try public evidence once before forcing a scope-limited response.
    if scope == "uploaded" and _needs_scope_limited_answer(query, citations):
        public_citations = fetch_context(query, "public")
        public_citations = _prune_public_citations(query, public_citations)
        if not public_citations and ENABLE_WEB_FALLBACK:
            # For non-academic entity questions, fall back to general web summaries.
            public_citations = fetch_context(query, "web")
        if public_citations:
            used_public_fallback = True
            public_hits = len(public_citations)
            citations = _rank_and_trim_citations(
                query,
                public_citations + citations,
                k,
                prefer_public=prefer_public,
            )
    rerank_ms = (time.perf_counter() - rerank_start) * 1000

    sense = resolve_sense(query, citations, chosen_sense=chosen_sense)
    if sense.get("is_ambiguous") and not compare_senses and not chosen_sense:
        return {
            "answer": "",
            "citations": [],
            "confidence": {
                "score": 0.2,
                "label": "Low",
                "needs_clarification": True,
                "factors": {
                    "top_sim": 0.0,
                    "top_rerank_norm": 0.0,
                    "citation_coverage": 0.0,
                    "evidence_margin": 0.0,
                    "ambiguity_penalty": 1.0,
                    "insufficiency_penalty": 0.0,
                },
                "explanation": "Query needs sense clarification before a grounded answer can be generated.",
            },
            "why_answer": {"rerank_changed_order": False, "top_chunks": []},
            "needs_clarification": True,
            "clarification": {
                "question": f"Do you mean {', '.join(sense.get('options', []))}?",
                "options": sense.get("options", []),
                "recommended_option": sense.get("recommended_option"),
                "rationale": sense.get("rationale"),
                "term": sense.get("term"),
            },
            "latency_breakdown_ms": {
                "retrieve": round(retrieval_ms, 2),
                "rerank": round(rerank_ms, 2),
                "generate": 0.0,
                "total": int((time.time() - started) * 1000),
            },
            "retrieval_policy": {
                "mode": "sense-resolver",
                "uploaded_hits": uploaded_hits,
                "public_hits": public_hits,
                "uploaded_strength": uploaded_strength,
                "uploaded_overlap": uploaded_overlap,
                "used_public_fallback": used_public_fallback,
            },
        }
    if chosen_sense and not compare_senses:
        citations = filter_citations_by_sense(citations, chosen_sense)

    if _is_company_intent_query(query):
        has_public = any((c.get("source") or "").lower() != "uploaded" for c in citations)
        has_profile = any((_source_scope(c) == "personal_profile") for c in citations)
        if has_profile and not has_public:
            return {
                "answer": (
                    "I only found company mentions in profile/resume context in your uploaded documents. "
                    "I don’t have reliable public evidence here to provide a company-level overview."
                ),
                "citations": [],
                "needs_clarification": True,
                "clarification": {
                    "question": "Do you want a profile-scoped summary from your docs, or a public company overview?",
                    "options": ["Profile-scoped summary", "Public company overview"],
                    "recommended_option": "Public company overview" if allow_general_background else "Profile-scoped summary",
                    "rationale": "Company intent detected but evidence is only personal profile context.",
                    "term": _primary_anchor_term(query),
                },
                "confidence": {
                    "score": 0.12,
                    "label": "Low",
                    "needs_clarification": True,
                    "factors": {
                        "top_sim": 0.0,
                        "top_rerank_norm": 0.0,
                        "citation_coverage": 0.0,
                        "evidence_margin": 0.0,
                        "ambiguity_penalty": 0.0,
                        "insufficiency_penalty": 1.0,
                    },
                    "explanation": "Detected company-intent query but only profile-scoped evidence was retrieved.",
                },
                "why_answer": {"rerank_changed_order": False, "top_chunks": []},
                "latency_breakdown_ms": {
                    "retrieve": round(retrieval_ms, 2),
                    "rerank": round(rerank_ms, 2),
                    "generate": 0.0,
                    "total": int((time.time() - started) * 1000),
                },
                "retrieval_policy": {
                    "mode": "company-intent-guard",
                    "uploaded_hits": uploaded_hits,
                    "public_hits": public_hits,
                    "uploaded_strength": uploaded_strength,
                    "uploaded_overlap": uploaded_overlap,
                    "used_public_fallback": used_public_fallback,
                },
            }

    context_lines = []
    for i, c in enumerate(citations, start=1):
        before_rank = int(c.get("initial_rank", i) or i)
        c["id"] = i
        c["rank_before"] = before_rank
        c["rank_after"] = i
        c["rank_delta"] = before_rank - i
        c["scope"] = _source_scope(c)
        c["rerank_raw"] = round(float(c.get("rerank_raw", _chunk_query_overlap(query, c)) or 0.0), 4)
        c["rerank_norm"] = round(float(c.get("rerank_norm", c.get("rerank_raw", 0.0)) or 0.0), 4)
        conf = float(c.get("confidence", 0.5))
        if c.get("source") == "uploaded":
            context_lines.append(
                f"[S{i}] doc {c.get('doc_id')} chunk {c.get('chunk_id')} page {c.get('page','?')} "
                f"(scope={c.get('scope')}, confidence={conf:.2f}): "
                f"{c.get('snippet','')}"
            )
        else:
            context_lines.append(
                f"[S{i}] {c.get('title','')} (scope={c.get('scope')}, confidence={conf:.2f}): {c.get('snippet','')}"
            )

    context = "\n\n".join(context_lines)
    compare_instruction = ""
    if compare_senses and sense.get("options"):
        compare_instruction = (
            "10) Compare senses mode is enabled. If multiple senses exist, write separate sections for each sense "
            f"from these options: {', '.join(sense.get('options', []))}. "
            "Do not merge senses in the same paragraph.\n"
        )

    prompt = (
        "You are ScholarRAG, a citation-grounded research assistant.\n"
        "Rules:\n"
        f"1) Use ONLY the provided sources. General background not in sources is {'allowed' if allow_general_background else 'NOT allowed'}.\n"
        "2) Keep answer concise and factual.\n"
        "3) Every paragraph or bullet that contains a claim must include at least one inline citation [S#].\n"
        "4) If you cannot support a claim from sources, do not state it.\n"
        "5) If evidence is weak, ask clarification or say: 'I don’t have enough evidence in your uploaded documents.'\n"
        "6) Do not invent sources.\n"
        "7) Do NOT say you cannot access documents/files; you can read the provided context.\n\n"
        "8) Respect source scope labels. If scope is personal_profile or course_material, do not generalize claims as "
        "global company/world facts unless corroborated by public_reference sources.\n"
        "9) When evidence only covers a person's role at a company, explicitly say that scope limitation.\n"
        f"{compare_instruction}\n"
        f"Question:\n{query}\n\nContext:\n{context}\n"
    )

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI client not configured. Set OPENAI_API_KEY (and install python-dotenv if relying on .env).",
        )

    gen_start = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=RESEARCH_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content or ""
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc
    generate_ms = (time.perf_counter() - gen_start) * 1000

    answer = _normalize_inline_citations(answer)
    answer = _humanize_answer_text(answer)
    citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)
    if scope == "uploaded" and _needs_scope_limited_answer(query, citations):
        answer = _scope_limited_answer(query, citations)
    if scope == "uploaded" and re.search(
        r"(cannot|can't|can not|not able to)\s+(access|see|view|read).*(documents|docs|files)",
        answer,
        flags=re.IGNORECASE,
    ):
        answer = (
            "I can use your uploaded documents through retrieval context. "
            "For this question, I need a bit more specific wording or clearer evidence in the indexed chunks. "
            "Insufficient evidence."
        )
    if unsupported_claims > 0:
        if scope != "uploaded" and citations:
            answer = _build_public_evidence_fallback(query, citations)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)
        else:
            evidence_label = _scope_evidence_label(scope)
            answer = (
                f"I don’t have enough evidence in the selected {evidence_label} to support all claims for this query. "
                "Please clarify the sense/topic or provide more specific sources."
            )
            # Do not present misleading evidence cards when answer is explicitly blocked.
            citations = []
    citations = _apply_usage_boost(citations, answer)
    cited_count = sum(1 for c in citations if c.get("used_in_answer"))
    avg_sim = (
        sum(float(c.get("sim_score", 0.0) or 0.0) for c in citations) / max(1, len(citations))
        if citations
        else 0.0
    )
    avg_rerank = (
        sum(float(c.get("rerank_score", 0.0) or 0.0) for c in citations) / max(1, len(citations))
        if citations
        else 0.0
    )
    citation_coverage = max(cited_count / max(1, len(citations)), citation_coverage_par)
    sorted_by_sim = sorted(citations, key=lambda x: float(x.get("sim_score", 0.0) or 0.0), reverse=True)
    top_sim = float(sorted_by_sim[0].get("sim_score", 0.0) or 0.0) if sorted_by_sim else 0.0
    top_rerank_norm = float(citations[0].get("rerank_norm", 0.0) or 0.0) if citations else 0.0
    if len(sorted_by_sim) > 1:
        evidence_margin = max(0.0, top_sim - float(sorted_by_sim[1].get("sim_score", 0.0) or 0.0))
    else:
        evidence_margin = top_sim
    ambiguity_penalty = 0.35 if (sense.get("is_ambiguous") and compare_senses) else 0.0
    insufficiency_penalty = 0.5 if unsupported_claims > 0 else (0.25 if "enough evidence" in (answer or "").lower() else 0.0)
    personal_only = bool(citations) and all((_source_scope(c) == "personal_profile") for c in citations)
    scope_penalty = 0.7 if (entity_query and personal_only) else 0.0
    confidence = build_confidence(
        top_sim=top_sim,
        top_rerank_norm=top_rerank_norm,
        citation_coverage=citation_coverage,
        evidence_margin=evidence_margin,
        ambiguity_penalty=ambiguity_penalty,
        insufficiency_penalty=insufficiency_penalty,
        scope_penalty=scope_penalty,
        needs_clarification=False,
    )
    trust = round(min(1.0, len(citations) / max(1, k)), 3)
    latency_ms = int((time.time() - started) * 1000)

    trace_chunks = []
    rerank_changed = False
    for c in citations:
        if (c.get("rank_delta") or 0) != 0:
            rerank_changed = True
        trace_chunks.append(
            {
                "id": c.get("id"),
                "title": c.get("title"),
                "doc_id": c.get("doc_id"),
                "chunk_id": c.get("chunk_id"),
                "page": c.get("page"),
                "snippet_preview": (c.get("snippet", "") or "")[:260],
                "sim_score": round(float(c.get("sim_score", 0.0) or 0.0), 4),
                "sim_raw": round(float(c.get("sim_raw", c.get("sim_score", 0.0)) or 0.0), 4),
                "rerank_raw": round(float(c.get("rerank_raw", 0.0) or 0.0), 4),
                "rerank_norm": round(float(c.get("rerank_norm", 0.0) or 0.0), 4),
                "rank_before": c.get("rank_before"),
                "rank_after": c.get("rank_after"),
                "rank_delta": c.get("rank_delta"),
                "cited": bool(c.get("used_in_answer")),
                "source": c.get("source"),
                "scope": c.get("scope"),
                "reranker_type": c.get("reranker_type"),
            }
        )

    if unsupported_claims > 0:
        trace_chunks = []
        rerank_changed = False

    log_json(
        REQUEST_LOG,
        {
            "ts": time.time(),
            "event": "assistant_answer",
            "query": query,
            "scope": scope,
            "doc_id": doc_id,
            "k": k,
            "multi_hop": multi_hop,
            "uploaded_hits": uploaded_hits,
            "public_hits": public_hits,
            "uploaded_strength": uploaded_strength,
            "uploaded_overlap": uploaded_overlap,
            "used_public_fallback": used_public_fallback,
            "context_count": len(citations),
            "citations": len(citations),
            "trust": trust,
            "confidence_score": confidence.get("score"),
            "latency_ms": latency_ms,
        },
    )

    citations_out = [{k: v for k, v in c.items() if k != "snippet"} for c in citations]
    for c in citations_out:
        c["confidence_obj"] = build_confidence(
            top_sim=float(c.get("sim_score", 0.0) or 0.0),
            top_rerank_norm=float(c.get("rerank_norm", 0.0) or 0.0),
            citation_coverage=1.0 if c.get("used_in_answer") else 0.0,
            evidence_margin=float(c.get("sim_score", 0.0) or 0.0),
            ambiguity_penalty=0.0,
            insufficiency_penalty=0.0,
            scope_penalty=0.0,
            needs_clarification=False,
        )
    if not debug_confidence:
        for c in citations_out:
            c.pop("_confidence_meta", None)
            c.pop("base_confidence", None)
            c.pop("usage_boost", None)

    if citations:
        if all((_source_scope(c) == "personal_profile") for c in citations):
            scope_label = "personal_document_context"
        elif any((c.get("source") or "").lower() != "uploaded" for c in citations):
            scope_label = "official_document_context"
        else:
            scope_label = "uploaded_document_context"
    else:
        scope_label = "retrieved_context"

    return {
        "answer": answer,
        "citations": citations_out,
        "confidence": confidence,
        "needs_clarification": False,
        "clarification": None,
        "answer_scope": chosen_sense or ("compare_senses" if compare_senses else scope_label),
        "unsupported_claims": unsupported_claims,
        "trust": trust,
        "latency_ms": latency_ms,
        "confidence_note": "Confidence is heuristic, not a ground-truth probability. Use debug_confidence=true for per-source breakdown.",
        "why_answer": {
            "rerank_changed_order": rerank_changed,
            "top_chunks": trace_chunks,
        },
        "scoring": {
            "similarity_metric": "cosine_similarity",
            "reranker_used": True,
            "reranker_type": "lexical_overlap",
            "rerank_score_fields": ["rerank_raw", "rerank_norm"],
        },
        "latency_breakdown_ms": {
            "retrieve": round(retrieval_ms, 2),
            "rerank": round(rerank_ms, 2),
            "generate": round(generate_ms, 2),
            "total": latency_ms,
        },
        "retrieval_policy": {
            "mode": "uploaded-first" if scope == "uploaded" else "public-only",
            "uploaded_hits": uploaded_hits,
            "public_hits": public_hits,
            "uploaded_strength": uploaded_strength,
            "uploaded_overlap": uploaded_overlap,
            "used_public_fallback": used_public_fallback,
            "source_breakdown": _source_breakdown(citations),
        },
    }


@app.post("/assistant/resolve_sense")
def assistant_resolve_sense(payload: dict = Body(...)):
    query = (payload.get("query") or "").strip()
    scope = payload.get("scope") or "uploaded"
    k = int(payload.get("k") or 8)
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    chunks = []
    if scope == "uploaded":
        rows = search_uploaded_chunks(query, k=k, doc_id=payload.get("doc_id"))["results"]
        for r in rows:
            chunks.append(
                {
                    "title": r.get("title"),
                    "snippet": r.get("text", ""),
                    "doc_id": r.get("document_id"),
                    "chunk_id": r.get("id"),
                    "page": r.get("page_no"),
                }
            )
    sense = resolve_sense(query, chunks, chosen_sense=payload.get("sense"))
    return sense


@app.get("/metrics/requests")
def metrics_requests():
    """
    Lightweight aggregation over logs/requests.jsonl (assistant_answer events).
    """
    import json

    path = LOG_DIR / "requests.jsonl"
    if not path.exists():
        return {"count": 0, "avg_latency_ms": None, "avg_trust": None, "avg_citations": None}

    latencies = []
    trusts = []
    cits = []
    count = 0
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event") != "assistant_answer":
                continue
            count += 1
            if rec.get("latency_ms") is not None:
                latencies.append(rec["latency_ms"])
            if rec.get("trust") is not None:
                trusts.append(rec["trust"])
            if rec.get("citations") is not None:
                cits.append(rec["citations"])

    def avg(arr):
        return sum(arr) / len(arr) if arr else None

    return {
        "count": count,
        "avg_latency_ms": avg(latencies),
        "avg_trust": avg(trusts),
        "avg_citations": avg(cits),
    }


def _eval_candidates_for_query(query: str, k: int) -> tuple[list[dict], list[dict], dict]:
    t_retrieve = time.perf_counter()
    raw = search_uploaded_chunks(query, k=max(10, k))["results"]
    retrieve_ms = (time.perf_counter() - t_retrieve) * 1000

    retrieval_only = []
    for idx, r in enumerate(raw, start=1):
        retrieval_only.append(
            {
                "doc_id": r.get("document_id"),
                "chunk_id": r.get("id"),
                "page": r.get("page_no"),
                "title": r.get("title") or f"Document {r.get('document_id')}",
                "distance": float(r.get("distance", 1.0) or 1.0),
                "snippet": r.get("text", ""),
                "initial_rank": idx,
                "confidence": 1.0 - min(1.0, float(r.get("distance", 1.0) or 1.0)),
            }
        )

    t_rerank = time.perf_counter()
    reranked = _rank_and_trim_citations(query, retrieval_only, k=max(10, k))
    rerank_ms = (time.perf_counter() - t_rerank) * 1000
    for i, c in enumerate(reranked, start=1):
        c["rank_after"] = i

    return retrieval_only[:k], reranked[:k], {"retrieve_ms": round(retrieve_ms, 2), "rerank_ms": round(rerank_ms, 2)}


@app.post("/eval/run")
def run_eval(payload: dict = Body(...)):
    name = (payload.get("name") or "Eval run").strip()
    scope = payload.get("scope") or "uploaded"
    k = int(payload.get("k") or 10)
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise HTTPException(status_code=400, detail="cases must be a non-empty list")
    if scope != "uploaded":
        raise HTTPException(status_code=400, detail="Eval currently supports uploaded scope only")

    retrieval_rows = []
    rerank_rows = []
    details = []
    lat_retrieve = []
    lat_rerank = []
    lat_generate = []

    for case in cases:
        query = (case.get("query") or "").strip()
        gold_doc_id = case.get("expected_doc_id")
        if not query:
            continue

        base, reranked, lat = _eval_candidates_for_query(query, k)
        lat_retrieve.append(lat["retrieve_ms"])
        lat_rerank.append(lat["rerank_ms"])
        lat_generate.append(0.0)

        retrieval_pred = [int(x.get("doc_id")) for x in base if x.get("doc_id") is not None]
        rerank_pred = [int(x.get("doc_id")) for x in reranked if x.get("doc_id") is not None]
        retrieval_rows.append({"pred_doc_ids": retrieval_pred, "gold_doc_id": gold_doc_id})
        rerank_rows.append({"pred_doc_ids": rerank_pred, "gold_doc_id": gold_doc_id})
        details.append(
            {
                "query": query,
                "gold_doc_id": gold_doc_id,
                "retrieval_only_top": base[:5],
                "rerank_top": reranked[:5],
                "latency_ms": lat,
            }
        )

    metrics_retrieval_only = aggregate_metrics(retrieval_rows)
    metrics_retrieval_rerank = aggregate_metrics(rerank_rows)
    latency_breakdown = {
        "retrieve_ms_avg": round(sum(lat_retrieve) / max(1, len(lat_retrieve)), 2),
        "rerank_ms_avg": round(sum(lat_rerank) / max(1, len(lat_rerank)), 2),
        "generate_ms_avg": round(sum(lat_generate) / max(1, len(lat_generate)), 2),
    }

    row = fetchone(
        """
        INSERT INTO eval_runs
        (name, scope, k, case_count, metrics_retrieval_only, metrics_retrieval_rerank, latency_breakdown, details)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb)
        RETURNING id, created_at
        """,
        [
            name,
            scope,
            k,
            len(details),
            json.dumps(metrics_retrieval_only),
            json.dumps(metrics_retrieval_rerank),
            json.dumps(latency_breakdown),
            json.dumps(details),
        ],
    )

    return {
        "run_id": row.get("id") if row else None,
        "created_at": row.get("created_at").isoformat() if row and row.get("created_at") else None,
        "name": name,
        "scope": scope,
        "k": k,
        "case_count": len(details),
        "metrics_retrieval_only": metrics_retrieval_only,
        "metrics_retrieval_rerank": metrics_retrieval_rerank,
        "latency_breakdown": latency_breakdown,
        "details": details,
    }


@app.get("/eval/runs")
def list_eval_runs(limit: int = 20):
    rows = fetchall(
        """
        SELECT id, name, scope, k, case_count, metrics_retrieval_only, metrics_retrieval_rerank, latency_breakdown, created_at
        FROM eval_runs
        ORDER BY id DESC
        LIMIT %s
        """,
        [max(1, min(limit, 100))],
    )
    out = []
    for r in rows:
        out.append(
            {
                "id": r.get("id"),
                "name": r.get("name"),
                "scope": r.get("scope"),
                "k": r.get("k"),
                "case_count": r.get("case_count"),
                "metrics_retrieval_only": r.get("metrics_retrieval_only"),
                "metrics_retrieval_rerank": r.get("metrics_retrieval_rerank"),
                "latency_breakdown": r.get("latency_breakdown"),
                "created_at": r.get("created_at").isoformat() if r.get("created_at") else None,
            }
        )
    return {"runs": out}

# ------------------------------
# Load FAISS index and metadata
# ------------------------------
INDEX_PATH = "data/scholar_index.faiss"
META_PATH = "data/metadata.json"
LOG_DIR = Path("logs")
RETRIEVAL_LOG = LOG_DIR / "retrieval.log"

try:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"✅ Loaded FAISS index with {index.ntotal} vectors and metadata for {len(meta)} papers")
except Exception as e:
    print(f"❌ Error loading FAISS index or metadata: {e}")
    index, meta = None, []

retriever = Retriever(index, meta) if index is not None else None

# ------------------------------
# OpenAI Embedding Config
# ------------------------------
try:
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)
except RuntimeError as err:
    print(f"❌ {err}")
    client = None

EMBED_MODEL = "text-embedding-3-small"  # aligned with 1536-dim pgvector schema

# Logger for observability
REQUEST_LOG = setup_file_logger(LOG_DIR / "requests.jsonl")

def get_embedding(text: str) -> np.ndarray:
    """Generate embedding vector for a query string."""
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured.")
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array([response.data[0].embedding], dtype="float32")




def log_request(entry: dict) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with RETRIEVAL_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def trust_score(sim: float, has_doi: bool) -> float:
    # Simple heuristic trust: similarity plus DOI bonus
    base = max(sim, 0.0)
    bonus = 0.05 if has_doi else 0.0
    return round(min(base + bonus, 1.0), 3)

# ------------------------------
# Endpoints
# ------------------------------

@app.get("/")
def home():
    return {"message": "ScholarRAG backend is live!"}

@app.get("/feed/latest")
def latest_papers(limit: int = 10):
    """Return latest papers from the global metadata by year (desc)."""
    if not meta:
        return {"results": []}
    def year_val(m):
        try:
            return int(m.get("year") or 0)
        except Exception:
            return 0
    sorted_meta = sorted(meta, key=year_val, reverse=True)
    out = []
    for m in sorted_meta[: max(1, limit)]:
        link = None
        if m.get("doi"):
            link = f"https://doi.org/{m.get('doi')}"
        elif m.get("id"):
            link = m.get("id")
        out.append(
            {
                "title": m.get("title", "Untitled"),
                "year": m.get("year"),
                "doi": m.get("doi"),
                "summary": (m.get("abstract") or m.get("summary") or "")[:280],
                "concepts": m.get("concepts", [])[:5],
                "url": link,
            }
        )
    return {"results": out}

@app.get("/search")
def search_papers(query: str = Query(..., description="Search query text"), k: int = 5):
    """Return top-k relevant papers for a given query."""
    if index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded.")

    q_emb = get_embedding(query)
    D, I = index.search(q_emb, k)
    results = []
    for rank, idx in enumerate(I[0]):
        if idx < len(meta):
            meta_entry = meta[idx]
            concepts = meta_entry.get("concepts") or []
            results.append({
                "rank": rank + 1,
                "title": meta_entry.get("title", "Unknown Title"),
                "year": meta_entry.get("year", "Unknown Year"),
                "doi": meta_entry.get("doi", ""),
                "concepts": concepts[:3],
                "similarity": round(float(D[0][rank]), 3)
            })
    return {"query": query, "results": results}

@app.get("/summarize")
def summarize(query: str = Query(..., description="Topic to summarize")):
    """Summarize top retrieved papers for a given query using GPT."""
    if index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded.")
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured.")

    q_emb = get_embedding(query)
    D, I = index.search(q_emb, 5)
    numbered_titles = [
        f"[P{i+1}] {meta[idx].get('title', 'Unknown Title')}"
        for i, idx in enumerate(I[0])
        if idx < len(meta)
    ]
    top_titles = "\n".join(numbered_titles)

    prompt = (
        f"Summarize key themes and insights for '{query}' using ONLY the papers below.\n"
        "Add inline citations for each claim using [P#].\n\n"
        f"Papers:\n{top_titles}"
    )
    completion = client.chat.completions.create(
        model=RESEARCH_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    summary = completion.choices[0].message.content
    return {"query": query, "summary": summary}

@app.post("/ask")
def ask(payload: dict = Body(...)):
    if index is None or retriever is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded.")
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured.")

    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'.")

    k = int(payload.get("k", 10))
    year_from = payload.get("year_from")
    year_to = payload.get("year_to")
    multi_hop = bool(payload.get("multi_hop", True))
    user_id = payload.get("user_id") or "guest"

    # Select per-user index if available, otherwise default global
    local_idx, local_meta = get_user_index(user_id)
    active_retriever = retriever
    if local_idx is not None and local_meta:
        active_retriever = Retriever(local_idx, local_meta)

    start = time.perf_counter()
    docs, stats = active_retriever.retrieve(query, k=k, min_pool=20, year_from=year_from, year_to=year_to, multi_hop=multi_hop)
    synthesis = synthesize_answer(query, docs)
    latency_ms = (time.perf_counter() - start) * 1000

    # Shape sources
    sources = []
    for d in docs:
        sim = round(float(d.get("_sim", 0.0)), 3)
        t_score = trust_score(sim, bool(d.get("doi")))
        sources.append({
            "title": d.get("title", "Unknown Title"),
            "year": d.get("year", "Unknown Year"),
            "doi": d.get("doi", ""),
            "openalex_id": d.get("id"),
            "arxiv_id": d.get("arxiv_id"),
            "concepts": d.get("concepts", [])[:5],
            "why_relevant": d.get("why_relevant", ""),
            "snippet": (d.get("abstract") or d.get("summary") or "")[:900],
            "similarity": sim,
            "trust_score": t_score,
            "authors": d.get("authors", []),
            "url": d.get("url"),
        })

    similarities = [s["similarity"] for s in sources if s.get("similarity") is not None]
    metrics = {
        "latency_ms": round(latency_ms, 2),
        "fallback_used": stats.get("fallback_used", False),
        "pool_size": stats.get("candidate_counts", {}).get("pool"),
        "ranked": stats.get("candidate_counts", {}).get("scored"),
        "openalex_added": stats.get("candidate_counts", {}).get("openalex"),
        "arxiv_added": stats.get("candidate_counts", {}).get("arxiv"),
        "max_similarity": max(similarities) if similarities else None,
        "mean_similarity": round(sum(similarities) / len(similarities), 3) if similarities else None,
        "token_prompt": synthesis.get("usage", {}).get("prompt_tokens"),
        "token_completion": synthesis.get("usage", {}).get("completion_tokens"),
        "token_total": synthesis.get("usage", {}).get("total_tokens"),
    }

    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "k": k,
        "year_from": year_from,
        "year_to": year_to,
        "multi_hop": multi_hop,
        "metrics": metrics,
        "candidate_counts": stats.get("candidate_counts", {}),
        "fallback_used": stats.get("fallback_used", False),
        "hops": stats.get("hops", []),
        "sources": [
            {
                "title": s.get("title"),
                "openalex_id": s.get("openalex_id"),
                "similarity": s.get("similarity"),
                "trust_score": s.get("trust_score"),
            }
            for s in sources
        ],
    }
    log_request(log_entry)

    return {
        "answer": synthesis.get("answer", ""),
        "sources": sources,
        "fallback_used": stats.get("fallback_used", False),
        "candidate_counts": stats.get("candidate_counts", {}),
        "metrics": metrics,
        "hops": stats.get("hops", []),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
