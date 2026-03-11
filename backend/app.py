# ------------------------------
# app.py — ScholarRAG Backend API
# ------------------------------

import json
import numpy as np
import os
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from fastapi import Body
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from utils.config import get_openai_api_key
from backend import auth
from backend import memory
from backend import agents
from backend import pdf_ingest
from backend import chat
from fastapi.responses import Response
from backend.pdf_ingest import search_chunks as search_uploaded_chunks
from utils.logging_utils import setup_file_logger, log_json
from backend.public_search import public_live_search
from backend.public_web import public_web_search
from backend.services.db import fetchall, fetchone, execute
from backend.services.assistant_utils import (
    _apply_usage_boost,
    _append_public_source_links,
    _base_confidence,
    _build_generation_prompt,
    _build_evidence_id,
    _build_multi_doc_uploaded_summary,
    _build_public_source_listing_answer,
    _build_public_synthesis_fallback,
    _build_uploaded_evidence_fallback,
    _build_uploaded_related_work_fallback,
    _classify_answer_mode,
    _chunk_query_overlap,
    _citation_coverage_stats,
    _clamp01,
    _compute_citation_msa,
    _confidence_breakdown,
    _has_official_company_docs,
    _humanize_answer_text,
    _is_company_intent_query,
    _is_doc_intent_query,
    _is_doc_visibility_query,
    _is_entity_level_query,
    _is_explicit_uploaded_summary_request,
    _is_general_knowledge_query,
    _is_related_work_query,
    _is_uploaded_doc_summary_query,
    _load_latest_calibration_weights,
    _normalize_forward,
    _needs_scope_limited_answer,
    _normalize_inline_citations,
    _normalize_source_url,
    _primary_anchor_term,
    _prune_public_citations,
    _prune_uploaded_citations,
    _query_overlap_strength,
    _rank_and_trim_citations,
    _rebalance_uploaded_multi_doc_citations,
    _requested_public_source,
    _resolve_effective_doc_id,
    _scope_evidence_label,
    _scope_limited_answer,
    _source_breakdown,
    _source_scope,
    _uploaded_evidence_strength,
)
from backend.services.embeddings import healthcheck_embeddings
from backend.services.research_feed import latest_research_feed
from backend.services.nli import entailment_prob
from backend.services.judge import aggregate_judge_report, evaluate_faithfulness
from backend.confidence import build_confidence, score_percent
from backend.eval_metrics import aggregate_metrics
from backend.sense_resolver import resolve_sense, filter_citations_by_sense
import statistics

# Initialize FastAPI app
app = FastAPI(title="ScholarRAG API", version="1.0")

_cors_env = os.environ.get("CORS_ORIGINS", "")
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else ["http://localhost:5173", "http://127.0.0.1:5173"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
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

# Fallback thresholds: only replace the LLM answer with a template when citation
# grounding is critically low.  Single-paragraph answers and answers with any inline
# citation are always preserved.
_FALLBACK_MIN_PARAGRAPHS = 3          # only enforce coverage on multi-paragraph answers
_FALLBACK_MIN_COVERAGE = 0.20         # < 20% of paragraphs cited → critically uncited


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


def _ensure_msa_schema() -> None:
    execute(
        """
        CREATE TABLE IF NOT EXISTS confidence_calibration (
            id SERIAL PRIMARY KEY,
            model_name TEXT DEFAULT 'msa_logistic_v1',
            label TEXT DEFAULT 'default',
            weights JSONB NOT NULL,
            metrics JSONB,
            dataset_size INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    execute(
        """
        CREATE TABLE IF NOT EXISTS evidence_scores (
            id SERIAL PRIMARY KEY,
            request_id TEXT,
            sentence_id INT,
            citation_id INT,
            evidence_id TEXT,
            m_score REAL,
            s_score REAL,
            a_score REAL,
            score REAL,
            created_at TIMESTAMP DEFAULT now()
        )
        """
    )
    execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_judge_runs (
            id SERIAL PRIMARY KEY,
            scope TEXT DEFAULT 'uploaded',
            query_count INT DEFAULT 0,
            metrics JSONB,
            details JSONB,
            created_at TIMESTAMP DEFAULT now()
        )
        """
    )


_ensure_eval_schema()
_ensure_msa_schema()


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
        examples={
            "default": {
                "summary": "Default assistant query payload",
                "value": {
                    "query": "What does the paper really address?",
                    "scope": "uploaded",
                    "doc_id": None,
                    "k": 10,
                },
            }
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
    raw_doc_ids = payload.get("doc_ids")
    doc_ids: list[int] | None = None
    try:
        doc_id = int(doc_id) if doc_id is not None else None
    except Exception:
        doc_id = None
    if isinstance(raw_doc_ids, list):
        try:
            doc_ids = [int(x) for x in raw_doc_ids if x is not None]
        except Exception:
            doc_ids = None
    k = int(payload.get("k") or 10)
    multi_hop = bool(payload.get("multi_hop"))
    debug_confidence = bool(payload.get("debug_confidence"))
    run_judge = bool(payload.get("run_judge"))
    run_judge_llm = bool(payload.get("run_judge_llm", True))
    allow_general_background = bool(payload.get("allow_general_background"))
    chosen_sense = payload.get("sense")
    compare_senses = bool(payload.get("compare_senses"))

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    qnorm = query.strip().lower()
    answer_mode = _classify_answer_mode(query)
    doc_summary_intent = _is_uploaded_doc_summary_query(query)
    related_work_intent = _is_related_work_query(query)
    if not doc_ids:
        doc_id = _resolve_effective_doc_id(doc_id, scope, query)
    requested_public_source = _requested_public_source(query)
    if related_work_intent and scope == "uploaded":
        # Sense-comparison is not useful for related-work extraction from one paper.
        compare_senses = False
        chosen_sense = None
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
    is_public_lookup = (
        _is_general_knowledge_query(query)
        or _is_related_work_query(query)
        or _is_company_intent_query(query)
    )

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
    if not is_research and not is_public_lookup and scope != "uploaded":
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

    if scope == "uploaded" and not doc_id and not doc_ids and not allow_general_background:
        if not _is_doc_intent_query(qnorm) and not doc_summary_intent and not related_work_intent:
            return {
                "answer": (
                    "No uploaded document is selected. Select one or more documents for document-grounded answers, "
                    "or switch to Public research for general questions."
                ),
                "citations": [],
                "why_answer": {"rerank_changed_order": False, "top_chunks": []},
                "latency_breakdown_ms": {"retrieve": 0.0, "rerank": 0.0, "generate": 0.0, "total": int((time.time() - started) * 1000)},
                "retrieval_policy": {"mode": "selection-required", "uploaded_hits": 0, "public_hits": 0, "uploaded_strength": 0.0, "uploaded_overlap": 0.0, "used_public_fallback": False},
            }

    retrieval_ms = 0.0
    rerank_ms = 0.0
    generate_ms = 0.0

    def fetch_context(q: str, mode: str):
        local_citations = []
        if mode == "uploaded":
            results = search_uploaded_chunks({"q": q, "k": k, "doc_id": doc_id, "doc_ids": doc_ids})["results"]
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
            nonlocal public_provider_status
            public_resp = public_live_search(q, k=min(k, 8), source_only=requested_public_source, return_metadata=True)
            docs = public_resp.get("results", [])
            for provider_name, meta in (public_resp.get("provider_status", {}) or {}).items():
                current = public_provider_status.get(provider_name, {})
                public_provider_status[provider_name] = {
                    "queried": True,
                    "variant": meta.get("variant") or current.get("variant"),
                    "fetched": max(int(current.get("fetched", 0) or 0), int(meta.get("fetched", 0) or 0)),
                    "selected": max(int(current.get("selected", 0) or 0), int(meta.get("selected", 0) or 0)),
                    "contributed": bool(current.get("contributed")) or bool(meta.get("contributed")),
                }
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
    public_provider_status: dict[str, dict] = {}

    if scope == "uploaded":
        if multi_hop and (" and " in query or ";" in query or "," in query):
            subqs = [q.strip() for q in re.split(r"and|;|,", query) if q.strip()]
            for sq in subqs:
                citations.extend(fetch_context(sq, "uploaded"))
        else:
            citations.extend(fetch_context(query, "uploaded"))

        # Dedicated recall boost for "related/similar work" requests on uploaded docs.
        if related_work_intent:
            related_probe = (
                "related work prior work baseline comparison compared with previous studies "
                "limitations future work " + query
            )
            citations.extend(fetch_context(related_probe, "uploaded"))

        explicit_uploaded_summary = _is_explicit_uploaded_summary_request(query)
        if allow_general_background and not explicit_uploaded_summary and (_is_general_knowledge_query(query) or _is_company_intent_query(query)):
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
        if weak_uploaded and allow_general_background and not explicit_uploaded_summary and answer_mode in {"research_synthesis", "source_listing"}:
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

    entity_query = _is_entity_level_query(query) and not doc_summary_intent
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

    if scope == "uploaded" and allow_general_background and not _is_explicit_uploaded_summary_request(query) and _is_general_knowledge_query(query):
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
    if scope == "uploaded" and not _is_explicit_uploaded_summary_request(query) and _needs_scope_limited_answer(query, citations):
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
    if sense.get("is_ambiguous") and not compare_senses and not chosen_sense and not related_work_intent:
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

    if _is_company_intent_query(query) and not doc_summary_intent:
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
        c["evidence_id"] = c.get("evidence_id") or _build_evidence_id(c)
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

    prompt = _build_generation_prompt(
        query=query,
        context=context,
        answer_mode=answer_mode,
        allow_general_background=allow_general_background,
        compare_instruction=compare_instruction,
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
    # Only fall back to template answers when the LLM output is critically uncited:
    # - zero inline [S#] citations anywhere in the answer, OR
    # - answer has _FALLBACK_MIN_PARAGRAPHS+ paragraphs but coverage < _FALLBACK_MIN_COVERAGE.
    # Intro sentences, transitions, and conclusions legitimately lack citations —
    # replacing a well-grounded answer just because one paragraph is citation-free
    # is the primary cause of retrieval-dump responses instead of coherent synthesis.
    # citation_coverage_par > 0 means _citation_coverage_stats already found at least one
    # cited paragraph — no need for a second regex pass over the answer.
    lacks_any_citations = citation_coverage_par == 0.0
    sparse_multi_para = paragraph_count >= _FALLBACK_MIN_PARAGRAPHS and citation_coverage_par < _FALLBACK_MIN_COVERAGE
    critically_uncited = lacks_any_citations or sparse_multi_para
    if critically_uncited:
        if scope != "uploaded" and citations:
            if answer_mode == "source_listing":
                answer = _build_public_source_listing_answer(citations)
            else:
                answer = _build_public_synthesis_fallback(citations)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)
        elif scope == "uploaded" and related_work_intent and citations:
            answer = _build_uploaded_related_work_fallback(citations)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)
        elif scope == "uploaded" and citations:
            answer = _build_uploaded_evidence_fallback(query, citations)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)
        else:
            evidence_label = _scope_evidence_label(scope)
            answer = (
                f"I don’t have enough evidence in the selected {evidence_label} to support a grounded answer. "
                "Try rephrasing or uploading a more relevant source."
            )
            # Do not present misleading evidence cards when answer is explicitly blocked.
            citations = []
    citations = _apply_usage_boost(citations, answer)
    msa_by_citation: dict[int, dict] = {}
    unsupported_by_msa = 0
    should_compute_msa = (
        bool(citations)
        and bool((answer or "").strip())
        and (
            answer_mode == "research_synthesis"
            or run_judge
        )
    )
    if should_compute_msa:
        msa_by_citation, unsupported_by_msa = _compute_citation_msa(
            query,
            answer,
            citations,
            scope=scope,
            k=k,
            doc_id=doc_id,
            source_only=requested_public_source,
        )

    if msa_by_citation:
        for c in citations:
            cid = int(c.get("id") or 0)
            msa = msa_by_citation.get(cid)
            if not msa:
                continue
            msa_payload = {
                "M": float(msa.get("M", 0.0)),
                "S": float(msa.get("S", 0.0)),
                "A": float(msa.get("A", 0.0)),
                "weights": _load_latest_calibration_weights(),
            }
            c["msa"] = {
                **msa_payload,
                "msa_score": float(msa.get("msa_score", 0.0)),
                "score_percent": score_percent(float(msa.get("msa_score", 0.0))),
            }
            c["msa_supported"] = bool(float(msa.get("M", 0.0)) >= 0.5)
            if c.get("used_in_answer") and not c["msa_supported"]:
                c["used_in_answer"] = False
            c["confidence_obj"] = build_confidence(
                top_sim=float(c.get("sim_score", 0.0) or 0.0),
                top_rerank_norm=float(c.get("rerank_norm", 0.0) or 0.0),
                citation_coverage=1.0 if c.get("used_in_answer") else 0.0,
                evidence_margin=float(c.get("sim_score", 0.0) or 0.0),
                ambiguity_penalty=0.0,
                insufficiency_penalty=0.0,
                scope_penalty=0.0,
                msa=msa_payload,
            )

    if msa_by_citation and any(c.get("msa_supported") is False for c in citations):
        unsupported_claims = max(unsupported_claims, unsupported_by_msa or unsupported_claims)

    # Only replace the answer when MSA drops ALL citations AND the answer carries no
    # inline [S#] references — meaning it genuinely hallucinated without grounding.
    # citation_coverage_par reflects the most recently recomputed answer (post-fallback),
    # so citation_coverage_par == 0.0 is equivalent to "no inline citations" without
    # an extra regex scan.
    all_used_dropped = citations and all(not c.get("used_in_answer") for c in citations)
    if all_used_dropped and answer.strip() and citation_coverage_par == 0.0:
        if scope == "uploaded" and citations:
            answer = _build_uploaded_evidence_fallback(query, citations)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)
        else:
            if answer_mode == "source_listing":
                answer = _build_public_source_listing_answer(citations)
            else:
                answer = _build_public_synthesis_fallback(citations)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)

    if scope == "uploaded" and doc_ids and len(doc_ids) > 1:
        citations = _rebalance_uploaded_multi_doc_citations(citations, doc_ids, k)
        for i, c in enumerate(citations, start=1):
            c["id"] = i
        if _is_uploaded_doc_summary_query(query):
            answer = _build_multi_doc_uploaded_summary(citations, doc_ids)
            citations = _apply_usage_boost(citations, answer)
            citation_coverage_par, unsupported_claims, paragraph_count = _citation_coverage_stats(answer)

    if scope != "uploaded" and answer_mode == "source_listing":
        answer = _append_public_source_links(answer, citations)

    cited_count = sum(1 for c in citations if c.get("used_in_answer"))
    if run_judge:
        faithfulness = evaluate_faithfulness(query, answer, citations, use_llm=run_judge_llm)
    else:
        faithfulness = None

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
            "answer_mode": answer_mode,
            "public_provider_status": public_provider_status,
            "trust": trust,
            "confidence_score": confidence.get("score"),
            "latency_ms": latency_ms,
        },
    )

    citations_out = [{k: v for k, v in c.items() if k != "snippet"} for c in citations]
    for c in citations_out:
        if "confidence_obj" not in c:
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
        c["confidence"] = float(c.get("confidence", c["confidence_obj"].get("score", 0.0)) or 0.0)
        c["confidence_percent"] = score_percent(float(c["confidence"]))
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
        "faithfulness": faithfulness if faithfulness is not None else None,
        "trust": trust,
        "latency_ms": latency_ms,
        "confidence_note": (
            "Confidence is heuristic/derived from evidence retrieval, retrieval stability, and optional MSA calibration. "
            "Use debug_confidence=true for per-source breakdown."
        ),
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
            "answer_mode": answer_mode,
            "uploaded_hits": uploaded_hits,
            "public_hits": public_hits,
            "uploaded_strength": uploaded_strength,
            "uploaded_overlap": uploaded_overlap,
            "used_public_fallback": used_public_fallback,
            "source_breakdown": _source_breakdown(citations),
            "public_provider_status": public_provider_status,
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


def _judge_label_to_binary(label: object) -> int | None:
    if label is None:
        return None
    v = str(label).strip().lower()
    if not v:
        return None
    if v in {"strong", "high", "positive", "yes", "1", "true", "supported"}:
        return 1
    if v in {"weak", "moderate", "medium", "low", "negative", "0", "false", "unsupported"}:
        return 0
    return None


def _sigmoid(x: float) -> float:
    # Clamp for numerical stability
    x = max(-60.0, min(60.0, float(x)))
    return 1.0 / (1.0 + np.exp(-x))


def _fit_logistic_weights(records: list[tuple[float, float, float, int]], iters: int = 2200) -> tuple[dict[str, float], dict[str, float]]:
    # records: [(m,s,a,label_int)]
    if not records:
        return (
            {"w1": 0.58, "w2": 0.22, "w3": 0.20, "b": 0.0},
            {"status": "empty"},
        )

    n = len(records)
    w1 = 0.58
    w2 = 0.22
    w3 = 0.20
    b = 0.0
    lr = 0.38
    l2 = 0.001

    for _ in range(max(1, iters)):
        g1 = g2 = g3 = gb = 0.0
        correct = 0
        brier = 0.0
        for m, s, a, y in records:
            z = b + w1 * m + w2 * s + w3 * a
            p = _sigmoid(z)
            y_f = float(y)
            diff = p - y_f
            g1 += diff * m
            g2 += diff * s
            g3 += diff * a
            gb += diff
            brier += (p - y_f) ** 2
            if (p >= 0.5) == bool(y_f):
                correct += 1

        g1 = g1 / n + l2 * w1
        g2 = g2 / n + l2 * w2
        g3 = g3 / n + l2 * w3
        gb = gb / n + l2 * b

        w1 -= lr * g1
        w2 -= lr * g2
        w3 -= lr * g3
        b -= lr * gb

    accuracy = correct / n if n else 0.0
    brier = brier / n if n else 0.0
    weights = {"w1": round(w1, 6), "w2": round(w2, 6), "w3": round(w3, 6), "b": round(b, 6)}
    metrics = {
        "n": n,
        "accuracy": round(accuracy, 4),
        "brier": round(brier, 4),
        "method": "gradient_logistic",
    }
    return weights, metrics


def _build_msa_records(payload: dict) -> list[tuple[float, float, float, int]]:
    rows: list[tuple[float, float, float, int]] = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        msa = item.get("msa") or {}
        if isinstance(msa, dict) and all(k in msa for k in ("M", "S", "A")):
            m = float(msa.get("M", 0.0))
            s = float(msa.get("S", 0.0))
            a = float(msa.get("A", 0.0))
        else:
            sentence = (item.get("sentence") or "").strip()
            evidence_text = (
                item.get("evidence")
                or item.get("evidence_text")
                or item.get("evidence_snippet")
                or ""
            )
            if sentence and evidence_text:
                m = entailment_prob(sentence, str(evidence_text))
                s = float(item.get("S", 0.5))
                a = float(item.get("A", 0.5))
            else:
                continue

        label = _judge_label_to_binary(item.get("label"))
        if label is None and "answer_supported" in item:
            label = 1 if bool(item.get("answer_supported")) else 0
        if label is None:
            continue
        rows.append((_clamp01(m), _clamp01(s), _clamp01(a), label))
    return rows


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


@app.post("/eval/judge")
def run_judge(payload: dict = Body(...)):
    scope = payload.get("scope") or "uploaded"
    k = int(payload.get("k") or 10)
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise HTTPException(status_code=400, detail="cases must be a non-empty list")
    if scope not in {"uploaded", "public"}:
        raise HTTPException(status_code=400, detail="scope must be uploaded or public")

    run_judge_llm = bool(payload.get("run_judge_llm", True))
    details = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        query = (case.get("query") or "").strip()
        if not query:
            continue

        answer = (case.get("answer") or "").strip()
        citations = case.get("citations")
        if not answer or not isinstance(citations, list):
            result = assistant_answer(
                {
                    "query": query,
                    "scope": scope,
                    "doc_id": case.get("doc_id"),
                    "k": k,
                    "run_judge": False,
                    "allow_general_background": bool(case.get("allow_general_background", False)),
                }
            )
            answer = (result.get("answer") or "").strip()
            citations = result.get("citations") or []
        report = evaluate_faithfulness(query, answer, citations if isinstance(citations, list) else [], use_llm=run_judge_llm)
        details.append(
            {
                "query": query,
                "answer": answer,
                "citations": citations if isinstance(citations, list) else [],
                "faithfulness": report,
                "scope": scope,
            }
        )

    if not details:
        raise HTTPException(status_code=400, detail="No valid cases found")

    aggregate = aggregate_judge_report([d.get("faithfulness", {}) for d in details])
    row = fetchone(
        """
        INSERT INTO evaluation_judge_runs
        (scope, query_count, metrics, details)
        VALUES (%s, %s, %s::jsonb, %s::jsonb)
        RETURNING id, created_at
        """,
        [
            scope,
            len(details),
            json.dumps(aggregate),
            json.dumps(details),
        ],
    )

    return {
        "run_id": row.get("id") if row else None,
        "created_at": row.get("created_at").isoformat() if row and row.get("created_at") else None,
        "scope": scope,
        "query_count": len(details),
        "metrics": aggregate,
        "details": details,
    }


@app.get("/eval/judge/runs")
def list_judge_runs(limit: int = 20):
    rows = fetchall(
        """
        SELECT id, scope, query_count, metrics, details, created_at
        FROM evaluation_judge_runs
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
                "scope": r.get("scope"),
                "query_count": r.get("query_count"),
                "metrics": r.get("metrics"),
                "created_at": r.get("created_at").isoformat() if r.get("created_at") else None,
            }
        )
    return {"runs": out}


@app.post("/confidence/calibrate")
def calibrate_confidence(payload: dict = Body(...)):
    records = payload.get("records") or []
    if not isinstance(records, list) or not records:
        raise HTTPException(status_code=400, detail="records must be a non-empty list")
    msarecords = _build_msa_records(records)
    if len(msarecords) < 5:
        raise HTTPException(status_code=400, detail="At least 5 labeled records required to fit calibration.")
    model_name = (payload.get("model_name") or "msa_logistic_v1").strip() or "msa_logistic_v1"
    label = (payload.get("label") or "default").strip() or "default"
    weights, metrics = _fit_logistic_weights(msarecords)
    row = fetchone(
        """
        INSERT INTO confidence_calibration
        (model_name, label, weights, metrics, dataset_size)
        VALUES (%s, %s, %s::jsonb, %s::jsonb, %s)
        RETURNING id, created_at
        """,
        [model_name, label, json.dumps(weights), json.dumps(metrics), len(msarecords)],
    )
    return {
        "run_id": row.get("id") if row else None,
        "created_at": row.get("created_at").isoformat() if row and row.get("created_at") else None,
        "model_name": model_name,
        "label": label,
        "records_used": len(msarecords),
        "weights": weights,
        "metrics": metrics,
    }


@app.get("/confidence/calibration")
def get_latest_calibration():
    row = fetchone(
        """
        SELECT id, model_name, label, weights, metrics, dataset_size, created_at
        FROM confidence_calibration
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    if not row:
        return {
            "model_name": "msa_logistic_v1",
            "label": "default",
            "weights": {"w1": 0.58, "w2": 0.22, "w3": 0.20, "b": 0.0},
            "metrics": None,
            "dataset_size": 0,
            "created_at": None,
        }
    created = row.get("created_at")
    return {
        "id": row.get("id"),
        "model_name": row.get("model_name"),
        "label": row.get("label"),
        "weights": row.get("weights") or {"w1": 0.58, "w2": 0.22, "w3": 0.20, "b": 0.0},
        "metrics": row.get("metrics"),
        "dataset_size": row.get("dataset_size"),
        "created_at": created.isoformat() if created else None,
    }

LOG_DIR = Path("logs")
RETRIEVAL_LOG = LOG_DIR / "retrieval.log"

# ------------------------------
# OpenAI Generation Config
# ------------------------------
try:
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)
except RuntimeError as err:
    print(f"❌ {err}")
    client = None

# Logger for observability
REQUEST_LOG = setup_file_logger(LOG_DIR / "requests.jsonl")

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


@app.get("/health/embeddings")
def embeddings_health():
    return healthcheck_embeddings()

@app.get("/research/latest")
def research_latest(
    topic: str | None = Query(default=None, description="Optional research topic"),
    limit: int = Query(default=8, ge=1, le=24),
    days: int = Query(default=45, ge=1, le=365),
):
    return latest_research_feed(topic=topic, limit=limit, days=days)

@app.get("/feed/latest")
def latest_papers(limit: int = 10):
    """Compatibility alias for the landing/latest research feed."""
    return latest_research_feed(limit=limit)

@app.get("/search")
def search_papers(query: str = Query(..., description="Search query text"), k: int = 5):
    """Return top-k live public scholarly results for a given query."""
    public_resp = public_live_search(query, k=max(1, k), return_metadata=True)
    results = []
    for rank, row in enumerate(public_resp.get("results", []), start=1):
        results.append(
            {
                "rank": rank,
                "title": row.get("title", "Unknown Title"),
                "year": row.get("year"),
                "doi": row.get("doi", ""),
                "source": row.get("source"),
                "url": row.get("url") or row.get("source_url"),
                "similarity": row.get("score") or row.get("similarity"),
                "summary": (row.get("abstract") or row.get("summary") or "")[:320],
            }
        )
    return {
        "query": query,
        "results": results,
        "provider_status": public_resp.get("provider_status", {}),
    }

@app.get("/summarize")
def summarize(query: str = Query(..., description="Topic to summarize")):
    """Summarize top live scholarly results for a given query using GPT."""
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured.")

    public_resp = public_live_search(query, k=5, return_metadata=True)
    rows = public_resp.get("results", [])
    if not rows:
        return {"query": query, "summary": "No relevant public sources found.", "provider_status": public_resp.get("provider_status", {})}

    numbered_sources = []
    for i, row in enumerate(rows, start=1):
        snippet = (row.get("abstract") or row.get("summary") or "").strip()
        if snippet:
            snippet = snippet[:400]
        numbered_sources.append(
            f"[P{i}] {row.get('title', 'Unknown Title')} ({row.get('source', 'unknown')}, {row.get('year', 'n/a')})\n{snippet}"
        )
    top_titles = "\n\n".join(numbered_sources)

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
    return {"query": query, "summary": summary, "provider_status": public_resp.get("provider_status", {})}

@app.post("/ask")
def ask(payload: dict = Body(...)):
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured.")

    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'.")

    k = int(payload.get("k", 10))
    start = time.perf_counter()
    public_resp = public_live_search(query, k=max(1, k), return_metadata=True)
    rows = public_resp.get("results", [])
    if not rows:
        return {
            "answer": "No relevant public sources found.",
            "sources": [],
            "candidate_counts": {},
            "metrics": {"latency_ms": round((time.perf_counter() - start) * 1000, 2)},
            "provider_status": public_resp.get("provider_status", {}),
        }

    context_blocks = []
    for i, row in enumerate(rows, start=1):
        snippet = (row.get("abstract") or row.get("summary") or "").strip()
        context_blocks.append(
            f"[P{i}] {row.get('title', 'Unknown Title')} | source={row.get('source', 'unknown')} | year={row.get('year', 'n/a')}\n{snippet}"
        )
    prompt = (
        "Answer the user's scholarly question using ONLY the provided sources.\n"
        "Every factual sentence must include an inline citation like [P1].\n"
        "If the evidence is weak or incomplete, say so explicitly.\n\n"
        f"Question: {query}\n\nSources:\n" + "\n\n".join(context_blocks)
    )
    completion = client.chat.completions.create(
        model=RESEARCH_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = completion.choices[0].message.content or ""
    latency_ms = (time.perf_counter() - start) * 1000

    # Shape sources
    sources = []
    for d in rows:
        sim = round(float(d.get("score") or d.get("similarity") or 0.0), 3)
        t_score = trust_score(sim, bool(d.get("doi")))
        sources.append({
            "title": d.get("title", "Unknown Title"),
            "year": d.get("year", "Unknown Year"),
            "doi": d.get("doi", ""),
            "openalex_id": d.get("id") or d.get("paper_id"),
            "arxiv_id": d.get("arxiv_id"),
            "concepts": (d.get("concepts") or [])[:5],
            "why_relevant": d.get("why_relevant", ""),
            "snippet": (d.get("abstract") or d.get("summary") or "")[:900],
            "similarity": sim,
            "trust_score": t_score,
            "authors": d.get("authors", []),
            "url": d.get("url") or d.get("source_url"),
            "source": d.get("source"),
        })

    similarities = [s["similarity"] for s in sources if s.get("similarity") is not None]
    metrics = {
        "latency_ms": round(latency_ms, 2),
        "fallback_used": False,
        "pool_size": len(rows),
        "ranked": len(rows),
        "max_similarity": max(similarities) if similarities else None,
        "mean_similarity": round(sum(similarities) / len(similarities), 3) if similarities else None,
        "token_prompt": completion.usage.prompt_tokens if completion.usage else None,
        "token_completion": completion.usage.completion_tokens if completion.usage else None,
        "token_total": completion.usage.total_tokens if completion.usage else None,
    }

    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "k": k,
        "metrics": metrics,
        "candidate_counts": {"public_results": len(rows)},
        "fallback_used": False,
        "public_provider_status": public_resp.get("provider_status", {}),
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
        "answer": answer,
        "sources": sources,
        "fallback_used": False,
        "candidate_counts": {"public_results": len(rows)},
        "metrics": metrics,
        "provider_status": public_resp.get("provider_status", {}),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
