from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backend.pdf_ingest import search_chunks
from backend.services.assistant_utils import _build_evidence_id, _extract_sentence_citation_ids
from backend.services.db import fetchall
from backend.services.judge import _split_sentences


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_int_list(values: object) -> list[int] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError("doc_ids must be a list of integers")
    out: list[int] = []
    for item in values:
        parsed = _coerce_int(item)
        if parsed is not None:
            out.append(parsed)
    return out or None


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def dump_json(data: Any, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def dump_jsonl(rows: Iterable[Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    out.write_text(("\n".join(lines) + "\n") if lines else "")


def normalize_query_entry(item: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"Query entry {index} must be an object")

    query = str(item.get("query") or "").strip()
    if not query:
        raise ValueError(f"Query entry {index} is missing `query`")

    query_id = str(item.get("query_id") or f"q{index}").strip() or f"q{index}"
    doc_scope = str(item.get("doc_scope") or item.get("scope") or "uploaded").strip().lower() or "uploaded"
    if doc_scope != "uploaded":
        raise ValueError(f"Query entry {query_id} has unsupported doc_scope `{doc_scope}`; only `uploaded` is supported")

    doc_id = _coerce_int(item.get("doc_id"))
    doc_ids = _coerce_int_list(item.get("doc_ids"))
    if doc_ids and doc_id is not None and doc_id not in doc_ids:
        doc_ids = [doc_id] + [value for value in doc_ids if value != doc_id]

    normalized = {
        "query_id": query_id,
        "query": query,
        "doc_scope": doc_scope,
    }
    if doc_id is not None:
        normalized["doc_id"] = doc_id
    if doc_ids:
        normalized["doc_ids"] = doc_ids
    if item.get("notes") is not None:
        normalized["notes"] = str(item.get("notes"))
    if item.get("paper_title") is not None:
        normalized["paper_title"] = str(item.get("paper_title"))
    if item.get("paper_titles") is not None and isinstance(item.get("paper_titles"), list):
        normalized["paper_titles"] = [str(x) for x in item.get("paper_titles") if x is not None]
    if item.get("allow_general_background") is not None:
        normalized["allow_general_background"] = bool(item.get("allow_general_background"))
    return normalized


def load_query_set(path: str | Path) -> dict[str, Any]:
    raw = load_json(path)
    if isinstance(raw, dict):
        query_items = raw.get("queries")
        meta = {k: v for k, v in raw.items() if k != "queries"}
    elif isinstance(raw, list):
        query_items = raw
        meta = {}
    else:
        raise ValueError("Query file must be a JSON object with `queries` or a JSON list")

    if not isinstance(query_items, list) or not query_items:
        raise ValueError("Query file must contain a non-empty `queries` list")

    queries = [normalize_query_entry(item, index + 1) for index, item in enumerate(query_items)]
    return {
        **meta,
        "queries": queries,
    }


def ready_documents(doc_ids: Iterable[int] | None = None) -> list[dict[str, Any]]:
    params: list[Any] = []
    where = ["status = 'ready'"]
    if doc_ids:
        clean_ids = [int(value) for value in doc_ids]
        where.append("id = ANY(%s)")
        params.append(clean_ids)
    rows = fetchall(
        f"""
        SELECT id, title, doc_type, status, created_at
        FROM documents
        WHERE {' AND '.join(where)}
        ORDER BY created_at DESC, id DESC
        """,
        params,
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "doc_id": int(row.get("id")),
                "title": row.get("title") or f"Document {row.get('id')}",
                "doc_type": row.get("doc_type") or "other",
                "status": row.get("status") or "ready",
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
            }
        )
    return out


_PER_DOC_TEMPLATES = [
    "What research problem does this paper address?",
    "What method or approach does this paper propose?",
    "What datasets, benchmarks, or evaluation settings are used in this paper?",
    "What are the main findings or results of this paper?",
    "What limitations or future work does this paper mention?",
]

_CROSS_DOC_TEMPLATES = [
    "Compare the methods proposed in these selected papers.",
    "What common research problems or themes appear across these selected papers?",
    "How do the evaluation setups differ across these selected papers?",
    "What similarities and differences appear in the findings across these selected papers?",
    "Which limitations or open questions recur across these selected papers?",
    "Summarize the main contribution of each selected paper and then compare them.",
]


def build_generated_query_set(
    documents: list[dict[str, Any]],
    *,
    per_doc: int = 4,
    cross_doc: int = 6,
) -> dict[str, Any]:
    if not documents:
        raise ValueError("At least one ready document is required to generate queries")

    queries: list[dict[str, Any]] = []
    query_index = 1
    per_doc = max(1, int(per_doc))
    cross_doc = max(0, int(cross_doc))

    for doc in documents:
        doc_templates = _PER_DOC_TEMPLATES[: min(per_doc, len(_PER_DOC_TEMPLATES))]
        if per_doc > len(_PER_DOC_TEMPLATES):
            repeats = per_doc - len(_PER_DOC_TEMPLATES)
            doc_templates += _PER_DOC_TEMPLATES[:repeats]
        for template in doc_templates[:per_doc]:
            queries.append(
                {
                    "query_id": f"q{query_index}",
                    "query": template,
                    "doc_scope": "uploaded",
                    "doc_id": int(doc["doc_id"]),
                    "paper_title": doc.get("title"),
                    "notes": "template_generated_per_doc",
                }
            )
            query_index += 1

    if len(documents) > 1 and cross_doc > 0:
        selected_doc_ids = [int(doc["doc_id"]) for doc in documents]
        selected_titles = [doc.get("title") for doc in documents]
        cross_templates = _CROSS_DOC_TEMPLATES[: min(cross_doc, len(_CROSS_DOC_TEMPLATES))]
        if cross_doc > len(_CROSS_DOC_TEMPLATES):
            repeats = cross_doc - len(_CROSS_DOC_TEMPLATES)
            cross_templates += _CROSS_DOC_TEMPLATES[:repeats]
        for template in cross_templates[:cross_doc]:
            queries.append(
                {
                    "query_id": f"q{query_index}",
                    "query": template,
                    "doc_scope": "uploaded",
                    "doc_ids": selected_doc_ids,
                    "paper_titles": selected_titles,
                    "notes": "template_generated_cross_doc",
                }
            )
            query_index += 1

    return {
        "mode": "open_corpus_queries",
        "generated_at": utc_now_iso(),
        "generator": "template",
        "queries": queries,
    }


def _query_scope_doc_ids(query_entry: dict[str, Any], all_docs: list[dict[str, Any]]) -> list[int]:
    if query_entry.get("doc_ids"):
        return [int(value) for value in query_entry.get("doc_ids") or []]
    if query_entry.get("doc_id") is not None:
        return [int(query_entry["doc_id"])]
    return [int(doc["doc_id"]) for doc in all_docs]


def export_retrieval_for_query(
    query_entry: dict[str, Any],
    *,
    k: int,
    all_docs: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "q": query_entry["query"],
        "k": max(int(k), 1),
    }
    if query_entry.get("doc_id") is not None:
        payload["doc_id"] = int(query_entry["doc_id"])
    if query_entry.get("doc_ids"):
        payload["doc_ids"] = [int(value) for value in query_entry["doc_ids"]]

    raw = search_chunks(payload=payload)["results"]
    retrieved: list[dict[str, Any]] = []
    seen_docs: set[int] = set()
    retrieved_docs: list[dict[str, Any]] = []

    for rank, row in enumerate(raw[: int(k)], start=1):
        doc_id = int(row.get("document_id"))
        title = row.get("title") or f"Document {doc_id}"
        chunk_id = int(row.get("id"))
        distance = float(row.get("distance", 1.0) or 1.0)
        score = max(0.0, 1.0 - distance)
        page = row.get("page_no")
        retrieved.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "title": title,
                "chunk_id": chunk_id,
                "score": round(score, 6),
                "distance": round(distance, 6),
                "page": int(page) if page is not None else None,
                "chunk_text": row.get("text", ""),
                "relevance_label": None,
            }
        )
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            retrieved_docs.append(
                {
                    "rank": len(retrieved_docs) + 1,
                    "first_chunk_rank": rank,
                    "doc_id": doc_id,
                    "title": title,
                    "relevance_label": None,
                }
            )

    allowed_doc_ids = set(_query_scope_doc_ids(query_entry, all_docs))
    corpus_docs = [
        {
            "doc_id": int(doc["doc_id"]),
            "title": doc.get("title"),
            "relevance_label": None,
        }
        for doc in all_docs
        if int(doc["doc_id"]) in allowed_doc_ids
    ]

    return {
        "query_id": query_entry["query_id"],
        "query": query_entry["query"],
        "doc_scope": query_entry.get("doc_scope", "uploaded"),
        "doc_id": query_entry.get("doc_id"),
        "doc_ids": query_entry.get("doc_ids"),
        "retrieved": retrieved,
        "retrieved_docs": retrieved_docs,
        "corpus_docs": corpus_docs,
    }


def _assistant_payload(
    query_entry: dict[str, Any],
    *,
    k: int,
    compute_msa: bool,
    run_judge_llm: bool,
    all_docs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "query": query_entry["query"],
        "scope": query_entry.get("doc_scope", "uploaded"),
        "k": int(k),
        "allow_general_background": bool(query_entry.get("allow_general_background", False)),
    }
    if query_entry.get("doc_id") is not None:
        payload["doc_id"] = int(query_entry["doc_id"])
    if query_entry.get("doc_ids"):
        payload["doc_ids"] = [int(value) for value in query_entry["doc_ids"]]
    elif all_docs:
        payload["doc_ids"] = [int(doc["doc_id"]) for doc in all_docs]
    if compute_msa:
        payload["run_judge"] = True
        payload["run_judge_llm"] = bool(run_judge_llm)
    return payload


def run_answer_for_query(
    query_entry: dict[str, Any],
    *,
    k: int,
    compute_msa: bool = False,
    run_judge_llm: bool = False,
    all_docs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from backend.app import assistant_answer

    return assistant_answer(
        _assistant_payload(
            query_entry,
            k=k,
            compute_msa=compute_msa,
            run_judge_llm=run_judge_llm,
            all_docs=all_docs,
        )
    )


def export_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for index, citation in enumerate(citations or [], start=1):
        citation_idx = _coerce_int(citation.get("id")) or index
        exported.append(
            {
                "citation_id": f"S{citation_idx}",
                "citation_index": citation_idx,
                "evidence_id": citation.get("evidence_id") or _build_evidence_id(citation),
                "doc_id": _coerce_int(citation.get("doc_id")),
                "title": citation.get("title"),
                "chunk_id": _coerce_int(citation.get("chunk_id")),
                "page": _coerce_int(citation.get("page")),
                "source": citation.get("source"),
                "snippet": citation.get("snippet", ""),
                "used_in_answer": bool(citation.get("used_in_answer")),
                "msa": citation.get("msa") if isinstance(citation.get("msa"), dict) else None,
            }
        )
    return exported


def build_claim_rows(query_id: str, answer: str, exported_citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citation_index = {int(c.get("citation_index") or idx): c for idx, c in enumerate(exported_citations, start=1)}
    claims: list[dict[str, Any]] = []

    for sentence_index, sentence in enumerate(_split_sentences(answer), start=1):
        clean_text = re.sub(r"\[(?:S)?(\d+)\]", "", sentence).strip()
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        if not clean_text:
            continue

        cited_ids = _extract_sentence_citation_ids(sentence)
        citation_ids: list[str] = []
        evidence_ids: list[str] = []
        evidence_parts: list[str] = []
        msa_rows: list[dict[str, float]] = []

        for cited_id in cited_ids:
            citation = citation_index.get(int(cited_id))
            if not citation:
                continue
            citation_ids.append(str(citation.get("citation_id") or f"S{cited_id}"))
            evidence_id = str(citation.get("evidence_id") or "")
            if evidence_id and evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
            snippet = str(citation.get("snippet") or "").strip()
            if snippet and snippet not in evidence_parts:
                evidence_parts.append(snippet)
            msa = citation.get("msa")
            if isinstance(msa, dict) and all(key in msa for key in ("M", "S", "A")):
                msa_rows.append(
                    {
                        "M": float(msa.get("M", 0.0)),
                        "S": float(msa.get("S", 0.0)),
                        "A": float(msa.get("A", 0.0)),
                    }
                )

        claim: dict[str, Any] = {
            "claim_id": f"{query_id}_c{sentence_index}",
            "text": clean_text,
            "citation_ids": citation_ids,
            "evidence_ids": evidence_ids,
            "evidence_text": " ".join(evidence_parts).strip(),
            "label": None,
            "citation_correct": None,
        }
        if msa_rows:
            claim["msa"] = {
                "M": round(sum(row["M"] for row in msa_rows) / len(msa_rows), 4),
                "S": round(sum(row["S"] for row in msa_rows) / len(msa_rows), 4),
                "A": round(sum(row["A"] for row in msa_rows) / len(msa_rows), 4),
            }
        claims.append(claim)

    return claims


def export_answer_for_query(
    query_entry: dict[str, Any],
    *,
    k: int,
    compute_msa: bool = False,
    run_judge_llm: bool = False,
    all_docs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = run_answer_for_query(
        query_entry,
        k=k,
        compute_msa=compute_msa,
        run_judge_llm=run_judge_llm,
        all_docs=all_docs,
    )
    citations = export_citations(result.get("citations") or [])
    claims = build_claim_rows(query_entry["query_id"], result.get("answer") or "", citations)
    return {
        "query_id": query_entry["query_id"],
        "query": query_entry["query"],
        "doc_scope": query_entry.get("doc_scope", "uploaded"),
        "doc_id": query_entry.get("doc_id"),
        "doc_ids": query_entry.get("doc_ids"),
        "answer": result.get("answer") or "",
        "answer_scope": result.get("answer_scope"),
        "citations": citations,
        "claims": claims,
    }


def build_claim_annotation_entry(answer_entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": answer_entry.get("query_id"),
        "query": answer_entry.get("query"),
        "answer": answer_entry.get("answer"),
        "claims": answer_entry.get("claims") or [],
    }
