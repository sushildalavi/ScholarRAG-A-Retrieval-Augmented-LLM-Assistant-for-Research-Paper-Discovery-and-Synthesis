from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from backend.open_eval_metrics import relevance_gain


QUERY_SUMMARY_FIELDS = [
    "query_id",
    "query",
    "generated_answer",
    "answer_level_notes",
]

RETRIEVAL_ANNOTATION_FIELDS = [
    "query_id",
    "query",
    "rank",
    "doc_id",
    "document_title",
    "chunk_id",
    "page",
    "retrieval_score",
    "chunk_text",
    "relevance_label",
]

CORPUS_DOC_FIELDS = [
    "query_id",
    "query",
    "doc_id",
    "document_title",
    "relevance_label",
]

CLAIM_ANNOTATION_FIELDS = [
    "query_id",
    "query",
    "claim_id",
    "claim_text",
    "evidence_ids",
    "evidence_text",
    "msa_M",
    "msa_S",
    "msa_A",
    "support_label",
    "citation_correct",
    "annotator_notes",
]


def dump_csv_rows(path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _to_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _join_values(values: object) -> str:
    if isinstance(values, list):
        return " | ".join(str(item) for item in values if item not in (None, ""))
    if values in (None, ""):
        return ""
    return str(values)


def _prefer_label(current: object, candidate: object) -> str:
    current_label = str(current or "").strip().lower()
    candidate_label = str(candidate or "").strip().lower()
    if relevance_gain(candidate_label) > relevance_gain(current_label):
        return candidate_label
    return current_label


def build_query_summary_rows(answer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for answer_row in answer_rows:
        rows.append(
            {
                "query_id": answer_row.get("query_id"),
                "query": answer_row.get("query"),
                "generated_answer": answer_row.get("answer") or "",
                "answer_level_notes": "",
            }
        )
    return rows


def build_retrieval_annotation_rows(retrieval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_row in retrieval_rows:
        for item in query_row.get("retrieved") or []:
            rows.append(
                {
                    "query_id": query_row.get("query_id"),
                    "query": query_row.get("query"),
                    "rank": item.get("rank"),
                    "doc_id": item.get("doc_id"),
                    "document_title": item.get("title"),
                    "chunk_id": item.get("chunk_id"),
                    "page": item.get("page"),
                    "retrieval_score": item.get("score"),
                    "chunk_text": item.get("chunk_text") or "",
                    "relevance_label": item.get("relevance_label") or "",
                }
            )
    return rows


def build_corpus_doc_rows(retrieval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_row in retrieval_rows:
        for item in query_row.get("corpus_docs") or []:
            rows.append(
                {
                    "query_id": query_row.get("query_id"),
                    "query": query_row.get("query"),
                    "doc_id": item.get("doc_id"),
                    "document_title": item.get("title"),
                    "relevance_label": item.get("relevance_label") or "",
                }
            )
    return rows


def build_claim_annotation_rows(answer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for answer_row in answer_rows:
        for claim in answer_row.get("claims") or []:
            msa = claim.get("msa") if isinstance(claim.get("msa"), dict) else {}
            rows.append(
                {
                    "query_id": answer_row.get("query_id"),
                    "query": answer_row.get("query"),
                    "claim_id": claim.get("claim_id"),
                    "claim_text": claim.get("text") or "",
                    "evidence_ids": _join_values(claim.get("evidence_ids")),
                    "evidence_text": claim.get("evidence_text") or "",
                    "msa_M": msa.get("M", "") if msa else "",
                    "msa_S": msa.get("S", "") if msa else "",
                    "msa_A": msa.get("A", "") if msa else "",
                    "support_label": claim.get("label") or "",
                    "citation_correct": claim.get("citation_correct"),
                    "annotator_notes": "",
                }
            )
    return rows


def load_retrieval_annotations_csv(
    retrieval_csv_path: str | Path,
    *,
    corpus_csv_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for row in load_csv_rows(retrieval_csv_path):
        query_id = str(row.get("query_id") or "").strip()
        query = str(row.get("query") or "").strip()
        if not query_id or not query:
            continue
        query_row = grouped.setdefault(
            query_id,
            {
                "query_id": query_id,
                "query": query,
                "retrieved_docs": [],
                "corpus_docs": [],
            },
        )
        doc_id = _to_int(row.get("doc_id"))
        rank = _to_int(row.get("rank"))
        if doc_id is None or rank is None:
            continue

        existing = next((item for item in query_row["retrieved_docs"] if item.get("doc_id") == doc_id), None)
        if existing is None:
            query_row["retrieved_docs"].append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "title": row.get("document_title") or "",
                    "relevance_label": str(row.get("relevance_label") or "").strip().lower(),
                }
            )
        else:
            if rank < int(existing.get("rank") or rank):
                existing["rank"] = rank
            existing["relevance_label"] = _prefer_label(existing.get("relevance_label"), row.get("relevance_label"))

    for query_row in grouped.values():
        query_row["retrieved_docs"] = sorted(query_row["retrieved_docs"], key=lambda item: int(item.get("rank") or 0))

    if corpus_csv_path:
        for row in load_csv_rows(corpus_csv_path):
            query_id = str(row.get("query_id") or "").strip()
            query = str(row.get("query") or "").strip()
            if not query_id or not query:
                continue
            query_row = grouped.setdefault(
                query_id,
                {
                    "query_id": query_id,
                    "query": query,
                    "retrieved_docs": [],
                    "corpus_docs": [],
                },
            )
            doc_id = _to_int(row.get("doc_id"))
            if doc_id is None:
                continue
            query_row["corpus_docs"].append(
                {
                    "doc_id": doc_id,
                    "title": row.get("document_title") or "",
                    "relevance_label": str(row.get("relevance_label") or "").strip().lower(),
                }
            )

    return list(grouped.values())


def build_calibration_records_from_claim_csv(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in load_csv_rows(path):
        label = str(row.get("support_label") or row.get("label") or "").strip().lower()
        if label not in {"supported", "unsupported"}:
            continue
        sentence = str(row.get("claim_text") or row.get("text") or "").strip()
        evidence_text = str(row.get("evidence_text") or "").strip()
        if not sentence or not evidence_text:
            continue
        record: dict[str, Any] = {
            "query_id": str(row.get("query_id") or "").strip() or None,
            "claim_id": str(row.get("claim_id") or "").strip() or None,
            "sentence": sentence,
            "evidence_text": evidence_text,
            "label": label,
        }
        m = _to_float(row.get("msa_M"))
        s = _to_float(row.get("msa_S"))
        a = _to_float(row.get("msa_A"))
        if m is not None and s is not None and a is not None:
            record["msa"] = {"M": m, "S": s, "A": a}
        records.append(record)
    return records
