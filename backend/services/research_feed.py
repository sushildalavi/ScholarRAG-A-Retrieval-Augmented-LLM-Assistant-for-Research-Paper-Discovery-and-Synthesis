import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any

import requests


logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org/works"
ARXIV_API_URL = os.getenv("ARXIV_API_URL", "https://export.arxiv.org/api/query")
RESEARCH_FEED_TIMEOUT = float(os.getenv("RESEARCH_FEED_TIMEOUT", "12"))
RESEARCH_FEED_CACHE_TTL_SECONDS = int(os.getenv("RESEARCH_FEED_CACHE_TTL_SECONDS", "900")) or 900
RESEARCH_FEED_MAX_WORKERS = int(os.getenv("RESEARCH_FEED_MAX_WORKERS", "2")) or 2

_FEED_CACHE: dict[tuple[str, int, int, str], tuple[float, dict[str, Any]]] = {}
DEFAULT_DISCOVERY_TOPIC = "artificial intelligence machine learning natural language processing computer vision"


def _reconstruct_abstract(inv_index: dict[str, list[int]] | None) -> str:
    if not inv_index:
        return ""
    flattened: list[tuple[int, str]] = []
    for word, positions in inv_index.items():
        for pos in positions:
            flattened.append((pos, word))
    flattened.sort(key=lambda x: x[0])
    return " ".join(word for _, word in flattened)


def _normalize_topic(topic: str | None) -> str:
    value = (topic or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def _normalize_sort(sort: str | None) -> str:
    value = (sort or "latest").strip().lower()
    if value not in {"latest", "top_cited", "trending"}:
        return "latest"
    return value


def _plain_text(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _iso_date_days_ago(days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    return dt.date().isoformat()


def _latest_openalex(topic: str | None, limit: int, days: int, sort: str) -> list[dict[str, Any]]:
    normalized_sort = _normalize_sort(sort)
    if normalized_sort == "top_cited":
        fetch_limit = max(24, min(limit * 5, 80))
        lookback_days = max(days * 8, 720)
        provider_sort = "cited_by_count:desc"
        why_relevant = "Highly cited work from OpenAlex ranked by citation count."
    elif normalized_sort == "trending":
        fetch_limit = max(24, min(limit * 4, 72))
        lookback_days = max(days * 3, 120)
        provider_sort = "cited_by_count:desc"
        why_relevant = "Recent OpenAlex work with strong citation activity."
    else:
        fetch_limit = max(18, min(limit * 3, 50))
        lookback_days = max(1, days)
        provider_sort = "publication_date:desc"
        why_relevant = "Recent work from OpenAlex ordered by publication date."

    params: dict[str, str] = {
        "per-page": str(fetch_limit),
        "sort": provider_sort,
        "select": ",".join(
            [
                "id",
                "display_name",
                "publication_year",
                "publication_date",
                "doi",
                "cited_by_count",
                "abstract_inverted_index",
                "authorships",
                "primary_location",
                "concepts",
            ]
        ),
    }
    openalex_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if openalex_key:
        params["api_key"] = openalex_key

    filters = [f"from_publication_date:{_iso_date_days_ago(lookback_days)}"]
    topic_value = _normalize_topic(topic) or DEFAULT_DISCOVERY_TOPIC
    if topic_value:
        params["search"] = topic_value
    params["filter"] = ",".join(filters)

    resp = requests.get(OPENALEX_BASE_URL, params=params, timeout=RESEARCH_FEED_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    results = []
    for work in payload.get("results", [])[:limit]:
        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        authors = []
        for authorship in work.get("authorships", [])[:6]:
            author = authorship.get("author") or {}
            if author.get("display_name"):
                authors.append(author.get("display_name"))
        results.append(
            {
                "provider": "openalex",
                "id": work.get("id"),
                "title": work.get("display_name"),
                "abstract": _plain_text(_reconstruct_abstract(work.get("abstract_inverted_index"))),
                "authors": authors,
                "year": work.get("publication_year"),
                "published_at": work.get("publication_date"),
                "url": primary_location.get("landing_page_url") or work.get("id"),
                "pdf_url": primary_location.get("pdf_url"),
                "venue": source.get("display_name"),
                "doi": work.get("doi"),
                "citation_count": work.get("cited_by_count") or 0,
                "topics": [c.get("display_name") for c in work.get("concepts", [])[:5] if c.get("display_name")],
                "why_relevant": why_relevant,
            }
        )
    return results


def _latest_arxiv(topic: str | None, limit: int, days: int, sort: str) -> list[dict[str, Any]]:
    normalized_sort = _normalize_sort(sort)
    if normalized_sort == "top_cited":
        return []

    query = _normalize_topic(topic) or "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max(1, min(limit * (3 if normalized_sort == "trending" else 2), 30)),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API_URL, params=params, timeout=RESEARCH_FEED_TIMEOUT)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    results = []
    for entry in root.findall("atom:entry", ns):
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        published_dt: datetime | None = None
        try:
            published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            published_dt = None
        if published_dt and published_dt < cutoff:
            continue
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        abstract = _plain_text(entry.findtext("atom:summary", default="", namespaces=ns))
        authors = [
            a.findtext("atom:name", default="", namespaces=ns)
            for a in entry.findall("atom:author", ns)
            if a.findtext("atom:name", default="", namespaces=ns)
        ]
        html_link = None
        pdf_link = None
        for link in entry.findall("atom:link", ns):
            href = link.attrib.get("href")
            if link.attrib.get("type") == "text/html":
                html_link = href
            if link.attrib.get("title") == "pdf" or (href and href.endswith(".pdf")):
                pdf_link = href
        results.append(
            {
                "provider": "arxiv",
                "id": entry.findtext("atom:id", default="", namespaces=ns),
                "title": title,
                "abstract": abstract,
                "authors": authors[:6],
                "year": published_dt.year if published_dt else None,
                "published_at": published_dt.date().isoformat() if published_dt else published,
                "url": html_link or entry.findtext("atom:id", default="", namespaces=ns),
                "pdf_url": pdf_link,
                "venue": "arXiv",
                "doi": entry.findtext("atom:doi", default="", namespaces=ns) or None,
                "citation_count": 0,
                "topics": [c.attrib.get("term") for c in entry.findall("atom:category", ns)[:5] if c.attrib.get("term")],
                "why_relevant": "Recent preprint from arXiv ordered by submission date."
                if normalized_sort == "latest"
                else "Recent arXiv preprint with strong topic overlap.",
            }
        )
        if len(results) >= limit:
            break
    return results


def _dedupe_papers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (
            (row.get("doi") or "").strip().lower()
            or (row.get("title") or "").strip().lower()
            or str(row.get("id") or "").strip().lower()
        )
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _published_timestamp(row: dict[str, Any]) -> float:
    published = row.get("published_at") or ""
    try:
        return datetime.fromisoformat(str(published).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _age_days(row: dict[str, Any]) -> float:
    ts = _published_timestamp(row)
    if not ts:
        return 9999.0
    delta = max(time.time() - ts, 1.0)
    return delta / 86400.0


def _trend_score(row: dict[str, Any]) -> float:
    cites = float(row.get("citation_count") or 0)
    age = max(_age_days(row), 3.0)
    return (cites + 5.0) / (age ** 0.72)


def _sort_rows(rows: list[dict[str, Any]], sort: str) -> None:
    if sort == "top_cited":
        rows.sort(key=lambda row: (float(row.get("citation_count") or 0), _published_timestamp(row)), reverse=True)
        return
    if sort == "trending":
        rows.sort(key=lambda row: (_trend_score(row), _published_timestamp(row)), reverse=True)
        return
    rows.sort(key=lambda row: (_published_timestamp(row), float(row.get("citation_count") or 0)), reverse=True)


def latest_research_feed(topic: str | None = None, limit: int = 8, days: int = 45, sort: str | None = "latest") -> dict[str, Any]:
    normalized_topic = _normalize_topic(topic)
    normalized_sort = _normalize_sort(sort)
    capped_limit = max(1, min(limit, 24))
    capped_days = max(1, min(days, 365))
    cache_key = (normalized_topic, capped_limit, capped_days, normalized_sort)
    cached = _FEED_CACHE.get(cache_key)
    now = time.time()
    if cached and (now - cached[0] <= RESEARCH_FEED_CACHE_TTL_SECONDS):
        return cached[1]

    providers = {
        "openalex": lambda: _latest_openalex(normalized_topic or None, capped_limit, capped_days, normalized_sort),
        "arxiv": lambda: _latest_arxiv(normalized_topic or None, capped_limit, capped_days, normalized_sort),
    }
    provider_status: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(RESEARCH_FEED_MAX_WORKERS, len(providers))) as executor:
        future_map = {executor.submit(fetcher): provider for provider, fetcher in providers.items()}
        for future in as_completed(future_map):
            provider = future_map[future]
            try:
                provider_rows = future.result() or []
                provider_status[provider] = {"ok": True, "count": len(provider_rows)}
                rows.extend(provider_rows)
            except Exception as exc:
                logger.warning("latest research provider failed provider=%s error=%s", provider, exc)
                provider_status[provider] = {"ok": False, "count": 0, "error": str(exc)}

    deduped = _dedupe_papers(rows)
    _sort_rows(deduped, normalized_sort)
    payload = {
        "topic": normalized_topic or None,
        "days": capped_days,
        "sort": normalized_sort,
        "results": deduped[:capped_limit],
        "provider_status": provider_status,
    }
    _FEED_CACHE[cache_key] = (now, payload)
    return payload
