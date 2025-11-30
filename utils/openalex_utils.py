import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests


OPENALEX_BASE_URL = "https://api.openalex.org/works"
MAX_PER_PAGE = 200
RATE_LIMIT_DELAY = 0.4  # seconds between requests to respect rate limits
MAX_RETRIES = 3
DEFAULT_MAX_RESULTS = int(os.getenv("OPENALEX_MAX_RESULTS", "0")) or None
REQUEST_TIMEOUT = float(os.getenv("OPENALEX_TIMEOUT", "12"))

logger = logging.getLogger(__name__)


def _build_filter(query: str, year_from: Optional[int], year_to: Optional[int]) -> str:
    filters = []
    if query:
        filters.append(f"title.search:{query}")
    if year_from:
        filters.append(f"from_publication_date:{int(year_from)}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{int(year_to)}-12-31")
    return ",".join(filters)


def _reconstruct_abstract(inv_index: Optional[Dict]) -> str:
    if not inv_index:
        return ""
    flattened = []
    for word, positions in inv_index.items():
        for pos in positions:
            flattened.append((pos, word))
    flattened.sort(key=lambda x: x[0])
    return " ".join(word for _, word in flattened)


def _backoff_seconds(attempt: int) -> float:
    # Exponential backoff with jitter
    return min(2 ** attempt + random.random(), 10.0)


def fetch_candidates_from_openalex(
    query: str,
    limit: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[Dict]:
    """
    Fetch candidate papers from OpenAlex API.

    Args:
        query: Search query (matched against title).
        limit: Optional cap on total results. If None, fetch all available
               pages (subject to OPENALEX_MAX_RESULTS env cap).
        year_from/year_to: Publication year bounds.
    """

    # Determine cap: explicit limit wins, otherwise env var, otherwise unlimited.
    max_results = limit if limit is not None else DEFAULT_MAX_RESULTS
    remaining = max_results if max_results is not None else float("inf")

    params: Dict[str, str] = {
        "per-page": str(min(MAX_PER_PAGE, int(remaining))) if remaining != float("inf") else str(MAX_PER_PAGE),
        "sort": "relevance_score:desc",
        "cursor": "*",
        "mailto": os.getenv("OPENALEX_MAILTO", "scholarrag@example.com"),
    }
    filter_expr = _build_filter(query, year_from, year_to)
    if filter_expr:
        params["filter"] = filter_expr

    fields = [
        "id",
        "display_name",
        "publication_year",
        "doi",
        "concepts",
        "abstract_inverted_index",
        "authorships",
    ]
    params["select"] = ",".join(fields)

    session = requests.Session()
    results: List[Dict] = []

    while remaining > 0 and params.get("cursor"):
        response = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = session.get(OPENALEX_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                logger.warning("OpenAlex request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
                if attempt == MAX_RETRIES:
                    return results
                time.sleep(_backoff_seconds(attempt))

        payload = response.json()
        works = payload.get("results", [])
        if not works:
            break

        for work in works:
            if remaining <= 0:
                break
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author") or {}
                name = author.get("display_name")
                authors.append(
                    {
                        "id": author.get("id"),
                        "display_name": name,
                        "family": (name.split()[-1] if name else None),
                    }
                )

            results.append(
                {
                    "id": work.get("id"),
                    "openalex_id": work.get("id"),
                    "title": work.get("display_name"),
                    "year": work.get("publication_year"),
                    "doi": work.get("doi"),
                    "concepts": [c.get("display_name") for c in work.get("concepts", [])],
                    "abstract": abstract,
                    "authors": authors,
                }
            )
            remaining -= 1

        params["cursor"] = payload.get("meta", {}).get("next_cursor")
        time.sleep(RATE_LIMIT_DELAY)

    return results
