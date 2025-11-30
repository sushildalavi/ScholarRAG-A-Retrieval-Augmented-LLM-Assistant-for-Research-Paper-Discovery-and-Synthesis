"""
OpenAlex client (metadata only).
Docs: https://docs.openalex.org/
"""
import logging
from typing import List, Dict, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org/works"
TIMEOUT = 10


def fetch_papers_openalex(query: str, limit: int = 20, year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[Dict]:
    params: Dict[str, str] = {
        "search": query,
        "per-page": str(limit),
        "sort": "cited_by_count:desc",
        "mailto": "scholarrag@example.com",
    }
    filters = []
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{year_to}-12-31")
    if filters:
        params["filter"] = ",".join(filters)

    try:
        resp = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("OpenAlex fetch failed: %s", exc)
        return []
