import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

CROSSREF_URL = "https://api.crossref.org/works"
CROSSREF_MAX = int(os.getenv("CROSSREF_MAX_RESULTS", "50")) or 50
REQUEST_TIMEOUT = float(os.getenv("CROSSREF_TIMEOUT", "10"))
MAX_RETRIES = 3


def _backoff(attempt: int) -> float:
    return min(2 ** attempt + random.random(), 8.0)


def fetch_from_crossref(query: str, limit: Optional[int] = None, year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[Dict]:
    remaining = limit if limit is not None else CROSSREF_MAX
    if remaining <= 0:
        return []
    params: Dict[str, str] = {
        "query": query,
        "rows": str(remaining),
        "sort": "published",
        "order": "desc",
    }
    filters = []
    if year_from:
        filters.append(f"from-pub-date:{int(year_from)}-01-01")
    if year_to:
        filters.append(f"until-pub-date:{int(year_to)}-12-31")
    if filters:
        params["filter"] = ",".join(filters)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(CROSSREF_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("message", {}).get("items", [])
            results = []
            for it in items:
                results.append(
                    {
                        "id": it.get("DOI"),
                        "title": (it.get("title") or ["Untitled"])[0],
                        "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
                        "doi": it.get("DOI"),
                        "abstract": it.get("abstract"),
                        "concepts": [s.get("subject") for s in it.get("subject", [])] if isinstance(it.get("subject"), list) else [],
                        "authors": [{"display_name": " ".join(a.get("given", "").split() + a.get("family", "").split())} for a in it.get("author", [])],
                        "url": it.get("URL"),
                        "source": "crossref",
                    }
                )
            return results
        except requests.RequestException as exc:
            logger.warning("CrossRef request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(_backoff(attempt))
    return []
