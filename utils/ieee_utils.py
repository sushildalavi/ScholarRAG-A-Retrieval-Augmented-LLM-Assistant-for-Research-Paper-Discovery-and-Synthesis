import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

IEEE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
IEEE_KEY = os.getenv("IEEE_API_KEY")
IEEE_MAX = int(os.getenv("IEEE_MAX_RESULTS", "30")) or 30
REQUEST_TIMEOUT = float(os.getenv("IEEE_TIMEOUT", "10"))
MAX_RETRIES = 3


def _backoff(attempt: int) -> float:
    return min(2 ** attempt + random.random(), 8.0)


def fetch_from_ieee(query: str, limit: Optional[int] = None, year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[Dict]:
    if not IEEE_KEY:
        logger.debug("IEEE_API_KEY not set; skipping IEEE fetch.")
        return []
    remaining = limit if limit is not None else IEEE_MAX
    if remaining <= 0:
        return []
    params: Dict[str, str] = {
        "querytext": query,
        "apikey": IEEE_KEY,
        "max_records": str(remaining),
        "sort_order": "desc",
        "sort_field": "publication_year",
    }
    if year_from:
        params["start_year"] = str(int(year_from))
    if year_to:
        params["end_year"] = str(int(year_to))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(IEEE_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", []) or []
            results = []
            for a in articles:
                yr = a.get("publication_year")
                if year_from and yr and int(yr) < int(year_from):
                    continue
                if year_to and yr and int(yr) > int(year_to):
                    continue
                results.append(
                    {
                        "id": a.get("article_number"),
                        "title": a.get("title"),
                        "year": yr,
                        "doi": a.get("doi"),
                        "abstract": a.get("abstract"),
                        "authors": [{"display_name": auth.get("full_name")} for auth in a.get("authors", []) if isinstance(a.get("authors"), list)],
                        "url": a.get("pdf_url") or a.get("html_url"),
                        "concepts": a.get("index_terms", []),
                        "source": "ieee",
                    }
                )
            return results
        except requests.RequestException as exc:
            logger.warning("IEEE request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(_backoff(attempt))
    return []
