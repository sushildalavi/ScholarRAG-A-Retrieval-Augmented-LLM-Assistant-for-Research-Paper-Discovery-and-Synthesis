import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

SPRINGER_URL = "http://api.springernature.com/metadata/json"
SPRINGER_KEY = os.getenv("SPRINGER_API_KEY")
SPRINGER_MAX = int(os.getenv("SPRINGER_MAX_RESULTS", "30")) or 30
REQUEST_TIMEOUT = float(os.getenv("SPRINGER_TIMEOUT", "10"))
MAX_RETRIES = 3


def _backoff(attempt: int) -> float:
    return min(2 ** attempt + random.random(), 8.0)


def fetch_from_springer(query: str, limit: Optional[int] = None, year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[Dict]:
    if not SPRINGER_KEY:
        logger.debug("SPRINGER_API_KEY not set; skipping Springer fetch.")
        return []
    remaining = limit if limit is not None else SPRINGER_MAX
    if remaining <= 0:
        return []
    params: Dict[str, str] = {
        "q": f"all:{query}",
        "p": str(remaining),
        "api_key": SPRINGER_KEY,
    }
    if year_from and year_to:
        params["date-facet-mode"] = "between"
        params["date-facet-min"] = str(int(year_from))
        params["date-facet-max"] = str(int(year_to))
    elif year_from:
        params["date-facet-mode"] = "after"
        params["date-facet-min"] = str(int(year_from))
    elif year_to:
        params["date-facet-mode"] = "before"
        params["date-facet-max"] = str(int(year_to))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(SPRINGER_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("records", [])
            results = []
            for r in records:
                yr = r.get("publicationDate") or r.get("publicationName")
                try:
                    yr_int = int(str(yr)[:4]) if yr else None
                except Exception:
                    yr_int = None
                results.append(
                    {
                        "id": r.get("doi") or r.get("identifier"),
                        "title": r.get("title"),
                        "year": yr_int,
                        "doi": r.get("doi"),
                        "abstract": r.get("abstract"),
                        "authors": [{"display_name": a.get("creator")} for a in r.get("creators", [])],
                        "url": r.get("url"),
                        "concepts": r.get("subject") if isinstance(r.get("subject"), list) else [],
                        "source": "springer",
                    }
                )
            return results
        except requests.RequestException as exc:
            logger.warning("Springer request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(_backoff(attempt))
    return []
