import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

ELSEVIER_URL = "https://api.elsevier.com/content/search/sciencedirect"
ELSEVIER_KEY = os.getenv("ELSEVIER_API_KEY")
ELSEVIER_MAX = int(os.getenv("ELSEVIER_MAX_RESULTS", "30")) or 30
REQUEST_TIMEOUT = float(os.getenv("ELSEVIER_TIMEOUT", "10"))
MAX_RETRIES = 3


def _backoff(attempt: int) -> float:
    return min(2 ** attempt + random.random(), 8.0)


def fetch_from_elsevier(query: str, limit: Optional[int] = None, year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[Dict]:
    if not ELSEVIER_KEY:
        logger.debug("ELSEVIER_API_KEY not set; skipping Elsevier fetch.")
        return []
    remaining = limit if limit is not None else ELSEVIER_MAX
    if remaining <= 0:
        return []
    params: Dict[str, str] = {
        "query": query,
        "count": str(remaining),
        "httpAccept": "application/json",
    }
    headers = {"X-ELS-APIKey": ELSEVIER_KEY}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(ELSEVIER_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("search-results", {}).get("entry", []) or []
            results = []
            for e in entries:
                yr = e.get("prism:coverDate", "")[:4]
                try:
                    yr_int = int(yr) if yr else None
                except Exception:
                    yr_int = None
                if year_from and yr_int and yr_int < int(year_from):
                    continue
                if year_to and yr_int and yr_int > int(year_to):
                    continue
                results.append(
                    {
                        "id": e.get("dc:identifier"),
                        "title": e.get("dc:title"),
                        "year": yr_int,
                        "doi": e.get("prism:doi"),
                        "abstract": e.get("dc:description"),
                        "authors": [{"display_name": a.get("given-name", "") + " " + a.get("surname", "")} for a in e.get("dc:creator", []) if isinstance(a, dict)],
                        "url": e.get("link", [{}])[0].get("@href") if isinstance(e.get("link"), list) else None,
                        "concepts": [],
                        "source": "elsevier",
                    }
                )
            return results
        except requests.RequestException as exc:
            logger.warning("Elsevier request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(_backoff(attempt))
    return []
