"""
IEEE Xplore client.
Docs: https://developer.ieee.org/docs/read/IEEE_Xplore_API
"""
import logging
import os
import time
from typing import List, Dict, Optional

import requests

logger = logging.getLogger(__name__)

API_KEY = os.getenv("IEEE_API_KEY")
BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
TIMEOUT = 10
MAX_RETRIES = 2


def fetch_papers_ieee(query: str, limit: int = 20, year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[Dict]:
    if not API_KEY:
        logger.debug("IEEE_API_KEY not set; skipping IEEE fetch.")
        return []
    params: Dict[str, str] = {
        "apikey": API_KEY,
        "querytext": query,
        "max_records": str(limit),
        "sort_order": "desc",
        "sort_field": "publication_year",
    }
    if year_from:
        params["start_year"] = str(int(year_from))
    if year_to:
        params["end_year"] = str(int(year_to))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", []) or []
            # If API returns status code in JSON indicating failure, log and break
            if isinstance(data, dict) and data.get("message"):
                logger.warning("IEEE returned message: %s", data.get("message"))
            return articles
        except Exception as exc:  # pragma: no cover
            logger.warning("IEEE fetch failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(min(2 ** attempt, 8.0))
