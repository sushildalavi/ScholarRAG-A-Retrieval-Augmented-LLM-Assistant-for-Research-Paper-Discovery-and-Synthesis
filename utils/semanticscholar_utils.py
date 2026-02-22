import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
# Keep default small to reduce 429s if no API key
S2_FIELDS = "title,year,abstract,externalIds,authors,venue,fieldsOfStudy,url"
MAX_RETRIES = 3


def _backoff(attempt: int) -> float:
    return min(2 ** attempt + random.random(), 8.0)


def fetch_from_s2(query: str, limit: Optional[int] = None, year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[Dict]:
    s2_max = int(os.getenv("S2_MAX_RESULTS", "20")) or 20
    s2_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    request_timeout = float(os.getenv("S2_TIMEOUT", "10"))
    remaining = limit if limit is not None else s2_max
    if remaining <= 0:
        return []
    params: Dict[str, str] = {
        "query": query,
        "limit": str(remaining),
        "fields": S2_FIELDS,
    }
    headers = {}
    if s2_api_key:
        headers["x-api-key"] = s2_api_key
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(S2_URL, params=params, headers=headers, timeout=request_timeout)
            if resp.status_code == 429:
                # Too many requests: back off more aggressively
                sleep_for = min(2 ** attempt + random.random(), 12.0)
                logger.warning("Semantic Scholar 429 (attempt %s/%s); backing off %.1fs", attempt, MAX_RETRIES, sleep_for)
                time.sleep(sleep_for)
                continue
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])
            results = []
            for it in items:
                y = it.get("year")
                if year_from and y and y < int(year_from):
                    continue
                if year_to and y and y > int(year_to):
                    continue
                doi = None
                ext = it.get("externalIds") or {}
                if isinstance(ext, dict):
                    doi = ext.get("DOI")
                results.append(
                    {
                        "id": it.get("paperId"),
                        "title": it.get("title"),
                        "year": y,
                        "doi": doi,
                        "abstract": it.get("abstract"),
                        "concepts": it.get("fieldsOfStudy") or [],
                        "authors": [{"display_name": a.get("name")} for a in it.get("authors", [])],
                        "url": it.get("url"),
                        "source": "semanticscholar",
                    }
                )
            return results
        except requests.RequestException as exc:
            logger.warning("Semantic Scholar request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(_backoff(attempt))
    return []
