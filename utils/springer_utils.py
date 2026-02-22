import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

SPRINGER_META_V2_URL = "https://api.springernature.com/meta/v2/json"
SPRINGER_LEGACY_URL = "https://api.springernature.com/metadata/json"
MAX_RETRIES = 3


def _backoff(attempt: int) -> float:
    return min(2 ** attempt + random.random(), 8.0)


def _build_query_expression(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "keyword:test"
    # If caller already provides a fielded query, pass through.
    if ":" in q and any(tag in q.lower() for tag in ("keyword:", "doi:", "issn:", "isbn:", "journal:", "title:")):
        return q
    # Basic-plan friendly default: keyword search.
    safe_q = q.replace('"', "")
    return f'keyword:"{safe_q}"'


def fetch_from_springer(query: str, limit: Optional[int] = None, year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[Dict]:
    springer_key = os.getenv("SPRINGER_API_KEY")
    springer_max = int(os.getenv("SPRINGER_MAX_RESULTS", "30")) or 30
    request_timeout = float(os.getenv("SPRINGER_TIMEOUT", "10"))
    springer_meta_version = (os.getenv("SPRINGER_META_VERSION", "v2") or "v2").strip().lower()
    if not springer_key:
        logger.debug("SPRINGER_API_KEY not set; skipping Springer fetch.")
        return []
    remaining = limit if limit is not None else springer_max
    if remaining <= 0:
        return []

    endpoints = [SPRINGER_META_V2_URL, SPRINGER_LEGACY_URL] if springer_meta_version == "v2" else [SPRINGER_LEGACY_URL, SPRINGER_META_V2_URL]
    query_expr = _build_query_expression(query)

    for endpoint in endpoints:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if endpoint == SPRINGER_META_V2_URL:
                    params: Dict[str, str] = {
                        "api_key": springer_key,
                        "q": query_expr,
                        "s": "1",
                        "p": str(remaining),
                    }
                else:
                    params = {
                        "q": f"all:{query}",
                        "p": str(remaining),
                        "api_key": springer_key,
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

                resp = requests.get(endpoint, params=params, timeout=request_timeout)
                resp.raise_for_status()
                data = resp.json()
                records = data.get("records", []) or []
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
                logger.warning("Springer request failed (%s, attempt %s/%s): %s", endpoint, attempt, MAX_RETRIES, exc)
                if attempt == MAX_RETRIES:
                    break
                time.sleep(_backoff(attempt))
    return []
