"""
Springer client.
Docs: https://dev.springernature.com/
"""
import logging
import os
from typing import List, Dict

import requests

logger = logging.getLogger(__name__)

API_KEY = os.getenv("SPRINGER_API_KEY")
BASE_URL = "http://api.springernature.com/metadata/json"
TIMEOUT = 10


def fetch_papers_springer(query: str, limit: int = 20) -> List[Dict]:
    if not API_KEY:
        logger.debug("SPRINGER_API_KEY not set; skipping Springer fetch.")
        return []
    params = {
        "q": f"all:{query}",
        "p": str(limit),
        "api_key": API_KEY,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("records", []) or []
    except Exception as exc:  # pragma: no cover
        logger.warning("Springer fetch failed: %s", exc)
        return []
