import logging
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

ARXIV_API = os.getenv("ARXIV_API_URL", "https://export.arxiv.org/api/query")
ARXIV_PAGE_SIZE = 50  # keep pages small to reduce 429s
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "50")) or 50
ARXIV_TIMEOUT = float(os.getenv("ARXIV_TIMEOUT", "12"))
ARXIV_MAX_RETRIES = 3


def _backoff_seconds(attempt: int) -> float:
    return min(2 ** attempt + random.random(), 8.0)


def _extract_year(date_str: str) -> Optional[int]:
    if not date_str:
        return None
    m = re.match(r"(\\d{4})", date_str)
    return int(m.group(1)) if m else None


def fetch_arxiv_candidates(
    query: str,
    limit: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> List[Dict]:
    """
    Fetch candidates from arXiv API. Returns basic metadata plus abstract.
    """
    max_results = limit if limit is not None else ARXIV_MAX_RESULTS
    remaining = max_results if max_results is not None else float("inf")
    start = 0
    results: List[Dict] = []

    def within_year(year: Optional[int]) -> bool:
        if year is None:
            return True
        if year_from and year < year_from:
            return False
        if year_to and year > year_to:
            return False
        return True

    while remaining > 0:
        page_size = min(ARXIV_PAGE_SIZE, int(remaining)) if remaining != float("inf") else ARXIV_PAGE_SIZE
        params = {
            "search_query": query,
            "start": start,
            "max_results": page_size,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        response = None
        for attempt in range(1, ARXIV_MAX_RETRIES + 1):
            try:
                response = requests.get(ARXIV_API, params=params, timeout=ARXIV_TIMEOUT)
                if response.status_code == 429:
                    # Back off and retry
                    time.sleep(_backoff_seconds(attempt))
                    continue
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                logger.warning("arXiv request failed (attempt %s/%s): %s", attempt, ARXIV_MAX_RETRIES, exc)
                if attempt == ARXIV_MAX_RETRIES:
                    return results
                time.sleep(_backoff_seconds(attempt))

        if response is None:
            break

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            logger.warning("arXiv parse failed (start=%s, remaining=%s): %s", start, remaining, exc)
            break
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            break

        for entry in entries:
            if remaining <= 0:
                break
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            abstract = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            link = entry.find("atom:link[@type='text/html']", ns)
            doi = entry.findtext("atom:doi", default="", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns)
            year = _extract_year(published)
            if not within_year(year):
                continue
            authors = [a.findtext("atom:name", default="", namespaces=ns) for a in entry.findall("atom:author", ns)]
            concept_tags = [c.attrib.get("term") for c in entry.findall("atom:category", ns)]

            results.append(
                {
                    "id": entry.findtext("atom:id", default="", namespaces=ns),
                    "arxiv_id": entry.findtext("atom:id", default="", namespaces=ns),
                    "title": title,
                    "year": year,
                    "doi": doi,
                    "abstract": abstract,
                    "authors": [{"display_name": a} for a in authors if a],
                    "concepts": [c for c in concept_tags if c],
                    "url": link.attrib.get("href") if link is not None else None,
                    "source": "arxiv",
                }
            )
            remaining -= 1

        start += page_size

    return results
