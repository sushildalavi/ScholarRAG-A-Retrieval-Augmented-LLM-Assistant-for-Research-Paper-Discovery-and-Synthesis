"""
Minimal arXiv client (Atom feed).
Docs: https://arxiv.org/help/api/index
"""
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import List, Dict

import requests

logger = logging.getLogger(__name__)

BASE_URL = "http://export.arxiv.org/api/query"
TIMEOUT = 10
MAX_RETRIES = 3


def _extract_year(published: str) -> int | None:
    m = re.match(r"(\\d{4})", published or "")
    return int(m.group(1)) if m else None


def fetch_papers_arxiv(query: str, limit: int = 20) -> List[Dict]:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            out: List[Dict] = []
            for e in entries:
                title = (e.findtext("atom:title", default="", namespaces=ns) or "").strip()
                abstract = (e.findtext("atom:summary", default="", namespaces=ns) or "").strip()
                link = e.find("atom:link[@type='text/html']", ns)
                pub = e.findtext("atom:published", default="", namespaces=ns)
                year = _extract_year(pub)
                authors = [a.findtext("atom:name", default="", namespaces=ns) for a in e.findall("atom:author", ns)]
                out.append(
                    {
                        "id": e.findtext("atom:id", default="", namespaces=ns),
                        "title": title,
                        "abstract": abstract,
                        "authors": [a for a in authors if a],
                        "year": year,
                        "url": link.attrib.get("href") if link is not None else None,
                    }
                )
            return out
        except Exception as exc:  # pragma: no cover
            logger.warning("arXiv fetch failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(min(2 ** attempt, 8.0))
