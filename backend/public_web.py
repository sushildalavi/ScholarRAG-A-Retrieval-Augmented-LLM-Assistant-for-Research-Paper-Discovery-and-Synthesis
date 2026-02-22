from __future__ import annotations

import re
from typing import List, Dict

import requests

HEADERS = {
    "User-Agent": "ScholarRAG/1.0 (research assistant; contact: local-dev)",
    "Accept": "application/json",
}


def _tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in toks if len(t) > 2}


def _overlap_score(query: str, text: str) -> float:
    q = _tokens(query)
    t = _tokens(text)
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def _wiki_search_titles(query: str, limit: int = 5) -> List[str]:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max(1, min(limit, 10)),
                "format": "json",
            },
            timeout=8,
            headers=HEADERS,
        )
        r.raise_for_status()
        data = r.json()
        return [x.get("title", "") for x in data.get("query", {}).get("search", []) if x.get("title")]
    except Exception:
        return []


def _wiki_summary(title: str) -> Dict:
    try:
        url_title = title.replace(" ", "_")
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{url_title}",
            timeout=8,
            headers=HEADERS,
        )
        r.raise_for_status()
        data = r.json()
        return {
            "title": data.get("title") or title,
            "snippet": data.get("extract") or "",
            "url": (data.get("content_urls", {}).get("desktop", {}) or {}).get("page"),
            "source": "wikipedia",
        }
    except Exception:
        return {"title": title, "snippet": "", "url": None, "source": "wikipedia"}


def _duckduckgo_instant(query: str) -> List[Dict]:
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8,
            headers=HEADERS,
        )
        r.raise_for_status()
        data = r.json()
        out = []
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            out.append(
                {
                    "title": data.get("Heading") or query,
                    "snippet": abstract,
                    "url": data.get("AbstractURL") or None,
                    "source": "duckduckgo",
                }
            )
        return out
    except Exception:
        return []


def public_web_search(query: str, k: int = 6) -> List[Dict]:
    candidates: List[Dict] = []
    for t in _wiki_search_titles(query, limit=max(3, k)):
        s = _wiki_summary(t)
        if s.get("snippet"):
            candidates.append(s)
    candidates.extend(_duckduckgo_instant(query))

    scored = []
    for c in candidates:
        text = f"{c.get('title','')} {c.get('snippet','')}"
        sim = _overlap_score(query, text)
        if sim <= 0:
            continue
        cc = dict(c)
        cc["_sim"] = round(sim, 3)
        scored.append(cc)
    scored.sort(key=lambda x: x.get("_sim", 0.0), reverse=True)
    return scored[: max(1, k)]
