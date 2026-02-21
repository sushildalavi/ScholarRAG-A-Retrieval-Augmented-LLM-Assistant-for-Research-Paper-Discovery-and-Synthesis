from models import Paper


def normalize_openalex(raw: dict) -> Paper:
    return Paper(
        paper_id=str(raw.get("id")),
        title=raw.get("title", ""),
        abstract=raw.get("abstract", "") or raw.get("summary", ""),
        authors=[a.get("display_name") for a in raw.get("authorships", []) if a.get("display_name")] if raw.get("authorships") else raw.get("authors", []),
        year=raw.get("publication_year") or raw.get("year"),
        source="openalex",
        source_url=raw.get("url") or raw.get("id"),
    )


def normalize_arxiv(raw: dict) -> Paper:
    return Paper(
        paper_id=str(raw.get("id")),
        title=raw.get("title", ""),
        abstract=raw.get("abstract", ""),
        authors=raw.get("authors", []),
        year=raw.get("year"),
        source="arxiv",
        source_url=raw.get("url"),
    )


def normalize_ieee(raw: dict) -> Paper:
    return Paper(
        paper_id=str(raw.get("id") or raw.get("article_number")),
        title=raw.get("title", ""),
        abstract=raw.get("abstract", ""),
        authors=[auth.get("full_name") for auth in raw.get("authors", []) if isinstance(raw.get("authors"), list)],
        year=raw.get("publication_year"),
        source="ieee",
        source_url=raw.get("pdf_url") or raw.get("html_url") or raw.get("url"),
    )


def normalize_springer(raw: dict) -> Paper:
    return Paper(
        paper_id=str(raw.get("doi") or raw.get("identifier") or raw.get("id")),
        title=raw.get("title", ""),
        abstract=raw.get("abstract", ""),
        authors=[c.get("creator") for c in raw.get("creators", []) if c.get("creator")] if isinstance(raw.get("creators"), list) else [],
        year=int(str(raw.get("publicationDate"))[:4]) if raw.get("publicationDate") else raw.get("year"),
        source="springer",
        source_url=raw.get("url"),
    )
