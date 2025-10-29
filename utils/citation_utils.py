from typing import Dict


def short_citation(doc: Dict) -> str:
    authors = doc.get("authors") or []
    surname = (authors[0]["family"] if isinstance(authors, list) and authors and isinstance(authors[0], dict) and authors[0].get("family") else None)
    if not surname:
        # Try first author string
        if isinstance(authors, list) and authors:
            surname = str(authors[0]).split()[-1]
        else:
            surname = (doc.get("author") or "Anon").split()[-1]
    year = doc.get("year") or "n.d."
    return f"{surname}, {year}"


def pick_url(doc: Dict) -> str:
    doi = doc.get("doi")
    if doi:
        return f"https://doi.org/{doi}"
    openalex_id = doc.get("openalex_id") or doc.get("id")
    if openalex_id:
        return f"https://openalex.org/{openalex_id}"
    return ""


def make_snippet(text: str, max_chars: int = 900) -> str:
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    return text[:max_chars]

