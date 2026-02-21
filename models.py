from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    authors: List[str] | str
    year: Optional[int]
    source: str
    source_url: Optional[str] = None


def paper_to_row(paper: Paper, embedding: list[float]) -> Tuple:
    authors_str = paper.authors if isinstance(paper.authors, str) else ", ".join(paper.authors)
    return (
        paper.paper_id,
        paper.title,
        paper.abstract,
        authors_str,
        paper.year,
        paper.source,
        paper.source_url,
        embedding,
    )
