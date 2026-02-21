"""
Ingestion and upsert pipeline.
"""
from typing import Callable, Dict

from backend.services.db import execute
from backend.services.embeddings import get_embedding
from models import Paper, paper_to_row
from normalization import (
    normalize_openalex,
    normalize_arxiv,
    normalize_ieee,
    normalize_springer,
)
from sources.openalex_client import fetch_papers_openalex
from sources.arxiv_client import fetch_papers_arxiv
from sources.ieee_client import fetch_papers_ieee
from sources.springer_client import fetch_papers_springer

FETCHERS: Dict[str, Callable] = {
    "openalex": fetch_papers_openalex,
    "arxiv": fetch_papers_arxiv,
    # "ieee": fetch_papers_ieee,  # Disabled (no API key)
    "springer": fetch_papers_springer,
}

NORMALIZERS: Dict[str, Callable] = {
    "openalex": normalize_openalex,
    "arxiv": normalize_arxiv,
    # "ieee": normalize_ieee,  # Disabled (no API key)
    "springer": normalize_springer,
}


def index_paper(paper: Paper) -> None:
    """Compute embedding and upsert a single paper."""
    emb = get_embedding(f"{paper.title}\n{paper.abstract}")
    row = paper_to_row(paper, emb)
    execute(
        """
        INSERT INTO papers (paper_id, title, abstract, authors, year, source, source_url, embedding)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (paper_id) DO UPDATE SET
            title = EXCLUDED.title,
            abstract = EXCLUDED.abstract,
            authors = EXCLUDED.authors,
            year = EXCLUDED.year,
            source = EXCLUDED.source,
            source_url = EXCLUDED.source_url,
            embedding = EXCLUDED.embedding,
            updated_at = now()
        """,
        row,
    )


def index_from_source(source: str, query: str, limit: int = 20) -> None:
    """
    Fetch, normalize, and index from a given source. Supports incremental updates via upsert.
    """
    fetcher = FETCHERS.get(source)
    normalizer = NORMALIZERS.get(source)
    if not fetcher or not normalizer:
        raise ValueError(f"Unsupported source: {source}")
    raw_list = fetcher(query, limit=limit)
    for raw in raw_list:
        paper = normalizer(raw)
        index_paper(paper)
    # Note: could add change detection to skip re-embedding unchanged records.
