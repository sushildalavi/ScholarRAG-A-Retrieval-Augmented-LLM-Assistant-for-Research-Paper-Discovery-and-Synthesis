"""
Semantic search over papers using pgvector.
"""
from typing import List, Optional

from db import fetchall
from embeddings import get_embedding


def search_papers(
    query: str,
    k: int = 10,
    year_min: Optional[int] = None,
    sources: Optional[List[str]] = None,
    hybrid: bool = True,
) -> List[dict]:
    """
    Compute query embedding; order by summary presence, optional keyword match, and vector distance.
    Optional filters: year_min, sources list.
    Returns dicts with paper fields + distance.
    """
    q_emb = get_embedding(query)
    params = [q_emb]
    filters = []
    if year_min is not None:
        filters.append("year >= %s")
        params.append(year_min)
    if sources:
        placeholders = ",".join(["%s"] * len(sources))
        filters.append(f"source IN ({placeholders})")
        params.extend(sources)

    kw_hit_select = ""
    kw_hit_order = ""
    if hybrid:
        kw_hit_select = " , CASE WHEN title ILIKE %s OR abstract ILIKE %s THEN 1 ELSE 0 END AS kw_hit "
        params.extend([f"%{query}%", f"%{query}%"])
        kw_hit_order = "kw_hit DESC,"

    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    sql = f"""
        SELECT paper_id,
               title,
               abstract,
               authors,
               year,
               source,
               source_url,
               (embedding <-> %s::vector) AS distance,
               CASE WHEN abstract IS NOT NULL AND length(abstract) > 0 THEN 1 ELSE 0 END AS has_summary
               {kw_hit_select}
        FROM papers
        {where_clause}
        ORDER BY has_summary DESC, {kw_hit_order} embedding <-> %s::vector
        LIMIT {k}
    """
    params.append(q_emb)
    return fetchall(sql, params)
