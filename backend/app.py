# ------------------------------
# app.py — ScholarRAG Backend API
# ------------------------------

import faiss
import json
import numpy as np
import time
import re
from datetime import datetime
from pathlib import Path
from fastapi import Body
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from utils.config import get_openai_api_key
from backend.retriever import Retriever
from backend.generator import synthesize_answer
from backend import auth
from backend import memory
from backend import agents
from backend import pdf_ingest
from backend.user_store import get_user_index
from backend import chat
from fastapi.responses import Response
from backend.pdf_ingest import search_chunks as search_uploaded_chunks
from utils.embedding_utils import embed_query, embed_batch_cached
from utils.logging_utils import setup_file_logger, log_json
from backend.public_search import public_live_search

# Initialize FastAPI app
app = FastAPI(title="ScholarRAG API", version="1.0")

# Allow frontend access
# CORS: allow local dev frontend
# CORS: allow local dev frontend explicitly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(memory.router)
app.include_router(agents.router)
app.include_router(pdf_ingest.router)
app.include_router(chat.router)


@app.get("/favicon.ico")
def favicon():
    """Avoid 404 spam from browsers requesting favicon."""
    return Response(status_code=204)


# ------------------------------
# Assistant endpoint (RAG + GPT-4o-mini)
# ------------------------------


@app.post("/assistant/answer")
def assistant_answer(
    payload: dict = Body(
        ...,
        example={
            "query": "What does the paper really address?",
            "scope": "uploaded",
            "doc_id": None,
            "k": 10,
        },
    )
):
    """
    Unified QA endpoint for uploaded docs (chunk RAG) or public papers (FAISS/external).
    Returns answer plus lightweight citations.
    """
    started = time.time()
    query = payload.get("query") or ""
    scope = payload.get("scope") or "uploaded"
    doc_id = payload.get("doc_id")
    k = int(payload.get("k") or 10)
    multi_hop = bool(payload.get("multi_hop"))

    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    def fetch_context(q: str):
        local_citations = []
        local_context = []
        if scope == "uploaded":
            results = search_uploaded_chunks(q, k=k, doc_id=doc_id)["results"]
            for r in results:
                local_citations.append(
                    {
                        "title": f"Document {r.get('document_id')}",
                        "source": "uploaded",
                        "doc_id": r.get("document_id"),
                        "chunk_id": r.get("id"),
                        "page": r.get("page_no"),
                    }
                )
                local_context.append(
                    f"[doc {r.get('document_id')} chunk {r.get('id')} page {r.get('page_no','?')}] {r.get('text','')}"
                )
        else:
            docs = public_live_search(q, k=min(k, 8))
            for d in docs:
                local_citations.append(
                    {
                        "title": d.get("title"),
                        "year": d.get("year"),
                        "source": d.get("source") or d.get("venue"),
                        "url": d.get("url") or d.get("doi"),
                    }
                )
                local_context.append(
                    f"[{d.get('title','')}] {d.get('abstract') or d.get('summary') or ''}"
                )
        return local_context, local_citations

    citations = []
    context_blocks = []

    if multi_hop and (" and " in query or ";" in query or "," in query):
        subqs = [q.strip() for q in re.split(r"and|;|,", query) if q.strip()]
        for sq in subqs:
            ctx, cits = fetch_context(sq)
            context_blocks.extend(ctx)
            citations.extend(cits)
    else:
        ctx, cits = fetch_context(query)
        context_blocks.extend(ctx)
        citations.extend(cits)

    context = "\n\n".join(context_blocks[:k]) if context_blocks else "No context found."
    prompt = (
        "You are a research assistant. Use the provided context to answer. "
        "Respond with a detailed answer (multiple paragraphs) and cite sources inline like [1], [2]. "
        "If context is weak, say so. Do not invent citations.\n\n"
        f"Question:\n{query}\n\nContext:\n{context}\n"
    )

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI client not configured. Set OPENAI_API_KEY (and install python-dotenv if relying on .env).",
        )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc

    trust = round(min(1.0, len(citations) / max(1, k)), 3)
    latency_ms = int((time.time() - started) * 1000)

    log_json(
        REQUEST_LOG,
        {
            "ts": time.time(),
            "event": "assistant_answer",
            "query": query,
            "scope": scope,
            "doc_id": doc_id,
            "k": k,
            "multi_hop": multi_hop,
            "context_count": len(context_blocks),
            "citations": len(citations),
            "trust": trust,
            "latency_ms": latency_ms,
        },
    )

    return {"answer": answer, "citations": citations, "trust": trust, "latency_ms": latency_ms}


@app.get("/metrics/requests")
def metrics_requests():
    """
    Lightweight aggregation over logs/requests.jsonl (assistant_answer events).
    """
    import json

    path = LOG_DIR / "requests.jsonl"
    if not path.exists():
        return {"count": 0, "avg_latency_ms": None, "avg_trust": None, "avg_citations": None}

    latencies = []
    trusts = []
    cits = []
    count = 0
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event") != "assistant_answer":
                continue
            count += 1
            if rec.get("latency_ms") is not None:
                latencies.append(rec["latency_ms"])
            if rec.get("trust") is not None:
                trusts.append(rec["trust"])
            if rec.get("citations") is not None:
                cits.append(rec["citations"])

    def avg(arr):
        return sum(arr) / len(arr) if arr else None

    return {
        "count": count,
        "avg_latency_ms": avg(latencies),
        "avg_trust": avg(trusts),
        "avg_citations": avg(cits),
    }

# ------------------------------
# Load FAISS index and metadata
# ------------------------------
INDEX_PATH = "data/scholar_index.faiss"
META_PATH = "data/metadata.json"
LOG_DIR = Path("logs")
RETRIEVAL_LOG = LOG_DIR / "retrieval.log"

try:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"✅ Loaded FAISS index with {index.ntotal} vectors and metadata for {len(meta)} papers")
except Exception as e:
    print(f"❌ Error loading FAISS index or metadata: {e}")
    index, meta = None, []

retriever = Retriever(index, meta) if index is not None else None

# ------------------------------
# OpenAI Embedding Config
# ------------------------------
try:
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)
except RuntimeError as err:
    print(f"❌ {err}")
    client = None

EMBED_MODEL = "text-embedding-3-small"  # aligned with 1536-dim pgvector schema

# Logger for observability
REQUEST_LOG = setup_file_logger(LOG_DIR / "requests.jsonl")

def get_embedding(text: str) -> np.ndarray:
    """Generate embedding vector for a query string."""
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured.")
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array([response.data[0].embedding], dtype="float32")




def log_request(entry: dict) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with RETRIEVAL_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def trust_score(sim: float, has_doi: bool) -> float:
    # Simple heuristic trust: similarity plus DOI bonus
    base = max(sim, 0.0)
    bonus = 0.05 if has_doi else 0.0
    return round(min(base + bonus, 1.0), 3)

# ------------------------------
# Endpoints
# ------------------------------

@app.get("/")
def home():
    return {"message": "ScholarRAG backend is live!"}

@app.get("/feed/latest")
def latest_papers(limit: int = 10):
    """Return latest papers from the global metadata by year (desc)."""
    if not meta:
        return {"results": []}
    def year_val(m):
        try:
            return int(m.get("year") or 0)
        except Exception:
            return 0
    sorted_meta = sorted(meta, key=year_val, reverse=True)
    out = []
    for m in sorted_meta[: max(1, limit)]:
        link = None
        if m.get("doi"):
            link = f"https://doi.org/{m.get('doi')}"
        elif m.get("id"):
            link = m.get("id")
        out.append(
            {
                "title": m.get("title", "Untitled"),
                "year": m.get("year"),
                "doi": m.get("doi"),
                "summary": (m.get("abstract") or m.get("summary") or "")[:280],
                "concepts": m.get("concepts", [])[:5],
                "url": link,
            }
        )
    return {"results": out}

@app.get("/search")
def search_papers(query: str = Query(..., description="Search query text"), k: int = 5):
    """Return top-k relevant papers for a given query."""
    if index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded.")

    q_emb = get_embedding(query)
    D, I = index.search(q_emb, k)
    results = []
    for rank, idx in enumerate(I[0]):
        if idx < len(meta):
            meta_entry = meta[idx]
            concepts = meta_entry.get("concepts") or []
            results.append({
                "rank": rank + 1,
                "title": meta_entry.get("title", "Unknown Title"),
                "year": meta_entry.get("year", "Unknown Year"),
                "doi": meta_entry.get("doi", ""),
                "concepts": concepts[:3],
                "similarity": round(float(D[0][rank]), 3)
            })
    return {"query": query, "results": results}

@app.get("/summarize")
def summarize(query: str = Query(..., description="Topic to summarize")):
    """Summarize top retrieved papers for a given query using GPT."""
    if index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded.")
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured.")

    q_emb = get_embedding(query)
    D, I = index.search(q_emb, 5)
    top_titles = "\n".join(
        [
            f"- {meta[idx].get('title', 'Unknown Title')}"
            for idx in I[0]
            if idx < len(meta)
        ]
    )

    prompt = f"Summarize the key themes and insights from the following recent papers about '{query}':\n{top_titles}"
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    summary = completion.choices[0].message.content
    return {"query": query, "summary": summary}

@app.post("/ask")
def ask(payload: dict = Body(...)):
    if index is None or retriever is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded.")
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured.")

    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query'.")

    k = int(payload.get("k", 10))
    year_from = payload.get("year_from")
    year_to = payload.get("year_to")
    multi_hop = bool(payload.get("multi_hop", True))
    user_id = payload.get("user_id") or "guest"

    # Select per-user index if available, otherwise default global
    local_idx, local_meta = get_user_index(user_id)
    active_retriever = retriever
    if local_idx is not None and local_meta:
        active_retriever = Retriever(local_idx, local_meta)

    start = time.perf_counter()
    docs, stats = active_retriever.retrieve(query, k=k, min_pool=20, year_from=year_from, year_to=year_to, multi_hop=multi_hop)
    synthesis = synthesize_answer(query, docs)
    latency_ms = (time.perf_counter() - start) * 1000

    # Shape sources
    sources = []
    for d in docs:
        sim = round(float(d.get("_sim", 0.0)), 3)
        t_score = trust_score(sim, bool(d.get("doi")))
        sources.append({
            "title": d.get("title", "Unknown Title"),
            "year": d.get("year", "Unknown Year"),
            "doi": d.get("doi", ""),
            "openalex_id": d.get("id"),
            "arxiv_id": d.get("arxiv_id"),
            "concepts": d.get("concepts", [])[:5],
            "why_relevant": d.get("why_relevant", ""),
            "snippet": (d.get("abstract") or d.get("summary") or "")[:900],
            "similarity": sim,
            "trust_score": t_score,
            "authors": d.get("authors", []),
            "url": d.get("url"),
        })

    similarities = [s["similarity"] for s in sources if s.get("similarity") is not None]
    metrics = {
        "latency_ms": round(latency_ms, 2),
        "fallback_used": stats.get("fallback_used", False),
        "pool_size": stats.get("candidate_counts", {}).get("pool"),
        "ranked": stats.get("candidate_counts", {}).get("scored"),
        "openalex_added": stats.get("candidate_counts", {}).get("openalex"),
        "arxiv_added": stats.get("candidate_counts", {}).get("arxiv"),
        "max_similarity": max(similarities) if similarities else None,
        "mean_similarity": round(sum(similarities) / len(similarities), 3) if similarities else None,
        "token_prompt": synthesis.get("usage", {}).get("prompt_tokens"),
        "token_completion": synthesis.get("usage", {}).get("completion_tokens"),
        "token_total": synthesis.get("usage", {}).get("total_tokens"),
    }

    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "k": k,
        "year_from": year_from,
        "year_to": year_to,
        "multi_hop": multi_hop,
        "metrics": metrics,
        "candidate_counts": stats.get("candidate_counts", {}),
        "fallback_used": stats.get("fallback_used", False),
        "hops": stats.get("hops", []),
        "sources": [
            {
                "title": s.get("title"),
                "openalex_id": s.get("openalex_id"),
                "similarity": s.get("similarity"),
                "trust_score": s.get("trust_score"),
            }
            for s in sources
        ],
    }
    log_request(log_entry)

    return {
        "answer": synthesis.get("answer", ""),
        "sources": sources,
        "fallback_used": stats.get("fallback_used", False),
        "candidate_counts": stats.get("candidate_counts", {}),
        "metrics": metrics,
        "hops": stats.get("hops", []),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
