# ------------------------------
# app.py — ScholarRAG Backend API
# ------------------------------

import faiss
import json
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from utils.config import get_openai_api_key
from backend.retriever import Retriever
from backend.generator import synthesize_answer

# Initialize FastAPI app
app = FastAPI(title="ScholarRAG API", version="1.0")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Load FAISS index and metadata
# ------------------------------
INDEX_PATH = "data/scholar_index.faiss"
META_PATH = "data/metadata.json"

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

EMBED_MODEL = "text-embedding-3-large"

def get_embedding(text: str) -> np.ndarray:
    """Generate embedding vector for a query string."""
    if client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured.")
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array([response.data[0].embedding], dtype="float32")

# ------------------------------
# Endpoints
# ------------------------------

@app.get("/")
def home():
    return {"message": "ScholarRAG backend is live!"}

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

    docs, stats = retriever.retrieve(query, k=k, min_pool=20, year_from=year_from, year_to=year_to)

    answer = synthesize_answer(query, docs)

    # Shape sources
    sources = []
    for d in docs:
        sources.append({
            "title": d.get("title", "Unknown Title"),
            "year": d.get("year", "Unknown Year"),
            "doi": d.get("doi", ""),
            "openalex_id": d.get("id"),
            "concepts": d.get("concepts", [])[:5],
            "why_relevant": d.get("why_relevant", ""),
            "snippet": (d.get("abstract") or d.get("summary") or "")[:900],
            "similarity": round(float(d.get("_sim", 0.0)), 3),
        })

    return {
        "answer": answer,
        "sources": sources,
        "fallback_used": stats.get("fallback_used", False),
        "candidate_counts": stats.get("candidate_counts", {}),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
