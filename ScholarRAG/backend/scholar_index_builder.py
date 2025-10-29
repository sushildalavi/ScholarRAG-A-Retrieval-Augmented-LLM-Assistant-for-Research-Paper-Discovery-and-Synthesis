"""
ScholarRAG Index Builder
------------------------
Fetches scholarly papers from OpenAlex, generates OpenAI embeddings,
and builds a FAISS index for semantic search.
"""

import os
import time
import json
import requests
import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI
from utils.config import get_openai_api_key

# ------------------------------
# CONFIGURATION
# ------------------------------
TOPIC = "machine learning"           # 🔹 Change to your desired research topic
FROM_DATE = "2024-01-01"             # 🔹 Start date for papers
MAX_PAPERS = 200                     # 🔹 Max number of papers to fetch
EMBED_MODEL = "text-embedding-3-large"  # 🔹 Embedding model
INDEX_FILE = "scholar_index.faiss"
META_FILE = "metadata.json"

# ------------------------------
# OPENAI CLIENT SETUP
# ------------------------------
client = OpenAI(api_key=get_openai_api_key())

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def reconstruct_abstract(inv_index):
    """Rebuild abstract text from OpenAlex's abstract_inverted_index."""
    if not inv_index:
        return ""
    words = sorted([(pos, word) for word, pos_list in inv_index.items() for pos in pos_list])
    return " ".join([w for _, w in words])


def get_openalex_papers(topic, from_date, max_papers):
    """Fetch papers from OpenAlex API."""
    print(f"🔎 Fetching up to {max_papers} papers about '{topic}' since {from_date}...")
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"title.search:{topic},from_publication_date:{from_date}",
        "per-page": 100,
        "sort": "publication_date:desc"
    }

    papers, cursor = [], "*"
    with tqdm(total=max_papers) as pbar:
        while len(papers) < max_papers:
            params["cursor"] = cursor
            resp = requests.get(url, params=params)
            if resp.status_code != 200:
                print(f"⚠️ Request failed: {resp.status_code}, retrying...")
                time.sleep(2)
                continue
            data = resp.json()
            results = data.get("results", [])
            papers.extend(results)
            pbar.update(len(results))
            cursor = data["meta"].get("next_cursor")
            if not cursor:
                break
            time.sleep(0.4)
    print(f"✅ Retrieved {len(papers)} papers")
    return papers


def get_openai_embeddings(texts, model=EMBED_MODEL):
    """Generate embeddings in batches using OpenAI API."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), 100), desc="🔢 Generating embeddings"):
        batch = texts[i:i+100]
        try:
            response = client.embeddings.create(model=model, input=batch)
            batch_embeds = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeds)
        except Exception as e:
            print(f"❌ Embedding batch {i} failed: {e}")
            time.sleep(3)
    return np.array(all_embeddings, dtype="float32")


# ------------------------------
# MAIN PIPELINE
# ------------------------------
if __name__ == "__main__":
    # 1️⃣ Fetch papers
    all_papers = get_openalex_papers(TOPIC, FROM_DATE, MAX_PAPERS)

    # 2️⃣ Extract text and metadata
    texts, meta = [], []
    for paper in all_papers:
        title = paper.get("display_name", "")
        abstract = reconstruct_abstract(paper.get("abstract_inverted_index"))
        if not abstract:
            continue
        full_text = f"{title}. {abstract}"
        texts.append(full_text)
        meta.append({
            "id": paper.get("id"),
            "title": title,
            "year": paper.get("publication_year"),
            "doi": paper.get("doi"),
            "concepts": [c["display_name"] for c in paper.get("concepts", [])]
        })

    print(f"🧩 Ready to embed {len(texts)} abstracts")

    if not texts:
        print("⚠️ No abstracts found — exiting.")
        exit()

    # 3️⃣ Generate embeddings
    embeddings = get_openai_embeddings(texts)

    # 4️⃣ Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine similarity (since normalized)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"📚 FAISS index built with {index.ntotal} vectors")

    # 5️⃣ Save index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"💾 Saved index to '{INDEX_FILE}' and metadata to '{META_FILE}'")

    # 6️⃣ Quick retrieval test
    query = "transformer architectures"
    print(f"\n🔍 Testing retrieval for query: '{query}'")
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    q_emb = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k=5)

    print("\n📄 Top 5 relevant papers:\n")
    for rank, idx in enumerate(I[0]):
        print(f"{rank+1}. {meta[idx]['title']} ({meta[idx]['year']})")
        print(f"   DOI: {meta[idx]['doi']}")
        print(f"   Similarity: {D[0][rank]:.3f}\n")
