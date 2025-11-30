"""
Utility to incrementally update the FAISS index with new documents.
Assumes metadata is a list of dicts with keys: id/title/abstract/year/doi.
"""

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from utils.embedding_utils import embed_batch_cached


INDEX_PATH = Path("data/scholar_index.faiss")
META_PATH = Path("data/metadata.json")


def load_meta() -> List[Dict]:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return []


def save_meta(meta: List[Dict]) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2))


def append_documents(new_docs: List[Dict]) -> None:
    meta = load_meta()
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        raise RuntimeError("Index not found; build a base index first.")

    # Embed abstracts/titles
    items = []
    for d in new_docs:
        doc_id = str(d.get("id") or d.get("doi") or len(meta))
        text = d.get("abstract") or d.get("summary") or d.get("title") or ""
        items.append((doc_id, text))

    emb_map = embed_batch_cached(items)
    vecs = np.vstack([emb_map[i[0]] for i in items if i[0] in emb_map])
    index.add(vecs)

    meta.extend(new_docs)
    save_meta(meta)
    faiss.write_index(index, str(INDEX_PATH))


if __name__ == "__main__":
    print("This script is intended to be imported and called with new documents.")
