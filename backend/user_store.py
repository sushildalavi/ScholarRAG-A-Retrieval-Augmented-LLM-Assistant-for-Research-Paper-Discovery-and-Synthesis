"""
Per-user vector store management (FAISS + JSON metadata).
Stores indices under data/user_indices/{user_id}/.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from utils.embedding_utils import embed_batch_cached

BASE_DIR = Path("data/user_indices")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def _paths(user_id: str) -> Tuple[Path, Path]:
    root = BASE_DIR / user_id
    return root / "index.faiss", root / "meta.json"


def _load(user_id: str) -> Tuple[Optional[faiss.Index], List[Dict]]:
    idx_path, meta_path = _paths(user_id)
    if not idx_path.exists() or not meta_path.exists():
        return None, []
    try:
        idx = faiss.read_index(str(idx_path))
        meta = json.loads(meta_path.read_text())
        return idx, meta
    except Exception:
        return None, []


def _save(user_id: str, idx: faiss.Index, meta: List[Dict]) -> None:
    idx_path, meta_path = _paths(user_id)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(idx_path))
    meta_path.write_text(json.dumps(meta))


def add_user_documents(user_id: str, docs: List[Dict]) -> int:
    if not docs:
        return 0
    idx, meta = _load(user_id)
    texts = []
    ids = []
    for i, d in enumerate(docs):
        doc_id = str(d.get("id") or f"{len(meta)+i}")
        text = d.get("abstract") or d.get("summary") or d.get("snippet") or ""
        ids.append(doc_id)
        texts.append((doc_id, text))

    emb_map = embed_batch_cached(texts)
    if not emb_map:
        return 0
    vecs = np.vstack([emb_map[i] for i in ids if i in emb_map]).astype("float32")
    if idx is None:
        d = vecs.shape[1]
        idx = faiss.IndexFlatIP(d)
    idx.add(vecs)
    meta.extend(docs)
    _save(user_id, idx, meta)
    return len(docs)


def get_user_index(user_id: str) -> Tuple[Optional[faiss.Index], List[Dict]]:
    return _load(user_id)
