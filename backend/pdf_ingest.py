import hashlib
import io
import os
import re
import time
import math
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from openai import OpenAI
from PyPDF2 import PdfReader

from backend.services.db import execute, fetchall, fetchone
from backend.services.embeddings import (
    embed_documents,
    embed_query,
    get_embedding_model,
    get_embedding_version,
    get_provider,
    get_raw_embedding_dims,
)

router = APIRouter(prefix="/documents", tags=["documents"])

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
ALLOWED_MIME_PREFIXES = ("text/",)
ALLOWED_EXACT_MIME = {"application/pdf"}
DOC_TYPES = {"resume", "research_paper", "official_doc", "assignment", "notes", "other"}


def _ensure_doc_type_schema() -> None:
    execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_type TEXT DEFAULT 'other'")
    execute("ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS provider TEXT DEFAULT 'unknown'")
    execute("ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS embedding_version TEXT DEFAULT 'v1'")
    execute("ALTER TABLE embedding_cache ADD COLUMN IF NOT EXISTS provider TEXT DEFAULT 'unknown'")
    execute("ALTER TABLE embedding_cache ADD COLUMN IF NOT EXISTS model TEXT DEFAULT 'unknown'")
    execute("ALTER TABLE embedding_cache ADD COLUMN IF NOT EXISTS embedding_version TEXT DEFAULT 'v1'")
    # Backfill obvious research-paper docs for older rows.
    execute(
        """
        UPDATE documents
        SET doc_type='research_paper'
        WHERE (doc_type IS NULL OR doc_type='other')
          AND (
            lower(title) ~ '\\m(arxiv|ieee|acm|conference|journal|paper)\\M'
            OR lower(title) ~ '\\m[0-9]{4}\\.[0-9]{4,5}(v[0-9]+)?\\M'
          )
        """
    )


_ensure_doc_type_schema()


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_pdf_text(data: bytes) -> List[Tuple[int, str]]:
    """Lightweight text extractor for PDFs; OCR is a TODO."""
    reader = PdfReader(io.BytesIO(data))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def _chunk_text(text: str, target_min: int = 420, target_max: int = 620, overlap: int = 100) -> List[str]:
    """
    Structure-aware, token-window chunking with overlap and recursive fallback.

    Strategy:
    1. Preserve obvious structural units (headings / paragraphs).
    2. Build chunks in a target token window.
    3. Carry overlap across chunk boundaries.
    4. Recursively split oversized blocks by paragraph, sentence, then raw tokens.
    """
    text = _sanitize_text(text or "")
    if not text.strip():
        return []

    target_min = max(220, min(int(target_min), 520))
    target_max = max(target_min + 80, min(int(target_max), 760))
    overlap = max(40, min(int(overlap), 140))

    def token_count(value: str) -> int:
        return len(value.split())

    def is_heading(block: str) -> bool:
        stripped = block.strip()
        if not stripped:
            return False
        words = stripped.split()
        if len(words) > 18:
            return False
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", stripped):
            return True
        if stripped.endswith(":") and len(words) <= 12:
            return True
        letters = [c for c in stripped if c.isalpha()]
        if not letters:
            return False
        uppercase_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
        return uppercase_ratio > 0.72

    def split_sentences(block: str) -> List[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", block) if s.strip()]
        return sentences or [block.strip()]

    def split_token_windows(block: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        words = block.split()
        if not words:
            return []
        windows: List[str] = []
        start = 0
        while start < len(words):
            end = min(len(words), start + max_tokens)
            piece = " ".join(words[start:end]).strip()
            if piece:
                windows.append(piece)
            if end >= len(words):
                break
            start = max(start + 1, end - overlap_tokens)
        return windows

    def split_recursive(block: str) -> List[str]:
        block = block.strip()
        if not block:
            return []
        if token_count(block) <= target_max:
            return [block]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", block) if p.strip()]
        if len(paragraphs) > 1:
            out: List[str] = []
            for paragraph in paragraphs:
                out.extend(split_recursive(paragraph))
            return out

        sentences = split_sentences(block)
        if len(sentences) > 1:
            out: List[str] = []
            current: List[str] = []
            current_tokens = 0
            for sentence in sentences:
                st = token_count(sentence)
                if current and current_tokens + st > target_max:
                    out.append(" ".join(current).strip())
                    current = [sentence]
                    current_tokens = st
                else:
                    current.append(sentence)
                    current_tokens += st
            if current:
                out.append(" ".join(current).strip())
            flattened: List[str] = []
            for item in out:
                if token_count(item) > target_max:
                    flattened.extend(split_token_windows(item, target_max, overlap))
                else:
                    flattened.append(item)
            return flattened

        return split_token_windows(block, target_max, overlap)

    normalized = re.sub(r"\r\n?", "\n", text)
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n+", normalized) if b.strip()]
    structure_blocks: List[str] = []
    buffer: List[str] = []
    for block in raw_blocks:
        if is_heading(block):
            if buffer:
                structure_blocks.append("\n\n".join(buffer).strip())
                buffer = []
            structure_blocks.append(block)
        else:
            buffer.append(block)
    if buffer:
        structure_blocks.append("\n\n".join(buffer).strip())

    units: List[str] = []
    for block in structure_blocks:
        units.extend(split_recursive(block))

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def build_overlap_context(parts: List[str]) -> tuple[list[str], int]:
        kept: List[str] = []
        kept_tokens = 0
        for piece in reversed(parts):
            pt = token_count(piece)
            if kept and kept_tokens + pt > overlap:
                break
            kept.append(piece)
            kept_tokens += pt
        kept = list(reversed(kept))
        return kept, sum(token_count(p) for p in kept)

    def flush(force: bool = False) -> None:
        nonlocal current, current_tokens
        if not current:
            return
        if not force and current_tokens < target_min and chunks:
            return
        chunk = "\n\n".join(current).strip()
        if chunk:
            chunks.append(chunk)
        current, current_tokens = build_overlap_context(current) if overlap > 0 else ([], 0)

    for unit in units:
        ut = token_count(unit)
        if ut == 0:
            continue
        if current and current_tokens + ut > target_max:
            flush(force=True)
        current.append(unit)
        current_tokens += ut

    flush(force=True)
    return [chunk for chunk in chunks if chunk.strip()]


def _sanitize_text(text: str) -> str:
    """
    Remove characters that Postgres text columns reject.
    """
    return text.replace("\x00", "")


def _is_supported_upload(filename: str, mime: str) -> bool:
    ext = Path(filename.lower()).suffix
    if ext in ALLOWED_EXTENSIONS:
        return True
    if mime in ALLOWED_EXACT_MIME:
        return True
    return any(mime.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES)


def _infer_doc_type(name: str) -> str:
    n = (name or "").lower()
    if any(k in n for k in ("resume", "cv", "curriculum vitae", "bio")):
        return "resume"
    if any(k in n for k in ("assignment", "homework", "problem set", "pset")):
        return "assignment"
    if any(k in n for k in ("notes", "lecture", "slides")):
        return "notes"
    if any(k in n for k in ("policy", "report", "spec", "manual", "company profile", "official")):
        return "official_doc"
    if any(k in n for k in ("paper", "arxiv", "ieee", "acm", "journal", "conference")):
        return "research_paper"
    # arXiv-style filenames like 2602.17037v2.pdf
    if re.search(r"\b\d{4}\.\d{4,5}(v\d+)?\b", n):
        return "research_paper"
    return "other"


def _embed_and_store_chunks(document_id: int, chunks: List[Tuple[int, int, str]]) -> int:
    clean_chunks: List[Tuple[int, int, str]] = []
    for page_no, chunk_idx, text in chunks:
        text = _sanitize_text(text)
        if not text.strip():
            continue
        clean_chunks.append((page_no, chunk_idx, text))

    if not clean_chunks:
        return 0

    embeddings = embed_documents([text for _, _, text in clean_chunks])
    provider = get_provider()
    model = get_embedding_model()
    version = get_embedding_version()
    raw_dim = get_raw_embedding_dims()
    inserted = 0
    for (page_no, chunk_idx, text), emb in zip(clean_chunks, embeddings):
        row = fetchone(
            """
            INSERT INTO chunks (document_id, page_no, chunk_index, text, tokens)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            [document_id, page_no, chunk_idx, text, len(text.split())],
        )
        cid = row["id"]
        execute(
            """
            INSERT INTO chunk_embeddings (chunk_id, provider, model, embedding_version, dim, vector)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            [cid, provider, model, version, raw_dim, emb],
        )
        inserted += 1
    return inserted


@router.get("")
def list_documents():
    # Show unique titles (best-effort) ordered by recency
    docs = fetchall(
        """
        SELECT DISTINCT ON (COALESCE(hash_sha256, title)) id, title, status, doc_type, pages, bytes, created_at
        FROM documents
        ORDER BY COALESCE(hash_sha256, title), created_at DESC
        LIMIT 100
        """
    )
    return {"documents": docs}


@router.get("/latest")
def latest_documents(limit: int = 10):
    docs = fetchall(
        """
        SELECT id, title, status, doc_type, pages, bytes, created_at
        FROM (
            SELECT DISTINCT ON (COALESCE(hash_sha256, title))
                id, title, status, doc_type, pages, bytes, created_at
            FROM documents
            ORDER BY COALESCE(hash_sha256, title), created_at DESC
        ) t
        ORDER BY created_at DESC
        LIMIT %s
        """,
        [limit],
    )
    return {"documents": docs}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), title: Optional[str] = None, background_tasks: BackgroundTasks = None):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    mime = file.content_type or "application/octet-stream"
    if not _is_supported_upload(file.filename or "", mime):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload PDF, TXT, or Markdown files.",
        )

    fname = f"{int(time.time())}_{file.filename}"
    fpath = STORAGE_DIR / fname
    fpath.write_bytes(data)

    sha = _hash_bytes(data)
    existing = fetchone(
        """
        SELECT id, status
        FROM documents
        WHERE hash_sha256=%s AND status IN ('processing','ready')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [sha],
    )
    if existing:
        # Duplicate upload of the same file bytes: keep canonical doc context.
        try:
            if fpath.exists():
                fpath.unlink()
        except Exception:
            pass
        return JSONResponse(
            {
                "document_id": existing["id"],
                "status": existing.get("status") or "ready",
                "deduplicated": True,
            }
        )

    inferred_doc_type = _infer_doc_type(title or file.filename or "")
    doc_row = fetchone(
        """
        INSERT INTO documents (title, source_path, mime_type, bytes, hash_sha256, status, doc_type)
        VALUES (%s, %s, %s, %s, %s, 'processing', %s)
        RETURNING id
        """,
        [title or file.filename, str(fpath), mime, len(data), sha, inferred_doc_type],
    )
    doc_id = doc_row["id"]

    # Offload heavy parsing/embedding to background to return fast
    if background_tasks is not None:
        background_tasks.add_task(_ingest_document, doc_id, data, mime, file.filename, title)
    else:
        _ingest_document(doc_id, data, mime, file.filename, title)

    return JSONResponse({"document_id": doc_id, "status": "processing"})


def _ingest_document(doc_id: int, data: bytes, mime: str, filename: str, title: Optional[str]):
    try:
        if mime == "application/pdf" or filename.lower().endswith(".pdf"):
            pages = _extract_pdf_text(data)
        else:
            decoded = data.decode("utf-8", errors="ignore")
            pages = [(1, _sanitize_text(decoded))]

        chunk_tuples: List[Tuple[int, int, str]] = []
        for page_no, text in pages:
            text = _sanitize_text(text)
            for idx, chunk in enumerate(_chunk_text(text)):
                chunk_tuples.append((page_no, idx, chunk))

        if chunk_tuples:
            inserted = _embed_and_store_chunks(doc_id, chunk_tuples)
            if inserted > 0:
                execute("UPDATE documents SET pages=%s, status='ready' WHERE id=%s", [len(pages), doc_id])
            else:
                execute("UPDATE documents SET status='error' WHERE id=%s", [doc_id])
        else:
            execute("UPDATE documents SET status='error' WHERE id=%s", [doc_id])
    except Exception as exc:
        execute("UPDATE documents SET status='error' WHERE id=%s", [doc_id])
        raise


@router.post("/search/chunks")
def search_chunks(payload: dict = None, q: str = None, k: int = 10, doc_id: Optional[int] = None, doc_ids: Optional[list[int]] = None):
    """
    Accepts JSON body {q, k, doc_id}. Allows query param q fallback.
    """
    # Normalize payload to dict
    if payload is None:
        payload = {}
    elif isinstance(payload, str):
        payload = {"q": payload}
    elif not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    q = payload.get("q") or q or ""
    if not q:
        raise HTTPException(status_code=400, detail="q (query) is required")
    k = int(payload.get("k") or k or 10)
    doc_id = payload.get("doc_id") if payload.get("doc_id") is not None else doc_id
    raw_doc_ids = payload.get("doc_ids")
    if raw_doc_ids is not None:
        try:
            doc_ids = [int(x) for x in raw_doc_ids if x is not None]
        except Exception:
            raise HTTPException(status_code=400, detail="doc_ids must be a list of integers")

    qvec = embed_query(q)
    provider = get_provider()
    model = get_embedding_model()
    version = get_embedding_version()

    if doc_ids:
        per_doc_limit = max(1, math.ceil(int(k) / max(1, len(doc_ids))))
        rows = fetchall(
            f"""
            WITH ranked AS (
              SELECT
                chunks.id,
                chunks.document_id,
                documents.title,
                documents.doc_type,
                chunks.text,
                chunks.page_no,
                chunks.chunk_index,
                chunk_embeddings.vector <-> %s::vector AS distance,
                ROW_NUMBER() OVER (
                  PARTITION BY chunks.document_id
                  ORDER BY chunk_embeddings.vector <-> %s::vector ASC
                ) AS doc_rank
              FROM chunk_embeddings
              JOIN chunks ON chunk_embeddings.chunk_id = chunks.id
              JOIN documents ON documents.id = chunks.document_id
              WHERE documents.status = 'ready'
                AND chunk_embeddings.provider = %s
                AND chunk_embeddings.model = %s
                AND chunk_embeddings.embedding_version = %s
                AND chunks.document_id = ANY(%s)
            )
            SELECT id, document_id, title, doc_type, text, page_no, chunk_index, distance
            FROM ranked
            WHERE doc_rank <= {per_doc_limit}
            ORDER BY distance ASC
            LIMIT {int(k)}
            """,
            [qvec, qvec, provider, model, version, doc_ids],
        )
    else:
        params = [qvec]
        where_clauses = ["documents.status = 'ready'"]
        where_clauses.extend(
            [
                "chunk_embeddings.provider = %s",
                "chunk_embeddings.model = %s",
                "chunk_embeddings.embedding_version = %s",
            ]
        )
        params.extend([provider, model, version])
        if doc_id:
            where_clauses.append("chunks.document_id = %s")
            params.append(doc_id)
        where = "WHERE " + " AND ".join(where_clauses)
        rows = fetchall(
            f"""
            SELECT chunks.id, chunks.document_id, documents.title, documents.doc_type, chunks.text, chunks.page_no, chunks.chunk_index,
                   chunk_embeddings.vector <-> %s::vector AS distance
            FROM chunk_embeddings
            JOIN chunks ON chunk_embeddings.chunk_id = chunks.id
            JOIN documents ON documents.id = chunks.document_id
            {where}
            ORDER BY distance ASC
            LIMIT {int(k)}
            """,
            params,
        )
    return {"results": rows}


@router.delete("/{doc_id}")
def delete_document(doc_id: int):
    """
    Delete a document and its chunks/embeddings. Also delete from chat_uploads if present.
    """
    row = fetchone("SELECT id, hash_sha256, source_path FROM documents WHERE id=%s", [doc_id])
    if not row:
        return {"ok": True, "document_id": doc_id, "deleted_ids": []}

    hash_sha = row.get("hash_sha256")
    if hash_sha:
        rel = fetchall("SELECT id, source_path FROM documents WHERE hash_sha256=%s", [hash_sha])
    else:
        rel = [row]
    ids = [int(r.get("id")) for r in rel if r.get("id") is not None]

    for did in ids:
        execute(
            "DELETE FROM chunk_embeddings USING chunks WHERE chunk_embeddings.chunk_id = chunks.id AND chunks.document_id=%s",
            [did],
        )
        execute("DELETE FROM chunks WHERE document_id=%s", [did])
        execute("DELETE FROM chat_uploads WHERE doc_id=%s", [did])
        execute("DELETE FROM documents WHERE id=%s", [did])

    for r in rel:
        sp = r.get("source_path")
        if not sp:
            continue
        try:
            p = Path(sp)
            if p.exists():
                p.unlink()
        except Exception:
            pass

    return {"ok": True, "document_id": doc_id, "deleted_ids": ids}


@router.put("/{doc_id}/type")
def update_document_type(doc_id: int, payload: dict):
    doc_type = (payload.get("doc_type") or "").strip().lower()
    if doc_type not in DOC_TYPES:
        raise HTTPException(status_code=400, detail=f"doc_type must be one of {sorted(DOC_TYPES)}")
    execute("UPDATE documents SET doc_type=%s WHERE id=%s", [doc_type, doc_id])
    return {"ok": True, "document_id": doc_id, "doc_type": doc_type}


@router.post("/qa")
def qa_over_chunks(q: str, k: int = 8, doc_id: Optional[int] = None):
    """
    Run RAG over uploaded documents (chunk store) and answer with GPT-4o-mini.
    Returns answer and citations (chunk ids + doc ids).
    """
    res = search_chunks(q, k=k, doc_id=doc_id)["results"]
    if not res:
        return {"answer": "No relevant chunks found.", "chunks_used": []}
    context = ""
    for r in res:
        context += f"### Document {r['document_id']} - Chunk {r['id']} (page {r.get('page_no','?')})\n{r['text']}\n\n"

    client = OpenAI()
    prompt = (
        "You are a research assistant. Answer concisely and cite the document/chunk ids you used.\n\n"
        f"Question: {q}\n\nContext:\n{context}"
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = completion.choices[0].message.content
    return {"answer": answer, "chunks_used": res}
