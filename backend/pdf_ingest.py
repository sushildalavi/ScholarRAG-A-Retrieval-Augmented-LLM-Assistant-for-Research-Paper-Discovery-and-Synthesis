import hashlib
import io
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from openai import OpenAI
from PyPDF2 import PdfReader

from backend.services.db import execute, fetchall, fetchone
from backend.services.embeddings import get_embedding, get_embeddings

router = APIRouter(prefix="/documents", tags=["documents"])

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
ALLOWED_MIME_PREFIXES = ("text/",)
ALLOWED_EXACT_MIME = {"application/pdf"}


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


def _chunk_text(text: str, target_min: int = 300, target_max: int = 700, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        end = min(len(words), i + target_max)
        if end - i < target_min and end != len(words):
            end = min(len(words), i + target_min)
        chunk = " ".join(words[i:end]).strip()
        if chunk:
            chunks.append(chunk)
        if overlap > 0 and end < len(words):
            i = end - overlap
        else:
            i = end
    return chunks


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


def _embed_and_store_chunks(document_id: int, chunks: List[Tuple[int, int, str]]) -> int:
    clean_chunks: List[Tuple[int, int, str]] = []
    for page_no, chunk_idx, text in chunks:
        text = _sanitize_text(text)
        if not text.strip():
            continue
        clean_chunks.append((page_no, chunk_idx, text))

    if not clean_chunks:
        return 0

    embeddings = get_embeddings([text for _, _, text in clean_chunks])
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
            INSERT INTO chunk_embeddings (chunk_id, model, dim, vector)
            VALUES (%s, %s, %s, %s)
            """,
            [cid, "text-embedding-3-small", 1536, emb],
        )
        inserted += 1
    return inserted


@router.get("")
def list_documents():
    # Show unique titles (best-effort) ordered by recency
    docs = fetchall(
        """
        SELECT DISTINCT ON (title) id, title, status, pages, bytes, created_at
        FROM documents
        ORDER BY title, created_at DESC
        LIMIT 100
        """
    )
    return {"documents": docs}


@router.get("/latest")
def latest_documents(limit: int = 10):
    docs = fetchall(
        "SELECT id, title, status, pages, bytes, created_at FROM documents ORDER BY created_at DESC LIMIT %s",
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

    doc_row = fetchone(
        """
        INSERT INTO documents (title, source_path, mime_type, bytes, hash_sha256, status)
        VALUES (%s, %s, %s, %s, %s, 'processing')
        RETURNING id
        """,
        [title or file.filename, str(fpath), mime, len(data), sha],
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
def search_chunks(payload: dict = None, q: str = None, k: int = 10, doc_id: Optional[int] = None):
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

    qvec = get_embedding(q)
    params = [qvec]
    where = ""
    if doc_id:
        where = "WHERE chunks.document_id = %s"
        params.append(doc_id)
    rows = fetchall(
        f"""
        SELECT chunks.id, chunks.document_id, chunks.text, chunks.page_no, chunks.chunk_index,
               chunk_embeddings.vector <-> %s::vector AS distance
        FROM chunk_embeddings
        JOIN chunks ON chunk_embeddings.chunk_id = chunks.id
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
    # Best-effort cleanup of chunk embeddings and chat uploads
    execute("DELETE FROM chunk_embeddings USING chunks WHERE chunk_embeddings.chunk_id = chunks.id AND chunks.document_id=%s", [doc_id])
    execute("DELETE FROM chunks WHERE document_id=%s", [doc_id])
    execute("DELETE FROM chat_uploads WHERE doc_id=%s", [doc_id])
    execute("DELETE FROM documents WHERE id=%s", [doc_id])
    return {"ok": True, "document_id": doc_id}


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
