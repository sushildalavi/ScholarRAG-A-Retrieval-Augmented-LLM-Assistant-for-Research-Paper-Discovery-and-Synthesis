import hashlib
import io
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader

from db import execute, fetchall, fetchone
from embeddings import get_embedding

router = APIRouter(prefix="/documents", tags=["documents"])

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


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


def _embed_and_store_chunks(document_id: int, chunks: List[Tuple[int, int, str]]) -> None:
    for page_no, chunk_idx, text in chunks:
        emb = get_embedding(text)
        execute(
            """
            INSERT INTO chunks (document_id, page_no, chunk_index, text, tokens)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            [document_id, page_no, chunk_idx, text, len(text.split())],
        )
        row = fetchone("SELECT currval(pg_get_serial_sequence('chunks','id')) as cid")
        cid = row["cid"]
        execute(
            """
            INSERT INTO chunk_embeddings (chunk_id, model, dim, vector)
            VALUES (%s, %s, %s, %s)
            """,
            [cid, "text-embedding-3-small", 1536, emb],
        )


@router.get("")
def list_documents():
    docs = fetchall("SELECT id, title, status, pages, bytes, created_at FROM documents ORDER BY created_at DESC LIMIT 100")
    return {"documents": docs}


@router.get("/latest")
def latest_documents(limit: int = 10):
    docs = fetchall(
        "SELECT id, title, status, pages, bytes, created_at FROM documents ORDER BY created_at DESC LIMIT %s",
        [limit],
    )
    return {"documents": docs}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), title: Optional[str] = None):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    fname = f"{int(time.time())}_{file.filename}"
    fpath = STORAGE_DIR / fname
    fpath.write_bytes(data)

    sha = _hash_bytes(data)
    mime = file.content_type or "application/octet-stream"

    execute(
        """
        INSERT INTO documents (title, source_path, mime_type, bytes, hash_sha256, status)
        VALUES (%s, %s, %s, %s, %s, 'processing')
        """,
        [title or file.filename, str(fpath), mime, len(data), sha],
    )
    doc_row = fetchone("SELECT currval(pg_get_serial_sequence('documents','id')) as did")
    doc_id = doc_row["did"]

    pages = []
    if mime == "application/pdf" or file.filename.lower().endswith(".pdf"):
        pages = _extract_pdf_text(data)
    else:
        pages = [(1, data.decode(errors="ignore"))]

    chunk_tuples: List[Tuple[int, int, str]] = []
    for page_no, text in pages:
        for idx, chunk in enumerate(_chunk_text(text)):
            chunk_tuples.append((page_no, idx, chunk))

    if chunk_tuples:
        _embed_and_store_chunks(doc_id, chunk_tuples)
        execute("UPDATE documents SET pages=%s, status='ready' WHERE id=%s", [len(pages), doc_id])
    else:
        execute("UPDATE documents SET status='error' WHERE id=%s", [doc_id])

    return JSONResponse({"document_id": doc_id, "pages": len(pages), "chunks": len(chunk_tuples)})


@router.post("/search/chunks")
def search_chunks(q: str, k: int = 10, doc_id: Optional[int] = None):
    qvec = get_embedding(q)
    params = [qvec]
    where = ""
    if doc_id:
        where = "WHERE chunks.document_id = %s"
        params.append(doc_id)
    rows = fetchall(
        f"""
        SELECT chunks.id, chunks.document_id, chunks.text, chunks.page_no, chunks.chunk_index,
               chunk_embeddings.vector <-> %s AS distance
        FROM chunk_embeddings
        JOIN chunks ON chunk_embeddings.chunk_id = chunks.id
        {where}
        ORDER BY distance ASC
        LIMIT {int(k)}
        """,
        params,
    )
    return {"results": rows}


@router.post("/qa")
def qa_over_chunks(q: str, k: int = 8, doc_id: Optional[int] = None):
    res = search_chunks(q, k=k, doc_id=doc_id)["results"]
    context = "\n\n".join([r["text"] for r in res])
    answer = f"(stub) Top chunks used: {len(res)}. Use this context to answer:\n{context[:1000]}"
    return {"answer": answer, "chunks_used": res}
