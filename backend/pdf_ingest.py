"""
PDF ingestion and routing into per-user vector stores.
Uses PyPDF2 for text extraction (replace with Grobid/ScienceParse for production).
"""

import io
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, Form

from backend.user_store import add_user_documents

router = APIRouter()


def _simple_chunk(text: str, max_len: int = 800) -> List[str]:
    words = text.split()
    chunks = []
    buf = []
    for w in words:
        buf.append(w)
        if len(" ".join(buf)) > max_len:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _extract_text_from_bytes(data: bytes) -> str:
    try:
        import PyPDF2  # type: ignore
    except ImportError:
        return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""


@router.post("/pdf/ingest")
async def ingest_pdf(user_id: str = Form(...), file: UploadFile = File(...), max_len: Optional[int] = 800):
    data = await file.read()
    text = _extract_text_from_bytes(data)
    if not text:
        raise HTTPException(status_code=422, detail="Could not extract text; install PyPDF2 or integrate Grobid.")
    chunks = _simple_chunk(text, max_len or 800)
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(
            {
                "id": f"{file.filename}-p{i}",
                "title": file.filename,
                "page": i + 1,
                "abstract": chunk,
                "snippet": chunk[:500],
                "year": None,
                "source": file.filename,
            }
        )
    added = add_user_documents(user_id, docs)
    return {"user_id": user_id, "added": added, "chunks": len(chunks)}
