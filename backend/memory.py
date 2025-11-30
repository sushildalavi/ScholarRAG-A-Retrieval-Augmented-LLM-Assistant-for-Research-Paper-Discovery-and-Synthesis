"""
Lightweight user memory/history storage.

Stores queries and notes per user into JSON on disk.
Not a production DB; replace with Redis/Postgres for real use.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException

MEMORY_PATH = Path("data/user_memory.json")

router = APIRouter()


def _load() -> Dict:
    if MEMORY_PATH.exists():
        try:
            return json.loads(MEMORY_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save(data: Dict) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(data, indent=2))


@router.post("/memory/log")
def log_interaction(payload: dict):
    user_id = payload.get("user_id") or "guest"
    query = payload.get("query")
    answer = payload.get("answer")
    notes = payload.get("notes", "")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")
    data = _load()
    history: List[Dict] = data.get(user_id, [])
    history.append(
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "answer": answer,
            "notes": notes,
        }
    )
    data[user_id] = history[-200:]  # cap to last 200 entries
    _save(data)
    return {"ok": True, "count": len(data[user_id])}


@router.get("/memory/history")
def get_history(user_id: str = "guest", limit: int = 20):
    data = _load()
    history = data.get(user_id, [])
    return {"user_id": user_id, "history": history[-limit:]}
