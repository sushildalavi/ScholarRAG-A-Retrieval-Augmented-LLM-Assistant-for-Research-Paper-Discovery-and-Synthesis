"""
Stub agentic workflow endpoints: register scheduled digests and list them.
For production, back with a scheduler (Celery/APS) and real jobs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException

DIGEST_PATH = Path("data/digests.json")

router = APIRouter()


def _load() -> List[Dict]:
    if DIGEST_PATH.exists():
        try:
            return json.loads(DIGEST_PATH.read_text())
        except Exception:
            return []
    return []


def _save(items: List[Dict]) -> None:
    DIGEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIGEST_PATH.write_text(json.dumps(items, indent=2))


@router.post("/agents/digest")
def create_digest(payload: dict):
    """
    Register a scheduled digest (stub). Expect fields:
    - user_id
    - query or topic
    - frequency (e.g., daily, weekly)
    """
    user_id = payload.get("user_id") or "guest"
    query = payload.get("query")
    frequency = payload.get("frequency", "weekly")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")
    items = _load()
    item = {
        "id": f"digest-{len(items)+1}",
        "user_id": user_id,
        "query": query,
        "frequency": frequency,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    items.append(item)
    _save(items)
    return item


@router.get("/agents/digest")
def list_digests(user_id: str = None):
    items = _load()
    if user_id:
        items = [i for i in items if i.get("user_id") == user_id]
    return {"digests": items}
