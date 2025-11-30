import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from google.oauth2 import id_token
from google.auth.transport import requests

router = APIRouter()


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    load_dotenv(override=False)


@router.post("/auth/google")
def auth_google(payload: dict):
    _load_dotenv_if_available()
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    token = payload.get("id_token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing id_token")
    if not client_id:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID not configured")
    try:
        info = id_token.verify_oauth2_token(token, requests.Request(), client_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

    # Minimal session payload; replace with DB lookup + JWT issuance if desired.
    return {
        "user_id": info.get("sub"),
        "email": info.get("email"),
        "name": info.get("name"),
        "picture": info.get("picture"),
    }
