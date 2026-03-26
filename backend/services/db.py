import os
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import psycopg2
from psycopg2.extensions import connection as PGConnection
from utils.config import _load_dotenv_if_available


def _database_connection_hint(database_url: str, exc: psycopg2.OperationalError) -> str:
    message = str(exc)
    try:
        parsed = urlparse(database_url)
    except Exception:
        parsed = None

    host = (parsed.hostname or "").lower() if parsed else ""
    if "pooler.supabase.com" in host and "tenant or user not found" in message.lower():
        return (
            "Supabase rejected DATABASE_URL with `Tenant or user not found`. "
            "For local development, set DATABASE_URL to "
            "`postgresql://scholarrag:scholarrag@127.0.0.1:5432/scholarrag_db` and run "
            "`docker compose up -d db`. "
            "If you intended to use Supabase, copy the exact Postgres connection string from "
            "Supabase Dashboard > Connect and verify the username, password, host, and port."
        )
    return f"DATABASE_URL connection failed: {message}"


def get_connection() -> PGConnection:
    _load_dotenv_if_available()
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            return psycopg2.connect(database_url)
        except psycopg2.OperationalError as exc:
            raise RuntimeError(_database_connection_hint(database_url, exc)) from exc

    return psycopg2.connect(
        host=os.getenv("PGHOST", "127.0.0.1"),
        port=int(os.getenv("PGPORT", "5432")),
        user=os.getenv("PGUSER", "scholarrag"),
        password=os.getenv("PGPASSWORD", "scholarrag"),
        dbname=os.getenv("PGDATABASE", "scholarrag_db"),
    )


def execute(query: str, params: Optional[Iterable[Any]] = None) -> None:
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, params or [])
    finally:
        conn.close()


def fetchall(query: str, params: Optional[Iterable[Any]] = None):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, params or [])
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def fetchone(query: str, params: Optional[Iterable[Any]] = None):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, params or [])
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [desc[0] for desc in cur.description]
                return dict(zip(cols, row))
    finally:
        conn.close()
