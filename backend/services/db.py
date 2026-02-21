import os
from typing import Any, Iterable, Optional

import psycopg2


def get_connection():
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
