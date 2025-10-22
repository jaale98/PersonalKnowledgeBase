from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from typing import Any, Dict, List, Optional
from .config import DATABASE_URL

def _normalize_conninfo(url: str) -> str:
    return url.replace("postgresql+psycopg", "postgresql")

pool = ConnectionPool(conninfo=DATABASE_URL, min_size=1, max_size=5)

def init_db() -> None:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()

def close_db() -> None:
    pool.close()


def db_diagnostics() -> Dict[str, Any]:
    """
    Collect a few quick facts to expose via /health:
      - postgres version
      - extensions we care about
      - schema existence
      - simple row counts
    """
    data: Dict[str, Any] = {}
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT current_database() AS db, version() AS version;")
            data.update(cur.fetchone())

            cur.execute("""
                SELECT extname
                FROM pg_extension
                WHERE extname IN ('vector', 'pg_trgm')
                ORDER BY extname;
            """)
            data["extensions"] = [r["extname"] for r in cur.fetchall()]

            cur.execute("""
                SELECT to_regclass('public.notes')      IS NOT NULL AS has_notes,
                       to_regclass('public.tags')       IS NOT NULL AS has_tags,
                       to_regclass('public.note_tags')  IS NOT NULL AS has_note_tags;
            """)
            data.update(cur.fetchone())

            if data.get("has_notes"):
                cur.execute("SELECT COUNT(*) AS notes_count FROM public.notes;")
                data["notes_count"] = cur.fetchone()["notes_count"]
            else:
                data["notes_count"] = None

            if data.get("has_tags"):
                cur.execute("SELECT COUNT(*) AS tags_count FROM public.tags;")
                data["tags_count"] = cur.fetchone()["tags_count"]
            else:
                data["tags_count"] = None

    return data

def fetchall(sql: str, params: Optional[tuple] = None) -> List[tuple]:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()

def fetchone(sql: str, params: Optional[tuple] = None) -> Optional[tuple]:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchone()

def execute(sql: str, params: Optional[tuple] = None) -> None:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            conn.commit()
