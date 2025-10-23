from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from typing import Any, Dict, List, Optional, Iterable, Tuple
from .config import DATABASE_URL, RESULT_LIMIT_DEFAULT, ANN_PROBES

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

def _norm_tag(name: str) -> str:
    return name.strip().lower()

def upsert_tag_get_id(name: str) -> int:
    """Upsert a single tag by name and return its id."""
    name = _norm_tag(name)
    sql = """
    INSERT INTO tags (name)
    VALUES (%s)
    ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
    RETURNING id;
    """
    row = fetchone(sql, (name,))
    return row[0]

def upsert_tags_get_ids(names: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for n in names:
        if not n or not n.strip():
            continue
        ids.append(upsert_tag_get_id(n))
    return ids

def _vector_literal(vec: Iterable[float]) -> str:
    """Format a Python list of floats into a pgvector literal string."""
    return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"

def insert_note_with_embedding(title: str, body: str, embedding: list[float]) -> int:
    """Insert a note and its embedding. Returns new note id."""
    vec_str = _vector_literal(embedding)
    sql = """
    INSERT INTO notes (title, body, embedding)
    VALUES (%s, %s, %s::vector)
    RETURNING id;
    """
    row = fetchone(sql, (title, body, vec_str))
    return row[0]

def link_note_tags(note_id: int, tag_ids: Iterable[int]) -> None:
    vals: list[Tuple[int, int]] = [(note_id, tid) for tid in tag_ids]
    if not vals:
        return
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO note_tags (note_id, tag_id) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                vals,
            )
            conn.commit()

def get_note_with_tags(note_id: int):
    """Return a single note with aggregated tags."""
    sql = """
    SELECT
      n.id, n.title, n.body,
      to_char(n.created_at, 'YYYY-MM-DD"T"HH24:MI:SSOF') AS created_at,
      to_char(n.updated_at, 'YYYY-MM-DD"T"HH24:MI:SSOF') AS updated_at,
      COALESCE(json_agg(t.name) FILTER (WHERE t.id IS NOT NULL), '[]') AS tags
    FROM notes n
    LEFT JOIN note_tags nt ON nt.note_id = n.id
    LEFT JOIN tags t ON t.id = nt.tag_id
    WHERE n.id = %s
    GROUP BY n.id;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (note_id,))
            return cur.fetchone()

def list_notes_with_tags(limit: int = RESULT_LIMIT_DEFAULT, offset: int = 0):
    """List notes (newest first) with aggregated tags."""
    sql = """
    SELECT
      n.id, n.title, n.body,
      to_char(n.created_at, 'YYYY-MM-DD"T"HH24:MI:SSOF') AS created_at,
      to_char(n.updated_at, 'YYYY-MM-DD"T"HH24:MI:SSOF') AS updated_at,
      COALESCE(json_agg(t.name) FILTER (WHERE t.id IS NOT NULL), '[]') AS tags
    FROM notes n
    LEFT JOIN note_tags nt ON nt.note_id = n.id
    LEFT JOIN tags t ON t.id = nt.tag_id
    GROUP BY n.id
    ORDER BY n.created_at DESC
    LIMIT %s OFFSET %s;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit, offset))
            return cur.fetchall()

def search_notes_by_vector(query_vec: list[float], limit: int | None = None):
    """Cosine similarity search over notes.embedding via pgvector."""
    vec_str = _vector_literal(query_vec)
    lim = limit or RESULT_LIMIT_DEFAULT
    sql = """
    WITH q AS (SELECT %s::vector AS v)
    SELECT
      n.id,
      n.title,
      n.body,
      to_char(n.created_at, 'YYYY-MM-DD"T"HH24:MI:SSOF') AS created_at,
      to_char(n.updated_at, 'YYYY-MM-DD"T"HH24:MI:SSOF') AS updated_at,
      COALESCE(json_agg(t.name) FILTER (WHERE t.id IS NOT NULL), '[]') AS tags,
      dist,
      1 - dist AS score
    FROM (
      SELECT
        n.*,
        (n.embedding <=> (SELECT v FROM q)) AS dist
      FROM notes n
      WHERE n.embedding IS NOT NULL
    ) AS n
    LEFT JOIN note_tags nt ON nt.note_id = n.id
    LEFT JOIN tags t ON t.id = nt.tag_id
    GROUP BY n.id, n.title, n.body, n.created_at, n.updated_at, dist
    ORDER BY dist ASC
    LIMIT %s;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT set_config('ivfflat.probes', %s, true);", (str(ANN_PROBES),))
            cur.execute(sql, (vec_str, lim))
            return cur.fetchall()