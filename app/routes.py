from fastapi import APIRouter, Body, HTTPException, Query
from typing import List
from .config import ENABLE_FTS, FUSION_ALPHA
from .db import (
    db_diagnostics,
    insert_note_with_embedding,
    upsert_tags_get_ids,
    link_note_tags,
    get_note_with_tags,
    list_notes_with_tags,
    execute, fetchone,
    search_notes_by_vector,
    search_notes_by_vector_filtered,
    update_note_and_embedding,
    replace_note_tags,
    search_notes_hybrid_filtered
)
from .embeddings import generate_embedding
from .models import NoteCreate, NoteOut, SearchIn, NoteUpdate

router = APIRouter()

@router.get("/health", include_in_schema=False)
def health():
    try:
        info = db_diagnostics()
        return {
            "status": "ok",
            "db": {
                "database": info.get("db"),
                "version": info.get("version"),
                "extensions": info.get("extensions", []),
                "tables": {
                    "notes": bool(info.get("has_notes")),
                    "tags": bool(info.get("has_tags")),
                    "note_tags": bool(info.get("has_note_tags")),
                },
                "counts": {
                    "notes": info.get("notes_count"),
                    "tags": info.get("tags_count"),
                },
            },
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
    
@router.post("/embed-test")
def embed_test(text: str = Body(..., embed=True)):
    """
    Temporary test route to verify embedding API connectivity.
    Returns first 10 dimensions only.
    """
    try:
        vec = generate_embedding(text)
        return {
            "status": "ok",
            "dims": len(vec),
            "sample": vec[:10]
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
    
@router.post("/notes", response_model=NoteOut)
def create_note(payload: NoteCreate):
    tag_ids = upsert_tags_get_ids(payload.tags)

    to_embed = f"{payload.title}\n\n{payload.body}"
    vec = generate_embedding(to_embed)

    note_id = insert_note_with_embedding(payload.title, payload.body, vec)

    link_note_tags(note_id, tag_ids)

    row = get_note_with_tags(note_id)
    if not row:
        raise HTTPException(status_code = 500, detail="Failed to load newly created note.")
    
    return NoteOut(
        id=row[0],
        title=row[1],
        body=row[2],
        created_at=row[3],
        updated_at=row[4],
        tags=row[5] or [],
    )

@router.get("/notes", response_model=List[NoteOut])
def list_notes(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)):
    rows = list_notes_with_tags(limit=limit, offset=offset)
    out: List[NoteOut] = []
    for r in rows:
        out.append(NoteOut(
            id=r[0],
            title=r[1],
            body=r[2],
            created_at=r[3],
            updated_at=r[4],
            tags=r[5] or [],
        ))
    return out

@router.get("/notes/{note_id}", response_model=NoteOut)
def get_note(note_id: int):
    row = get_note_with_tags(note_id)
    if not row:
        raise HTTPException(status_code=404, detail="Note not found")
    return NoteOut(
        id=row[0],
        title=row[1],
        body=row[2],
        created_at=row[3],
        updated_at=row[4],
        tags=row[5] or [],
    )

@router.delete("/notes/{note_id}")
def delete_note(note_id: int):
    res = fetchone("DELETE FROM notes WHERE id = %s RETURNING id;", (note_id,))
    if not res:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"status": "ok", "deleted": note_id}


@router.post("/search", response_model=List[NoteOut])
def search(payload: SearchIn):
    q = (payload.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query text 'q' is required")

    vec = generate_embedding(q)

    use_hybrid = (getattr(payload, "mode", "hybrid") == "hybrid") and ENABLE_FTS

    if use_hybrid:
        rows = search_notes_hybrid_filtered(
            query_text=q,
            query_vec=vec,
            limit=payload.limit,
            tags=(payload.tags or None),
            match=payload.match,
            alpha=FUSION_ALPHA,
        )
    else:
        rows = search_notes_by_vector_filtered(
            query_vec=vec,
            limit=payload.limit,
            tags=(payload.tags or None),
            match=payload.match,
        )

    out: List[NoteOut] = []
    for r in rows:
        score_idx = 8 if (use_hybrid and len(r) > 8) else 7
        score_val = float(r[score_idx]) if (len(r) > score_idx and r[score_idx] is not None) else None
        out.append(NoteOut(
            id=r[0], title=r[1], body=r[2],
            created_at=r[3], updated_at=r[4],
            tags=r[5] or [],
            score=score_val,
        ))
    return out


@router.put("/notes/{note_id}", response_model=NoteOut)
def update_note(note_id: int, payload: NoteUpdate):
    row = get_note_with_tags(note_id)
    if not row:
        raise HTTPException(status_code=404, detail="Note not found")
    cur_title, cur_body = row[1], row[2]

    new_title = (payload.title if payload.title is not None else cur_title).strip()
    new_body  = (payload.body  if payload.body  is not None else cur_body).strip()
    if not new_title or not new_body:
        raise HTTPException(status_code=400, detail="Title and body cannot be empty")

    to_embed = f"{new_title}\n\n{new_body}"
    vec = generate_embedding(to_embed)

    ok = update_note_and_embedding(note_id, new_title, new_body, vec)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to update note")

    if payload.tags is not None:
        tag_ids = upsert_tags_get_ids(payload.tags)
        replace_note_tags(note_id, tag_ids)

    row = get_note_with_tags(note_id)
    return NoteOut(
        id=row[0], title=row[1], body=row[2],
        created_at=row[3], updated_at=row[4], tags=row[5] or []
    )