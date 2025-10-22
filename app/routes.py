from fastapi import APIRouter, Body
from .db import db_diagnostics
from .embeddings import generate_embedding

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
