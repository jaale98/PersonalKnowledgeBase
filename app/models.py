from typing import List, Optional
from pydantic import BaseModel, Field

class NoteCreate(BaseModel):
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    tags: List[str] = []

class NoteOut(BaseModel):
    id: int
    title: str
    body: str
    tags: List[str]
    created_at: str
    updated_at: str
    score: Optional[float] = None

class NoteListParams(BaseModel):
    limit: int = 20
    offset: int = 0

class SearchIn(BaseModel):
    q: str
    limit: int | None = None