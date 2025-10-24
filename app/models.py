from typing import List, Optional, Literal
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

class NoteUpdate(BaseModel):
    title: Optional[str] = Field(default=None)
    body: Optional[str] = Field(default=None)
    tags: Optional[List[str]] = None

class SearchIn(BaseModel):
    q: str
    limit: int | None = None
    tags: Optional[List[str]] = None
    match: Literal["any", "all"] = "any"
    mode: Literal["vector", "hybrid"] = "hybrid" 