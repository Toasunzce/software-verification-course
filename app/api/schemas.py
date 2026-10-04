from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import DocumentStatus


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    status: DocumentStatus
    category: str | None
    summary: str | None
    error: str | None
    chunks_count: int
    created_at: datetime


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    document_id: int | None = None


class SearchHitOut(BaseModel):
    text: str
    document_id: int
    chunk_index: int
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: int | None = None


class AnswerOut(BaseModel):
    answer: str
    sources: list[SearchHitOut]


class SummaryOut(BaseModel):
    document_id: int
    summary: str


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1)


class ClassifyOut(BaseModel):
    category: str
    score: float