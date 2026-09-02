from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    id: str
    filename: str
    page_count: int
    chunk_count: int
    created_at: datetime


class Citation(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_id: str
    excerpt: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[str] | None = None


class AskResponse(BaseModel):
    answer: str
    status: Literal["answered", "abstained", "blocked"]
    confidence: float
    citations: list[Citation] = []
    trace_id: str
    latency_ms: int


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: Literal["helpful", "not_helpful"]
    comment: str = Field(default="", max_length=1000)

