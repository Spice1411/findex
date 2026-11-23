"""Pydantic v2 boundary schemas for the HTTP API (internal dataclasses stay in findex.models)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field

from findex.models import SearchResult


class SearchHit(BaseModel):
    """Single ranked search result serialized at the API boundary."""

    doc_id: int
    title: str
    score: float = Field(..., description="Relevance score from the selected scorer.")
    snippet: str = Field(default="", description="Highlighted ±80-char context window.")

    @classmethod
    def from_domain(cls, item: SearchResult) -> Self:
        return cls(
            doc_id=item.doc_id,
            title=item.title,
            score=round(item.score, 6),
            snippet=item.snippet,
        )


class SearchResponse(BaseModel):
    """Paginated search response returned by ``GET /search``."""

    query: str
    scorer: Literal["bm25", "tfidf"]
    page: int = Field(..., ge=1)
    k: int = Field(..., ge=1, le=100)
    total: int = Field(..., ge=0)
    total_pages: int = Field(..., ge=0)
    took_ms: float = Field(..., ge=0.0)
    results: list[SearchHit]


class DocumentResponse(BaseModel):
    """Full document payload returned by ``GET /docs/{doc_id}``."""

    doc_id: int
    source_id: str
    title: str
    token_count: int
    text: str


class StatsResponse(BaseModel):
    """Index and service runtime statistics returned by ``GET /stats``."""

    num_docs: int
    num_terms: int
    total_tokens: int
    avg_doc_length: float
    has_positions: bool
    index_size_bytes: int
    uptime_seconds: float


class HealthResponse(BaseModel):
    """Health-check payload returned by ``GET /health``."""

    status: Literal["ok", "unhealthy"]
    num_docs: int = 0
    uptime_seconds: float = 0.0
    detail: str | None = None
