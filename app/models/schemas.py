"""API / validation schemas for InsurIntellect (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    """One retrieved clause chunk returned to API / UI."""

    chunk_id: Union[int, str] = -1
    document_id: Union[int, str] = -1
    document_name: str = "未知文档"
    content: str = ""
    page_number: Optional[int] = None
    similarity_score: float = Field(default=0.0, description="cosine 证据相关性；不表示答案正确概率")
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")


class QueryRequest(BaseModel):
    """POST /api/v1/queries/ask body."""

    question: str = Field(..., min_length=1, description="用户问题")
    query_type: str = Field(default="general", description="查询类型")
    session_id: Optional[str] = None
    stream: Optional[bool] = False
    show_sources: Optional[bool] = True

    model_config = ConfigDict(extra="ignore")


class QueryResponse(BaseModel):
    """Non-streaming ask response."""

    question: str
    answer: str
    query_type: str = "general"
    response_time: float = 0.0
    chunks_used: int = 0
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, description="RAG 主路径为最佳 cosine 证据相关性；不是正确概率")
    query_id: Optional[int] = None
    # UI state: answer | refusal | advice | llm_unavailable | degraded
    answer_kind: str = "answer"
    embedding_provider: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class QueryHistoryResponse(BaseModel):
    id: int
    question: str
    answer: str
    query_type: Optional[str] = None
    response_time: Optional[float] = None
    chunks_used: int = 0
    similarity_scores: Optional[List[float]] = None
    feedback_rating: Optional[int] = None
    feedback_comment: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")


class QueryHistoryListResponse(BaseModel):
    items: List[QueryHistoryResponse] = Field(default_factory=list)
    total_count: int = 0


class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class QueryStatistics(BaseModel):
    total_queries: int = 0
    avg_response_time: float = 0.0
    avg_rating: float = 0.0
    today_queries: int = 0
    query_type_distribution: Dict[str, int] = Field(default_factory=dict)
    period_days: int = 30
    avg_chunks_used: float = 0.0
    avg_confidence_score: float = 0.0
    feedback_stats: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: bool = True
    message: str
    status_code: int = 500
    detail: Optional[str] = None


class HealthCheck(BaseModel):
    status: str = "healthy"
    database_status: bool = False
    vector_db_status: bool = False
    llm_status: bool = False
    timestamp: Optional[datetime] = None
    version: Optional[str] = None


class SystemStats(BaseModel):
    total_documents: int = 0
    total_chunks: int = 0
    total_queries: int = 0
    avg_response_time: float = 0.0
    storage_used: int = 0
    uptime: float = 0.0


class DocumentMetadata(BaseModel):
    """Schema used by ingest Digital Archivist extraction."""

    document_title: Optional[str] = None
    product_name: Optional[str] = None
    effective_date: Optional[str] = None
    document_type: Optional[str] = None
    plan_name: Optional[str] = None
    status: Optional[str] = "active"
    chunk_id: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class QueryIntent(BaseModel):
    """Intent classification for advanced routing (default off on main path)."""

    intent: str = "general"
    metadata_filter: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class RegulatoryCheck(BaseModel):
    """Structured yes/no for regulatory relevance checks."""

    is_regulatory: bool = False
    reasoning: Optional[str] = None
    confidence: Optional[float] = None

    model_config = ConfigDict(extra="ignore")
