"""API / validation schemas for InsurIntellect (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievedChunk(BaseModel):
    """One retrieved clause chunk returned to API / UI."""

    source_status: str = "active"
    chunk_id: Union[int, str] = -1
    document_id: Union[int, str] = -1
    document_name: str = "未知文档"
    content: str = ""
    page_number: Optional[int] = None
    similarity_score: float = Field(default=0.0, description="cosine 证据相关性；不表示答案正确概率")
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")


class DocumentResponse(BaseModel):
    document_id: int
    name: str
    processing_status: str
    deletion_status: str
    deletion_stage: Optional[str] = None
    deletion_error: Optional[str] = None
    deletion_task_status: Optional[str] = None
    ready: bool
    page_count: Optional[int] = None
    language: Optional[str] = None
    file_size: Optional[int] = None
    attempt: int
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_pages: List[int] = Field(default_factory=list)
    file_url: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    upload_max_bytes: int
    max_pages: int
    design_capacity: int
    capacity_warning: Optional[str] = None


class QueryRequest(BaseModel):
    """POST /api/v1/queries/ask body."""

    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    document_ids: List[Annotated[int, Field(strict=True, gt=0)]] = Field(min_length=1)
    query_mode: Literal["auto", "qa", "compare"] = "auto"
    answer_language: Literal["auto", "zh", "en"] = "auto"
    query_type: str = Field(default="general", description="查询类型")
    session_id: Optional[str] = None
    scope_epoch: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    stream: Optional[bool] = False
    show_sources: Optional[bool] = True

    model_config = ConfigDict(extra="ignore")

    @field_validator("document_ids")
    @classmethod
    def normalize_scope(cls, values):
        return sorted(set(values))


class SessionSelectionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    document_ids: List[Annotated[int, Field(strict=True, gt=0)]]

    @field_validator("document_ids")
    @classmethod
    def normalize_scope(cls, values):
        return sorted(set(values))


class BatchQueryRequest(BaseModel):
    questions: List[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(min_length=1, max_length=10)
    document_ids: List[Annotated[int, Field(strict=True, gt=0)]] = Field(min_length=1)
    query_type: str = "general"
    query_mode: Literal["auto", "qa", "compare"] = "auto"
    answer_language: Literal["auto", "zh", "en"] = "auto"

    def queries(self):
        shared = self.model_dump(exclude={"questions"})
        return [QueryRequest(question=q, **shared) for q in self.questions]


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
    session_id: Optional[str] = None
    scope_epoch: Optional[int] = None
    scope_fingerprint: Optional[str] = None
    answer_language: str = "zh"
    index_commit_id: Optional[int] = None
    source_status: str = "active"
    document_ids: List[int] = Field(default_factory=list)
    query_mode: str = "qa"
    document_coverage: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class QueryHistoryResponse(BaseModel):
    document_ids: Optional[List[int]] = None
    session_id: Optional[str] = None
    scope_epoch: Optional[int] = None
    scope_fingerprint: Optional[str] = None
    answer_language: Optional[str] = None
    query_mode: Optional[str] = None
    index_commit_id: Optional[int] = None
    source_status: str = "unknown"
    answer_kind: Optional[str] = None
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
