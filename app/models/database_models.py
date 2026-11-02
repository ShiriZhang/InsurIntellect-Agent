"""SQLAlchemy ORM models aligned with existing service/API usage."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class QueryHistory(Base):
    """Persisted Q&A turns."""

    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    model_used = Column(String(128), nullable=True)  # reused as query_type
    response_time = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    cost = Column(Float, nullable=True)
    rating = Column(Integer, nullable=True)
    feedback = Column(Text, nullable=True)
    session_id = Column(String(128), nullable=True, index=True)
    rewritten_query = Column(Text, nullable=True)
    rewriting_metadata_json = Column(Text, nullable=True)
    retrieved_chunks = Column(Text, nullable=True)
    # Unknown legacy scope must never be inferred from retrieved citations.
    scope_status = Column(String(32), nullable=False, default="unknown", server_default="unknown")
    document_ids_json = Column(Text, nullable=True)
    scope_fingerprint = Column(String(64), nullable=True)
    scope_epoch = Column(Integer, nullable=True)
    answer_language = Column(String(16), nullable=True)
    query_mode = Column(String(16), nullable=True)
    index_commit_id = Column(Integer, ForeignKey("index_commits.id"), nullable=True)
    source_status = Column(String(32), nullable=False, default="unknown", server_default="unknown")
    created_time = Column(DateTime, default=datetime.utcnow, index=True)


class Document(Base):
    """Indexed insurance document (clause PDF)."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(512), nullable=False)
    original_name = Column(String(512), nullable=True)
    file_path = Column(String(1024), nullable=True)
    file_hash = Column(String(128), nullable=True, index=True)
    file_size = Column(Integer, nullable=True)
    page_count = Column(Integer, nullable=True)
    document_type = Column(String(128), nullable=True)
    product_name = Column(String(256), nullable=True)
    status = Column(String(64), default="active")
    processing_status = Column(String(32), nullable=False, default="pending", server_default="pending")
    language = Column(String(32), nullable=True)
    effective_date = Column(String(64), nullable=True)
    error_code = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    error_pages_json = Column(Text, nullable=True)
    attempt = Column(Integer, nullable=False, default=0, server_default="0")
    published_attempt = Column(Integer, nullable=True)
    published_fingerprint = Column(String(64), nullable=True)
    index_commit_id = Column(Integer, ForeignKey("index_commits.id"), nullable=True)
    deletion_status = Column(String(32), nullable=False, default="none", server_default="none")
    deletion_epoch = Column(Integer, nullable=False, default=0, server_default="0")
    deleted_time = Column(DateTime, nullable=True)
    created_time = Column(DateTime, default=datetime.utcnow)
    updated_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    # Prevent integer identity reuse even if an operator removes the highest row.
    __table_args__ = {"sqlite_autoincrement": True}

    @property
    def is_queryable(self) -> bool:
        return (self.status == "active" and self.processing_status == "ready"
                and self.deletion_status == "none" and self.published_attempt is not None
                and self.published_attempt == self.attempt
                and bool(self.published_fingerprint) and self.index_commit_id is not None)


class DocumentChunk(Base):
    """Text chunk belonging to a document (SQLite side; vectors live in Chroma)."""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, default=0)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    content_hash = Column(String(128), nullable=True)
    vector_id = Column(String(128), nullable=True)
    attempt = Column(Integer, nullable=True)  # NULL identifies unversioned legacy chunks.
    section_title = Column(Text, nullable=True)
    page_chunk_index = Column(Integer, nullable=True)
    normalized_content = Column(Text, nullable=True)
    embedding_fingerprint = Column(String(64), nullable=True)
    source_start = Column(Integer, nullable=True)
    source_end = Column(Integer, nullable=True)
    context_before = Column(Text, nullable=True)
    embedding_json = Column(Text, nullable=True)  # Unpublished vector checkpoint; never a ready index.
    created_time = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (UniqueConstraint("document_id", "attempt", "page_number",
                                       "page_chunk_index", name="uq_chunk_position"),)


class IngestionAttempt(Base):
    """Durable ingestion queue and retry cleanup checkpoint."""

    __tablename__ = "ingestion_attempts"
    document_id = Column(Integer, ForeignKey("documents.id"), primary_key=True)
    attempt = Column(Integer, primary_key=True)
    status = Column(String(32), nullable=False, default="queued")
    deletion_epoch = Column(Integer, nullable=False, default=0)
    cleanup_json = Column(Text, nullable=True)
    embedding_profile_json = Column(Text, nullable=True)
    diagnostics_json = Column(Text, nullable=True)
    expected_chunks = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_time = Column(DateTime, default=datetime.utcnow)
    updated_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ActiveContentClaim(Base):
    """One owner per content hash, including failed and deleting sources."""

    __tablename__ = "active_content_claims"
    file_hash = Column(String(128), primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, unique=True)


class IndexCommit(Base):
    """Immutable publication identity; only IndexPointer selects the live commit."""

    __tablename__ = "index_commits"
    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding_fingerprint = Column(String(64), nullable=False)
    document_attempts_json = Column(Text, nullable=False)
    vector_collection = Column(String(256), nullable=False)
    snapshot_path = Column(String(1024), nullable=False)
    created_time = Column(DateTime, default=datetime.utcnow)
    __table_args__ = {"sqlite_autoincrement": True}


class IndexPointer(Base):
    __tablename__ = "index_pointer"
    name = Column(String(32), primary_key=True, default="current")
    commit_id = Column(Integer, ForeignKey("index_commits.id"), nullable=False)


class SessionScope(Base):
    """An epoch remains distinct even when a session returns to an earlier scope."""

    __tablename__ = "session_scopes"
    session_id = Column(String(128), primary_key=True)
    epoch = Column(Integer, primary_key=True)
    scope_fingerprint = Column(String(64), nullable=False)
    document_ids_json = Column(Text, nullable=False)
    invalidated = Column(Boolean, nullable=False, default=False)
    created_time = Column(DateTime, default=datetime.utcnow)


class DeletionTask(Base):
    """Persistent tombstone and resumable cleanup progress, never source text."""

    __tablename__ = "deletion_tasks"
    document_id = Column(Integer, ForeignKey("documents.id"), primary_key=True)
    deletion_epoch = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    stage = Column(String(64), nullable=False, default="invalidate")
    completed_stages_json = Column(Text, nullable=False, default="[]")
    error_message = Column(Text, nullable=True)
    created_time = Column(DateTime, default=datetime.utcnow)
    updated_time = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DocumentMetadata(Base):
    """Structured metadata rows written by ingest."""

    __tablename__ = "document_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(128), nullable=False, unique=True, index=True)
    product_name = Column(String(256), nullable=True, index=True)
    effective_date = Column(String(64), nullable=True)
    document_type = Column(String(128), nullable=True)
    status = Column(String(64), nullable=True, default="active")
    created_time = Column(DateTime, default=datetime.utcnow)


class GraphEntity(Base):
    """Knowledge-graph entity (optional advanced path)."""

    __tablename__ = "graph_entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256), nullable=False, index=True)
    entity_type = Column(String(128), nullable=False, index=True)
    created_time = Column(DateTime, default=datetime.utcnow)


class GraphRelation(Base):
    """Knowledge-graph edge (optional advanced path)."""

    __tablename__ = "graph_relations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_entity_id = Column(Integer, ForeignKey("graph_entities.id"), nullable=False, index=True)
    target_entity_id = Column(Integer, ForeignKey("graph_entities.id"), nullable=False, index=True)
    relation_type = Column(String(128), nullable=False)
    chunk_id = Column(String(128), nullable=True, index=True)
    created_time = Column(DateTime, default=datetime.utcnow)


class SystemMetrics(Base):
    """Optional ops metrics rows for admin endpoints."""

    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cpu_usage = Column(Float, nullable=True)
    memory_usage = Column(Float, nullable=True)
    disk_usage = Column(Float, nullable=True)
    active_connections = Column(Integer, nullable=True)
    query_count = Column(Integer, nullable=True)
    error_count = Column(Integer, nullable=True)
    response_time_avg = Column(Float, nullable=True)
