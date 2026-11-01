"""Response-shaping helpers for stable public query schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.models.schemas import RetrievedChunk
from app.services.evidence_scores import chunk_evidence_score


def _coerce_id(value: Any) -> Any:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            return int(text)
        except Exception:
            return text
    if value is not None:
        return str(value)
    return -1


def _coerce_page(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except Exception:
            return None
    return None


def to_retrieved_chunks(chunks: List[Dict[str, Any]]) -> List[RetrievedChunk]:
    """Normalize raw retrieval dicts into public RetrievedChunk models."""
    normalized: List[RetrievedChunk] = []
    for chunk in chunks or []:
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        chunk_id = _coerce_id(
            chunk.get("chunk_id") if chunk.get("chunk_id") is not None else metadata.get("chunk_id")
        )
        document_id = _coerce_id(
            chunk.get("document_id")
            if chunk.get("document_id") is not None
            else metadata.get("document_id")
        )
        document_name = (
            chunk.get("display_name")
            or chunk.get("document_name")
            or metadata.get("display_name")
            or metadata.get("document_title")
            or metadata.get("filename")
            or "未知文档"
        )

        content = chunk.get("content") if chunk.get("content") is not None else ""
        if not isinstance(content, str):
            try:
                content = str(content)
            except Exception:
                content = ""

        page_raw = chunk.get("page_number")
        if page_raw is None:
            page_raw = metadata.get("page_number")
        page_number = _coerce_page(page_raw)

        similarity = chunk_evidence_score(chunk)

        public_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        if chunk.get("display_name") and "display_name" not in public_metadata:
            public_metadata["display_name"] = chunk.get("display_name")
        if not public_metadata:
            public_metadata = None

        normalized.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                document_name=document_name,
                content=content,
                page_number=page_number,
                similarity_score=similarity,
                metadata=public_metadata,
            )
        )
    return normalized
