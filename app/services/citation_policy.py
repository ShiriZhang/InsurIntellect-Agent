"""Citation curation and public evidence policy for clause-grounded answers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from app.services.evidence_scores import chunk_evidence_score, normalize_evidence_score, finite_number
from app.services.retrieval_text import tokenize_retrieval

WEAK_META_MARKERS = (
    "文档名称：",
    "产品名称：",
    "文档类型：",
    "生效日期：",
    "状态：演示样本",
    "状态：演示",
)

# 条款标识：多字短语直接 substring 命中即成立；
# 单字 "第"/"条" 在中文常见文本里太多（第一道菜/面条/条件），
# 单独 substring 会误判离题文本为"是条款"→ 过滤闸失守 → 离题带引用。
# 故从 LITERAL 里删除单字，改由 _has_clause_marker 用 regex 匹配"第N条"类格式。
CLAUSE_MARKERS = (
    "等待期",
    "犹豫期",
    "责任免除",
    "免赔",
    "保险责任",
    "保险金",
    "本合同",
    "投保人",
    "被保险人",
    "理赔",
    "除外",
)

# 匹配 "第三条" / "第十三条" / "第 3 条" 等条款序号格式（含全角致密空格）
_CLAUSE_NUM_RE = re.compile(r"第\s*[一二三四五六七八九十百零\d]{1,4}\s*条")


def _has_clause_marker(text: str) -> bool:
    """是否含条款标识：多字短语 substring 命中，或 "第N条" regex 命中。"""
    if not text:
        return False
    if any(marker in text for marker in CLAUSE_MARKERS):
        return True
    return bool(_CLAUSE_NUM_RE.search(text))


def chunk_text(chunk: Dict[str, Any]) -> str:
    content = chunk.get("content")
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def is_weak_citation_chunk(chunk: Dict[str, Any]) -> bool:
    """Return True for cover/metadata-only snippets that should not lead citations."""
    text = chunk_text(chunk).strip()
    if not text:
        return True
    has_clause = _has_clause_marker(text)
    meta_hits = sum(1 for marker in WEAK_META_MARKERS if marker in text)
    # clause_hits 仅统计多字 LITERAL 命中 + regex 命中（合计 0/1 的阈值语义不变）
    clause_hits = sum(1 for marker in CLAUSE_MARKERS if marker in text) + (1 if has_clause else 0)
    if meta_hits >= 2 and clause_hits <= 1 and len(text) < 280:
        return True
    if meta_hits >= 3 and clause_hits == 0:
        return True
    return False


def doc_page_key(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    name = (
        chunk.get("document_name")
        or metadata.get("document_title")
        or metadata.get("filename")
        or metadata.get("source")
        or "unknown"
    )
    page = chunk.get("page_number")
    if page is None:
        page = metadata.get("page_number")
    return f"{str(name).strip().lower()}|{page}"


def relevance_bonus(question: str, chunk: Dict[str, Any]) -> float:
    question_tokens = set(tokenize_retrieval(question))
    text = chunk_text(chunk)
    text_tokens = set(tokenize_retrieval(text))
    # Lexical overlap can break ranking ties; it is never used by the evidence gate.
    bonus = 0.05 * min(3, len(question_tokens & text_tokens))
    if _has_clause_marker(text):
        bonus += 0.05
    if is_weak_citation_chunk(chunk):
        bonus -= 0.5
    return bonus


def normalize_sim(score: Any) -> float:
    return normalize_evidence_score(score)


def chunk_gate_score(chunk: Dict[str, Any]) -> float:
    return chunk_evidence_score(chunk)


def eligible_evidence(chunks, threshold):
    """Only independently supported chunks may enter generation, at any lexical rank."""
    return [c for c in chunks or [] if chunk_gate_score(c) > 0
            and chunk_gate_score(c) >= threshold]


def best_similarity(chunks: List[Dict[str, Any]]) -> float:
    best = 0.0
    for chunk in chunks or []:
        try:
            best = max(best, chunk_gate_score(chunk))
        except Exception:
            continue
    return best


def curate_citations(
    chunks: List[Dict[str, Any]],
    question: str = "",
    *,
    limit: int = 4,
) -> List[Dict[str, Any]]:
    """Dedupe + drop weak cover chunks + keep top-N for answer [1]..[N]."""
    if not chunks:
        return []
    limit = max(1, int(limit or 4))
    selected, seen_ids, seen_text = [], set(), set()
    for raw in chunks:
        if not isinstance(raw, dict) or is_weak_citation_chunk(raw):
            continue
        chunk = dict(raw)
        metadata = chunk.get("metadata") or {}
        document = chunk.get("document_id", metadata.get("document_id"))
        # Legacy chunks without document IDs are isolated by their displayed name.
        if document is None:
            document = chunk.get("document_name") or metadata.get("filename")
        attempt = metadata.get("attempt")
        identity = chunk.get("chunk_id", metadata.get("chunk_id"))
        identity_key = (document, attempt, identity)
        text_key = (document, attempt, chunk.get("page_number", metadata.get("page_number")), chunk_text(chunk).strip())
        if (identity is not None and identity_key in seen_ids) or text_key in seen_text:
            continue
        seen_ids.add(identity_key)
        seen_text.add(text_key)
        chunk.setdefault("document_name", metadata.get("document_title") or metadata.get("filename") or "未知文档")
        if chunk.get("page_number") is None:
            chunk["page_number"] = metadata.get("page_number")
        selected.append(chunk)
        if len(selected) >= limit:
            break
    return selected


def public_citations(
    chunks: List[Dict[str, Any]],
    *,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Keep only chunks with a real similarity for UI/API honesty."""
    public: List[Dict[str, Any]] = []
    for chunk in chunks or []:
        if not isinstance(chunk, dict):
            continue
        score = chunk_gate_score(chunk)
        if score <= float(min_score or 0.0):
            continue
        if is_weak_citation_chunk(chunk):
            continue
        item = dict(chunk)
        item["similarity_score"] = round(max(0.0, min(1.0, score)), 4)
        public.append(item)
    return public


def citations_for_kind(
    kind: str,
    chunks: List[Dict[str, Any]],
    *,
    min_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Policy: refusal/advice/degraded -> no public sources; answer keeps scored ones."""
    normalized_kind = (kind or "answer").lower()
    if normalized_kind in ("refusal", "advice", "degraded", "insufficient_evidence"):
        return []
    threshold = min_score
    if threshold is None or float(threshold) <= 0.0:
        threshold = 0.05
    return public_citations(chunks, min_score=float(threshold))
