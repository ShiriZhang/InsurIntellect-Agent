"""
查询服务
整合向量搜索和LLM生成,提供完整的问答功能
"""

from datetime import datetime
import asyncio
from typing import List, Dict, Any, Optional
import json
from functools import lru_cache
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.app_logging import get_logger
from app.models.database_models import QueryHistory
from app.models.schemas import QueryRequest, QueryResponse, RetrievedChunk, QueryStatistics
from app.core.rag_workflow import get_agent

from app.services import citation_policy, refusal_policy, response_shaping, query_history_service

logger = get_logger(__name__)


class QueryService:
    """查询服务"""
    
    def __init__(self):
        """初始化查询服务,使用InsurIntellectAgent RAG工作流"""
        # 延迟初始化agent,避免在服务启动时就创建
        self._agent = None
        from app.services.scoped_query import ScopedQueryProcessor
        self.scoped = ScopedQueryProcessor()
        # 保留对后台 query 历史保存任务的强引用，防 GC 静默回收 task
        # （裸 asyncio.create_task 未持引用则可能被 GC，Python 官方文档明确警告）
        self._bg_tasks: set = set()

    def _spawn_bg(self, coro):
        """启动后台任务并持有强引用；任务结束后自动从集合退出。"""
        try:
            task = asyncio.create_task(coro)
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
            return task
        except Exception as exc:
            # create_task 失败（如无 running loop）时手动关闭 coro 防
            # "coroutine was never awaited" 警告与资源泄漏
            logger.debug(f"后台任务启动失败：{exc}")
            try:
                coro.close()
            except Exception:
                pass
            return None

    @property
    def agent(self):
        """获取agent实例,使用延迟初始化"""
        if self._agent is None:
            logger.info("初始化agent实例...")
            try:
                self._agent = get_agent()
                logger.info(f"Agent实例初始化成功: {type(self._agent)}")
            except Exception as e:
                logger.error(f"Agent实例初始化失败: {e}", exc_info=True)
                raise
        return self._agent

    @property
    def embedding_service(self):
        """Expose agent embedding service for provider metadata on responses."""
        try:
            return getattr(self.agent, "embedding_service", None)
        except Exception:
            return None

    def _chunk_text(self, c: Dict[str, Any]) -> str:
        return citation_policy.chunk_text(c)

    def _is_weak_citation_chunk(self, c: Dict[str, Any]) -> bool:
        return citation_policy.is_weak_citation_chunk(c)

    def _doc_page_key(self, c: Dict[str, Any]) -> str:
        return citation_policy.doc_page_key(c)

    def _relevance_bonus(self, question: str, c: Dict[str, Any]) -> float:
        return citation_policy.relevance_bonus(question, c)

    def curate_citations(
        self,
        chunks: List[Dict[str, Any]],
        question: str = "",
        *,
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        return citation_policy.curate_citations(chunks, question, limit=limit)

    def public_citations(
        self,
        chunks: List[Dict[str, Any]],
        *,
        for_ui: bool = True,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        return citation_policy.public_citations(chunks, min_score=min_score)

    def citations_for_kind(
        self,
        kind: str,
        chunks: List[Dict[str, Any]],
        *,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return citation_policy.citations_for_kind(kind, chunks, min_score=min_score)

    def _to_retrieved_chunks(self, chunks: List[Dict[str, Any]]) -> List[RetrievedChunk]:
        return response_shaping.to_retrieved_chunks(chunks)

    async def _get_chat_history(
        self,
        db: AsyncSession,
        session_id: Optional[str],
        limit: int = 10,
        scope_fingerprint: Optional[str] = None,
        scope_epoch: Optional[int] = None,
    ) -> str:
        """按会话获取最近若干轮用户/助理对话，拼接为简洁文本。

        - 必须提供会话、已知范围指纹和 epoch；旧未知范围历史不进入追问。
        - 返回按时间升序的最近 N 条，格式："用户: ...\n助理: ..."，以便提示词引用。
        """
        try:
            if not session_id or not scope_fingerprint or scope_epoch is None:
                return ""
            stmt = (
                select(QueryHistory)
                .where(QueryHistory.session_id == session_id)
                .where(QueryHistory.scope_status == "known",
                       QueryHistory.scope_fingerprint == scope_fingerprint,
                       QueryHistory.scope_epoch == scope_epoch,
                       QueryHistory.source_status == "active")
                .order_by(QueryHistory.created_time.desc())
                .limit(limit)
            )
            result = await db.execute(stmt)
            recent = result.scalars().all() or []
            # 反转为时间升序，便于阅读
            recent = list(reversed(recent))
            lines: List[str] = []
            for r in recent:
                q = (getattr(r, "query", None) or "").strip()
                a = (getattr(r, "response", None) or "").strip()
                if q:
                    lines.append(f"用户: {q}")
                if a:
                    lines.append(f"助理: {a}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"获取会话历史失败：{e}")
            return ""
    

    def _normalize_sim(self, score: Any) -> float:
        return citation_policy.normalize_sim(score)

    def _chunk_gate_score(self, c: Dict[str, Any]) -> float:
        return citation_policy.chunk_gate_score(c)

    def _best_similarity(self, chunks: List[Dict[str, Any]]) -> float:
        return citation_policy.best_similarity(chunks)

    def _refusal_answer(self, reason: str = "insufficient_evidence") -> str:
        return refusal_policy.refusal_answer(reason)

    def _is_advice_or_guarantee_question(self, question: str) -> bool:
        return refusal_policy.is_advice_or_guarantee_question(question)

    def _has_llm_credentials(self) -> bool:
        return settings.generation_configured

    def _is_off_topic(self, question: str, chunks: List[Dict[str, Any]]) -> bool:
        return refusal_policy.is_off_topic(question, chunks)


    async def process_query(self, db, request, stream=False, *, coordinator=None):
        if coordinator is None:
            from app.services.index_coordinator import default_coordinator
            coordinator = default_coordinator()
        return await self.scoped.process(db, request, coordinator=coordinator, stream=stream)

    async def _save_query_history(
        self,
        db: AsyncSession,
        question: str,
        answer: str,
        query_type: str,
        response_time: float,
        chunks_used: int,
        session_id: Optional[str] = None,
        rewritten_query: Optional[str] = None,
        rewriting_metadata_json: Optional[str] = None,
        retrieved_chunks_json: Optional[str] = None,
    ) -> Optional[int]:
        return await query_history_service.save_query_history(
            db=db,
            question=question,
            answer=answer,
            query_type=query_type,
            response_time=response_time,
            chunks_used=chunks_used,
            session_id=session_id,
            rewritten_query=rewritten_query,
            rewriting_metadata_json=rewriting_metadata_json,
            retrieved_chunks_json=retrieved_chunks_json,
        )

    async def _save_query_history_bg(
        self,
        question: str,
        answer: str,
        query_type: str,
        response_time: float,
        chunks_used: int,
        session_id: Optional[str] = None,
        rewritten_query: Optional[str] = None,
        rewriting_metadata_json: Optional[str] = None,
        retrieved_chunks_json: Optional[str] = None,
    ) -> None:
        await query_history_service.save_query_history_bg(
            question=question,
            answer=answer,
            query_type=query_type,
            response_time=response_time,
            chunks_used=chunks_used,
            session_id=session_id,
            rewritten_query=rewritten_query,
            rewriting_metadata_json=rewriting_metadata_json,
            retrieved_chunks_json=retrieved_chunks_json,
        )

    async def get_query_history(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        query_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[QueryHistory]:
        return await query_history_service.get_query_history(
            db=db,
            skip=skip,
            limit=limit,
            query_type=query_type,
            start_date=start_date,
            end_date=end_date,
        )

    async def update_query_feedback(
        self,
        db: AsyncSession,
        query_id: int,
        rating: int,
        comment: Optional[str] = None,
    ) -> bool:
        return await query_history_service.update_query_feedback(
            db=db,
            query_id=query_id,
            rating=rating,
            comment=comment,
        )

    async def get_query_statistics(self, db: AsyncSession, days: int = 30) -> QueryStatistics:
        return await query_history_service.get_query_statistics(db, days=days)


@lru_cache(maxsize=1)
def _load_ontology() -> dict:
    """加载保险术语本体库JSON，使用缓存减少I/O。"""
    try:
        with open(settings.ONTOLOGY_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载本体库失败，使用空本体：{e}")
        return {
            "synonyms": {},
            "product_types": [],
            "coverage_terms": [],
            "claim_terms": [],
            "exclusions": [],
        }
