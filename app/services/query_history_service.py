"""Persistence helpers for query history and demo statistics."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.app_logging import get_logger
from app.core.database import db_manager
from app.models.database_models import QueryHistory
from app.models.schemas import QueryStatistics

logger = get_logger(__name__)


async def ensure_query_history_columns(db: AsyncSession) -> None:
    """Compatibility entry point for scripts; use the same startup migration."""
    from app.core.schema_migrations import migrate_schema

    connection = await db.connection()
    await connection.run_sync(migrate_schema)


async def save_query_history(
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
    """Persist one query row; failures stay non-fatal for the public ask path."""
    try:
        await ensure_query_history_columns(db)
        query_history = QueryHistory(
            query=question,
            response=answer,
            model_used=query_type,
            response_time=response_time,
            retrieved_chunks=retrieved_chunks_json,
            tokens_used=None,
            cost=None,
            rating=None,
            feedback=None,
            session_id=session_id,
            rewritten_query=rewritten_query,
            rewriting_metadata_json=rewriting_metadata_json,
        )
        db.add(query_history)
        await db.commit()
        await db.refresh(query_history)
        logger.info(f"查询历史保存成功,ID: {query_history.id}")
        return query_history.id
    except Exception as exc:
        logger.error(f"保存查询历史失败: {exc}")
        try:
            await db.rollback()
        except Exception:
            pass
        return None


async def save_query_history_bg(
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
    """Background-save one query row using a fresh session."""
    try:
        async with db_manager.SessionLocal() as session:
            await save_query_history(
                db=session,
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
    except Exception as exc:
        logger.debug(f"后台保存查询历史失败: {exc}")


async def get_query_history(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
    query_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[QueryHistory]:
    """Return query history rows with optional type/date filters."""
    try:
        stmt = select(QueryHistory)
        if query_type:
            stmt = stmt.where(QueryHistory.model_used == query_type)
        if start_date:
            stmt = stmt.where(QueryHistory.created_time >= start_date)
        if end_date:
            stmt = stmt.where(QueryHistory.created_time <= end_date)
        stmt = stmt.order_by(QueryHistory.created_time.desc()).offset(skip).limit(limit)
        result = await db.execute(stmt)
        history = result.scalars().all()
        logger.info(f"获取查询历史成功,共 {len(history)} 条记录")
        return history
    except Exception as exc:
        logger.error(f"获取查询历史失败: {exc}")
        return []


async def update_query_feedback(
    db: AsyncSession,
    query_id: int,
    rating: int,
    comment: Optional[str] = None,
) -> bool:
    """Update rating/comment for a saved query row."""
    try:
        result = await db.execute(select(QueryHistory).where(QueryHistory.id == query_id))
        query_history = result.scalar_one_or_none()
        if not query_history:
            logger.warning(f"查询记录不存在: {query_id}")
            return False
        query_history.rating = rating
        query_history.feedback = comment
        await db.commit()
        logger.info(f"查询反馈更新成功: query_id={query_id}, rating={rating}")
        return True
    except Exception as exc:
        logger.error(f"更新查询反馈失败: {exc}")
        try:
            await db.rollback()
        except Exception:
            pass
        return False


async def get_query_statistics(db: AsyncSession, days: int = 30) -> QueryStatistics:
    """Return lightweight query statistics for the admin/API surface."""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        total_queries = (
            await db.execute(
                select(func.count()).select_from(QueryHistory).where(
                    QueryHistory.created_time >= start_date
                )
            )
        ).scalar() or 0

        avg_response_time = (
            await db.execute(
                select(func.avg(QueryHistory.response_time)).where(
                    QueryHistory.created_time >= start_date
                )
            )
        ).scalar() or 0

        avg_rating = (
            await db.execute(
                select(func.avg(QueryHistory.rating)).where(
                    QueryHistory.rating.isnot(None),
                    QueryHistory.created_time >= start_date,
                )
            )
        ).scalar() or 0

        today = datetime.utcnow().date()
        today_queries = (
            await db.execute(
                select(func.count()).select_from(QueryHistory).where(
                    func.date(QueryHistory.created_time) == today
                )
            )
        ).scalar() or 0

        type_stmt = (
            select(QueryHistory.model_used, func.count(QueryHistory.model_used).label("count"))
            .where(QueryHistory.created_time >= start_date)
            .group_by(QueryHistory.model_used)
        )
        type_result = await db.execute(type_stmt)
        query_type_distribution = {stat[0]: stat[1] for stat in type_result.all()}

        logger.info("查询统计信息获取成功")
        return QueryStatistics(
            total_queries=total_queries,
            avg_response_time=round(avg_response_time, 3),
            avg_rating=round(avg_rating, 2),
            today_queries=today_queries,
            query_type_distribution=query_type_distribution,
            period_days=days,
            avg_chunks_used=0.0,
            avg_confidence_score=0.0,
            feedback_stats={},
        )
    except Exception as exc:
        logger.error(f"获取查询统计信息失败: {exc}")
        return QueryStatistics(
            total_queries=0,
            avg_response_time=0,
            avg_rating=0,
            today_queries=0,
            query_type_distribution={},
            period_days=days,
            avg_chunks_used=0.0,
            avg_confidence_score=0.0,
            feedback_stats={},
        )
