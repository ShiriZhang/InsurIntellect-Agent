"""
查询处理API端点
"""

from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, and_

from app.core.database import get_db
from app.core.app_logging import get_logger
from app.models.database_models import QueryHistory
from app.models.schemas import (
    QueryRequest, QueryResponse, QueryHistoryResponse, QueryHistoryListResponse,
    FeedbackRequest, QueryStatistics, ErrorResponse, BatchQueryRequest
)
from app.services.query_service import QueryService
from app.services.query_sessions import history_fields

logger = get_logger(__name__)
router = APIRouter()

# 模块级单例：避免每次请求都重新初始化 Agent 等重资源
query_service = QueryService()


@router.get("/test")
async def test_endpoint():
    """测试端点"""
    logger.info("测试端点被调用")
    return {"message": "测试成功", "status": "ok"}


@router.post("/ask", summary="提交问题查询（统一：支持非流与SSE流）")
async def ask_question(
    query_request: QueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    stream: Optional[bool] = None,
):
    """
    提交问题并获取基于文档的回答

    支持的查询类型:
    - general: 一般问题
    - specific: 特定问题
    - comparison: 比较问题
    - summary: 总结问题
    """
    try:
        # 验证查询内容
        if not query_request.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        if len(query_request.question) > 1000:
            raise HTTPException(status_code=400, detail="问题长度不能超过1000个字符")

        # 统一：优先使用请求体中的 stream 字段；若为空回退 querystring
        use_stream = (
            query_request.stream if hasattr(query_request, "stream") else None
        )
        if use_stream is None:
            use_stream = bool(stream)

        if use_stream:
            sse_iter = await query_service.process_query(db=db, request=query_request, stream=True, coordinator=request.app.state.index_coordinator)
            logger.info(f"查询流式处理开始: {query_request.question[:50]}...")
            return StreamingResponse(sse_iter, media_type="text/event-stream")

        # 非流式统一入口
        response = await query_service.process_query(db=db, request=query_request, stream=False, coordinator=request.app.state.index_coordinator)
        logger.info(f"查询处理完成: {query_request.question[:50]}...")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


# 说明：已统一到 POST /ask + body.stream=true，移除旧版 GET /ask/stream 兼容层


@router.get("/history", response_model=QueryHistoryListResponse, summary="获取查询历史")
async def get_query_history(
    skip: int = 0,
    # 将默认 limit 提高，避免在历史较多时被 50 条截断导致外部统计误判
    limit: int = 1000,
    query_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    count_only: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    获取查询历史记录
    支持分页、类型过滤和时间范围过滤
    """
    try:
        history = await query_service.get_query_history(
            db=db,
            skip=skip,
            limit=limit,
            query_type=query_type,
            start_date=start_date,
            end_date=end_date
        )

        results = []
        for record in history:
            # 解析检索块JSON以计算 chunks_used 与相似度/最终分数
            retrieved_chunks = []
            try:
                if getattr(record, "retrieved_chunks", None):
                    import json as _json
                    retrieved_chunks = _json.loads(record.retrieved_chunks)
            except Exception:
                retrieved_chunks = []

            chunks_used = len(retrieved_chunks) if retrieved_chunks else 0
            similarity_scores = []
            try:
                for c in retrieved_chunks:
                    rd = c.get("ranking_details") or {}
                    if "final_score" in rd:
                        similarity_scores.append(float(rd.get("final_score", 0)))
                    else:
                        similarity_scores.append(float(c.get("similarity_score", 0)))
            except Exception:
                similarity_scores = []

            # 组合元数据（重写信息与原始检索块）
            metadata_dict = {}
            try:
                import json as _json
                rewriting_meta = _json.loads(record.rewriting_metadata_json) if record.rewriting_metadata_json else None
                metadata_dict = {
                    "rewritten_query": getattr(record, "rewritten_query", None),
                    "rewriting_metadata": rewriting_meta,
                    "retrieved_chunks": retrieved_chunks,
                }
            except Exception:
                metadata_dict = {"rewritten_query": getattr(record, "rewritten_query", None)}

            results.append(
                QueryHistoryResponse(
                    id=record.id,
                    **history_fields(record),
                    question=record.query,
                    answer=record.response,
                    query_type=record.model_used,
                    response_time=record.response_time,
                    chunks_used=chunks_used,
                    similarity_scores=similarity_scores or None,
                    feedback_rating=getattr(record, "rating", None),
                    feedback_comment=getattr(record, "feedback", None),
                    created_at=record.created_time,
                    metadata=metadata_dict or None,
                )
            )

        # 计算总数（与筛选条件一致）
        conditions = []
        if query_type:
            conditions.append(QueryHistory.model_used == query_type)
        if start_date:
            conditions.append(QueryHistory.created_time >= start_date)
        if end_date:
            conditions.append(QueryHistory.created_time <= end_date)

        count_query = select(func.count()).select_from(QueryHistory)
        if conditions:
            count_query = count_query.where(and_(*conditions))
        total_count = (await db.execute(count_query)).scalar_one()

        if count_only:
            # 仅返回总数，避免大列表传输开销
            return QueryHistoryListResponse(items=[], total_count=total_count)

        return QueryHistoryListResponse(items=results, total_count=total_count)

    except Exception as e:
        logger.error(f"获取查询历史失败: {e}")
        raise HTTPException(status_code=500, detail="获取查询历史失败")


@router.get("/history/{query_id}", response_model=QueryHistoryResponse, summary="获取查询详情")
async def get_query_detail(query_id: int, db: AsyncSession = Depends(get_db)):
    """获取指定查询的详细信息"""
    try:
        result = await db.execute(select(QueryHistory).where(QueryHistory.id == query_id))
        query_record = result.scalar_one_or_none()
        if not query_record:
            raise HTTPException(status_code=404, detail="查询记录不存在")

        # 解析检索块与重写元数据
        import json as _json
        retrieved_chunks = []
        try:
            if getattr(query_record, "retrieved_chunks", None):
                retrieved_chunks = _json.loads(query_record.retrieved_chunks)
        except Exception:
            retrieved_chunks = []

        chunks_used = len(retrieved_chunks) if retrieved_chunks else 0
        similarity_scores = []
        try:
            for c in retrieved_chunks:
                rd = c.get("ranking_details") or {}
                if "final_score" in rd:
                    similarity_scores.append(float(rd.get("final_score", 0)))
                else:
                    similarity_scores.append(float(c.get("similarity_score", 0)))
        except Exception:
            similarity_scores = []

        rewriting_meta = None
        try:
            rewriting_meta = _json.loads(query_record.rewriting_metadata_json) if query_record.rewriting_metadata_json else None
        except Exception:
            rewriting_meta = None

        return QueryHistoryResponse(
            id=query_record.id,
            **history_fields(query_record),
            question=query_record.query,
            answer=query_record.response,
            query_type=query_record.model_used,
            response_time=query_record.response_time,
            chunks_used=chunks_used,
            similarity_scores=similarity_scores or None,
            feedback_rating=getattr(query_record, "rating", None),
            feedback_comment=getattr(query_record, "feedback", None),
            created_at=query_record.created_time,
            metadata={
                "rewritten_query": getattr(query_record, "rewritten_query", None),
                "rewriting_metadata": rewriting_meta,
                "retrieved_chunks": retrieved_chunks,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取查询详情失败: {e}")
        raise HTTPException(status_code=500, detail="获取查询详情失败")


@router.post("/history/{query_id}/feedback", summary="提交查询反馈")
async def submit_feedback(
    query_id: int,
    feedback: FeedbackRequest,
    db: AsyncSession = Depends(get_db)
):
    """为查询结果提交反馈"""
    try:
        success = await query_service.update_query_feedback(
            db=db,
            query_id=query_id,
            rating=feedback.rating,
            comment=feedback.comment
        )
        if not success:
            raise HTTPException(status_code=404, detail="查询记录不存在")

        logger.info(f"查询反馈提交成功: {query_id}, 评分: {feedback.rating}")
        return {"message": "反馈提交成功", "query_id": query_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交查询反馈失败: {e}")
        raise HTTPException(status_code=500, detail="提交查询反馈失败")


@router.get("/statistics", response_model=QueryStatistics, summary="获取查询统计信息")
async def get_query_statistics(
    days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """
    获取查询统计信息
    包括查询数量、平均响应时间、用户满意度
    """
    try:
        stats = await query_service.get_query_statistics(db=db, days=days)
        return stats
    except Exception as e:
        logger.error(f"获取查询统计失败: {e}")
        raise HTTPException(status_code=500, detail="获取查询统计失败")


@router.post("/batch", response_model=List[QueryResponse], summary="批量查询")
async def batch_query(
    batch: BatchQueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    from app.services.query_scope import validate_scope
    from app.services.query_mode import resolve_mode

    queries = batch.queries()
    # Validate every shared ID and every resolved mode before the first child.
    await validate_scope(db, queries[0])
    for item in queries:
        if not item.question.strip():
            raise HTTPException(400, "问题不能为空")
        resolve_mode(item)
    responses = []
    for item in queries:
        responses.append(await query_service.process_query(
            db=db, request=item, coordinator=request.app.state.index_coordinator))
    return responses


@router.post("/suggest", response_model=List[str], summary="获取查询建议")
async def get_query_suggestions(
    partial_query: str,
    limit: int = 5,
    db: AsyncSession = Depends(get_db)
):
    """
    基于历史查询获取查询建议
    """
    try:
        if len(partial_query.strip()) < 2:
            return []

        # 从历史查询中搜索相似问题
        stmt = (
            select(QueryHistory.query)
            .where(QueryHistory.query.contains(partial_query.strip()))
            .distinct()
            .limit(limit)
        )
        result = await db.execute(stmt)
        suggestions = [row[0] for row in result.all()]

        # 如果历史查询不足,可以添加一些预定义的建议
        if len(suggestions) < limit:
            predefined_suggestions = [
                "这份保险的保障范围是什么？",
                "保险费用如何计算？",
                "理赔流程是怎样的？",
                "保险的免赔额是多少？",
                "保险期限是多长时间？"
            ]

            for suggestion in predefined_suggestions:
                if partial_query.lower() in suggestion.lower() and suggestion not in suggestions:
                    suggestions.append(suggestion)
                    if len(suggestions) >= limit:
                        break

        return suggestions[:limit]

    except Exception as e:
        logger.error(f"获取查询建议失败: {e}")
        return []


@router.delete("/history/{query_id}", summary="删除查询记录")
async def delete_query_history(query_id: int, db: AsyncSession = Depends(get_db)):
    """删除指定的查询历史记录"""
    try:
        result = await db.execute(select(QueryHistory).where(QueryHistory.id == query_id))
        query_record = result.scalar_one_or_none()
        if not query_record:
            raise HTTPException(status_code=404, detail="查询记录不存在")

        db.delete(query_record)
        await db.commit()

        logger.info(f"查询记录删除成功: {query_id}")
        return {"message": "查询记录删除成功", "query_id": query_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除查询记录失败: {e}")
        raise HTTPException(status_code=500, detail="删除查询记录失败")


@router.post("/clear-history", summary="清空查询历史")
async def clear_query_history(
    confirm: bool = False,
    days: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    清空查询历史记录
    confirm: 确认清空
    days: 清空指定天数前的记录,不指定则清空全部
    """
    try:
        if not confirm:
            raise HTTPException(status_code=400, detail="请确认清空操作")

        # 计算删除数量并执行删除
        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            count_stmt = select(func.count()).select_from(QueryHistory).where(
                QueryHistory.created_time < cutoff_date
            )
            deleted_count = (await db.execute(count_stmt)).scalar() or 0
            delete_stmt = delete(QueryHistory).where(QueryHistory.created_time < cutoff_date)
        else:
            count_stmt = select(func.count()).select_from(QueryHistory)
            deleted_count = (await db.execute(count_stmt)).scalar() or 0
            delete_stmt = delete(QueryHistory)

        await db.execute(delete_stmt)
        await db.commit()
        logger.info(f"查询历史清空完成: 删除了{deleted_count}条记录")

        return {
            "message": "查询历史清空成功",
            "deleted_count": deleted_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清空查询历史失败: {e}")
        raise HTTPException(status_code=500, detail="清空查询历史失败")

