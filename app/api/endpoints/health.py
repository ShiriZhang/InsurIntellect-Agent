"""
健康检查 API 端点
"""

import time
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db, db_manager
from app.core.app_logging import get_logger
from app.models.schemas import HealthCheck, SystemStats

logger = get_logger(__name__)
router = APIRouter()


async def diagnostics(request: Request):
    """Passive observations only: never load models, download or call a provider."""
    from app.core.config import settings
    ingestion = getattr(request.app.state, "document_ingestion", None)
    coordinator = getattr(request.app.state, "index_coordinator", None)
    processor = getattr(request.app.state, "query_processor", None)
    embedding = getattr(processor, "embedding", None) or getattr(ingestion, "embedding", None)
    local = (embedding.diagnostics() if embedding is not None else {
        "status": "not_checked", "device": None, "cache_status": "not_checked",
        "model": settings.LOCAL_EMBEDDING_MODEL,
        "revision": settings.LOCAL_EMBEDDING_REVISION,
    })
    if embedding is not None:
        local["cache_status"] = "loaded"  # Actual load, not just a directory existing.
    local["cache_directory"] = settings.LOCAL_EMBEDDING_CACHE_DIR
    if not settings.LOCAL_EMBEDDING_REVISION:
        local["status"] = "unavailable"
        local["reason"] = "revision_not_configured"
    generation = {"status": "not_checked" if settings.generation_configured else "unavailable",
                  "configured": settings.generation_configured}
    observation = getattr(request.app.state, "generation_check", None)
    if observation and observation[0] > time.monotonic():
        generation.update(observation[1])
    index = {"status": "unavailable", "commit_id": None}
    if coordinator is not None:
        with coordinator.state.lock:
            current = coordinator.state.current
            index = {"status": "ready" if coordinator.state.serving and current else "not_ready",
                     "commit_id": current.id if current else None}
            if current and local.get("embedding_profile") != current.profile:
                index["status"] = "profile_not_checked_or_mismatched"
    database = await db_manager.health_check()
    ready = (database and index["status"] == "ready" and local["status"] == "ready"
             and generation["status"] == "available")
    return {"status": "ready" if ready else "not_ready", "database_status": database,
            "index": index, "local_embedding": local, "generation": generation}


@router.get("/", response_model=HealthCheck, summary="被动健康诊断")
async def health_check(request: Request):
    data = await diagnostics(request)
    return HealthCheck(status="healthy" if data["status"] == "ready" else "degraded",
                       database_status=data["database_status"],
                       vector_db_status=data["index"]["status"] == "ready",
                       llm_status=data["generation"]["status"] == "available")


@router.get("/stats", response_model=SystemStats, summary="系统统计信息")
async def get_system_stats(request: Request, db: AsyncSession = Depends(get_db)):
    """
    获取系统统计信息：包括文档数量、查询数量、响应时间等
    """
    try:
        from app.models.database_models import Document, DocumentChunk, QueryHistory
        from sqlalchemy import func
        import os

        # 统计文档数量
        total_documents = (
            await db.execute(select(func.count()).select_from(Document))
        ).scalar() or 0

        # 统计文档块数量
        total_chunks = (
            await db.execute(select(func.count()).select_from(DocumentChunk))
        ).scalar() or 0

        # 统计查询数量
        total_queries = (
            await db.execute(select(func.count()).select_from(QueryHistory))
        ).scalar() or 0

        # 计算平均响应时间
        avg_response_time = (
            await db.execute(select(func.avg(QueryHistory.response_time)))
        ).scalar() or 0.0

        # 计算存储使用量
        storage_used = 0
        try:
            from app.core.config import settings
            pdf_path = settings.PDF_STORAGE_PATH
            if os.path.exists(pdf_path):
                for root, dirs, files in os.walk(pdf_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            storage_used += os.path.getsize(file_path)
        except Exception:
            pass

        # 运行时间（示例值）
        uptime = 3600.0

        return SystemStats(
            total_documents=total_documents,
            total_chunks=total_chunks,
            total_queries=total_queries,
            avg_response_time=round(avg_response_time, 3),
            storage_used=storage_used,
            uptime=uptime,
        )

    except Exception as e:
        logger.error(f"获取系统统计信息失败: {e}")
        return SystemStats(total_documents=0, total_chunks=0, total_queries=0, avg_response_time=0.0, storage_used=0, uptime=0.0)


@router.get("/ready", summary="被动就绪诊断")
async def readiness_check(request: Request):
    from fastapi.responses import JSONResponse
    data = await diagnostics(request)
    return JSONResponse(data, status_code=200 if data["status"] == "ready" else 503)


@router.get("/model", summary="被动模型诊断")
async def get_model_info(request: Request):
    data = await diagnostics(request)
    return {"local_embedding": data["local_embedding"], "generation": data["generation"]}


@router.post("/check-local", summary="显式本地模型预检（仅缓存，不下载）")
async def check_local(request: Request):
    import asyncio
    from app.services.embedding_service import LocalEmbeddingUnavailable
    processor = getattr(request.app.state, "query_processor", None)
    if processor is None:
        return {"status": "unavailable", "reason": "query_processor_not_initialized"}
    try:
        return await asyncio.to_thread(processor.preflight)
    except LocalEmbeddingUnavailable:
        return {"status": "unavailable", "reason": "local_model_preflight_failed"}


@router.post("/check-generation", summary="显式生成 API 预检（可能收费）")
async def check_generation(request: Request):
    from app.services.llm_service import LLMService
    service = LLMService()
    try:
        result = await service.preflight()
        # Short-lived observation, never a permanent connectivity guarantee.
        request.app.state.generation_check = (time.monotonic() + 60, result)
        return result
    finally:
        if service.client is not None:
            await service.client.close()


@router.get("/live", summary="存活检查")
async def liveness_check():
    """
    存活检查端点（用于 Kubernetes 等容器编排系统）
    """
    return {"status": "alive"}

