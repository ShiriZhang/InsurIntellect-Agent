"""
InsurIntellect Agent - 主应用入口
智能保险文档问答系统
"""

import logging
import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db, async_engine
from app.api.routes import api_router
from app.core.structured_logger import get_structured_logger
from app.core.auto_restart import setup_auto_restart

# 设置结构化日志
structured_logger = get_structured_logger("insurintellect", "logs/app.log")

# 配置标准日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("InsurIntellect Agent 正在启动...")
    structured_logger.log_system_event("application_startup", version="1.0.0")
    
    # 创建必要的目录
    directories = ["logs", "uploads", "temp", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"确保目录存在: {directory}")
    
    # 初始化数据库
    try:
        await init_db()
        logger.info("数据库初始化完成")
        structured_logger.info("database_initialized")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        structured_logger.log_error(e, {"operation": "database_initialization"})
        raise
    
    # 设置自动重启监控
    if settings.ENABLE_AUTO_RESTART:
        monitor = setup_auto_restart(
            check_interval=30,
            max_restart_attempts=3,
            restart_cooldown=60,
            log_file="logs/auto_restart.log"
        )
        
        # 添加健康检查
        def database_health_check():
            """数据库健康检查"""
            try:
                # 这里可以添加实际的数据库连接检查
                return True
            except:
                return False
        
        monitor.add_health_check("database", database_health_check)
        monitor.start_monitoring()
        logger.info("自动重启监控已启动")
        structured_logger.info("auto_restart_monitoring_started")

    from app.services.document_ingestion import DocumentIngestionService
    ingestion = DocumentIngestionService()
    app.state.document_ingestion = ingestion
    app.state.index_coordinator = ingestion.coordinator
    from app.services.document_deletion import DocumentDeletionService
    deletion = DocumentDeletionService(ingestion)
    app.state.document_deletion = deletion
    await asyncio.to_thread(deletion.recover)
    deletion.start()
    try:
        await asyncio.to_thread(ingestion.coordinator.recover)
    except Exception as exc:
        # Liveness remains available; index readers stay blocked until recovery.
        logger.error("Committed index unavailable (%s)", type(exc).__name__)
    await asyncio.to_thread(ingestion.start)
    try:
        yield
    finally:
        await asyncio.to_thread(deletion.stop)
        await asyncio.to_thread(ingestion.stop)

    # 关闭时执行
    logger.info("InsurIntellect Agent 正在关闭...")
    # 关闭异步数据库引擎
    try:
        logger.info("正在关闭异步数据库引擎...")
        await async_engine.dispose()
        logger.info("异步数据库引擎已关闭")
    except Exception as e:
        logger.warning(f"关闭异步数据库引擎失败: {e}")
    # 关闭 ChromaDB 线程池与持久化客户端
    try:
        from app.core.chromadb_manager import chroma_manager
        chroma_manager.close()
        logger.info("ChromaDB 资源已关闭")
    except Exception as e:
        logger.warning(f"关闭 ChromaDB 资源失败: {e}")
    structured_logger.log_system_event("application_shutdown")

    # Close file handlers to avoid ResourceWarning on test teardown
    for handler in logging.getLogger().handlers:
        try:
            handler.close()
        except Exception:
            pass
    try:
        structured_logger.close()
    except Exception:
        pass


# 创建FastAPI应用实例
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="智能保险文档问答系统 - 基于大语言模型的文档检索与问答服务",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# 添加请求日志中间件
@app.middleware("http")
async def log_requests(request, call_next):
    """记录所有HTTP请求"""
    logger.info(f"HTTP请求: {request.method} {request.url}")
    
    response = await call_next(request)
    
    logger.info(f"HTTP响应: {response.status_code}")
    
    return response

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加请求日志中间件
@app.middleware("http")
async def log_requests_structured(request, call_next):
    """记录HTTP请求的结构化日志"""
    from fastapi import Request
    start_time = time.time()
    
    # 记录请求开始
    structured_logger.debug("request_started", 
                          method=request.method,
                          path=request.url.path,
                          query_params=str(request.query_params),
                          client_ip=request.client.host if request.client else None)
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # 记录请求完成
        structured_logger.log_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            response_time=process_time,
            client_ip=request.client.host if request.client else None
        )
        
        # 添加响应头
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        
        # 记录请求错误
        structured_logger.log_error(e, {
            "operation": "http_request",
            "method": request.method,
            "path": request.url.path,
            "response_time": process_time
        })
        
        raise


# 全局异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500
        }
    )


# 根路由 - 重定向到前端界面
@app.get("/")
async def root():
    """根路由 - 重定向到前端界面"""
    from fastapi.responses import FileResponse
    import os
    
    # 使用绝对路径确保文件能被找到
    current_dir = Path(__file__).parent.parent  # 获取项目根目录
    static_path = current_dir / "static" / "index.html"
    
    if static_path.exists():
        return FileResponse(static_path)
    else:
        # 如果静态文件不存在,返回简单的API信息
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "message": "InsurIntellect Agent API is running",
            "docs": "/docs",
            "health": "/api/v1/health/"
        }

# API信息路径
@app.get("/api")
async def api_info():
    """API信息"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "智能保险文档问答系统",
        "status": "running",
        "docs_url": "/docs" if settings.DEBUG else None
    }


# 包含API路由
app.include_router(api_router, prefix="/api/v1")


# 挂载静态文件
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Sample / corpus PDFs for ChatPDF-style left viewer (chat-pdf fileUrl pattern)
_repo_root = Path(__file__).parent.parent
_samples_dir = _repo_root / "samples"
if _samples_dir.exists():
    app.mount("/samples", StaticFiles(directory=str(_samples_dir)), name="samples")



if __name__ == "__main__":
    import uvicorn
    from app.core.port_manager import PortManager
    
    # 使用端口管理器获取可用端口
    try:
        port = PortManager.get_port_with_fallback(
            preferred_port=settings.PORT,
            fallback_start=8000,
            max_attempts=10
        )
        logger.info(f"服务将在端口 {port} 上启动")
        
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=port,
            reload=settings.RELOAD and settings.DEBUG,
            log_level=settings.LOG_LEVEL.lower()
        )
    except RuntimeError as e:
        logger.error(f"无法启动服务: {e}")
        exit(1)


