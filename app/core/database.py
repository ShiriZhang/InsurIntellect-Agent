"""
数据库配置和管理模块（异步版）
"""

from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.pool import StaticPool
from app.core.config import settings
from app.core.app_logging import get_logger

logger = get_logger(__name__)


def _derive_async_url(sync_url: str) -> str:
    """从同步URL推导异步URL（仅支持SQLite的安全回退）。"""
    if sync_url.startswith("sqlite:///"):
        # sqlite:///path/to.db -> sqlite+aiosqlite:///path/to.db
        return "sqlite+aiosqlite:///" + sync_url.replace("sqlite:///", "", 1)
    return sync_url  # 非SQLite需通过 settings.DATABASE_URL_ASYNC 显式提供


# 选择异步数据库URL（空白视为未配置，回退到 SQLite 异步推导）
_async_override = (settings.DATABASE_URL_ASYNC or "").strip()
ASYNC_DB_URL = _async_override or _derive_async_url((settings.DATABASE_URL or "").strip())


# 异步数据库引擎配置
if ASYNC_DB_URL.startswith("sqlite+aiosqlite"):
    async_engine = create_async_engine(
        ASYNC_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.DEBUG,
    )
else:
    async_engine = create_async_engine(
        ASYNC_DB_URL,
        echo=settings.DEBUG,
    )

# 异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# 基础模型与元数据
Base = declarative_base()
metadata = MetaData()


async def init_db():
    """初始化数据库（异步）"""
    try:
        # Ensure ORM models are registered on Base.metadata before create_all
        import app.models.database_models  # noqa: F401
        from app.core.schema_migrations import migrate_schema

        async with async_engine.begin() as conn:
            await conn.run_sync(migrate_schema)
        logger.info("数据库初始化完成（异步）")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise


async def get_db():
    """获取异步数据库会话"""
    async with AsyncSessionLocal() as session:
        yield session


class DatabaseManager:
    """数据库管理器（异步）"""

    def __init__(self) -> None:
        self.engine = async_engine
        self.SessionLocal = AsyncSessionLocal

    async def create_session(self) -> AsyncSession:
        """创建异步数据库会话"""
        return self.SessionLocal()

    async def close_session(self, session: AsyncSession):
        """关闭异步数据库会话"""
        await session.close()

    async def health_check(self) -> bool:
        """数据库健康检查（异步）"""
        try:
            async with self.SessionLocal() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"数据库健康检查失败: {e}")
            return False


# 全局数据库管理器实例
db_manager = DatabaseManager()


