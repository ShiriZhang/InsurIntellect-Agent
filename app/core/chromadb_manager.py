#!/usr/bin/env python3
"""
ChromaDB 单例管理器
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Optional, Callable, Any
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.core.config import settings
from app.core.app_logging import get_logger

logger = get_logger(__name__)


class ChromaDBManager:
    """
    提供 ChromaDB 客户端与集合的单例管理
    """

    _instance: Optional["ChromaDBManager"] = None
    # 使用可重入锁以避免在 get_collection 调用 get_client 时的二次加锁死锁
    _lock = threading.RLock()
    _client: Optional[chromadb.Client] = None
    _collection: Optional[chromadb.Collection] = None
    _executor: Optional[ThreadPoolExecutor] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        logger.info("初始化 ChromaDB 管理器")

    def get_client(self) -> chromadb.Client:
        """
        获取 ChromaDB 客户端（单例）
        """
        if self._client is None:
            with self._lock:
                if self._client is None:
                    chroma_path = Path(settings.CHROMA_PERSIST_DIRECTORY)
                    chroma_path.mkdir(parents=True, exist_ok=True)

                    logger.info(f"创建 ChromaDB 持久化客户端, 路径 {chroma_path}（禁用匿名遥测）")
                    self._client = chromadb.PersistentClient(
                        path=str(chroma_path),
                        settings=ChromaSettings(
                            anonymized_telemetry=False,
                        ),
                    )
        return self._client

    def get_executor(self) -> ThreadPoolExecutor:
        """获取或创建用于包装同步调用的线程池。"""
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    size = settings.CHROMA_THREAD_MAX_WORKERS
                    logger.info(f"创建 ChromaDB 线程池，max_workers={size}")
                    self._executor = ThreadPoolExecutor(max_workers=size, thread_name_prefix="chroma-worker")
        return self._executor

    async def run_in_thread(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """在专用线程池中执行同步函数，并返回结果。"""
        loop = asyncio.get_running_loop()
        executor = self.get_executor()
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

    def get_collection(self, collection_name: str = "insurance_documents") -> chromadb.Collection:
        """
        获取或创建 ChromaDB 集合（单例）
        """
        if self._collection is None:
            with self._lock:
                if self._collection is None:
                    client = self.get_client()
                    try:
                        self._collection = client.get_collection(name=collection_name)
                        logger.info(f"获取现有 ChromaDB 集合: {collection_name}")
                    except Exception:
                        logger.info(f"创建新 ChromaDB 集合: {collection_name}")
                        self._collection = client.create_collection(
                            name=collection_name,
                            metadata={"hnsw:space": settings.EMBEDDING_DISTANCE},
                        )

                    logger.info(f"ChromaDB 集合初始化完成, 向量数 {self._collection.count()}")
        return self._collection

    def reset_collection(self, collection_name: str = "insurance_documents") -> None:
        """
        重置集合（删除并重新创建）
        """
        with self._lock:
            client = self.get_client()
            try:
                client.delete_collection(name=collection_name)
                logger.info(f"删除 ChromaDB 集合: {collection_name}")
            except Exception as e:
                logger.warning(f"删除集合失败（可能不存在）: {e}")

            self._collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": settings.EMBEDDING_DISTANCE},
            )
            logger.info(f"重新创建 ChromaDB 集合: {collection_name}")

    def close(self) -> None:
        """
        关闭 ChromaDB 连接
        """
        with self._lock:
            if self._client is not None:
                logger.info("关闭 ChromaDB 连接")
                self._client = None
                self._collection = None
            if self._executor is not None:
                try:
                    logger.info("关闭 ChromaDB 线程池")
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                finally:
                    self._executor = None

    @classmethod
    def get_instance(cls) -> "ChromaDBManager":
        """获取单例实例"""
        return cls()


# 全局单例实例
chroma_manager = ChromaDBManager()
