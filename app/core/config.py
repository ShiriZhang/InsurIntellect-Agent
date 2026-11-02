"""
配置管理模块
使用Pydantic Settings进行环境变量管理
"""

from typing import Optional, List, Literal
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """应用配置设置"""

    # 应用基本信息
    APP_NAME: str = Field(default="InsurIntellect Agent", description="应用名称")
    APP_VERSION: str = Field(default="1.0.0", description="应用版本")
    DEBUG: bool = Field(default=True, description="调试模式")
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")

    # 服务器配置（作品集演示默认本机 8766，避免 5000/8765/8748）
    HOST: str = Field(default="127.0.0.1", description="服务器主机")
    PORT: int = Field(default=8766, description="服务器端口")
    RELOAD: bool = Field(default=True, description="自动重载")

    # 本地向量化与生成 API 分别配置，互不借用参数或凭据。
    LOCAL_EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3", description="本地模型 ID 或目录")
    LOCAL_EMBEDDING_REVISION: str = Field(default="", description="固定模型 revision，建议 commit SHA")
    LOCAL_EMBEDDING_CACHE_DIR: str = Field(default="./data/models", description="模型缓存目录")
    LOCAL_EMBEDDING_DEVICE: str = Field(default="auto", description="auto/cpu/cuda/cuda:N")
    LOCAL_EMBEDDING_BATCH_SIZE: int = Field(default=32, ge=1, description="本地编码批大小")
    LOCAL_EMBEDDING_MAX_BATCH_RETRIES: int = Field(default=3, ge=0, le=10, description="GPU OOM 减小批次次数上限")
    LOCAL_EMBEDDING_MAX_INPUT_TOKENS: int = Field(default=512, ge=1, description="含前缀/特殊 token 的编码输入上限")
    LOCAL_EMBEDDING_TEXT_NORMALIZATION: Literal["preserve", "NFKC"] = "preserve"
    LOCAL_EMBEDDING_QUERY_PREFIX: str = ""
    LOCAL_EMBEDDING_DOCUMENT_PREFIX: str = ""
    LOCAL_EMBEDDING_NORMALIZE: bool = True
    EMBEDDING_DISTANCE: Literal["cosine"] = "cosine"

    GENERATION_API_BASE_URL: Optional[str] = Field(default=None, repr=False)
    GENERATION_API_KEY: Optional[str] = Field(default=None, repr=False)
    GENERATION_MODEL: Optional[str] = None

    # 旧生成配置别名；新字段为 None 时才回退，显式空值不回退。
    OPENAI_API_KEY: str = Field(default="", repr=False, description="旧生成 API 密钥别名")
    OPENAI_BASE_URL: str = Field(
        default="", repr=False,
        description="OpenAI-compatible API 基础 URL",
    )
    OPENAI_MODEL: str = Field(default="", description="旧生成模型别名")
    OPENAI_MODEL_LIGHT: str = Field(default="", description="旧轻量生成模型别名")
    OPENAI_MODEL_CORE: str = Field(default="", description="旧核心生成模型别名")
    # 仅接纳旧 .env 字段；实际向量化只读取 LOCAL_EMBEDDING_*。
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="",
        description="已停用，请使用 LOCAL_EMBEDDING_MODEL",
    )
    OPENAI_MAX_TOKENS: int = Field(default=1000, description="最大令牌数")
    OPENAI_TEMPERATURE: float = Field(default=0.2, description="温度参数")
    OPENAI_TIMEOUT_SECS: int = Field(default=90, ge=1, le=3600, description="LLM客户端超时（秒）")

    # 硅基流动配置（兼容回退；演示默认清空以免误走）
    SILICONFLOW_API_KEY: str = Field(default="", repr=False, description="旧生成密钥别名")
    SILICONFLOW_BASE_URL: str = Field(default="", repr=False, description="旧生成地址别名")
    SILICONFLOW_MODEL: str = Field(default="", description="硅基流动模型（可选回退）")

    # 向量数据库配置
    VECTOR_DB_TYPE: str = Field(default="chroma", description="向量数据库类型")
    CHROMA_PERSIST_DIRECTORY: str = Field(
        default="./data/vector_db/chroma", description="ChromaDB持久化目录"
    )
    PINECONE_API_KEY: Optional[str] = Field(default=None, description="Pinecone API密钥")
    PINECONE_ENVIRONMENT: Optional[str] = Field(default=None, description="Pinecone环境")
    PINECONE_INDEX_NAME: str = Field(default="insurintellect", description="Pinecone索引名称")

    # 兼容性配置
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3", description="嵌入模型（兼容性）")

    # 数据库配置
    DATABASE_URL: str = Field(default="sqlite:///./data/database/app.db", description="数据库URL")

    # 文件上传配置（作品集 v1 不提供 UI 上传；预置 samples/ + simple_ingest.py）
    ALLOWED_FILE_TYPES: List[str] = Field(default=["pdf"], description="允许的文件类型")
    PDF_STORAGE_PATH: str = Field(default="./data/documents/pdfs", description="PDF文件存储路径")
    UPLOAD_MAX_BYTES: int = Field(default=20 * 1024 * 1024, gt=0, description="单文件大小限制")
    PROCESSED_DATA_PATH: str = Field(default="./data/processed", description="处理后数据存储路径")

    # 文档处理配置
    CHUNK_SIZE: int = Field(default=400, description="文档块大小")
    CHUNK_OVERLAP: int = Field(default=40, description="文档块重叠")

    # 查询配置
    MAX_RETRIEVED_CHUNKS: int = Field(default=8, description="最大检索块数")
    SIMILARITY_THRESHOLD: float = Field(
        default=0.32, ge=0, le=1, allow_inf_nan=False,
        description="cosine 证据门槛；须按真实本地模型/语料校准，不是回答正确概率",
    )

    # 查询重写与路由配置（主演示默认关闭）
    ENABLE_QUERY_REWRITING: bool = Field(default=False, description="启用口语化查询意图转换引擎")
    ENABLE_QUERY_ROUTING: bool = Field(
        default=False, description="启用基于 LLM 的 RAG/SQL 路由（关闭则默认走 RAG）"
    )
    SIMPLE_RAG_MODE: bool = Field(
        default=True,
        description="主路径：retrieve topk → generate；关闭后才走 SQL/KG 等 advanced 旁路",
    )
    ONTOLOGY_JSON_PATH: str = Field(
        default="tools/insurance_ontology.json", description="保险术语本体库JSON路径"
    )

    # 时序与降级超时（演示需能完成一次真实生成；勿过小误伤）
    CONTEXT_BUILD_TIMEOUT_SECS: int = Field(default=20, description="检索上下文构建最大等待秒数")
    LLM_ANSWER_TIMEOUT_SECS: int = Field(
        default=60, description="非流式回答最大等待秒数，超时将返回降级提示"
    )
    LLM_QUEUE_TIMEOUT_SECS: float = Field(
        default=5.0, description="等待全局并发队列信号量的最大秒数，超时则快速降级"
    )

    # 监控和自动重启设置（默认关闭；演示用 uvicorn 自带重载即可）
    ENABLE_AUTO_RESTART: bool = Field(default=False, description="启用自动重启功能")
    MAX_RESTART_ATTEMPTS: int = Field(default=3, description="最大重启尝试次数")
    RESTART_COOLDOWN: int = Field(default=60, description="重启冷却时间（秒）")

    # 结构化日志设置
    ENABLE_STRUCTURED_LOGGING: bool = Field(default=True, description="启用结构化日志")
    STRUCTURED_LOG_FILE: str = Field(default="logs/structured.log", description="结构化日志文件路径")

    # 后台任务配置
    CELERY_BROKER_URL: Optional[str] = Field(default=None, description="Celery代理URL（仅配置占位，当前主路径不依赖 Celery）")
    CELERY_RESULT_BACKEND: Optional[str] = Field(default=None, description="Celery结果后端")

    # LLM 并发与可靠性控制
    LLM_MAX_CONCURRENCY: int = Field(default=20, description="LLM 并发限制（全局Semaphore）")
    LLM_MAX_RETRIES: int = Field(default=3, ge=1, le=10, description="LLM 最大尝试次数（含初次调用）")
    LLM_BACKOFF_BASE: float = Field(default=0.25, ge=0, le=60, description="LLM 退避基准秒")
    LLM_BACKOFF_FACTOR: float = Field(default=2.0, ge=1, le=10, description="LLM 退避增长因子")
    LLM_BACKOFF_MAX: float = Field(default=2.5, ge=0, le=60, description="LLM 最大退避秒")
    LLM_CIRCUIT_FAILURE_THRESHOLD: int = Field(default=10, description="LLM 熔断失败阈值")
    LLM_CIRCUIT_RESET_TIMEOUT: int = Field(default=30, description="LLM 熔断冷却秒")
    LLM_DAILY_TOKENS_LIMIT: int = Field(
        default=0, description="每日令牌上限（0表示不限制，需结合外部监控）"
    )
    CHROMA_THREAD_MAX_WORKERS: int = Field(default=32, description="ChromaDB 同步调用线程池大小")
    DATABASE_URL_ASYNC: str = Field(default="", description="异步数据库URL（postgresql+asyncpg...）")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Windows .env often uses CRLF; strip CR so empty/async URLs stay valid.
        self._sanitize_string_fields()
        self._create_directories()

    @property
    def generation_api_key(self) -> str:
        if self.GENERATION_API_KEY is not None:
            return self.GENERATION_API_KEY
        return self.OPENAI_API_KEY or self.SILICONFLOW_API_KEY

    @property
    def generation_api_base_url(self) -> str:
        if self.GENERATION_API_BASE_URL is not None:
            return self.GENERATION_API_BASE_URL
        return self.OPENAI_BASE_URL or self.SILICONFLOW_BASE_URL

    def generation_model(self, role: str = "core") -> str:
        if self.GENERATION_MODEL is not None:
            return self.GENERATION_MODEL
        legacy_role = self.OPENAI_MODEL_LIGHT if role == "light" else self.OPENAI_MODEL_CORE
        return legacy_role or self.OPENAI_MODEL or self.SILICONFLOW_MODEL

    @property
    def generation_configured(self) -> bool:
        return bool(self.generation_api_base_url and self.generation_api_key
                    and self.generation_model())

    def _sanitize_string_fields(self) -> None:
        for name, value in list(self.__dict__.items()):
            if isinstance(value, str):
                # Prefix spaces/newlines are part of the semantic input and fingerprint.
                cleaned = (value if name in {"LOCAL_EMBEDDING_QUERY_PREFIX", "LOCAL_EMBEDDING_DOCUMENT_PREFIX"}
                           else value.replace(chr(13), "").strip())
                if cleaned != value:
                    object.__setattr__(self, name, cleaned)

    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            Path("./data/documents/pdfs"),
            Path("./data/vector_db/chroma"),
            Path("./data/processed"),
            Path("./data/database"),
            Path("./logs"),
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# 全局配置实例
settings = Settings()
