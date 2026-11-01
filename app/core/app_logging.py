"""日志配置模块
使用标准库logging而不是loguru
"""

import logging
import json
import sys
import traceback
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, parse_qsl, quote, quote_plus

from app.core.config import settings


class CredentialFilter(logging.Filter):
    """Remove configured generation secrets, even when echoed by an exception."""

    def filter(self, record: logging.LogRecord) -> bool:
        secrets = [settings.generation_api_key, settings.OPENAI_API_KEY,
                   settings.SILICONFLOW_API_KEY]
        for url in (settings.generation_api_base_url, settings.OPENAI_BASE_URL,
                    settings.SILICONFLOW_BASE_URL):
            try:
                parsed = urlsplit(url)
                secrets.extend([parsed.username, parsed.password, parsed.query, parsed.fragment])
                secrets.extend(value for _, value in parse_qsl(parsed.query) if value)
            except ValueError:
                secrets.append(url)
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + "".join(traceback.format_exception(*record.exc_info))
            record.exc_info = None
            record.exc_text = None
        representations = {representation for secret in secrets if secret for representation in (
            secret, quote(secret, safe=""), quote_plus(secret),
            json.dumps(secret, ensure_ascii=True)[1:-1],
        )}
        for secret in sorted(representations, key=len, reverse=True):
            message = message.replace(secret, "[REDACTED]")
        record.msg, record.args = message, ()
        return True


def install_credential_redaction() -> None:
    """Redact at record creation, including HTTP clients and future/root handlers.

    Logger filters do not run for propagated child records. Wrapping the existing
    factory preserves any host logging integration while protecting every sink.
    """
    previous_factory = logging.getLogRecordFactory()
    if getattr(previous_factory, "_insurintellect_credential_redaction", False):
        return
    credential_filter = CredentialFilter()

    def safe_record_factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        credential_filter.filter(record)
        return record

    safe_record_factory._insurintellect_credential_redaction = True
    logging.setLogRecordFactory(safe_record_factory)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        
    Returns:
        配置好的logger实例
    """
    # 创建logger
    logger = get_logger("insurintellect")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的handlers
    logger.handlers.clear()
    
    # 创建formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler（如果指定了文件路径）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "insurintellect") -> logging.Logger:
    """
    获取logger实例
    
    Args:
        name: logger名称
        
    Returns:
        logger实例
    """
    logger = logging.getLogger(name)
    if not any(isinstance(f, CredentialFilter) for f in logger.filters):
        logger.addFilter(CredentialFilter())
    return logger


# 初始化默认logger
install_credential_redaction()
default_logger = setup_logging(
    log_level=settings.LOG_LEVEL,
    log_file="logs/app.log" if Path("logs").exists() else None
)


