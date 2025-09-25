"""Data persistence and storage module."""

from .cache import CacheManager
from .database import DatabaseManager
from .models import (
    DownloadTask,
    EngineType,
    GlobalConfig,
    ProgressInfo,
    ProxyConfig,
    TaskConfig,
    TaskStatus,
)
from .validation import (
    ConfigValidator,
    FileSystemValidator,
    ModelValidator,
    URLValidator,
    ValidationResult,
    get_engine_for_url,
    is_supported_url,
    validate_config_data,
    validate_download_task_data,
)

__all__ = [
    # Core models
    "DownloadTask",
    "EngineType",
    "GlobalConfig",
    "ProgressInfo",
    "ProxyConfig",
    "TaskConfig",
    "TaskStatus",
    # Storage managers
    "CacheManager",
    "DatabaseManager",
    # Validation utilities
    "ConfigValidator",
    "FileSystemValidator",
    "ModelValidator",
    "URLValidator",
    "ValidationResult",
    "get_engine_for_url",
    "is_supported_url",
    "validate_config_data",
    "validate_download_task_data",
]
