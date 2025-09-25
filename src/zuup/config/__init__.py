"""Configuration management module."""

from .defaults import (
    get_default_global_config,
    get_default_proxy_config,
    get_default_task_config,
)
from .manager import ConfigManager, SecureStorage, ValidationResult
from .settings import GlobalConfig, ProxyConfig, TaskConfig

__all__ = [
    "ConfigManager",
    "GlobalConfig",
    "ProxyConfig",
    "SecureStorage",
    "TaskConfig",
    "ValidationResult",
    "get_default_global_config",
    "get_default_proxy_config",
    "get_default_task_config",
]
