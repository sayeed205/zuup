"""Configuration settings models."""

# Re-export from storage.models for convenience
from ..storage.models import GlobalConfig, ProxyConfig, TaskConfig

__all__ = ["GlobalConfig", "ProxyConfig", "TaskConfig"]
