"""Download engine implementations."""

from .base import BaseDownloadEngine
from .ftp_engine import FTPEngine
from .http_engine import HTTPEngine
from .media_engine import MediaEngine
from .pycurl_models import (
    AuthConfig,
    AuthMethod,
    CompletedSegment,
    CurlError,
    DownloadSegment,
    ErrorAction,
    ErrorCategory,
    HttpFtpConfig,
    ProxyConfig,
    ProxyType,
    SegmentMergeInfo,
    SegmentStatus,
    SshConfig,
    WorkerProgress,
    WorkerStatus,
)
from .registry import (
    EngineRegistry,
    detect_engine_for_url,
    get_engine_for_url,
    get_registry,
    initialize_default_engines,
)
from .torrent_engine import TorrentEngine

__all__ = [
    "BaseDownloadEngine",
    "EngineRegistry",
    "FTPEngine",
    "HTTPEngine",
    "MediaEngine",
    "TorrentEngine",
    "detect_engine_for_url",
    "get_engine_for_url",
    "get_registry",
    "initialize_default_engines",
    # PyCurl models
    "AuthConfig",
    "AuthMethod",
    "CompletedSegment",
    "CurlError",
    "DownloadSegment",
    "ErrorAction",
    "ErrorCategory",
    "HttpFtpConfig",
    "ProxyConfig",
    "ProxyType",
    "SegmentMergeInfo",
    "SegmentStatus",
    "SshConfig",
    "WorkerProgress",
    "WorkerStatus",
]
