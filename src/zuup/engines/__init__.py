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
from .segment_merger import SegmentMerger
from .torrent_engine import TorrentEngine

__all__ = [
    "AuthConfig",
    "AuthMethod",
    "BaseDownloadEngine",
    "CompletedSegment",
    "CurlError",
    "DownloadSegment",
    "EngineRegistry",
    "ErrorAction",
    "ErrorCategory",
    "FTPEngine",
    "HTTPEngine",
    "HttpFtpConfig",
    "MediaEngine",
    "ProxyConfig",
    "ProxyType",
    "SegmentMergeInfo",
    "SegmentMerger",
    "SegmentStatus",
    "SshConfig",
    "TorrentEngine",
    "WorkerProgress",
    "WorkerStatus",
    "detect_engine_for_url",
    "get_engine_for_url",
    "get_registry",
    "initialize_default_engines",
]
