"""Download engine implementations."""

from .base import BaseDownloadEngine
from .http_ftp_engine import HttpFtpEngine
from .media_engine import MediaEngine
from .media_models import (
    AuthConfig as MediaAuthConfig,
)
from .media_models import (
    AuthMethod as MediaAuthMethod,
)
from .media_models import (
    BatchDownloadConfig,
    BatchProgress,
    ChapterInfo,
    FormatPreferences,
    MediaConfig,
    MediaFormat,
    MediaInfo,
    MediaMetadata,
    PlaylistInfo,
    ProcessingResult,
    ProcessingStep,
    SubtitleInfo,
)
from .media_models import (
    DownloadProgress as MediaDownloadProgress,
)
from .media_models import (
    DownloadStatus as MediaDownloadStatus,
)
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
    "BatchDownloadConfig",
    "BatchProgress",
    "ChapterInfo",
    "CompletedSegment",
    "CurlError",
    "DownloadSegment",
    "EngineRegistry",
    "ErrorAction",
    "ErrorCategory",
    "FormatPreferences",
    "HttpFtpConfig",
    "HttpFtpEngine",
    "MediaAuthConfig",
    "MediaAuthMethod",
    "MediaConfig",
    "MediaDownloadProgress",
    "MediaDownloadStatus",
    "MediaEngine",
    "MediaFormat",
    "MediaInfo",
    "MediaMetadata",
    "PlaylistInfo",
    "ProcessingResult",
    "ProcessingStep",
    "ProxyConfig",
    "ProxyType",
    "SegmentMergeInfo",
    "SegmentMerger",
    "SegmentStatus",
    "SshConfig",
    "SubtitleInfo",
    "TorrentEngine",
    "WorkerProgress",
    "WorkerStatus",
    "detect_engine_for_url",
    "get_engine_for_url",
    "get_registry",
    "initialize_default_engines",
]
