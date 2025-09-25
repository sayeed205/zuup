"""Data models for the download manager."""

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from cuid import cuid
from pydantic import BaseModel, Field


class TaskStatus(Enum):
    """Download task status enumeration."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETED = "completed"
    SEEDING = "seeding"  # Torrent-specific: actively seeding
    FAILED = "failed"
    CANCELLED = "cancelled"


class EngineType(Enum):
    """Download engine type enumeration."""

    HTTP = "http"
    FTP = "ftp"
    TORRENT = "torrent"
    MEDIA = "media"


class ProgressInfo(BaseModel):
    """Progress information for a download task."""

    downloaded_bytes: int = 0
    total_bytes: int | None = None
    download_speed: float = 0.0  # bytes per second
    eta: timedelta | None = None
    status: TaskStatus = TaskStatus.PENDING
    error_message: str | None = None

    # Torrent-specific fields (optional)
    upload_speed: float | None = None  # bytes per second
    peers_connected: int | None = None
    peers_total: int | None = None
    seeds_connected: int | None = None
    seeds_total: int | None = None
    ratio: float | None = None  # upload/download ratio
    is_seeding: bool | None = None

    @property
    def progress_percentage(self) -> float | None:
        """Calculate progress percentage."""
        if self.total_bytes and self.total_bytes > 0:
            return (self.downloaded_bytes / self.total_bytes) * 100
        return None


class ProxyConfig(BaseModel):
    """Proxy configuration."""

    enabled: bool = False
    http_proxy: str | None = None
    https_proxy: str | None = None
    socks_proxy: str | None = None
    username: str | None = None
    password: str | None = None


class TaskConfig(BaseModel):
    """Task-specific configuration."""

    max_connections: int | None = None
    download_speed_limit: int | None = None  # bytes per second
    upload_speed_limit: int | None = None  # bytes per second (torrents)
    retry_attempts: int = 3
    timeout: int = 30
    headers: dict[str, str] = Field(default_factory=dict)
    cookies: dict[str, str] = Field(default_factory=dict)

    # Torrent-specific settings
    enable_seeding: bool = True  # Whether to seed after download completion
    seed_ratio_limit: float | None = None  # Stop seeding at this ratio
    seed_time_limit: int | None = None  # Stop seeding after this many seconds

    # Proxy settings (can override global)
    proxy: ProxyConfig | None = None


class GlobalConfig(BaseModel):
    """Global application configuration."""

    max_concurrent_downloads: int = 3
    default_download_path: Path = Path.home() / "Downloads"
    temp_directory: Path = Path.home() / ".cache" / "download-manager"
    max_connections_per_download: int = 8
    user_agent: str = "Zuup/0.1.0"
    proxy_settings: ProxyConfig | None = None
    logging_level: str = "INFO"

    # Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 8080

    # GUI settings
    theme: str = "dark"
    auto_start_downloads: bool = True
    show_notifications: bool = True


class DownloadTask(BaseModel):
    """Download task model."""

    id: str = Field(default_factory=cuid)  # CUID for unique identification
    url: str
    destination: Path
    filename: str | None = None  # Auto-detected if not provided
    engine_type: EngineType
    config: TaskConfig = Field(default_factory=TaskConfig)
    status: TaskStatus = TaskStatus.PENDING
    progress: ProgressInfo = Field(default_factory=ProgressInfo)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Additional metadata
    file_size: int | None = None
    mime_type: str | None = None
    checksum: str | None = None

    def update_progress(self, progress: ProgressInfo) -> None:
        """Update task progress and timestamp."""
        self.progress = progress
        self.updated_at = datetime.now()

    def mark_completed(self) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.progress.status = TaskStatus.COMPLETED
        self.updated_at = datetime.now()

    def mark_failed(self, error_message: str) -> None:
        """Mark task as failed with error message."""
        self.status = TaskStatus.FAILED
        self.progress.status = TaskStatus.FAILED
        self.progress.error_message = error_message
        self.updated_at = datetime.now()
