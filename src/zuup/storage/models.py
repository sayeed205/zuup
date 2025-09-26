"""Data models for the download manager."""

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

from cuid import cuid
from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("downloaded_bytes", "total_bytes")
    @classmethod
    def validate_bytes(cls, v: int | None) -> int | None:
        """Validate byte counts are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Byte counts must be non-negative")
        return v

    @field_validator("download_speed", "upload_speed")
    @classmethod
    def validate_speeds(cls, v: float | None) -> float | None:
        """Validate speeds are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Speeds must be non-negative")
        return v

    @field_validator("peers_connected", "peers_total", "seeds_connected", "seeds_total")
    @classmethod
    def validate_peer_counts(cls, v: int | None) -> int | None:
        """Validate peer counts are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Peer counts must be non-negative")
        return v

    @field_validator("ratio")
    @classmethod
    def validate_ratio(cls, v: float | None) -> float | None:
        """Validate ratio is non-negative."""
        if v is not None and v < 0:
            raise ValueError("Ratio must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_progress_consistency(self) -> "ProgressInfo":
        """Validate progress information consistency."""
        # Ensure downloaded bytes doesn't exceed total bytes
        if (self.total_bytes is not None and
            self.downloaded_bytes > self.total_bytes):
            raise ValueError("Downloaded bytes cannot exceed total bytes")

        # Validate peer relationships
        if (self.peers_connected is not None and
            self.peers_total is not None and
            self.peers_connected > self.peers_total):
            raise ValueError("Connected peers cannot exceed total peers")

        if (self.seeds_connected is not None and
            self.seeds_total is not None and
            self.seeds_connected > self.seeds_total):
            raise ValueError("Connected seeds cannot exceed total seeds")

        return self

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

    @field_validator("http_proxy", "https_proxy", "socks_proxy")
    @classmethod
    def validate_proxy_url(cls, v: str | None) -> str | None:
        """Validate proxy URL format."""
        if v is None:
            return v

        try:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Proxy URL must include scheme and host")
            if parsed.scheme not in ("http", "https", "socks4", "socks5"):
                raise ValueError("Proxy scheme must be http, https, socks4, or socks5")
            return v
        except Exception as e:
            raise ValueError(f"Invalid proxy URL format: {e}") from e

    @model_validator(mode="after")
    def validate_proxy_config(self) -> "ProxyConfig":
        """Validate proxy configuration consistency."""
        if self.enabled:
            if not any([self.http_proxy, self.https_proxy, self.socks_proxy]):
                raise ValueError("At least one proxy URL must be provided when proxy is enabled")
        return self


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

    @field_validator("max_connections")
    @classmethod
    def validate_max_connections(cls, v: int | None) -> int | None:
        """Validate max connections is positive."""
        if v is not None and v <= 0:
            raise ValueError("max_connections must be positive")
        return v

    @field_validator("download_speed_limit", "upload_speed_limit")
    @classmethod
    def validate_speed_limits(cls, v: int | None) -> int | None:
        """Validate speed limits are positive."""
        if v is not None and v <= 0:
            raise ValueError("Speed limits must be positive")
        return v

    @field_validator("retry_attempts")
    @classmethod
    def validate_retry_attempts(cls, v: int) -> int:
        """Validate retry attempts is non-negative."""
        if v < 0:
            raise ValueError("retry_attempts must be non-negative")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("timeout must be positive")
        return v

    @field_validator("seed_ratio_limit")
    @classmethod
    def validate_seed_ratio_limit(cls, v: float | None) -> float | None:
        """Validate seed ratio limit is positive."""
        if v is not None and v <= 0:
            raise ValueError("seed_ratio_limit must be positive")
        return v

    @field_validator("seed_time_limit")
    @classmethod
    def validate_seed_time_limit(cls, v: int | None) -> int | None:
        """Validate seed time limit is positive."""
        if v is not None and v <= 0:
            raise ValueError("seed_time_limit must be positive")
        return v

    @field_validator("headers", "cookies")
    @classmethod
    def validate_string_dict(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate that dictionary values are strings."""
        for key, value in v.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Headers and cookies must be string key-value pairs")
        return v


class GlobalConfig(BaseModel):
    """Global application configuration."""

    max_concurrent_downloads: int = 3
    default_download_path: Path = Path.home() / "Downloads"
    temp_directory: Path = Path.home() / ".cache" / "download-manager"
    max_connections_per_download: int = 8
    user_agent: str = "Zuup/0.1.0"
    proxy_settings: ProxyConfig | None = None
    logging_level: str = "INFO"

    # Database settings
    database_path: Path = Path.home() / ".cache" / "download-manager" / "tasks.db"
    enable_database_backup: bool = True

    # Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 8080

    # GUI settings
    theme: str = "dark"
    auto_start_downloads: bool = True
    show_notifications: bool = True

    # Monitoring settings
    enable_monitoring: bool = True

    @field_validator("max_concurrent_downloads", "max_connections_per_download")
    @classmethod
    def validate_positive_integers(cls, v: int) -> int:
        """Validate that integer values are positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("default_download_path", "temp_directory")
    @classmethod
    def validate_directory_paths(cls, v: Path) -> Path:
        """Validate and create directory paths if they don't exist."""
        try:
            # Ensure path is absolute
            if not v.is_absolute():
                v = v.expanduser().resolve()

            # Create directory if it doesn't exist
            v.mkdir(parents=True, exist_ok=True)

            # Check if directory is writable
            if not v.exists() or not v.is_dir():
                raise ValueError(f"Path {v} is not a valid directory")

            return v
        except Exception as e:
            raise ValueError(f"Invalid path {v}: {e}") from e

    @field_validator("database_path")
    @classmethod
    def validate_database_path(cls, v: Path) -> Path:
        """Validate database path and create parent directory if needed."""
        try:
            # Ensure path is absolute
            if not v.is_absolute():
                v = v.expanduser().resolve()

            # Create parent directory if it doesn't exist
            v.parent.mkdir(parents=True, exist_ok=True)

            # Check if parent directory is writable
            if not v.parent.exists() or not v.parent.is_dir():
                raise ValueError(f"Database parent directory {v.parent} is not valid")

            return v
        except Exception as e:
            raise ValueError(f"Invalid database path {v}: {e}") from e

    @field_validator("logging_level")
    @classmethod
    def validate_logging_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"logging_level must be one of {valid_levels}")
        return v.upper()

    @field_validator("server_host")
    @classmethod
    def validate_server_host(cls, v: str) -> str:
        """Validate server host."""
        if not v.strip():
            raise ValueError("server_host cannot be empty")
        return v

    @field_validator("server_port")
    @classmethod
    def validate_server_port(cls, v: int) -> int:
        """Validate server port range."""
        if not (1 <= v <= 65535):
            raise ValueError("server_port must be between 1 and 65535")
        return v

    @field_validator("theme")
    @classmethod
    def validate_theme(cls, v: str) -> str:
        """Validate theme selection."""
        valid_themes = {"light", "dark", "auto"}
        if v.lower() not in valid_themes:
            raise ValueError(f"theme must be one of {valid_themes}")
        return v.lower()


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

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format and supported schemes."""
        if not v.strip():
            raise ValueError("URL cannot be empty")

        try:
            parsed = urlparse(v)
            if not parsed.scheme:
                raise ValueError("URL must include a scheme")

            # Check supported schemes
            supported_schemes = {
                "http", "https",  # HTTP engine
                "ftp", "ftps", "sftp",  # FTP engine
                "magnet",  # Torrent engine
                "file",  # Local files
            }

            scheme = parsed.scheme.lower()
            if scheme not in supported_schemes:
                raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

            # Special validation for different schemes
            if scheme == "magnet":
                # Magnet URLs don't need netloc, but should have xt parameter
                if "xt=" not in v:
                    raise ValueError("Magnet URL must contain xt parameter")
            elif scheme in ("http", "https", "ftp", "ftps", "sftp"):
                # These schemes need netloc (host)
                if not parsed.netloc:
                    raise ValueError(f"{scheme.upper()} URL must include host")
            elif scheme == "file":
                # File URLs should have a path
                if not parsed.path:
                    raise ValueError("File URL must include path")

            return v
        except Exception as e:
            raise ValueError(f"Invalid URL format: {e}") from e

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, v: Path) -> Path:
        """Validate destination path."""
        try:
            # Ensure path is absolute
            if not v.is_absolute():
                v = v.expanduser().resolve()

            # Check if parent directory exists or can be created
            parent = v.parent
            parent.mkdir(parents=True, exist_ok=True)

            # Check if parent is writable
            if not parent.exists() or not parent.is_dir():
                raise ValueError(f"Parent directory {parent} is not accessible")

            return v
        except Exception as e:
            raise ValueError(f"Invalid destination path {v}: {e}") from e

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str | None) -> str | None:
        """Validate filename for filesystem compatibility."""
        if v is None:
            return v

        if not v.strip():
            raise ValueError("Filename cannot be empty")

        # Check for invalid characters
        invalid_chars = '<>:"/\\|?*'
        if any(char in v for char in invalid_chars):
            raise ValueError(f"Filename contains invalid characters: {invalid_chars}")

        # Check length (most filesystems have 255 character limit)
        if len(v) > 255:
            raise ValueError("Filename too long (max 255 characters)")

        return v.strip()

    @field_validator("file_size")
    @classmethod
    def validate_file_size(cls, v: int | None) -> int | None:
        """Validate file size is non-negative."""
        if v is not None and v < 0:
            raise ValueError("file_size must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_task_consistency(self) -> "DownloadTask":
        """Validate task configuration consistency."""
        # Ensure progress status matches task status
        if self.progress.status != self.status:
            self.progress.status = self.status

        # Validate engine type matches URL scheme
        url_scheme = urlparse(self.url).scheme.lower()

        engine_scheme_mapping = {
            EngineType.HTTP: {"http", "https"},
            EngineType.FTP: {"ftp", "ftps", "sftp"},
            EngineType.TORRENT: {"magnet", "http", "https", "file"},  # Torrents can be magnet, HTTP, or local .torrent files
            EngineType.MEDIA: {"http", "https"},  # Media engine can handle HTTP URLs
        }

        valid_schemes = engine_scheme_mapping.get(self.engine_type, set())
        if url_scheme not in valid_schemes:
            raise ValueError(f"Engine type {self.engine_type.value} incompatible with URL scheme {url_scheme}")

        # Additional validation for torrent engine with HTTP/file URLs
        if (self.engine_type == EngineType.TORRENT and
            url_scheme in {"http", "https", "file"} and
            not self.url.lower().endswith('.torrent')):
            raise ValueError("HTTP/file URLs for torrent engine must point to .torrent files")

        return self

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
