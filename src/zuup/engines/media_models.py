"""Media-specific data models for yt-dlp integration."""

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class MediaFormat(BaseModel):
    """Media format information extracted from yt-dlp."""

    format_id: str
    ext: str
    resolution: str | None = None
    fps: float | None = None
    vcodec: str | None = None
    acodec: str | None = None
    abr: float | None = None  # Audio bitrate
    vbr: float | None = None  # Video bitrate
    filesize: int | None = None
    filesize_approx: int | None = None
    quality: float | None = None
    format_note: str | None = None
    language: str | None = None
    preference: int | None = None

    @field_validator("fps", "abr", "vbr", "quality")
    @classmethod
    def validate_positive_floats(cls, v: float | None) -> float | None:
        """Validate that float values are positive."""
        if v is not None and v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @field_validator("filesize", "filesize_approx", "preference")
    @classmethod
    def validate_non_negative_ints(cls, v: int | None) -> int | None:
        """Validate that integer values are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Value must be non-negative")
        return v


class SubtitleInfo(BaseModel):
    """Subtitle track information."""

    url: str
    ext: str
    language: str
    name: str | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate subtitle URL is not empty."""
        if not v.strip():
            raise ValueError("Subtitle URL cannot be empty")
        return v.strip()


class ChapterInfo(BaseModel):
    """Chapter information for media content."""

    title: str
    start_time: float
    end_time: float

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_times(cls, v: float) -> float:
        """Validate that times are non-negative."""
        if v < 0:
            raise ValueError("Time values must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_time_order(self) -> "ChapterInfo":
        """Validate that end time is after start time."""
        if self.end_time <= self.start_time:
            raise ValueError("End time must be after start time")
        return self


class MediaInfo(BaseModel):
    """Comprehensive media information extracted from yt-dlp."""

    id: str
    title: str
    description: str | None = None
    uploader: str | None = None
    upload_date: str | None = None
    duration: float | None = None
    view_count: int | None = None
    like_count: int | None = None
    thumbnail: str | None = None
    formats: list[MediaFormat]
    subtitles: dict[str, list[SubtitleInfo]] = Field(default_factory=dict)
    chapters: list[ChapterInfo] = Field(default_factory=list)
    webpage_url: str
    extractor: str
    extractor_key: str
    is_playlist: bool = False

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v: float | None) -> float | None:
        """Validate duration is positive."""
        if v is not None and v <= 0:
            raise ValueError("Duration must be positive")
        return v

    @field_validator("view_count", "like_count")
    @classmethod
    def validate_counts(cls, v: int | None) -> int | None:
        """Validate counts are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Count values must be non-negative")
        return v

    @field_validator("formats")
    @classmethod
    def validate_formats_not_empty(cls, v: list[MediaFormat]) -> list[MediaFormat]:
        """Validate that at least one format is available."""
        if not v:
            raise ValueError("At least one format must be available")
        return v


class PlaylistInfo(BaseModel):
    """Playlist information for batch downloads."""

    id: str
    title: str
    description: str | None = None
    uploader: str | None = None
    entry_count: int
    entries: list[MediaInfo]
    webpage_url: str

    @field_validator("entry_count")
    @classmethod
    def validate_entry_count(cls, v: int) -> int:
        """Validate entry count is positive."""
        if v <= 0:
            raise ValueError("Entry count must be positive")
        return v

    @model_validator(mode="after")
    def validate_entries_consistency(self) -> "PlaylistInfo":
        """Validate that entry count matches actual entries."""
        if len(self.entries) != self.entry_count:
            raise ValueError("Entry count must match number of entries")
        return self


class DownloadStatus(Enum):
    """Download status enumeration for media downloads."""

    DOWNLOADING = "downloading"
    FINISHED = "finished"
    ERROR = "error"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class DownloadProgress(BaseModel):
    """Progress information for media downloads."""

    status: DownloadStatus
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    download_speed: float | None = None  # bytes per second
    eta: timedelta | None = None
    filename: str | None = None
    fragment_index: int | None = None
    fragment_count: int | None = None

    @field_validator("downloaded_bytes")
    @classmethod
    def validate_downloaded_bytes(cls, v: int) -> int:
        """Validate downloaded bytes is non-negative."""
        if v < 0:
            raise ValueError("Downloaded bytes must be non-negative")
        return v

    @field_validator("total_bytes")
    @classmethod
    def validate_total_bytes(cls, v: int | None) -> int | None:
        """Validate total bytes is positive."""
        if v is not None and v <= 0:
            raise ValueError("Total bytes must be positive")
        return v

    @field_validator("download_speed")
    @classmethod
    def validate_download_speed(cls, v: float | None) -> float | None:
        """Validate download speed is non-negative."""
        if v is not None and v < 0:
            raise ValueError("Download speed must be non-negative")
        return v

    @field_validator("fragment_index", "fragment_count")
    @classmethod
    def validate_fragment_values(cls, v: int | None) -> int | None:
        """Validate fragment values are positive."""
        if v is not None and v <= 0:
            raise ValueError("Fragment values must be positive")
        return v

    @model_validator(mode="after")
    def validate_progress_consistency(self) -> "DownloadProgress":
        """Validate progress information consistency."""
        # Ensure downloaded bytes doesn't exceed total bytes
        if (self.total_bytes is not None and
            self.downloaded_bytes > self.total_bytes):
            raise ValueError("Downloaded bytes cannot exceed total bytes")

        # Validate fragment relationships
        if (self.fragment_index is not None and
            self.fragment_count is not None and
            self.fragment_index > self.fragment_count):
            raise ValueError("Fragment index cannot exceed fragment count")

        return self

    @property
    def progress_percentage(self) -> float | None:
        """Calculate progress percentage."""
        if self.total_bytes and self.total_bytes > 0:
            return (self.downloaded_bytes / self.total_bytes) * 100
        return None


class ProcessingStep(Enum):
    """Post-processing step enumeration."""

    CONVERT_FORMAT = "convert_format"
    EMBED_METADATA = "embed_metadata"
    EMBED_THUMBNAIL = "embed_thumbnail"
    EMBED_SUBTITLES = "embed_subtitles"
    ORGANIZE_FILES = "organize_files"


class ProcessingResult(BaseModel):
    """Result of post-processing operations."""

    success: bool
    output_path: Path
    processing_time: float
    steps_completed: list[ProcessingStep]
    errors: list[str] = Field(default_factory=list)

    @field_validator("processing_time")
    @classmethod
    def validate_processing_time(cls, v: float) -> float:
        """Validate processing time is non-negative."""
        if v < 0:
            raise ValueError("Processing time must be non-negative")
        return v


class BatchProgress(BaseModel):
    """Progress information for batch/playlist downloads."""

    total_items: int
    completed_items: int
    failed_items: int
    current_item: str | None = None
    current_progress: DownloadProgress | None = None
    overall_downloaded_bytes: int = 0
    overall_total_bytes: int | None = None

    @field_validator("total_items", "completed_items", "failed_items")
    @classmethod
    def validate_item_counts(cls, v: int) -> int:
        """Validate item counts are non-negative."""
        if v < 0:
            raise ValueError("Item counts must be non-negative")
        return v

    @field_validator("overall_downloaded_bytes")
    @classmethod
    def validate_overall_downloaded(cls, v: int) -> int:
        """Validate overall downloaded bytes is non-negative."""
        if v < 0:
            raise ValueError("Overall downloaded bytes must be non-negative")
        return v

    @field_validator("overall_total_bytes")
    @classmethod
    def validate_overall_total(cls, v: int | None) -> int | None:
        """Validate overall total bytes is positive."""
        if v is not None and v <= 0:
            raise ValueError("Overall total bytes must be positive")
        return v

    @model_validator(mode="after")
    def validate_batch_consistency(self) -> "BatchProgress":
        """Validate batch progress consistency."""
        # Validate item counts don't exceed total
        if self.completed_items + self.failed_items > self.total_items:
            raise ValueError("Completed + failed items cannot exceed total items")

        # Validate overall bytes consistency
        if (self.overall_total_bytes is not None and
            self.overall_downloaded_bytes > self.overall_total_bytes):
            raise ValueError("Overall downloaded bytes cannot exceed total bytes")

        return self

    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_items > 0:
            return (self.completed_items / self.total_items) * 100
        return 0.0

    @property
    def remaining_items(self) -> int:
        """Calculate remaining items to process."""
        return self.total_items - self.completed_items - self.failed_items


class AuthMethod(Enum):
    """Authentication method enumeration."""

    NONE = "none"
    USERNAME_PASSWORD = "username_password"
    COOKIES = "cookies"
    NETRC = "netrc"
    OAUTH = "oauth"


class ProxyType(Enum):
    """Proxy type enumeration."""
    
    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"


class ProxyConfig(BaseModel):
    """Advanced proxy configuration for geo-bypass and network routing."""
    
    # Basic proxy settings
    proxy_url: str | None = None
    proxy_type: ProxyType = ProxyType.HTTP
    
    # Proxy authentication
    proxy_username: str | None = None
    proxy_password: str | None = None
    
    # Proxy rotation and fallback
    proxy_list: list[str] = Field(default_factory=list)
    enable_proxy_rotation: bool = False
    proxy_rotation_interval: int = 10  # requests before rotation
    
    # Proxy testing and validation
    test_proxy_connectivity: bool = True
    proxy_timeout: float = 10.0
    
    # Geo-bypass specific settings
    geo_bypass_proxy: str | None = None  # Specific proxy for geo-blocked content
    auto_detect_geo_blocking: bool = True

    @field_validator("proxy_rotation_interval")
    @classmethod
    def validate_rotation_interval(cls, v: int) -> int:
        """Validate rotation interval is positive."""
        if v <= 0:
            raise ValueError("Proxy rotation interval must be positive")
        return v

    @field_validator("proxy_timeout")
    @classmethod
    def validate_proxy_timeout(cls, v: float) -> float:
        """Validate proxy timeout is positive."""
        if v <= 0:
            raise ValueError("Proxy timeout must be positive")
        return v

    @model_validator(mode="after")
    def validate_proxy_config(self) -> "ProxyConfig":
        """Validate proxy configuration consistency."""
        # If proxy rotation is enabled, ensure proxy list is provided
        if self.enable_proxy_rotation and not self.proxy_list:
            raise ValueError("Proxy list required when proxy rotation is enabled")
        
        # If proxy authentication is provided, ensure proxy URL is set
        if (self.proxy_username or self.proxy_password) and not self.proxy_url:
            raise ValueError("Proxy URL required when proxy authentication is provided")
        
        return self


class NetworkConfig(BaseModel):
    """Advanced network configuration for timeouts, retries, and headers."""
    
    # Timeout settings
    socket_timeout: int = 30
    read_timeout: int = 60
    connect_timeout: int = 30
    
    # Retry configuration
    retries: int = 10
    fragment_retries: int = 10
    retry_sleep: float = 1.0
    max_retry_sleep: float = 60.0
    retry_backoff_factor: float = 2.0
    
    # Rate limiting and delays
    sleep_interval: float = 0.0
    max_sleep_interval: float = 5.0
    sleep_interval_subtitles: float = 0.0
    
    # User agent and headers
    user_agent: str | None = None
    custom_headers: dict[str, str] = Field(default_factory=dict)
    referer: str | None = None
    
    # Platform-specific user agents
    platform_user_agents: dict[str, str] = Field(default_factory=dict)
    rotate_user_agents: bool = False
    
    # Connection settings
    prefer_ipv4: bool = False
    prefer_ipv6: bool = False
    source_address: str | None = None
    
    # SSL/TLS settings
    no_check_certificate: bool = False
    client_certificate: str | None = None
    client_certificate_key: str | None = None
    client_certificate_password: str | None = None

    @field_validator("socket_timeout", "read_timeout", "connect_timeout", "retries", "fragment_retries")
    @classmethod
    def validate_positive_ints(cls, v: int) -> int:
        """Validate positive integer values."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("retry_sleep", "max_retry_sleep", "sleep_interval", "max_sleep_interval", 
                    "sleep_interval_subtitles", "retry_backoff_factor")
    @classmethod
    def validate_non_negative_floats(cls, v: float) -> float:
        """Validate non-negative float values."""
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_network_config(self) -> "NetworkConfig":
        """Validate network configuration consistency."""
        # Validate retry sleep ranges
        if self.retry_sleep > self.max_retry_sleep:
            raise ValueError("Retry sleep cannot exceed max retry sleep")
        
        if self.sleep_interval > self.max_sleep_interval:
            raise ValueError("Sleep interval cannot exceed max sleep interval")
        
        # Validate backoff factor
        if self.retry_backoff_factor <= 1.0:
            raise ValueError("Retry backoff factor must be greater than 1.0")
        
        # Validate IP preference consistency
        if self.prefer_ipv4 and self.prefer_ipv6:
            raise ValueError("Cannot prefer both IPv4 and IPv6")
        
        return self


class GeoBypassConfig(BaseModel):
    """Geo-bypass configuration with country selection and automatic detection."""
    
    # Basic geo-bypass settings
    geo_bypass: bool = False
    geo_bypass_country: str | None = None
    geo_bypass_ip_block: str | None = None
    
    # Advanced geo-bypass features
    auto_detect_geo_blocking: bool = True
    geo_bypass_fallback_countries: list[str] = Field(default_factory=list)
    
    # Country-specific settings
    preferred_countries: list[str] = Field(default_factory=list)
    blocked_countries: list[str] = Field(default_factory=list)
    
    # Geo-bypass testing
    test_geo_bypass: bool = True
    geo_bypass_timeout: float = 15.0

    @field_validator("geo_bypass_country")
    @classmethod
    def validate_country_code(cls, v: str | None) -> str | None:
        """Validate country code format."""
        if v is not None:
            if len(v) != 2 or not v.isalpha():
                raise ValueError("Country code must be a 2-letter ISO code")
            return v.upper()
        return v

    @field_validator("geo_bypass_fallback_countries", "preferred_countries", "blocked_countries")
    @classmethod
    def validate_country_lists(cls, v: list[str]) -> list[str]:
        """Validate country code lists."""
        validated = []
        for country in v:
            if len(country) != 2 or not country.isalpha():
                raise ValueError(f"Invalid country code: {country}. Must be 2-letter ISO code")
            validated.append(country.upper())
        return validated

    @field_validator("geo_bypass_timeout")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("Geo-bypass timeout must be positive")
        return v


class AuthConfig(BaseModel):
    """Authentication configuration for media downloads."""

    method: AuthMethod = AuthMethod.NONE
    username: str | None = None
    password: str | None = None
    cookies_file: Path | None = None
    netrc_file: Path | None = None
    oauth_token: str | None = None

    @model_validator(mode="after")
    def validate_auth_config(self) -> "AuthConfig":
        """Validate authentication configuration consistency."""
        if self.method == AuthMethod.USERNAME_PASSWORD:
            if not self.username or not self.password:
                raise ValueError("Username and password required for username_password auth")
        elif self.method == AuthMethod.COOKIES:
            if not self.cookies_file:
                raise ValueError("Cookies file required for cookies auth")
        elif self.method == AuthMethod.NETRC:
            if not self.netrc_file:
                # Default to ~/.netrc if not specified
                self.netrc_file = Path.home() / ".netrc"
        elif self.method == AuthMethod.OAUTH and not self.oauth_token:
            raise ValueError("OAuth token required for oauth auth")

        return self


class FormatPreferences(BaseModel):
    """Advanced format selection preferences with quality control."""

    # Basic preferences
    prefer_free_formats: bool = False
    max_height: int | None = None
    max_width: int | None = None
    preferred_codecs: list[str] = Field(default_factory=list)
    preferred_containers: list[str] = Field(default_factory=list)
    audio_only: bool = False
    video_only: bool = False
    
    # Advanced quality control
    target_quality: str | None = None  # "ultra_high", "high", "medium", "low", "very_low"
    adaptive_quality: bool = True
    quality_fallback: bool = True
    
    # Bitrate preferences
    max_video_bitrate: int | None = None  # kbps
    min_video_bitrate: int | None = None  # kbps
    max_audio_bitrate: int | None = None  # kbps
    min_audio_bitrate: int | None = None  # kbps
    target_audio_bitrate: int | None = None  # kbps for audio-only
    
    # Frame rate preferences
    max_fps: float | None = None
    min_fps: float | None = None
    prefer_60fps: bool = False
    
    # Codec-specific preferences
    prefer_hardware_decodable: bool = True
    avoid_experimental_codecs: bool = True
    
    # Format conversion preferences
    allow_format_conversion: bool = True
    prefer_native_formats: bool = True

    @field_validator("max_height", "max_width")
    @classmethod
    def validate_dimensions(cls, v: int | None) -> int | None:
        """Validate dimensions are positive."""
        if v is not None and v <= 0:
            raise ValueError("Dimensions must be positive")
        return v

    @field_validator("target_quality")
    @classmethod
    def validate_target_quality(cls, v: str | None) -> str | None:
        """Validate target quality value."""
        if v is not None:
            valid_qualities = {"ultra_high", "high", "medium", "low", "very_low"}
            if v not in valid_qualities:
                raise ValueError(f"Invalid target quality: {v}. Must be one of {valid_qualities}")
        return v

    @field_validator("max_video_bitrate", "min_video_bitrate", "max_audio_bitrate", 
                    "min_audio_bitrate", "target_audio_bitrate")
    @classmethod
    def validate_bitrates(cls, v: int | None) -> int | None:
        """Validate bitrate values are positive."""
        if v is not None and v <= 0:
            raise ValueError("Bitrate values must be positive")
        return v

    @field_validator("max_fps", "min_fps")
    @classmethod
    def validate_fps(cls, v: float | None) -> float | None:
        """Validate FPS values are positive."""
        if v is not None and v <= 0:
            raise ValueError("FPS values must be positive")
        return v

    @model_validator(mode="after")
    def validate_format_preferences(self) -> "FormatPreferences":
        """Validate format preferences consistency."""
        if self.audio_only and self.video_only:
            raise ValueError("Cannot specify both audio_only and video_only")
        
        # Validate bitrate ranges
        if (self.min_video_bitrate is not None and self.max_video_bitrate is not None and
            self.min_video_bitrate > self.max_video_bitrate):
            raise ValueError("Min video bitrate cannot exceed max video bitrate")
        
        if (self.min_audio_bitrate is not None and self.max_audio_bitrate is not None and
            self.min_audio_bitrate > self.max_audio_bitrate):
            raise ValueError("Min audio bitrate cannot exceed max audio bitrate")
        
        # Validate FPS ranges
        if (self.min_fps is not None and self.max_fps is not None and
            self.min_fps > self.max_fps):
            raise ValueError("Min FPS cannot exceed max FPS")
        
        return self


class MediaConfig(BaseModel):
    """Comprehensive configuration for media downloads using yt-dlp."""

    # Quality and format preferences
    format_selector: str = "best[height<=1080]"
    audio_format: str | None = None
    video_format: str | None = None
    format_preferences: FormatPreferences = Field(default_factory=FormatPreferences)

    # Output configuration
    output_template: str = "%(title)s.%(ext)s"
    output_directory: Path
    create_subdirectories: bool = True

    # Post-processing
    extract_audio: bool = False
    audio_quality: str = "192"
    embed_subtitles: bool = True
    embed_thumbnail: bool = True
    embed_metadata: bool = True

    # Download behavior
    download_archive: Path | None = None
    ignore_errors: bool = False
    max_downloads: int | None = None
    playlist_start: int = 1
    playlist_end: int | None = None

    # Authentication
    auth_config: AuthConfig = Field(default_factory=AuthConfig)

    # Advanced network configuration
    network_config: NetworkConfig = Field(default_factory=NetworkConfig)
    proxy_config: ProxyConfig = Field(default_factory=ProxyConfig)
    geo_bypass_config: GeoBypassConfig = Field(default_factory=GeoBypassConfig)

    # Legacy network settings (for backward compatibility)
    proxy: str | None = None
    socket_timeout: int = 30
    retries: int = 10
    fragment_retries: int = 10
    geo_bypass: bool = False
    geo_bypass_country: str | None = None

    # Extractor options
    extractor_args: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Concurrent downloads
    concurrent_downloads: int = 1
    delay_between_downloads: float = 0.0

    @field_validator("audio_quality")
    @classmethod
    def validate_audio_quality(cls, v: str) -> str:
        """Validate audio quality setting."""
        try:
            # Check if it's a valid bitrate
            int(v)
            return v
        except ValueError:
            # Check if it's a valid quality string
            valid_qualities = {"best", "worst", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
            if v not in valid_qualities:
                raise ValueError(f"Invalid audio quality: {v}") from None
            return v

    @field_validator("socket_timeout", "retries", "fragment_retries")
    @classmethod
    def validate_positive_ints(cls, v: int) -> int:
        """Validate positive integer values."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("playlist_start")
    @classmethod
    def validate_playlist_start(cls, v: int) -> int:
        """Validate playlist start is positive."""
        if v <= 0:
            raise ValueError("Playlist start must be positive")
        return v

    @field_validator("playlist_end", "max_downloads")
    @classmethod
    def validate_optional_positive_ints(cls, v: int | None) -> int | None:
        """Validate optional positive integer values."""
        if v is not None and v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("concurrent_downloads")
    @classmethod
    def validate_concurrent_downloads(cls, v: int) -> int:
        """Validate concurrent downloads is positive."""
        if v <= 0:
            raise ValueError("Concurrent downloads must be positive")
        return v

    @field_validator("delay_between_downloads")
    @classmethod
    def validate_delay(cls, v: float) -> float:
        """Validate delay is non-negative."""
        if v < 0:
            raise ValueError("Delay must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_media_config(self) -> "MediaConfig":
        """Validate media configuration consistency."""
        # Validate playlist range
        if (self.playlist_end is not None and
            self.playlist_end < self.playlist_start):
            raise ValueError("Playlist end must be >= playlist start")

        # Validate output directory
        try:
            self.output_directory = self.output_directory.expanduser().resolve()
            self.output_directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Invalid output directory: {e}") from e

        # Validate download archive path
        if self.download_archive is not None:
            try:
                self.download_archive = self.download_archive.expanduser().resolve()
                self.download_archive.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Invalid download archive path: {e}") from e

        # Handle backward compatibility for legacy network settings
        self._sync_legacy_network_settings()

        return self

    def _sync_legacy_network_settings(self) -> None:
        """Synchronize legacy network settings with new configuration objects."""
        # Sync proxy settings
        if self.proxy and not self.proxy_config.proxy_url:
            self.proxy_config.proxy_url = self.proxy
        elif self.proxy_config.proxy_url and not self.proxy:
            self.proxy = self.proxy_config.proxy_url

        # Sync network timeouts and retries
        if self.socket_timeout != 30:  # Non-default value
            self.network_config.socket_timeout = self.socket_timeout
        elif self.network_config.socket_timeout != 30:
            self.socket_timeout = self.network_config.socket_timeout

        if self.retries != 10:  # Non-default value
            self.network_config.retries = self.retries
        elif self.network_config.retries != 10:
            self.retries = self.network_config.retries

        if self.fragment_retries != 10:  # Non-default value
            self.network_config.fragment_retries = self.fragment_retries
        elif self.network_config.fragment_retries != 10:
            self.fragment_retries = self.network_config.fragment_retries

        # Sync geo-bypass settings
        if self.geo_bypass:
            self.geo_bypass_config.geo_bypass = self.geo_bypass
        elif self.geo_bypass_config.geo_bypass:
            self.geo_bypass = self.geo_bypass_config.geo_bypass

        if self.geo_bypass_country:
            self.geo_bypass_config.geo_bypass_country = self.geo_bypass_country
        elif self.geo_bypass_config.geo_bypass_country:
            self.geo_bypass_country = self.geo_bypass_config.geo_bypass_country


class BatchDownloadConfig(BaseModel):
    """Configuration for batch/playlist downloads."""

    concurrent_downloads: int = 1
    delay_between_downloads: float = 0.0
    skip_existing: bool = True
    continue_on_error: bool = True
    max_failures: int = 5
    archive_file: Path | None = None

    @field_validator("concurrent_downloads")
    @classmethod
    def validate_concurrent_downloads(cls, v: int) -> int:
        """Validate concurrent downloads is positive."""
        if v <= 0:
            raise ValueError("Concurrent downloads must be positive")
        return v

    @field_validator("delay_between_downloads")
    @classmethod
    def validate_delay(cls, v: float) -> float:
        """Validate delay is non-negative."""
        if v < 0:
            raise ValueError("Delay must be non-negative")
        return v

    @field_validator("max_failures")
    @classmethod
    def validate_max_failures(cls, v: int) -> int:
        """Validate max failures is non-negative."""
        if v < 0:
            raise ValueError("Max failures must be non-negative")
        return v


class MediaMetadata(BaseModel):
    """Metadata for media files."""

    title: str | None = None
    artist: str | None = None
    album: str | None = None
    description: str | None = None
    upload_date: datetime | None = None
    duration: float | None = None
    thumbnail_url: str | None = None
    uploader: str | None = None
    view_count: int | None = None
    like_count: int | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v: float | None) -> float | None:
        """Validate duration is positive."""
        if v is not None and v <= 0:
            raise ValueError("Duration must be positive")
        return v

    @field_validator("view_count", "like_count")
    @classmethod
    def validate_counts(cls, v: int | None) -> int | None:
        """Validate counts are non-negative."""
        if v is not None and v < 0:
            raise ValueError("Count values must be non-negative")
        return v
