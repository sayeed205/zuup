"""Configuration integration and validation for the yt-dlp media engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import threading
import time
from typing import Any

# Optional watchdog import for hot-reload functionality
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..storage.models import GlobalConfig, TaskConfig
from ..storage.models import ProxyConfig as CoreProxyConfig
from .media_models import (
    AuthMethod,
    MediaConfig,
    ProxyConfig,
    ProxyType,
)

logger = logging.getLogger(__name__)


class MediaConfigurationError(Exception):
    """Raised when media configuration validation or mapping fails."""

    def __init__(self, message: str, field: str | None = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


class MediaConfigValidationResult(BaseModel):
    """Result of media configuration validation."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    config: MediaConfig | None = None


class MediaConfigProfile(BaseModel):
    """Configuration profile for different media download use cases."""

    name: str
    description: str
    config: MediaConfig
    tags: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate profile name is not empty."""
        if not v.strip():
            raise ValueError("Profile name cannot be empty")
        return v.strip()


class FormatSelectorValidator:
    """Validates yt-dlp format selectors for correctness."""

    # Common format selector patterns
    VALID_PATTERNS = {
        # Quality-based selectors
        "best",
        "worst",
        "bestvideo",
        "worstvideo",
        "bestaudio",
        "worstaudio",
        # Resolution-based
        "best[height<=720]",
        "best[height<=1080]",
        "best[height<=1440]",
        "best[height<=2160]",
        # Codec-based
        "best[vcodec^=avc1]",
        "best[acodec^=mp4a]",
        "best[vcodec!=none]",
        # Container-based
        "best[ext=mp4]",
        "best[ext=webm]",
        "best[ext=mkv]",
        # Combined selectors
        "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
    }

    @classmethod
    def validate_format_selector(cls, selector: str) -> tuple[bool, list[str]]:
        """
        Validate a yt-dlp format selector.

        Args:
            selector: Format selector string

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not selector or not selector.strip():
            errors.append("Format selector cannot be empty")
            return False, errors

        selector = selector.strip()

        # Check for selectors that start with [ (should have a base selector first)
        if selector.startswith("["):
            errors.append(
                "Format selector should not start with brackets - use a base selector like 'best[...]'"
            )

        # Check for selectors that end with ] but have no opening bracket
        if selector.endswith("]") and "[" not in selector:
            errors.append("Format selector ends with ] but has no opening bracket")

        # Check for unmatched brackets
        bracket_count = selector.count("[") - selector.count("]")
        if bracket_count != 0:
            errors.append("Unmatched brackets in format selector")

        # Check for invalid operators
        invalid_ops = ["==", "!=!", "<=<", ">=<"]
        for op in invalid_ops:
            if op in selector:
                errors.append(f"Invalid operator '{op}' in format selector")

        # Validate field names in conditions
        valid_fields = {
            "ext",
            "acodec",
            "vcodec",
            "container",
            "protocol",
            "format_id",
            "height",
            "width",
            "tbr",
            "abr",
            "vbr",
            "asr",
            "fps",
            "filesize",
            "filesize_approx",
            "quality",
            "format_note",
            "language",
            "preference",
        }

        # Extract field names from conditions (simplified parsing)
        import re

        field_pattern = r"\[([a-zA-Z_]+)(?:[<>=!^$*]|!=)"
        fields_used = re.findall(field_pattern, selector)

        for field in fields_used:
            if field not in valid_fields:
                errors.append(f"Unknown field '{field}' in format selector")

        return len(errors) == 0, errors


class ExtractorConfigValidator:
    """Validates extractor-specific configuration options."""

    # Known extractor options for popular sites
    EXTRACTOR_OPTIONS = {
        "youtube": {
            "skip_dash_manifest",
            "player_skip_js",
            "include_ads",
            "max_comments",
            "comment_sort",
            "max_comment_replies",
            "player_client",
            "innertube_host",
            "innertube_key",
            "skip_hls_native",
        },
        "twitch": {"api_base", "client_id", "token", "disable_ads"},
        "twitter": {"api_base", "guest_token", "syndication_api"},
        "tiktok": {"api_base", "app_version", "manifest_app_version"},
        "instagram": {"api_base", "include_stories", "include_highlights"},
        "facebook": {"api_base", "include_stories"},
        "vimeo": {"api_base", "player_url"},
        "dailymotion": {"family_filter", "geo_bypass_country"},
        "soundcloud": {"client_id", "api_base"},
        "spotify": {"client_id", "client_secret"},
    }

    @classmethod
    def validate_extractor_args(
        cls, extractor_args: dict[str, dict[str, Any]]
    ) -> tuple[bool, list[str]]:
        """
        Validate extractor-specific arguments.

        Args:
            extractor_args: Dictionary of extractor arguments

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        for extractor, args in extractor_args.items():
            if not isinstance(args, dict):
                errors.append(f"Extractor args for '{extractor}' must be a dictionary")
                continue

            # Check if we know about this extractor
            if extractor in cls.EXTRACTOR_OPTIONS:
                valid_options = cls.EXTRACTOR_OPTIONS[extractor]
                for option in args.keys():
                    if option not in valid_options:
                        errors.append(
                            f"Unknown option '{option}' for extractor '{extractor}'. "
                            f"Valid options: {', '.join(sorted(valid_options))}"
                        )

        return len(errors) == 0, errors


class MediaConfigurationMapper:
    """Maps TaskConfig and GlobalConfig to MediaConfig with validation."""

    def __init__(self):
        """Initialize the media configuration mapper."""
        self._default_profiles = self._create_default_profiles()
        self._format_validator = FormatSelectorValidator()
        self._extractor_validator = ExtractorConfigValidator()

    def map_task_config(
        self,
        task_config: TaskConfig,
        global_config: GlobalConfig | None = None,
        profile_name: str | None = None,
        url: str | None = None,
    ) -> MediaConfig:
        """
        Map TaskConfig and GlobalConfig to MediaConfig.

        Args:
            task_config: Task-specific configuration
            global_config: Global application configuration
            profile_name: Optional profile to use as base
            url: Optional URL for extractor-specific configuration

        Returns:
            Mapped MediaConfig

        Raises:
            MediaConfigurationError: If mapping fails
        """
        try:
            # Start with profile if specified
            if profile_name:
                base_config = self.get_profile(profile_name).config.model_copy()
            else:
                base_config = MediaConfig(output_directory=Path.home() / "Downloads")

            # Apply global config overrides
            if global_config:
                self._apply_global_config(base_config, global_config)

            # Apply task-specific overrides
            self._apply_task_config(base_config, task_config)

            # Apply URL-specific configuration if URL provided
            if url:
                self._apply_url_specific_config(base_config, url)

            return base_config

        except Exception as e:
            raise MediaConfigurationError(f"Failed to map configuration: {e}") from e

    def _apply_global_config(
        self, config: MediaConfig, global_config: GlobalConfig
    ) -> None:
        """Apply global configuration settings to MediaConfig."""
        # Output directory
        if global_config.default_download_path:
            config.output_directory = global_config.default_download_path

        # Concurrent downloads - use global config value
        config.concurrent_downloads = global_config.max_concurrent_downloads

        # User agent (for network config)
        if global_config.user_agent:
            config.network_config.user_agent = global_config.user_agent

        # Proxy settings
        if global_config.proxy_settings:
            config.proxy_config = self._map_proxy_config(global_config.proxy_settings)
            # Also set legacy proxy field for backward compatibility
            if global_config.proxy_settings.http_proxy:
                config.proxy = global_config.proxy_settings.http_proxy
            elif global_config.proxy_settings.https_proxy:
                config.proxy = global_config.proxy_settings.https_proxy

    def _apply_task_config(self, config: MediaConfig, task_config: TaskConfig) -> None:
        """Apply task-specific configuration settings to MediaConfig."""
        # Timeout settings
        if task_config.timeout > 0:
            config.socket_timeout = task_config.timeout
            config.network_config.socket_timeout = task_config.timeout
            config.network_config.connect_timeout = min(task_config.timeout, 30)
            config.network_config.read_timeout = task_config.timeout

        # Retry settings
        config.retries = task_config.retry_attempts
        config.network_config.retries = task_config.retry_attempts

        # Headers and cookies
        if task_config.headers:
            config.network_config.custom_headers.update(task_config.headers)

        if task_config.cookies:
            # Convert cookies dict to cookies file format if needed
            # For now, we'll store them in the network config
            pass  # yt-dlp handles cookies differently

        # Speed limiting
        if task_config.download_speed_limit:
            # yt-dlp doesn't have direct speed limiting, but we can use sleep intervals
            # Calculate sleep interval based on desired speed limit
            # This is a rough approximation
            if task_config.download_speed_limit < 1024 * 1024:  # < 1MB/s
                config.network_config.sleep_interval = 0.1
            elif task_config.download_speed_limit < 5 * 1024 * 1024:  # < 5MB/s
                config.network_config.sleep_interval = 0.05

        # Authentication settings
        if task_config.auth_username or task_config.auth_password:
            config.auth_config.method = AuthMethod.USERNAME_PASSWORD
            config.auth_config.username = task_config.auth_username
            config.auth_config.password = task_config.auth_password

        # Proxy override
        if task_config.proxy:
            config.proxy_config = self._map_proxy_config(task_config.proxy)
            config.proxy = task_config.proxy.http_proxy or task_config.proxy.https_proxy

    def _apply_url_specific_config(self, config: MediaConfig, url: str) -> None:
        """Apply URL-specific configuration optimizations."""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # YouTube-specific optimizations
            if "youtube.com" in domain or "youtu.be" in domain:
                self._apply_youtube_config(config)
            # Twitch-specific optimizations
            elif "twitch.tv" in domain:
                self._apply_twitch_config(config)
            # TikTok-specific optimizations
            elif "tiktok.com" in domain:
                self._apply_tiktok_config(config)
            # Instagram-specific optimizations
            elif "instagram.com" in domain:
                self._apply_instagram_config(config)
            # Twitter/X-specific optimizations
            elif "twitter.com" in domain or "x.com" in domain:
                self._apply_twitter_config(config)

        except Exception as e:
            logger.warning(f"Failed to apply URL-specific config for {url}: {e}")

    def _apply_youtube_config(self, config: MediaConfig) -> None:
        """Apply YouTube-specific configuration."""
        # Prefer mp4 format for better compatibility
        if config.format_selector == "best[height<=1080]":
            config.format_selector = "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best[height<=1080]"

        # YouTube-specific extractor args
        if "youtube" not in config.extractor_args:
            config.extractor_args["youtube"] = {}

        youtube_args = config.extractor_args["youtube"]

        # Skip DASH manifest for faster extraction
        if "skip_dash_manifest" not in youtube_args:
            youtube_args["skip_dash_manifest"] = True

        # Use specific player client for better reliability
        if "player_client" not in youtube_args:
            youtube_args["player_client"] = ["android", "web"]

    def _apply_twitch_config(self, config: MediaConfig) -> None:
        """Apply Twitch-specific configuration."""
        # Twitch prefers specific formats
        if config.format_selector == "best[height<=1080]":
            config.format_selector = "best[height<=1080]/best"

        # Twitch-specific settings
        if "twitch" not in config.extractor_args:
            config.extractor_args["twitch"] = {}

    def _apply_tiktok_config(self, config: MediaConfig) -> None:
        """Apply TikTok-specific configuration."""
        # TikTok often has geo-restrictions
        config.geo_bypass_config.geo_bypass = True
        config.geo_bypass_config.auto_detect_geo_blocking = True

        # TikTok-specific settings
        if "tiktok" not in config.extractor_args:
            config.extractor_args["tiktok"] = {}

    def _apply_instagram_config(self, config: MediaConfig) -> None:
        """Apply Instagram-specific configuration."""
        # Instagram requires careful handling
        config.network_config.sleep_interval = 1.0  # Be respectful
        config.network_config.retries = 3  # Don't be too aggressive

        # Instagram-specific settings
        if "instagram" not in config.extractor_args:
            config.extractor_args["instagram"] = {}

    def _apply_twitter_config(self, config: MediaConfig) -> None:
        """Apply Twitter/X-specific configuration."""
        # Twitter has rate limiting
        config.network_config.sleep_interval = 0.5
        config.network_config.retries = 5

        # Twitter-specific settings
        if "twitter" not in config.extractor_args:
            config.extractor_args["twitter"] = {}

    def _map_proxy_config(self, core_proxy: CoreProxyConfig) -> ProxyConfig:
        """Map core ProxyConfig to media ProxyConfig."""
        # Determine proxy type and extract host/port from URLs
        proxy_type = ProxyType.HTTP  # Default
        proxy_url = None

        # Check which proxy URL is provided
        if core_proxy.http_proxy:
            proxy_url = core_proxy.http_proxy
            proxy_type = ProxyType.HTTP
        elif core_proxy.https_proxy:
            proxy_url = core_proxy.https_proxy
            proxy_type = ProxyType.HTTPS
        elif core_proxy.socks_proxy:
            proxy_url = core_proxy.socks_proxy
            # Determine SOCKS version from URL scheme
            if proxy_url.startswith("socks4"):
                proxy_type = ProxyType.SOCKS4
            elif proxy_url.startswith("socks5"):
                proxy_type = ProxyType.SOCKS5
            else:
                proxy_type = ProxyType.SOCKS5  # Default to SOCKS5

        return ProxyConfig(
            proxy_url=proxy_url,
            proxy_type=proxy_type,
            proxy_username=core_proxy.username,
            proxy_password=core_proxy.password,
            test_proxy_connectivity=True,
            auto_detect_geo_blocking=True,
        )

    def validate_config(self, config: MediaConfig) -> MediaConfigValidationResult:
        """
        Validate MediaConfig with comprehensive error checking.

        Args:
            config: Configuration to validate

        Returns:
            Validation result with errors and warnings
        """
        errors = []
        warnings = []

        try:
            # Validate using Pydantic
            config.model_validate(config.model_dump())

            # Additional business logic validation
            self._validate_format_settings(config, errors, warnings)
            self._validate_output_settings(config, errors, warnings)
            self._validate_network_settings(config, errors, warnings)
            self._validate_auth_settings(config, errors, warnings)
            self._validate_extractor_settings(config, errors, warnings)
            self._validate_performance_settings(config, errors, warnings)

            return MediaConfigValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                config=config if len(errors) == 0 else None,
            )

        except ValidationError as e:
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                errors.append(f"{field_path}: {error['msg']}")

            return MediaConfigValidationResult(
                valid=False, errors=errors, warnings=warnings, config=None
            )

    def _validate_format_settings(
        self, config: MediaConfig, errors: list[str], warnings: list[str]
    ) -> None:
        """Validate format selection settings."""
        # Validate format selector
        is_valid, format_errors = self._format_validator.validate_format_selector(
            config.format_selector
        )
        if not is_valid:
            errors.extend([f"Format selector: {err}" for err in format_errors])

        # Check for conflicting format preferences
        if (
            config.format_preferences.audio_only
            and config.format_preferences.video_only
        ):
            errors.append("Cannot specify both audio_only and video_only")

        # Validate audio quality setting
        if config.extract_audio:
            try:
                int(config.audio_quality)
            except ValueError:
                valid_qualities = {
                    "best",
                    "worst",
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                }
                if config.audio_quality not in valid_qualities:
                    errors.append(f"Invalid audio quality: {config.audio_quality}")

        # Check for reasonable quality limits
        if (
            config.format_preferences.max_height
            and config.format_preferences.max_height > 4320
        ):
            warnings.append(
                "Very high resolution limit (>4K) may not be available for most content"
            )

        if (
            config.format_preferences.max_video_bitrate
            and config.format_preferences.max_video_bitrate > 50000
        ):
            warnings.append(
                "Very high video bitrate limit may not be available for most content"
            )

    def _validate_output_settings(
        self, config: MediaConfig, errors: list[str], warnings: list[str]
    ) -> None:
        """Validate output and file organization settings."""
        # Check output directory
        try:
            config.output_directory.mkdir(parents=True, exist_ok=True)
            if not config.output_directory.is_dir():
                errors.append(
                    f"Output directory is not a directory: {config.output_directory}"
                )
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")

        # Validate output template
        if not config.output_template:
            errors.append("Output template cannot be empty")
        elif (
            "%(title)s" not in config.output_template
            and "%(id)s" not in config.output_template
        ):
            warnings.append(
                "Output template should include %(title)s or %(id)s for unique filenames"
            )

        # Check download archive path
        if config.download_archive:
            try:
                config.download_archive.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create download archive directory: {e}")

        # Validate playlist range
        if config.playlist_end and config.playlist_end < config.playlist_start:
            errors.append("Playlist end must be >= playlist start")

    def _validate_network_settings(
        self, config: MediaConfig, errors: list[str], warnings: list[str]
    ) -> None:
        """Validate network and connection settings."""
        # Check timeout consistency
        if config.socket_timeout <= 0:
            errors.append("Socket timeout must be positive")

        if (
            config.network_config.connect_timeout
            >= config.network_config.socket_timeout
        ):
            warnings.append("Connect timeout should be less than socket timeout")

        # Check retry settings
        if config.retries > 20:
            warnings.append(
                "Very high retry count may cause long delays on persistent failures"
            )

        if config.network_config.retry_backoff_factor > 5.0:
            warnings.append(
                "Very high retry backoff factor may cause extremely long delays"
            )

        # Check sleep intervals
        if config.network_config.sleep_interval > 10.0:
            warnings.append(
                "Very high sleep interval will significantly slow downloads"
            )

        # Validate proxy settings
        if config.proxy_config.proxy_url:
            from urllib.parse import urlparse

            try:
                parsed = urlparse(config.proxy_config.proxy_url)
                if not parsed.scheme or not parsed.netloc:
                    errors.append("Invalid proxy URL format")
            except Exception:
                errors.append("Invalid proxy URL format")

    def _validate_auth_settings(
        self, config: MediaConfig, errors: list[str], warnings: list[str]
    ) -> None:
        """Validate authentication settings."""
        if config.auth_config.method == AuthMethod.USERNAME_PASSWORD:
            if not config.auth_config.username or not config.auth_config.password:
                errors.append(
                    "Username and password required for username_password auth"
                )

        elif config.auth_config.method == AuthMethod.COOKIES:
            if not config.auth_config.cookies_file:
                errors.append("Cookies file required for cookies auth")
            elif not config.auth_config.cookies_file.exists():
                warnings.append(
                    f"Cookies file does not exist: {config.auth_config.cookies_file}"
                )

        elif config.auth_config.method == AuthMethod.NETRC:
            if (
                config.auth_config.netrc_file
                and not config.auth_config.netrc_file.exists()
            ):
                warnings.append(
                    f"Netrc file does not exist: {config.auth_config.netrc_file}"
                )

    def _validate_extractor_settings(
        self, config: MediaConfig, errors: list[str], warnings: list[str]
    ) -> None:
        """Validate extractor-specific settings."""
        # Validate extractor arguments
        is_valid, extractor_errors = self._extractor_validator.validate_extractor_args(
            config.extractor_args
        )
        if not is_valid:
            warnings.extend(extractor_errors)  # These are warnings, not errors

        # Check geo-bypass settings
        if config.geo_bypass_config.geo_bypass_country:
            if len(config.geo_bypass_config.geo_bypass_country) != 2:
                errors.append("Geo-bypass country must be a 2-letter ISO code")

    def _validate_performance_settings(
        self, config: MediaConfig, errors: list[str], warnings: list[str]
    ) -> None:
        """Validate performance-related settings."""
        # Check concurrent downloads
        if config.concurrent_downloads > 10:
            warnings.append(
                "Very high concurrent downloads may overwhelm servers or cause rate limiting"
            )

        # Check delay settings
        if config.delay_between_downloads > 60.0:
            warnings.append(
                "Very high delay between downloads will significantly slow batch operations"
            )

        # Check download limits
        if config.max_downloads and config.max_downloads > 10000:
            warnings.append(
                "Very high download limit may cause extremely long operations"
            )

    def get_profile(self, name: str) -> MediaConfigProfile:
        """
        Get a media configuration profile by name.

        Args:
            name: Profile name

        Returns:
            Configuration profile

        Raises:
            MediaConfigurationError: If profile not found
        """
        if name not in self._default_profiles:
            raise MediaConfigurationError(f"Profile '{name}' not found")

        return self._default_profiles[name]

    def list_profiles(self) -> list[str]:
        """List available configuration profile names."""
        return list(self._default_profiles.keys())

    def _create_default_profiles(self) -> dict[str, MediaConfigProfile]:
        """Create default media configuration profiles."""
        profiles = {}

        # Audio-only profile
        audio_config = MediaConfig(
            output_directory=Path.home() / "Downloads" / "Audio",
            format_selector="bestaudio/best",
            extract_audio=True,
            audio_quality="192",
            audio_format="mp3",
            embed_metadata=True,
            embed_thumbnail=True,
            output_template="%(uploader)s - %(title)s.%(ext)s",
        )
        audio_config.format_preferences.audio_only = True
        audio_config.format_preferences.target_audio_bitrate = 192

        profiles["audio_only"] = MediaConfigProfile(
            name="audio_only",
            description="Optimized for audio-only downloads with MP3 conversion",
            config=audio_config,
            tags=["audio", "music", "podcast"],
        )

        # Best quality profile
        best_quality_config = MediaConfig(
            output_directory=Path.home() / "Downloads" / "Videos",
            format_selector="bestvideo[height<=2160]+bestaudio/best[height<=2160]/best",
            embed_subtitles=True,
            embed_thumbnail=True,
            embed_metadata=True,
            output_template="%(uploader)s/%(upload_date)s - %(title)s.%(ext)s",
            create_subdirectories=True,
        )
        best_quality_config.format_preferences.max_height = 2160  # 4K
        best_quality_config.format_preferences.prefer_free_formats = False
        best_quality_config.format_preferences.quality_fallback = True

        profiles["best_quality"] = MediaConfigProfile(
            name="best_quality",
            description="Download best available quality up to 4K with all metadata",
            config=best_quality_config,
            tags=["quality", "4k", "metadata"],
        )

        # Fast download profile
        fast_config = MediaConfig(
            output_directory=Path.home() / "Downloads",
            format_selector="best[height<=720]/best",
            embed_subtitles=False,
            embed_thumbnail=False,
            embed_metadata=False,
            concurrent_downloads=3,
            delay_between_downloads=0.0,
        )
        fast_config.format_preferences.max_height = 720
        fast_config.format_preferences.adaptive_quality = True
        fast_config.network_config.retries = 3
        fast_config.network_config.retry_sleep = 0.5

        profiles["fast"] = MediaConfigProfile(
            name="fast",
            description="Fast downloads with minimal processing and 720p max quality",
            config=fast_config,
            tags=["fast", "720p", "minimal"],
        )

        # Mobile-friendly profile
        mobile_config = MediaConfig(
            output_directory=Path.home() / "Downloads" / "Mobile",
            format_selector="best[height<=480][ext=mp4]/best[ext=mp4][height<=480]/best[height<=480]",
            embed_metadata=True,
            output_template="%(title)s.%(ext)s",
        )
        mobile_config.format_preferences.max_height = 480
        mobile_config.format_preferences.preferred_containers = ["mp4"]
        mobile_config.format_preferences.prefer_hardware_decodable = True
        mobile_config.network_config.sleep_interval = (
            1.0  # Be gentle on mobile networks
        )

        profiles["mobile"] = MediaConfigProfile(
            name="mobile",
            description="Mobile-optimized downloads with small file sizes and MP4 format",
            config=mobile_config,
            tags=["mobile", "480p", "mp4", "small"],
        )

        # Playlist profile
        playlist_config = MediaConfig(
            output_directory=Path.home() / "Downloads" / "Playlists",
            format_selector="best[height<=1080]/best",
            embed_metadata=True,
            embed_thumbnail=True,
            output_template="%(playlist)s/%(playlist_index)02d - %(title)s.%(ext)s",
            create_subdirectories=True,
            concurrent_downloads=2,
            delay_between_downloads=1.0,
            ignore_errors=True,
        )
        playlist_config.download_archive = (
            Path.home() / "Downloads" / ".download_archive"
        )

        profiles["playlist"] = MediaConfigProfile(
            name="playlist",
            description="Optimized for playlist downloads with archive support",
            config=playlist_config,
            tags=["playlist", "batch", "archive"],
        )

        # Archive profile (for backing up channels/playlists)
        archive_config = MediaConfig(
            output_directory=Path.home() / "Downloads" / "Archive",
            format_selector="bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            embed_subtitles=True,
            embed_thumbnail=True,
            embed_metadata=True,
            output_template="%(uploader)s/%(upload_date)s - %(title)s [%(id)s].%(ext)s",
            create_subdirectories=True,
            concurrent_downloads=1,
            delay_between_downloads=2.0,
            ignore_errors=True,
        )
        archive_config.download_archive = (
            Path.home() / "Downloads" / "Archive" / ".archive"
        )

        profiles["archive"] = MediaConfigProfile(
            name="archive",
            description="Conservative settings for archiving channels with full metadata",
            config=archive_config,
            tags=["archive", "conservative", "metadata", "backup"],
        )

        return profiles


# Hot-reload support (similar to the HTTP engine)
if WATCHDOG_AVAILABLE:

    class MediaConfigFileHandler(FileSystemEventHandler):
        """File system event handler for media configuration files."""

        def __init__(self, callback: callable):
            """Initialize with callback function."""
            self.callback = callback

        def on_modified(self, event):
            """Handle file modification events."""
            if not event.is_directory:
                self.callback(event.src_path)
else:

    class MediaConfigFileHandler:
        """Dummy handler when watchdog is not available."""

        def __init__(self, callback: callable):
            """Initialize with callback function."""
            self.callback = callback


class MediaConfigurationHotReloader:
    """Provides hot-reload support for media configuration files during development."""

    def __init__(self, config_path: Path, callback: callable | None = None):
        """
        Initialize hot reloader.

        Args:
            config_path: Path to configuration file to watch
            callback: Optional callback to call when config changes
        """
        self.config_path = config_path
        self.callback = callback
        self.observer = None
        self._lock = threading.Lock()
        self._last_reload = 0
        self._reload_debounce = 1.0  # 1 second debounce

        if not WATCHDOG_AVAILABLE:
            logger.warning(
                "Watchdog not available - media configuration hot-reload disabled. "
                "Install with: pip install watchdog"
            )

    def start(self) -> None:
        """Start watching for configuration file changes."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("Cannot start hot-reload: watchdog not available")
            return

        if self.observer is not None:
            return

        event_handler = MediaConfigFileHandler(self._on_config_changed)
        self.observer = Observer()
        self.observer.schedule(
            event_handler, str(self.config_path.parent), recursive=False
        )
        self.observer.start()
        logger.info(f"Started media configuration hot-reload for {self.config_path}")

    def stop(self) -> None:
        """Stop watching for configuration file changes."""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped media configuration hot-reload")

    def _on_config_changed(self, file_path: str) -> None:
        """Handle configuration file change event."""
        if Path(file_path) != self.config_path:
            return

        with self._lock:
            current_time = time.time()
            if current_time - self._last_reload < self._reload_debounce:
                return

            self._last_reload = current_time

        try:
            logger.info(f"Media configuration file changed: {file_path}")
            if self.callback:
                self.callback(file_path)
        except Exception as e:
            logger.error(f"Error in media configuration reload callback: {e}")


class MediaConfigurationManager:
    """Main configuration manager for the yt-dlp media engine."""

    def __init__(self):
        """Initialize media configuration manager."""
        self.mapper = MediaConfigurationMapper()
        self.hot_reloader = None
        self._config_cache = {}
        self._cache_lock = threading.Lock()
        self._validation_cache = {}  # Cache validation results

    def create_engine_config(
        self,
        task_config: TaskConfig,
        global_config: GlobalConfig | None = None,
        profile_name: str | None = None,
        url: str | None = None,
        validate: bool = True,
    ) -> MediaConfig:
        """
        Create MediaConfig from TaskConfig and GlobalConfig.

        Args:
            task_config: Task-specific configuration
            global_config: Global application configuration
            profile_name: Optional profile to use as base
            url: Optional URL for extractor-specific configuration
            validate: Whether to validate the resulting configuration

        Returns:
            Configured MediaConfig

        Raises:
            MediaConfigurationError: If configuration is invalid
        """
        # Create cache key
        cache_key = self._create_cache_key(
            task_config, global_config, profile_name, url
        )

        # Check cache
        with self._cache_lock:
            if cache_key in self._config_cache:
                return self._config_cache[cache_key].model_copy()

        # Map configuration
        config = self.mapper.map_task_config(
            task_config, global_config, profile_name, url
        )

        # Validate if requested
        if validate:
            result = self.mapper.validate_config(config)
            if not result.valid:
                error_msg = "Media configuration validation failed:\n" + "\n".join(
                    result.errors
                )
                raise MediaConfigurationError(error_msg)

            # Log warnings
            for warning in result.warnings:
                logger.warning(f"Media configuration warning: {warning}")

        # Cache the result
        with self._cache_lock:
            self._config_cache[cache_key] = config.model_copy()

        return config

    def _create_cache_key(
        self,
        task_config: TaskConfig,
        global_config: GlobalConfig | None,
        profile_name: str | None,
        url: str | None,
    ) -> str:
        """Create a cache key for configuration."""
        import hashlib

        # Create a deterministic hash of the configuration
        config_data = {
            "task": task_config.model_dump(mode="json"),
            "global": global_config.model_dump(mode="json") if global_config else None,
            "profile": profile_name,
            "url": url,
        }

        config_json = json.dumps(config_data, sort_keys=True)
        return hashlib.md5(config_json.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        with self._cache_lock:
            self._config_cache.clear()
            self._validation_cache.clear()
        logger.info("Media configuration cache cleared")

    def enable_hot_reload(self, config_file: Path) -> None:
        """
        Enable hot-reload for a configuration file.

        Args:
            config_file: Path to configuration file to watch
        """
        if self.hot_reloader is not None:
            self.hot_reloader.stop()

        def on_config_change(file_path: str):
            logger.info(f"Media configuration changed, clearing cache: {file_path}")
            self.clear_cache()

        self.hot_reloader = MediaConfigurationHotReloader(config_file, on_config_change)
        self.hot_reloader.start()

    def disable_hot_reload(self) -> None:
        """Disable configuration hot-reload."""
        if self.hot_reloader is not None:
            self.hot_reloader.stop()
            self.hot_reloader = None

    def get_profile_names(self) -> list[str]:
        """Get list of available configuration profile names."""
        return self.mapper.list_profiles()

    def get_profile_info(self, name: str) -> dict[str, Any]:
        """
        Get information about a configuration profile.

        Args:
            name: Profile name

        Returns:
            Profile information dictionary
        """
        profile = self.mapper.get_profile(name)
        return {
            "name": profile.name,
            "description": profile.description,
            "tags": profile.tags,
            "config_summary": {
                "format_selector": profile.config.format_selector,
                "output_directory": str(profile.config.output_directory),
                "extract_audio": profile.config.extract_audio,
                "embed_metadata": profile.config.embed_metadata,
                "concurrent_downloads": profile.config.concurrent_downloads,
                "max_height": profile.config.format_preferences.max_height,
            },
        }

    def validate_task_config(
        self, task_config: TaskConfig, global_config: GlobalConfig | None = None
    ) -> MediaConfigValidationResult:
        """
        Validate a TaskConfig by mapping it to MediaConfig and checking for issues.

        Args:
            task_config: Task configuration to validate
            global_config: Optional global configuration

        Returns:
            Validation result
        """
        # Create cache key for validation results
        validation_key = self._create_validation_cache_key(task_config, global_config)

        # Check validation cache
        with self._cache_lock:
            if validation_key in self._validation_cache:
                return self._validation_cache[validation_key]

        try:
            mapped_config = self.mapper.map_task_config(task_config, global_config)
            result = self.mapper.validate_config(mapped_config)

            # Cache the validation result
            with self._cache_lock:
                self._validation_cache[validation_key] = result

            return result
        except Exception as e:
            result = MediaConfigValidationResult(
                valid=False,
                errors=[f"Media configuration mapping failed: {e}"],
                warnings=[],
                config=None,
            )

            # Cache the error result too
            with self._cache_lock:
                self._validation_cache[validation_key] = result

            return result

    def _create_validation_cache_key(
        self, task_config: TaskConfig, global_config: GlobalConfig | None
    ) -> str:
        """Create a cache key for validation results."""
        import hashlib

        config_data = {
            "task": task_config.model_dump(mode="json"),
            "global": global_config.model_dump(mode="json") if global_config else None,
        }

        config_json = json.dumps(config_data, sort_keys=True)
        return f"validation_{hashlib.md5(config_json.encode()).hexdigest()}"

    def get_extractor_options(self, extractor_name: str) -> set[str]:
        """
        Get available options for a specific extractor.

        Args:
            extractor_name: Name of the extractor

        Returns:
            Set of available option names
        """
        return ExtractorConfigValidator.EXTRACTOR_OPTIONS.get(extractor_name, set())

    def validate_format_selector(self, selector: str) -> tuple[bool, list[str]]:
        """
        Validate a yt-dlp format selector.

        Args:
            selector: Format selector string

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return FormatSelectorValidator.validate_format_selector(selector)

    def suggest_format_selector(self, requirements: dict[str, Any]) -> str:
        """
        Suggest a format selector based on requirements.

        Args:
            requirements: Dictionary of requirements (height, codec, etc.)

        Returns:
            Suggested format selector string
        """
        parts = []

        # Video quality
        if requirements.get("max_height"):
            parts.append(f"height<={requirements['max_height']}")

        # Video codec
        if requirements.get("video_codec"):
            parts.append(f"vcodec^={requirements['video_codec']}")

        # Audio codec
        if requirements.get("audio_codec"):
            parts.append(f"acodec^={requirements['audio_codec']}")

        # Container format
        if requirements.get("container"):
            parts.append(f"ext={requirements['container']}")

        # Build selector
        if parts:
            conditions = "[" + "][".join(parts) + "]"
            if requirements.get("audio_only"):
                return f"bestaudio{conditions}/bestaudio"
            elif requirements.get("video_only"):
                return f"bestvideo{conditions}/bestvideo"
            else:
                return f"bestvideo{conditions}+bestaudio/best{conditions}/best"
        elif requirements.get("audio_only"):
            return "bestaudio/best"
        elif requirements.get("video_only"):
            return "bestvideo/best"
        else:
            return "best"
