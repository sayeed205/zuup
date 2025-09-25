"""Comprehensive validation utilities for data models."""

from pathlib import Path
import re
from typing import Any, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from .models import DownloadTask, GlobalConfig, ProgressInfo, ProxyConfig, TaskConfig

T = TypeVar('T', bound=BaseModel)


class ValidationResult:
    """Result of a validation operation."""

    def __init__(self, is_valid: bool, errors: list[str] | None = None, data: Any = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.data = data

    def __bool__(self) -> bool:
        """Return True if validation passed."""
        return self.is_valid

    def __str__(self) -> str:
        """String representation of validation result."""
        if self.is_valid:
            return "Validation passed"
        return f"Validation failed: {'; '.join(self.errors)}"


class ModelValidator:
    """Comprehensive model validation utilities."""

    @staticmethod
    def validate_model(model_class: type[T], data: dict[str, Any]) -> ValidationResult:
        """
        Validate data against a Pydantic model.
        
        Args:
            model_class: The Pydantic model class to validate against
            data: Dictionary of data to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        try:
            validated_model = model_class(**data)
            return ValidationResult(is_valid=True, data=validated_model)
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error["loc"])
                message = error["msg"]
                errors.append(f"{field}: {message}")
            return ValidationResult(is_valid=False, errors=errors)
        except Exception as e:
            return ValidationResult(is_valid=False, errors=[f"Unexpected error: {e!s}"])

    @staticmethod
    def validate_download_task(data: dict[str, Any]) -> ValidationResult:
        """Validate download task data."""
        return ModelValidator.validate_model(DownloadTask, data)

    @staticmethod
    def validate_progress_info(data: dict[str, Any]) -> ValidationResult:
        """Validate progress info data."""
        return ModelValidator.validate_model(ProgressInfo, data)

    @staticmethod
    def validate_task_config(data: dict[str, Any]) -> ValidationResult:
        """Validate task configuration data."""
        return ModelValidator.validate_model(TaskConfig, data)

    @staticmethod
    def validate_global_config(data: dict[str, Any]) -> ValidationResult:
        """Validate global configuration data."""
        return ModelValidator.validate_model(GlobalConfig, data)

    @staticmethod
    def validate_proxy_config(data: dict[str, Any]) -> ValidationResult:
        """Validate proxy configuration data."""
        return ModelValidator.validate_model(ProxyConfig, data)


class URLValidator:
    """URL validation utilities."""

    SUPPORTED_SCHEMES = {
        "http", "https", "ftp", "ftps", "sftp", "magnet", "file"
    }

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid and supported."""
        try:
            parsed = urlparse(url)
            scheme = parsed.scheme.lower()

            if scheme not in URLValidator.SUPPORTED_SCHEMES:
                return False

            # Special validation for different schemes
            if scheme == "magnet":
                # Magnet URLs don't need netloc, but should have xt parameter
                return "xt=" in url
            elif scheme in ("http", "https", "ftp", "ftps", "sftp"):
                # These schemes need netloc (host)
                return bool(parsed.netloc)
            elif scheme == "file":
                # File URLs should have a path
                return bool(parsed.path)

            return True
        except Exception:
            return False

    @staticmethod
    def get_url_scheme(url: str) -> str | None:
        """Extract scheme from URL."""
        try:
            return urlparse(url).scheme.lower()
        except Exception:
            return None

    @staticmethod
    def is_torrent_url(url: str) -> bool:
        """Check if URL is a torrent magnet link or .torrent file (local or remote)."""
        return (url.startswith("magnet:") or
                (URLValidator.is_valid_url(url) and url.lower().endswith('.torrent')))

    @staticmethod
    def is_media_url(url: str) -> bool:
        """Check if URL might be a media URL (basic heuristic)."""
        if not URLValidator.is_valid_url(url):
            return False

        # Common video/media domains (basic check)
        media_domains = {
            "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
            "twitch.tv", "soundcloud.com", "bandcamp.com"
        }

        try:
            domain = urlparse(url).netloc.lower()
            return any(media_domain in domain for media_domain in media_domains)
        except Exception:
            return False


class FileSystemValidator:
    """File system validation utilities."""

    @staticmethod
    def is_valid_path(path: str | Path, must_exist: bool = False) -> bool:
        """Check if path is valid."""
        try:
            path_obj = Path(path)

            if must_exist:
                return path_obj.exists()

            # Check if parent directory exists or can be created
            parent = path_obj.parent
            return parent.exists() or FileSystemValidator._can_create_directory(parent)
        except Exception:
            return False

    @staticmethod
    def _can_create_directory(path: Path) -> bool:
        """Check if directory can be created."""
        try:
            # Try to create the directory structure
            path.mkdir(parents=True, exist_ok=True)
            return path.exists() and path.is_dir()
        except Exception:
            return False

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        if not filename.strip():
            return "download"

        # Remove invalid characters
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(invalid_chars, "_", filename)

        # Remove leading/trailing whitespace and dots
        sanitized = sanitized.strip(" .")

        # Ensure filename is not empty and not too long
        if not sanitized:
            sanitized = "download"
        elif len(sanitized) > 255:
            sanitized = sanitized[:255]

        return sanitized

    @staticmethod
    def get_available_space(path: str | Path) -> int | None:
        """Get available disk space in bytes."""
        try:
            import shutil
            return shutil.disk_usage(Path(path)).free
        except Exception:
            return None


class ConfigValidator:
    """Configuration validation utilities."""

    @staticmethod
    def validate_port(port: int) -> bool:
        """Validate port number range."""
        return 1 <= port <= 65535

    @staticmethod
    def validate_logging_level(level: str) -> bool:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        return level.upper() in valid_levels

    @staticmethod
    def validate_theme(theme: str) -> bool:
        """Validate theme selection."""
        valid_themes = {"light", "dark", "auto"}
        return theme.lower() in valid_themes

    @staticmethod
    def validate_user_agent(user_agent: str) -> bool:
        """Validate user agent string."""
        # Basic validation - not empty and reasonable length
        return bool(user_agent.strip()) and len(user_agent) <= 500


# Convenience functions for common validation tasks
def validate_download_task_data(data: dict[str, Any]) -> ValidationResult:
    """Validate download task data."""
    return ModelValidator.validate_download_task(data)


def validate_config_data(data: dict[str, Any], config_type: str = "global") -> ValidationResult:
    """
    Validate configuration data.
    
    Args:
        data: Configuration data to validate
        config_type: Type of config ("global", "task", "proxy")
        
    Returns:
        ValidationResult with validation status
    """
    validators = {
        "global": ModelValidator.validate_global_config,
        "task": ModelValidator.validate_task_config,
        "proxy": ModelValidator.validate_proxy_config,
    }

    validator = validators.get(config_type)
    if not validator:
        return ValidationResult(is_valid=False, errors=[f"Unknown config type: {config_type}"])

    return validator(data)


def is_supported_url(url: str) -> bool:
    """Check if URL is supported by the download manager."""
    return URLValidator.is_valid_url(url)


def get_engine_for_url(url: str) -> str | None:
    """Determine appropriate engine for URL."""
    if not URLValidator.is_valid_url(url):
        return None

    scheme = URLValidator.get_url_scheme(url)
    if not scheme:
        return None

    # Check for torrent URLs first (magnet or .torrent files)
    if URLValidator.is_torrent_url(url):
        return "torrent"

    # Check if it might be a media URL
    if scheme in ("http", "https") and URLValidator.is_media_url(url):
        return "media"

    # Default engine mapping
    engine_mapping = {
        "http": "http",
        "https": "http",
        "ftp": "ftp",
        "ftps": "ftp",
        "sftp": "ftp",
        "file": "http",  # File URLs can be handled by HTTP engine
    }

    return engine_mapping.get(scheme)
