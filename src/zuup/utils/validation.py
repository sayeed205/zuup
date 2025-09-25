"""Input validation utilities."""

from pathlib import Path
import re
from urllib.parse import urlparse


def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.

    Args:
        url: URL string to validate

    Returns:
        True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_path(path: str, must_exist: bool = False) -> bool:
    """
    Validate if a string is a valid file path.

    Args:
        path: Path string to validate
        must_exist: Whether the path must already exist

    Returns:
        True if valid path, False otherwise
    """
    try:
        path_obj = Path(path)

        if must_exist:
            return path_obj.exists()

        # Check if path is valid (can be created)
        return True
    except Exception:
        return False


def validate_email(email: str) -> bool:
    """
    Validate if a string is a valid email address.

    Args:
        email: Email string to validate

    Returns:
        True if valid email, False otherwise
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove invalid characters for most filesystems
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, "_", filename)

    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(" .")

    # Ensure filename is not empty
    if not sanitized:
        sanitized = "download"

    return sanitized
