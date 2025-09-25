"""Common utility functions."""


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    if bytes_value == 0:
        return "0 B"

    BYTES_PER_UNIT = 1024
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    size = float(bytes_value)

    while size >= BYTES_PER_UNIT and unit_index < len(units) - 1:
        size /= BYTES_PER_UNIT
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def format_speed(bytes_per_second: float) -> str:
    """
    Format download speed into human-readable string.

    Args:
        bytes_per_second: Speed in bytes per second

    Returns:
        Formatted string (e.g., "1.5 MB/s")
    """
    return f"{format_bytes(int(bytes_per_second))}/s"


def format_duration(seconds: float | None) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds is None or seconds <= 0:
        return "Unknown"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def calculate_eta(downloaded: int, total: int, speed: float) -> float | None:
    """
    Calculate estimated time of arrival for download completion.

    Args:
        downloaded: Bytes already downloaded
        total: Total bytes to download
        speed: Current download speed in bytes per second

    Returns:
        ETA in seconds, or None if cannot be calculated
    """
    if speed <= 0 or total <= 0 or downloaded >= total:
        return None

    remaining_bytes = total - downloaded
    return remaining_bytes / speed
