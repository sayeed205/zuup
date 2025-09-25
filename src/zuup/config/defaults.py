"""Default configuration values."""

from pathlib import Path

from .settings import GlobalConfig, ProxyConfig, TaskConfig


def get_default_global_config() -> GlobalConfig:
    """
    Get default global configuration.

    Returns:
        Default global configuration
    """
    return GlobalConfig(
        max_concurrent_downloads=3,
        default_download_path=Path.home() / "Downloads",
        temp_directory=Path.home() / ".cache" / "zuup",
        max_connections_per_download=8,
        user_agent="Zuup/0.1.0",
        proxy_settings=None,
        logging_level="INFO",
        server_host="127.0.0.1",
        server_port=8080,
        theme="dark",
        auto_start_downloads=True,
        show_notifications=True,
    )


def get_default_task_config() -> TaskConfig:
    """
    Get default task configuration.

    Returns:
        Default task configuration
    """
    return TaskConfig(
        max_connections=None,  # Use global default
        download_speed_limit=None,  # No limit
        upload_speed_limit=None,  # No limit
        retry_attempts=3,
        timeout=30,
        headers={},
        cookies={},
        enable_seeding=True,
        seed_ratio_limit=None,  # No limit
        seed_time_limit=None,  # No limit
        proxy=None,  # Use global proxy settings
    )


def get_default_proxy_config() -> ProxyConfig:
    """
    Get default proxy configuration.

    Returns:
        Default proxy configuration (disabled)
    """
    return ProxyConfig(
        enabled=False,
        http_proxy=None,
        https_proxy=None,
        socks_proxy=None,
        username=None,
        password=None,
    )
