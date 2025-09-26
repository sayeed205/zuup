"""Logging configuration utilities and structured logging system."""

from datetime import datetime
import json
import logging
import logging.handlers
from pathlib import Path
import sys
import traceback
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from ..storage.models import EngineType, ProgressInfo


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "task_id"):
            log_entry["task_id"] = record.task_id
        if hasattr(record, "engine_type"):
            log_entry["engine_type"] = record.engine_type
        if hasattr(record, "download_speed"):
            log_entry["download_speed"] = record.download_speed
        if hasattr(record, "progress_percentage"):
            log_entry["progress_percentage"] = record.progress_percentage
        if hasattr(record, "error_code"):
            log_entry["error_code"] = record.error_code
        if hasattr(record, "url"):
            log_entry["url"] = record.url

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry, default=str)


class DownloadLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for download-specific logging with context."""

    def __init__(self, logger: logging.Logger, task_id: str, engine_type: EngineType):
        self.task_id = task_id
        self.engine_type = engine_type
        super().__init__(logger, {"task_id": task_id, "engine_type": engine_type.value})

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Process log message and add context."""
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs

    def log_progress(self, progress: ProgressInfo, url: str = "") -> None:
        """Log download progress information."""
        extra = {
            "download_speed": progress.download_speed,
            "progress_percentage": progress.progress_percentage,
            "downloaded_bytes": progress.downloaded_bytes,
            "total_bytes": progress.total_bytes,
            "status": progress.status.value,
            "url": url,
        }

        if progress.upload_speed is not None:
            extra["upload_speed"] = progress.upload_speed
        if progress.peers_connected is not None:
            extra["peers_connected"] = progress.peers_connected
        if progress.ratio is not None:
            extra["ratio"] = progress.ratio

        self.info(
            f"Progress: {progress.progress_percentage:.1f}% "
            f"({progress.downloaded_bytes}/{progress.total_bytes or 'unknown'} bytes) "
            f"Speed: {progress.download_speed / 1024 / 1024:.2f} MB/s",
            extra=extra,
        )

    def log_error(self, error: Exception, error_code: str = "", url: str = "") -> None:
        """Log download error with context."""
        extra = {
            "error_code": error_code,
            "error_type": type(error).__name__,
            "url": url,
        }

        self.error(f"Download error: {error}", extra=extra, exc_info=True)

    def log_completion(self, final_size: int, duration: float, url: str = "") -> None:
        """Log download completion."""
        extra = {
            "final_size": final_size,
            "duration_seconds": duration,
            "average_speed": final_size / duration if duration > 0 else 0,
            "url": url,
        }

        self.info(
            f"Download completed: {final_size} bytes in {duration:.2f}s "
            f"(avg: {extra['average_speed'] / 1024 / 1024:.2f} MB/s)",
            extra=extra,
        )


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    rich_console: bool = True,
    structured_logging: bool = False,
    max_log_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Set up comprehensive logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        rich_console: Whether to use rich console handler
        structured_logging: Whether to use JSON structured logging for files
        max_log_size: Maximum size of log files before rotation
        backup_count: Number of backup log files to keep
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Create formatters
    standard_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if rich_console:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            console=Console(stderr=True),
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_handler.setLevel(numeric_level)
        root_logger.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(standard_formatter)
        console_handler.setLevel(numeric_level)
        root_logger.addHandler(console_handler)

    # File handler with rotation (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler to prevent huge log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_log_size, backupCount=backup_count, encoding="utf-8"
        )

        if structured_logging:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(detailed_formatter)

        file_handler.setLevel(logging.DEBUG)  # File logs capture everything
        root_logger.addHandler(file_handler)

        # Separate error log file
        error_log_file = log_file.parent / f"{log_file.stem}_errors{log_file.suffix}"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_log_size,
            backupCount=backup_count,
            encoding="utf-8",
        )

        if structured_logging:
            error_handler.setFormatter(StructuredFormatter())
        else:
            error_handler.setFormatter(detailed_formatter)

        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)

    # Set third-party library log levels to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    # Set specific levels for download engines
    logging.getLogger("zuup.engines").setLevel(logging.INFO)
    logging.getLogger("zuup.core").setLevel(logging.INFO)
    logging.getLogger("zuup.storage").setLevel(logging.INFO)


def get_download_logger(task_id: str, engine_type: EngineType) -> DownloadLoggerAdapter:
    """
    Get a logger adapter for download tasks with context.

    Args:
        task_id: Unique task identifier
        engine_type: Type of download engine

    Returns:
        Logger adapter with download context
    """
    logger = logging.getLogger(f"zuup.engines.{engine_type.value}")
    return DownloadLoggerAdapter(logger, task_id, engine_type)


def log_system_info() -> None:
    """Log system information for debugging."""
    import platform

    import psutil

    logger = logging.getLogger("zuup.system")

    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    logger.info(f"Disk space: {psutil.disk_usage('/').free / 1024**3:.1f} GB free")


def setup_debug_logging(log_dir: Path) -> None:
    """
    Set up debug logging configuration for development.

    Args:
        log_dir: Directory to store debug logs
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level="DEBUG",
        log_file=log_dir / "zuup_debug.log",
        rich_console=True,
        structured_logging=True,
        max_log_size=50 * 1024 * 1024,  # 50MB for debug logs
        backup_count=3,
    )

    # Log system info at startup
    log_system_info()

    logger = logging.getLogger("zuup.debug")
    logger.info("Debug logging initialized")


class LogCapture:
    """Context manager for capturing logs during testing."""

    def __init__(self, logger_name: str = "", level: int = logging.INFO):
        self.logger_name = logger_name
        self.level = level
        self.records: list[logging.LogRecord] = []
        self.handler: logging.Handler | None = None

    def __enter__(self) -> "LogCapture":
        """Start capturing logs."""
        self.handler = logging.Handler()
        self.handler.emit = self.records.append
        self.handler.setLevel(self.level)

        logger = logging.getLogger(self.logger_name)
        logger.addHandler(self.handler)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop capturing logs."""
        if self.handler:
            logger = logging.getLogger(self.logger_name)
            logger.removeHandler(self.handler)

    def get_messages(self) -> list[str]:
        """Get captured log messages."""
        return [record.getMessage() for record in self.records]

    def has_message_containing(self, text: str) -> bool:
        """Check if any captured message contains the given text."""
        return any(text in record.getMessage() for record in self.records)
