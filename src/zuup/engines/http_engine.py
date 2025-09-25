"""HTTP/HTTPS download engine using pycurl."""

from collections.abc import AsyncIterator
import logging
from urllib.parse import urlparse

from ..storage.models import DownloadTask, ProgressInfo
from .base import BaseDownloadEngine

logger = logging.getLogger(__name__)


class HTTPEngine(BaseDownloadEngine):
    """HTTP/HTTPS download engine using pycurl."""

    def __init__(self) -> None:
        """Initialize HTTP engine."""
        super().__init__()
        logger.info("HTTPEngine initialized")

    async def download(self, task: DownloadTask) -> AsyncIterator[ProgressInfo]:
        """
        Download using HTTP/HTTPS protocol.

        Args:
            task: Download task to execute

        Yields:
            Progress information updates
        """
        # Implementation will be added in later tasks
        logger.info(f"Starting HTTP download for task {task.id}")
        raise NotImplementedError("HTTP download will be implemented in later tasks")
        yield  # type: ignore[unreachable]

    async def pause(self, task_id: str) -> None:
        """Pause HTTP download."""
        logger.info(f"Pausing HTTP download {task_id}")
        raise NotImplementedError("HTTP pause will be implemented in task 4")

    async def resume(self, task_id: str) -> None:
        """Resume HTTP download."""
        logger.info(f"Resuming HTTP download {task_id}")
        raise NotImplementedError("HTTP resume will be implemented in task 4")

    async def cancel(self, task_id: str) -> None:
        """Cancel HTTP download."""
        logger.info(f"Cancelling HTTP download {task_id}")
        raise NotImplementedError("HTTP cancel will be implemented in task 4")

    def get_progress(self, task_id: str) -> ProgressInfo:
        """Get HTTP download progress."""
        logger.info(f"Getting HTTP progress for {task_id}")
        raise NotImplementedError("HTTP progress will be implemented in task 4")

    def supports_protocol(self, url: str) -> bool:
        """
        Check if URL uses HTTP/HTTPS protocol.

        Args:
            url: URL to check

        Returns:
            True if HTTP/HTTPS, False otherwise
        """
        try:
            parsed = urlparse(url)
            return parsed.scheme.lower() in ("http", "https")
        except Exception:
            return False
