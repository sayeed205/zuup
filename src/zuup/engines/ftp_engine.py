"""FTP/SFTP download engine using paramiko and pycurl."""

from collections.abc import AsyncIterator
import logging
from urllib.parse import urlparse

from ..storage.models import DownloadTask, ProgressInfo
from .base import BaseDownloadEngine

logger = logging.getLogger(__name__)


class FTPEngine(BaseDownloadEngine):
    """FTP/SFTP download engine using paramiko and pycurl."""

    def __init__(self) -> None:
        """Initialize FTP engine."""
        super().__init__()
        logger.info("FTPEngine initialized")

    async def download(self, task: DownloadTask) -> AsyncIterator[ProgressInfo]:
        """
        Download using FTP/SFTP protocol.

        Args:
            task: Download task to execute

        Yields:
            Progress information updates
        """
        # Implementation will be added in later tasks
        logger.info(f"Starting FTP download for task {task.id}")
        raise NotImplementedError("FTP download will be implemented in later tasks")
        yield  # type: ignore[unreachable]

    async def pause(self, task_id: str) -> None:
        """Pause FTP download."""
        logger.info(f"Pausing FTP download {task_id}")
        raise NotImplementedError("FTP pause will be implemented in task 4")

    async def resume(self, task_id: str) -> None:
        """Resume FTP download."""
        logger.info(f"Resuming FTP download {task_id}")
        raise NotImplementedError("FTP resume will be implemented in task 4")

    async def cancel(self, task_id: str) -> None:
        """Cancel FTP download."""
        logger.info(f"Cancelling FTP download {task_id}")
        raise NotImplementedError("FTP cancel will be implemented in task 4")

    def get_progress(self, task_id: str) -> ProgressInfo:
        """Get FTP download progress."""
        logger.info(f"Getting FTP progress for {task_id}")
        raise NotImplementedError("FTP progress will be implemented in task 4")

    def supports_protocol(self, url: str) -> bool:
        """
        Check if URL uses FTP/SFTP protocol.

        Args:
            url: URL to check

        Returns:
            True if FTP/SFTP, False otherwise
        """
        try:
            parsed = urlparse(url)
            return parsed.scheme.lower() in ("ftp", "ftps", "sftp")
        except Exception:
            return False
