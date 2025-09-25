"""BitTorrent download engine using libtorrent."""

from collections.abc import AsyncIterator
import logging
from urllib.parse import urlparse

from ..storage.models import DownloadTask, ProgressInfo
from .base import BaseDownloadEngine

logger = logging.getLogger(__name__)


class TorrentEngine(BaseDownloadEngine):
    """BitTorrent download engine using libtorrent."""

    def __init__(self) -> None:
        """Initialize Torrent engine."""
        super().__init__()
        logger.info("TorrentEngine initialized")

    async def download(self, task: DownloadTask) -> AsyncIterator[ProgressInfo]:
        """
        Download using BitTorrent protocol.

        Args:
            task: Download task to execute

        Yields:
            Progress information updates
        """
        # Implementation will be added in task 4
        logger.info(f"Starting torrent download for task {task.id}")
        # Temporary implementation to satisfy type checker
        if False:  # pragma: no cover
            yield ProgressInfo()
        raise NotImplementedError("Torrent download will be implemented in task 4")

    async def pause(self, task_id: str) -> None:
        """Pause torrent download."""
        logger.info(f"Pausing torrent download {task_id}")
        raise NotImplementedError("Torrent pause will be implemented in task 4")

    async def resume(self, task_id: str) -> None:
        """Resume torrent download."""
        logger.info(f"Resuming torrent download {task_id}")
        raise NotImplementedError("Torrent resume will be implemented in task 4")

    async def cancel(self, task_id: str) -> None:
        """Cancel torrent download."""
        logger.info(f"Cancelling torrent download {task_id}")
        raise NotImplementedError("Torrent cancel will be implemented in task 4")

    def get_progress(self, task_id: str) -> ProgressInfo:
        """Get torrent download progress."""
        logger.info(f"Getting torrent progress for {task_id}")
        raise NotImplementedError("Torrent progress will be implemented in task 4")

    def supports_protocol(self, url: str) -> bool:
        """
        Check if URL is a torrent (magnet link or .torrent file).

        Args:
            url: URL to check

        Returns:
            True if torrent, False otherwise
        """
        try:
            # Check for magnet links
            if url.lower().startswith("magnet:"):
                return True

            # Check for .torrent files
            parsed = urlparse(url)
            return parsed.path.lower().endswith(".torrent")
        except Exception:
            return False
