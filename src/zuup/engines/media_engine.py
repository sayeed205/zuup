"""Media download engine using yt-dlp."""

from collections.abc import AsyncIterator
import logging
from urllib.parse import urlparse

from ..storage.models import DownloadTask, ProgressInfo
from .base import BaseDownloadEngine

logger = logging.getLogger(__name__)


class MediaEngine(BaseDownloadEngine):
    """Media download engine using yt-dlp."""

    def __init__(self) -> None:
        """Initialize Media engine."""
        super().__init__()
        logger.info("MediaEngine initialized")

    async def download(self, task: DownloadTask) -> AsyncIterator[ProgressInfo]:
        """
        Download using yt-dlp for media content.

        Args:
            task: Download task to execute

        Yields:
            Progress information updates
        """
        # Implementation will be added in task 4
        logger.info(f"Starting media download for task {task.id}")
        # Temporary implementation to satisfy type checker
        if False:  # pragma: no cover
            yield ProgressInfo()
        raise NotImplementedError("Media download will be implemented in task 4")

    async def pause(self, task_id: str) -> None:
        """Pause media download."""
        logger.info(f"Pausing media download {task_id}")
        raise NotImplementedError("Media pause will be implemented in task 4")

    async def resume(self, task_id: str) -> None:
        """Resume media download."""
        logger.info(f"Resuming media download {task_id}")
        raise NotImplementedError("Media resume will be implemented in task 4")

    async def cancel(self, task_id: str) -> None:
        """Cancel media download."""
        logger.info(f"Cancelling media download {task_id}")
        raise NotImplementedError("Media cancel will be implemented in task 4")

    def get_progress(self, task_id: str) -> ProgressInfo:
        """Get media download progress."""
        logger.info(f"Getting media progress for {task_id}")
        raise NotImplementedError("Media progress will be implemented in task 4")

    def supports_protocol(self, url: str) -> bool:
        """
        Check if URL is supported by yt-dlp.

        Args:
            url: URL to check

        Returns:
            True if supported by yt-dlp, False otherwise
        """
        try:
            # Common video/media platforms supported by yt-dlp
            media_domains = [
                "youtube.com",
                "youtu.be",
                "vimeo.com",
                "dailymotion.com",
                "twitch.tv",
                "soundcloud.com",
                "bandcamp.com",
                "spotify.com",
                "tiktok.com",
                "instagram.com",
                "twitter.com",
                "facebook.com",
            ]

            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www. prefix if present
            if domain.startswith("www."):
                domain = domain[4:]

            return any(media_domain in domain for media_domain in media_domains)
        except Exception:
            return False
