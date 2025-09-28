"""Media download engine using yt-dlp."""

from collections.abc import AsyncIterator
import logging
from pathlib import Path
from urllib.parse import urlparse

from ..storage.models import DownloadTask, ProgressInfo
from .base import BaseDownloadEngine
from .format_extractor import FormatExtractor
from .media_downloader import MediaDownloader
from .media_models import MediaConfig

logger = logging.getLogger(__name__)


class MediaEngine(BaseDownloadEngine):
    """Media download engine using yt-dlp."""

    def __init__(self, config: MediaConfig | None = None) -> None:
        """Initialize Media engine."""
        super().__init__()
        
        # Create default config if none provided
        if config is None:
            config = MediaConfig(
                output_directory=Path.home() / "Downloads"
            )
        
        self.config = config
        self.extractor = FormatExtractor(config)
        self.downloader = MediaDownloader(config)
        
        logger.info("MediaEngine initialized")

    async def download(self, task: DownloadTask) -> AsyncIterator[ProgressInfo]:
        """
        Download using yt-dlp for media content.

        Args:
            task: Download task to execute

        Yields:
            Progress information updates
        """
        logger.info(f"Starting media download for task {task.id}")
        
        try:
            # Validate task
            self._validate_task(task)
            self._register_task_start(task.id)
            
            # Extract media information
            logger.info(f"Extracting media info for {task.url}")
            media_info = await self.extractor.extract_info(task.url)
            
            # Start download with progress tracking
            async for progress in self.downloader.download_media(media_info, task.id):
                # Convert to ProgressInfo and update internal tracking
                progress_info = self.downloader.convert_to_progress_info(progress)
                self._update_progress(task.id, progress_info)
                yield progress_info
                
        except Exception as e:
            logger.error(f"Media download failed for task {task.id}: {e}")
            # Create error progress info
            error_progress = ProgressInfo(
                downloaded_bytes=0,
                total_bytes=0,
                download_speed=0.0,
                status=task.status,
                error_message=str(e)
            )
            self._update_progress(task.id, error_progress)
            yield error_progress
            raise
        finally:
            self._unregister_task(task.id)

    async def pause(self, task_id: str) -> None:
        """Pause media download."""
        logger.info(f"Pausing media download {task_id}")
        try:
            await self.downloader.pause_download(task_id)
        except Exception as e:
            logger.error(f"Failed to pause media download {task_id}: {e}")
            raise

    async def resume(self, task_id: str) -> None:
        """Resume media download."""
        logger.info(f"Resuming media download {task_id}")
        # Note: Resume requires re-extracting info and restarting download
        # This is a limitation of yt-dlp - it doesn't support true pause/resume
        # The downloader will handle resume by restarting the download
        logger.warning(f"Resume for media downloads restarts the download: {task_id}")

    async def cancel(self, task_id: str) -> None:
        """Cancel media download."""
        logger.info(f"Cancelling media download {task_id}")
        try:
            await self.downloader.cancel_download(task_id)
        except Exception as e:
            logger.error(f"Failed to cancel media download {task_id}: {e}")
            raise

    def get_progress(self, task_id: str) -> ProgressInfo:
        """Get media download progress."""
        try:
            return self._task_progress[task_id]
        except KeyError:
            logger.error(f"No progress found for task {task_id}")
            raise

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

    async def cleanup(self) -> None:
        """Clean up engine resources."""
        logger.info("Cleaning up MediaEngine")
        
        # Clean up downloader resources
        await self.downloader.cleanup()
        
        # Call parent cleanup
        await super().cleanup()
        
        logger.info("MediaEngine cleanup completed")
