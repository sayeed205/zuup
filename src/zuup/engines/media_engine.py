"""Media download engine using yt-dlp."""

from collections.abc import AsyncIterator
import logging
from pathlib import Path
from urllib.parse import urlparse

from ..storage.models import DownloadTask, ProgressInfo, TaskStatus
from .base import BaseDownloadEngine
from .format_extractor import FormatExtractor
from .media_downloader import MediaDownloader
from .media_models import BatchDownloadConfig, BatchProgress, MediaConfig, MediaInfo
from .playlist_manager import PlaylistManager

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
        self.playlist_manager = PlaylistManager(config)

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

            # Extract media information to determine if it's a playlist
            logger.info(f"Extracting media info for {task.url}")
            media_info = await self.extractor.extract_info(task.url)

            # Check if this is a playlist or single video
            if media_info.is_playlist or self._is_playlist_url(task.url):
                # Handle as playlist/batch download
                async for progress in self._download_playlist(task):
                    yield progress
            else:
                # Handle as single video download
                async for progress in self._download_single_video(task, media_info):
                    yield progress

        except Exception as e:
            logger.error(f"Media download failed for task {task.id}: {e}")
            # Create error progress info
            error_progress = ProgressInfo(
                downloaded_bytes=0,
                total_bytes=0,
                download_speed=0.0,
                status=TaskStatus.FAILED,
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

    async def _download_single_video(self, task: DownloadTask, media_info: MediaInfo) -> AsyncIterator[ProgressInfo]:
        """
        Download a single video.

        Args:
            task: Download task
            media_info: Extracted media information

        Yields:
            Progress information updates
        """
        # Start download with progress tracking
        async for progress in self.downloader.download_media(media_info, task.id):
            # Convert to ProgressInfo and update internal tracking
            progress_info = self.downloader.convert_to_progress_info(progress)
            self._update_progress(task.id, progress_info)
            yield progress_info

    async def _download_playlist(self, task: DownloadTask) -> AsyncIterator[ProgressInfo]:
        """
        Download a playlist/batch of videos.

        Args:
            task: Download task

        Yields:
            Progress information updates
        """
        # Extract playlist information
        playlist_info = await self.playlist_manager.extract_playlist(task.url)

        # Create batch download configuration
        batch_config = BatchDownloadConfig(
            concurrent_downloads=self.config.concurrent_downloads,
            delay_between_downloads=self.config.delay_between_downloads,
            skip_existing=True,
            continue_on_error=True,
            max_failures=5,
            archive_file=self.config.download_archive,
        )

        # Download playlist with batch progress reporting
        async for batch_progress in self.playlist_manager.download_playlist(playlist_info, batch_config):
            # Convert BatchProgress to ProgressInfo
            progress_info = self._convert_batch_to_progress_info(batch_progress, task)
            self._update_progress(task.id, progress_info)
            yield progress_info

    def _is_playlist_url(self, url: str) -> bool:
        """
        Check if URL is likely a playlist based on URL patterns.

        Args:
            url: URL to check

        Returns:
            True if URL appears to be a playlist
        """
        # Common playlist URL patterns
        playlist_patterns = [
            "playlist",
            "list=",
            "channel",
            "user/",
            "c/",
            "@",  # YouTube handles
            "/sets/",  # SoundCloud sets
            "/albums/",  # Various platforms
        ]

        url_lower = url.lower()
        return any(pattern in url_lower for pattern in playlist_patterns)

    def _convert_batch_to_progress_info(self, batch_progress: BatchProgress, _task: DownloadTask) -> ProgressInfo:
        """
        Convert BatchProgress to ProgressInfo for engine interface.

        Args:
            batch_progress: Batch progress information
            _task: Download task (unused but kept for interface consistency)

        Returns:
            Generic ProgressInfo for engine interface
        """

        # Determine status based on batch progress
        if batch_progress.completed_items == batch_progress.total_items:
            status = TaskStatus.COMPLETED
        elif batch_progress.failed_items > 0 and batch_progress.remaining_items == 0:
            # All remaining items failed
            status = TaskStatus.FAILED
        else:
            status = TaskStatus.DOWNLOADING

        # Estimate total bytes if not available
        total_bytes = batch_progress.overall_total_bytes
        if total_bytes is None and batch_progress.total_items > 0:
            # Rough estimate: 50MB per video on average
            total_bytes = batch_progress.total_items * 50 * 1024 * 1024

        return ProgressInfo(
            downloaded_bytes=batch_progress.overall_downloaded_bytes,
            total_bytes=total_bytes,
            download_speed=0.0,  # Batch downloads don't have a single speed
            status=status,
            error_message=None,
        )

    async def cleanup(self) -> None:
        """Clean up engine resources."""
        logger.info("Cleaning up MediaEngine")

        # Clean up playlist manager resources
        await self.playlist_manager.cleanup()

        # Clean up downloader resources
        await self.downloader.cleanup()

        # Call parent cleanup
        await super().cleanup()

        logger.info("MediaEngine cleanup completed")
