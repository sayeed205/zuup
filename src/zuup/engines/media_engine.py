"""Media download engine using yt-dlp."""

import asyncio
from collections.abc import AsyncIterator
import logging
from pathlib import Path
from urllib.parse import urlparse

from ..storage.models import DownloadTask, ProgressInfo, TaskStatus
from .base import BaseDownloadEngine
from .format_extractor import FormatExtractor
from .media_downloader import MediaDownloader
from .media_error_handler import (
    DownloadError,
    ExtractionError,
    MediaErrorHandler,
)
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
            config = MediaConfig(output_directory=Path.home() / "Downloads")

        self.config = config
        self.extractor = FormatExtractor(config)
        self.downloader = MediaDownloader(config)
        self.playlist_manager = PlaylistManager(config)

        # Initialize error handler with configuration
        self.error_handler = MediaErrorHandler(
            max_retry_attempts=config.retries,
            base_retry_delay=1.0,
            max_retry_delay=60.0,
            backoff_factor=2.0,
            enable_fallback_extractors=True,
            enable_format_alternatives=True,
        )

        logger.info("MediaEngine initialized with error handling")

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

            # Extract media information with error handling
            media_info = await self._extract_with_retry(task.url)

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
                error_message=str(e),
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

    async def _download_single_video(
        self, task: DownloadTask, media_info: MediaInfo
    ) -> AsyncIterator[ProgressInfo]:
        """
        Download a single video with error handling.

        Args:
            task: Download task
            media_info: Extracted media information

        Yields:
            Progress information updates
        """
        # Use error handling download method
        async for progress in self._download_with_error_handling(task, media_info):
            yield progress

    async def _download_playlist(
        self, task: DownloadTask
    ) -> AsyncIterator[ProgressInfo]:
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
        async for batch_progress in self.playlist_manager.download_playlist(
            playlist_info, batch_config
        ):
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

    def _convert_batch_to_progress_info(
        self, batch_progress: BatchProgress, _task: DownloadTask
    ) -> ProgressInfo:
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

    async def _extract_with_retry(self, url: str, max_attempts: int = 3) -> MediaInfo:
        """
        Extract media information with retry logic and error handling.

        Args:
            url: URL to extract information from
            max_attempts: Maximum number of extraction attempts

        Returns:
            Extracted media information

        Raises:
            ExtractionError: If extraction fails after all retries
        """
        last_error = None

        for attempt in range(max_attempts):
            try:
                logger.info(f"Extracting media info for {url} (attempt {attempt + 1})")
                return await self.extractor.extract_info(url)

            except Exception as e:
                extraction_error = ExtractionError(
                    message=str(e),
                    error_code=getattr(e, "code", None),
                    extractor=getattr(e, "extractor", None),
                )

                # Handle the error and determine action
                action = await self.error_handler.handle_extraction_error(
                    extraction_error, url, attempt
                )

                last_error = extraction_error

                # Check if we should retry
                if not self.error_handler.should_retry_extraction(
                    extraction_error, attempt
                ):
                    logger.error(f"Extraction failed permanently for {url}: {e}")
                    break

                # Handle different actions
                if action.name.startswith("RETRY"):
                    if action.name == "RETRY_WITH_FALLBACK":
                        # Try fallback extractor
                        fallback = self.error_handler.get_fallback_extractor(url)
                        if fallback:
                            logger.info(
                                f"Trying fallback extractor {fallback} for {url}"
                            )
                            # Note: Actual fallback implementation would require
                            # modifying the extractor to use specific extractors

                    # Calculate delay for retry
                    if action.name == "RETRY_WITH_DELAY":
                        # Use a default category for delay calculation
                        from .media_error_handler import MediaErrorCategory

                        delay = await self.error_handler.calculate_retry_delay(
                            attempt, MediaErrorCategory.EXTRACTION
                        )
                        logger.info(f"Retrying extraction after {delay:.1f}s delay")
                        await asyncio.sleep(delay)

                elif action.name == "FAIL_DOWNLOAD":
                    logger.error(f"Extraction failed permanently for {url}: {e}")
                    break

        # If we get here, all attempts failed
        if last_error:
            raise last_error
        else:
            raise ExtractionError(f"Failed to extract information from {url}")

    async def _download_with_error_handling(
        self, task: DownloadTask, media_info: MediaInfo
    ) -> AsyncIterator[ProgressInfo]:
        """
        Download with comprehensive error handling and format alternatives.

        Args:
            task: Download task
            media_info: Media information

        Yields:
            Progress information updates
        """
        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                # Start download with progress tracking
                async for progress in self.downloader.download_media(
                    media_info, task.id
                ):
                    # Convert to ProgressInfo and update internal tracking
                    progress_info = self.downloader.convert_to_progress_info(progress)
                    self._update_progress(task.id, progress_info)
                    yield progress_info

                # If we get here, download succeeded
                return

            except Exception as e:
                download_error = DownloadError(
                    message=str(e),
                    error_code=getattr(e, "code", None),
                    format_id=getattr(e, "format_id", None),
                )

                # Handle the error and determine action
                action = await self.error_handler.handle_download_error(
                    download_error, media_info, attempt
                )

                last_error = download_error

                # Handle different actions
                if action.name == "USE_ALTERNATIVE_FORMAT":
                    # Get current format and suggest alternatives
                    current_formats = media_info.formats
                    if current_formats:
                        failed_format = current_formats[
                            0
                        ]  # Assume first format was tried
                        alternatives = self.error_handler.suggest_format_alternatives(
                            failed_format, current_formats
                        )

                        if alternatives:
                            logger.info(
                                f"Trying alternative format: {alternatives[0].format_id}"
                            )
                            # Update media_info with alternative format
                            media_info.formats = alternatives
                            continue

                elif action.name == "REDUCE_QUALITY":
                    # Try to find lower quality formats
                    current_formats = media_info.formats
                    lower_quality_formats = [
                        fmt
                        for fmt in current_formats
                        if fmt.quality
                        and fmt.quality < (current_formats[0].quality or 100)
                    ]

                    if lower_quality_formats:
                        logger.info(
                            f"Trying lower quality format: {lower_quality_formats[0].format_id}"
                        )
                        media_info.formats = lower_quality_formats
                        continue

                elif action.name.startswith("RETRY"):
                    if action.name == "RETRY_WITH_DELAY":
                        from .media_error_handler import MediaErrorCategory

                        delay = await self.error_handler.calculate_retry_delay(
                            attempt, MediaErrorCategory.NETWORK
                        )
                        logger.info(f"Retrying download after {delay:.1f}s delay")
                        await asyncio.sleep(delay)
                    continue

                elif action.name in ["FAIL_DOWNLOAD", "SKIP_ITEM"]:
                    logger.error(f"Download failed permanently: {e}")
                    break

        # If we get here, all attempts failed
        if last_error:
            raise last_error
        else:
            raise DownloadError(f"Failed to download {media_info.title}")

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
