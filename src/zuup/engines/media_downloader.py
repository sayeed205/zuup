"""Media downloader with yt-dlp integration and progress tracking."""

import asyncio
from collections.abc import AsyncIterator
import contextlib
from datetime import timedelta
import logging
from typing import Any

import yt_dlp  # type: ignore[import-untyped]

from ..storage.models import ProgressInfo, TaskStatus
from .media_models import (
    DownloadProgress,
    DownloadStatus,
    MediaConfig,
    MediaInfo,
)

logger = logging.getLogger(__name__)

# Constants for progress filtering
PROGRESS_BYTES_THRESHOLD = 1024 * 1024  # 1MB
PROGRESS_PERCENT_THRESHOLD = 1.0  # 1%
SPEED_CHANGE_THRESHOLD = 10.0  # 10%


class MediaDownloader:
    """Handles media downloads with yt-dlp and real-time progress tracking."""

    def __init__(self, config: MediaConfig) -> None:
        """
        Initialize MediaDownloader with configuration.

        Args:
            config: Media configuration for downloads
        """
        self.config = config
        self.progress_queue: asyncio.Queue[DownloadProgress] = asyncio.Queue()
        self._download_states: dict[str, DownloadStatus] = {}
        self._download_tasks: dict[str, asyncio.Task[None]] = {}
        self._yt_dlp_opts = self._create_yt_dlp_options()
        logger.info("MediaDownloader initialized")

    def _create_yt_dlp_options(self) -> dict[str, Any]:
        """
        Create yt-dlp options from configuration.

        Returns:
            Dictionary of yt-dlp options
        """
        opts = {
            # Output settings
            "outtmpl": str(self.config.output_directory / self.config.output_template),
            "restrictfilenames": False,
            "windowsfilenames": False,

            # Format selection
            "format": self.config.format_selector,

            # Network settings
            "socket_timeout": self.config.socket_timeout,
            "retries": self.config.retries,
            "fragment_retries": self.config.fragment_retries,

            # Progress hooks
            "progress_hooks": [self._progress_hook],

            # Proxy settings
            "proxy": self.config.proxy,

            # Geo-bypass settings
            "geo_bypass": self.config.geo_bypass,
            "geo_bypass_country": self.config.geo_bypass_country,

            # Extractor arguments
            "extractor_args": self.config.extractor_args,

            # Post-processing
            "writesubtitles": self.config.embed_subtitles,
            "writeautomaticsub": self.config.embed_subtitles,
            "writethumbnail": self.config.embed_thumbnail,
            "embedsubtitles": self.config.embed_subtitles,
            "embedthumbnail": self.config.embed_thumbnail,

            # Download behavior
            "ignoreerrors": self.config.ignore_errors,
            "download_archive": str(self.config.download_archive) if self.config.download_archive else None,
            "playliststart": self.config.playlist_start,
            "playlistend": self.config.playlist_end,
            "max_downloads": self.config.max_downloads,

            # Logging
            "quiet": False,
            "no_warnings": False,
        }

        # Add authentication if configured
        if self.config.auth_config.username:
            opts["username"] = self.config.auth_config.username
        if self.config.auth_config.password:
            opts["password"] = self.config.auth_config.password
        if self.config.auth_config.cookies_file:
            opts["cookiefile"] = str(self.config.auth_config.cookies_file)
        if self.config.auth_config.netrc_file:
            opts["usenetrc"] = True
            opts["netrc_location"] = str(self.config.auth_config.netrc_file)

        # Audio extraction settings
        if self.config.extract_audio:
            opts["format"] = "bestaudio/best"
            opts["postprocessors"] = [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": self.config.audio_format or "mp3",
                "preferredquality": self.config.audio_quality,
            }]

        return opts

    def _progress_hook(self, d: dict[str, Any]) -> None:
        """
        yt-dlp progress hook for real-time updates.

        Args:
            d: Progress dictionary from yt-dlp
        """
        try:
            progress = self._parse_progress_dict(d)
            # Put progress in queue for async consumption
            task = asyncio.create_task(self.progress_queue.put(progress))
            # Store reference to prevent garbage collection
            task.add_done_callback(lambda _: None)
        except Exception as e:
            logger.error(f"Error in progress hook: {e}")

    def _parse_progress_dict(self, d: dict[str, Any]) -> DownloadProgress:
        """
        Parse yt-dlp progress dictionary into DownloadProgress model.

        Args:
            d: Progress dictionary from yt-dlp

        Returns:
            Parsed DownloadProgress object
        """
        status_map = {
            "downloading": DownloadStatus.DOWNLOADING,
            "finished": DownloadStatus.FINISHED,
            "error": DownloadStatus.ERROR,
            "extracting": DownloadStatus.EXTRACTING,
        }

        status = status_map.get(d.get("status", "downloading"), DownloadStatus.DOWNLOADING)

        # Extract progress information
        downloaded_bytes = d.get("downloaded_bytes", 0) or 0
        total_bytes = d.get("total_bytes") or d.get("total_bytes_estimate")
        download_speed = d.get("speed") or 0.0
        filename = d.get("filename")

        # Fragment information for segmented downloads
        fragment_index = d.get("fragment_index")
        fragment_count = d.get("fragment_count")

        # Calculate ETA
        eta = None
        if download_speed and total_bytes and downloaded_bytes < total_bytes:
            remaining_bytes = total_bytes - downloaded_bytes
            eta_seconds = remaining_bytes / download_speed
            eta = timedelta(seconds=eta_seconds)

        return DownloadProgress(
            status=status,
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            download_speed=download_speed,
            eta=eta,
            filename=filename,
            fragment_index=fragment_index,
            fragment_count=fragment_count,
        )

    async def download_media(
        self,
        info: MediaInfo,
        task_id: str,
        format_spec: str | None = None
    ) -> AsyncIterator[DownloadProgress]:
        """
        Download media with real-time progress tracking.

        Args:
            info: Media information from extraction
            task_id: Unique task identifier
            format_spec: Optional format specification override

        Yields:
            DownloadProgress updates

        Raises:
            RuntimeError: If download fails
        """
        logger.info(f"Starting media download for task {task_id}: {info.title}")

        # Set initial state
        self._download_states[task_id] = DownloadStatus.DOWNLOADING

        try:
            # Create task-specific options
            opts = self._yt_dlp_opts.copy()
            if format_spec:
                opts["format"] = format_spec

            # Create download task
            download_task = asyncio.create_task(
                self._download_with_yt_dlp(info.webpage_url, opts, task_id)
            )
            self._download_tasks[task_id] = download_task

            # Yield progress updates
            async for progress in self._stream_progress(task_id):
                yield progress

                # Check if download is complete or failed
                if progress.status in (DownloadStatus.FINISHED, DownloadStatus.ERROR):
                    break

                # Check if task was cancelled
                if self._download_states.get(task_id) == DownloadStatus.CANCELLED:
                    break

            # Wait for download task to complete
            await download_task

        except asyncio.CancelledError:
            logger.info(f"Download cancelled for task {task_id}")
            self._download_states[task_id] = DownloadStatus.CANCELLED
            raise
        except Exception as e:
            logger.error(f"Download failed for task {task_id}: {e}")
            self._download_states[task_id] = DownloadStatus.ERROR
            raise RuntimeError(f"Media download failed: {e}") from e
        finally:
            # Cleanup
            self._cleanup_task(task_id)

    async def _download_with_yt_dlp(
        self,
        url: str,
        opts: dict[str, Any],
        task_id: str
    ) -> None:
        """
        Execute yt-dlp download in thread pool.

        Args:
            url: URL to download
            opts: yt-dlp options
            task_id: Task identifier
        """
        def download_sync() -> None:
            """Synchronous download function for thread pool."""
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])

                # Mark as finished if not already set
                if self._download_states.get(task_id) != DownloadStatus.CANCELLED:
                    self._download_states[task_id] = DownloadStatus.FINISHED

            except yt_dlp.DownloadError as e:
                logger.error(f"yt-dlp download error for task {task_id}: {e}")
                self._download_states[task_id] = DownloadStatus.ERROR
                raise
            except Exception as e:
                logger.error(f"Unexpected download error for task {task_id}: {e}")
                self._download_states[task_id] = DownloadStatus.ERROR
                raise

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, download_sync)

    async def _stream_progress(self, task_id: str) -> AsyncIterator[DownloadProgress]:
        """
        Stream progress updates for a download task.

        Args:
            task_id: Task identifier

        Yields:
            DownloadProgress updates
        """
        last_progress = None

        while True:
            try:
                # Wait for progress update with timeout
                progress = await asyncio.wait_for(
                    self.progress_queue.get(),
                    timeout=1.0
                )

                # Only yield if progress has changed significantly
                if self._should_yield_progress(progress, last_progress):
                    yield progress
                    last_progress = progress

                # Check for completion
                if progress.status in (DownloadStatus.FINISHED, DownloadStatus.ERROR):
                    break

            except asyncio.TimeoutError:
                # Check if task is still active
                current_state = self._download_states.get(task_id)
                if current_state in (DownloadStatus.FINISHED, DownloadStatus.ERROR, DownloadStatus.CANCELLED):
                    # Create final progress update
                    final_progress = DownloadProgress(
                        status=current_state,
                        downloaded_bytes=last_progress.downloaded_bytes if last_progress else 0,
                        total_bytes=last_progress.total_bytes if last_progress else None,
                        download_speed=0.0,
                        filename=last_progress.filename if last_progress else None,
                    )
                    yield final_progress
                    break

                # Continue waiting for progress
                continue

    def _should_yield_progress(
        self,
        current: DownloadProgress,
        last: DownloadProgress | None
    ) -> bool:
        """
        Determine if progress update should be yielded.

        Args:
            current: Current progress
            last: Last yielded progress

        Returns:
            True if progress should be yielded
        """
        if last is None:
            return True

        # Always yield status changes
        if current.status != last.status:
            return True

        # Yield if significant bytes downloaded (1MB or 1% change)
        if current.downloaded_bytes != last.downloaded_bytes:
            bytes_diff = abs(current.downloaded_bytes - last.downloaded_bytes)

            # Check for 1MB threshold
            if bytes_diff >= 1024 * 1024:  # 1MB
                return True

            # Check for 1% threshold
            if current.total_bytes and current.total_bytes > 0:
                percent_diff = (bytes_diff / current.total_bytes) * 100
                if percent_diff >= 1.0:  # 1%
                    return True

        # Yield if speed changed significantly (>10%)
        if current.download_speed and last.download_speed and last.download_speed > 0:
            speed_diff = abs(current.download_speed - last.download_speed)
            speed_change_percent = (speed_diff / last.download_speed) * 100
            if speed_change_percent >= SPEED_CHANGE_THRESHOLD:
                return True

        return False

    async def pause_download(self, task_id: str) -> None:
        """
        Pause a download task.

        Args:
            task_id: Task identifier

        Raises:
            ValueError: If task not found or cannot be paused
        """
        if task_id not in self._download_tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self._download_tasks[task_id]
        if task.done():
            raise ValueError(f"Task {task_id} is already completed")

        logger.info(f"Pausing download task {task_id}")

        # Cancel the download task
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Update state
        self._download_states[task_id] = DownloadStatus.PAUSED

    async def resume_download(
        self,
        task_id: str,
        info: MediaInfo,
        format_spec: str | None = None
    ) -> AsyncIterator[DownloadProgress]:
        """
        Resume a paused download task.

        Args:
            task_id: Task identifier
            info: Media information
            format_spec: Optional format specification

        Yields:
            DownloadProgress updates

        Raises:
            ValueError: If task not found or cannot be resumed
        """
        if task_id not in self._download_states:
            raise ValueError(f"Task {task_id} not found")

        if self._download_states[task_id] != DownloadStatus.PAUSED:
            raise ValueError(f"Task {task_id} is not paused")

        logger.info(f"Resuming download task {task_id}")

        # Resume download (yt-dlp handles resume automatically)
        async for progress in self.download_media(info, task_id, format_spec):
            yield progress

    def is_playlist_supported(self, url: str) -> bool:
        """
        Check if URL is a supported playlist format.

        Args:
            url: URL to check

        Returns:
            True if playlist is supported
        """
        # Use the format extractor to check support
        return self.extractor.supports_url(url) if hasattr(self, 'extractor') else True

    async def cancel_download(self, task_id: str) -> None:
        """
        Cancel a download task.

        Args:
            task_id: Task identifier

        Raises:
            ValueError: If task not found
        """
        if task_id not in self._download_tasks:
            raise ValueError(f"Task {task_id} not found")

        logger.info(f"Cancelling download task {task_id}")

        # Update state first
        self._download_states[task_id] = DownloadStatus.CANCELLED

        # Cancel the download task
        task = self._download_tasks[task_id]
        if not task.done():
            task.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await task

    def get_download_state(self, task_id: str) -> DownloadStatus:
        """
        Get current download state for a task.

        Args:
            task_id: Task identifier

        Returns:
            Current download status

        Raises:
            KeyError: If task not found
        """
        if task_id not in self._download_states:
            raise KeyError(f"Task {task_id} not found")

        return self._download_states[task_id]

    def _cleanup_task(self, task_id: str) -> None:
        """
        Clean up task resources.

        Args:
            task_id: Task identifier to clean up
        """
        self._download_tasks.pop(task_id, None)
        # Keep state for potential resume operations
        logger.debug(f"Cleaned up resources for task {task_id}")

    def setup_progress_hooks(self) -> None:
        """Configure yt-dlp progress hooks for real-time updates."""
        # Progress hooks are already set up in _create_yt_dlp_options
        logger.debug("Progress hooks already configured")

    def create_yt_dlp_options(self) -> dict[str, Any]:
        """
        Get current yt-dlp options.

        Returns:
            Dictionary of yt-dlp options
        """
        return self._yt_dlp_opts.copy()

    async def cleanup(self) -> None:
        """Clean up downloader resources and cancel active downloads."""
        logger.info("Cleaning up MediaDownloader")

        # Cancel all active downloads
        for task_id in list(self._download_tasks.keys()):
            try:
                await self.cancel_download(task_id)
            except Exception as e:
                logger.error(f"Error cancelling task {task_id} during cleanup: {e}")

        # Clear all state
        self._download_states.clear()
        self._download_tasks.clear()

        # Clear progress queue
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("MediaDownloader cleanup completed")

    def convert_to_progress_info(self, progress: DownloadProgress) -> ProgressInfo:
        """
        Convert DownloadProgress to ProgressInfo for engine interface.

        Args:
            progress: Media-specific progress information

        Returns:
            Generic ProgressInfo for engine interface
        """
        # Map media status to task status
        status_map = {
            DownloadStatus.DOWNLOADING: TaskStatus.DOWNLOADING,
            DownloadStatus.FINISHED: TaskStatus.COMPLETED,
            DownloadStatus.ERROR: TaskStatus.FAILED,
            DownloadStatus.EXTRACTING: TaskStatus.DOWNLOADING,
            DownloadStatus.PAUSED: TaskStatus.PAUSED,
            DownloadStatus.CANCELLED: TaskStatus.CANCELLED,
        }

        return ProgressInfo(
            downloaded_bytes=progress.downloaded_bytes,
            total_bytes=progress.total_bytes,
            download_speed=progress.download_speed or 0.0,
            eta=progress.eta,
            status=status_map.get(progress.status, TaskStatus.DOWNLOADING),
            error_message=None,  # Error details would be in exception
        )
