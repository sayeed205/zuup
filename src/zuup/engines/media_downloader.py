"""Media downloader with yt-dlp integration and progress tracking."""

import asyncio
from collections.abc import AsyncIterator
import contextlib
from datetime import timedelta
import logging
from typing import Any
from urllib.parse import urlparse

import yt_dlp  # type: ignore[import-untyped]

from ..storage.models import ProgressInfo, TaskStatus
from .auth_error_handler import AuthenticationErrorHandler
from .cookie_manager import AuthenticationError, AuthenticationManager
from .format_selector import FormatSelector
from .quality_controller import QualityController
from .media_models import (
    DownloadProgress,
    DownloadStatus,
    FormatPreferences,
    MediaConfig,
    MediaFormat,
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
        
        # Initialize authentication components
        cookies_file = config.auth_config.cookies_file
        self.auth_manager = AuthenticationManager(cookies_file)
        self.auth_error_handler = AuthenticationErrorHandler(self.auth_manager)
        
        # Initialize quality control components
        self.format_selector = FormatSelector()
        self.quality_controller = QualityController(self.format_selector)
        
        # Track progress history for quality adaptation
        self._progress_history: dict[str, list[DownloadProgress]] = {}
        
        self._yt_dlp_opts = self._create_yt_dlp_options()
        logger.info("MediaDownloader initialized with authentication support")

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

        # Authentication will be set up dynamically per URL
        # This allows for URL-specific authentication and error handling

        # Audio extraction settings
        if self.config.extract_audio:
            opts["format"] = "bestaudio/best"
            opts["postprocessors"] = [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": self.config.audio_format or "mp3",
                "preferredquality": self.config.audio_quality,
            }]

        return opts
    
    async def _setup_authentication_for_url(self, url: str, opts: dict[str, Any]) -> dict[str, Any]:
        """
        Set up authentication for a specific URL.
        
        Args:
            url: URL being accessed
            opts: Current yt-dlp options
            
        Returns:
            Updated options with authentication
        """
        try:
            # Convert auth config to dictionary format
            auth_dict = {
                "method": self.config.auth_config.method.value,
                "username": self.config.auth_config.username,
                "password": self.config.auth_config.password,
                "cookies_file": str(self.config.auth_config.cookies_file) if self.config.auth_config.cookies_file else None,
                "netrc_file": str(self.config.auth_config.netrc_file) if self.config.auth_config.netrc_file else None,
                "oauth_token": self.config.auth_config.oauth_token
            }
            
            # Set up authentication
            auth_opts = await self.auth_manager.setup_authentication(auth_dict, url)
            opts.update(auth_opts)
            
            logger.debug(f"Authentication configured for {urlparse(url).netloc}")
            
        except AuthenticationError as e:
            logger.warning(f"Authentication setup failed for {url}: {e}")
            # Continue without authentication - some sites may work without it
        except Exception as e:
            logger.error(f"Unexpected error setting up authentication for {url}: {e}")
        
        return opts
    
    async def _is_authentication_error(self, error: Exception) -> bool:
        """
        Check if an error is authentication-related.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is authentication-related
        """
        error_message = str(error).lower()
        auth_keywords = [
            "login", "password", "authentication", "unauthorized", "forbidden",
            "cookie", "session", "token", "captcha", "verify", "sign in",
            "403", "401", "please log in", "access denied"
        ]
        
        return any(keyword in error_message for keyword in auth_keywords)
    
    async def _handle_authentication_error(
        self, 
        error: Exception, 
        url: str, 
        current_opts: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Handle authentication error and attempt recovery.
        
        Args:
            error: Authentication error
            url: URL that failed
            current_opts: Current yt-dlp options
            
        Returns:
            Updated options for retry, or None if recovery failed
        """
        try:
            # Use authentication error handler
            action = await self.auth_error_handler.handle_auth_error(
                error, self.config.auth_config, url
            )
            
            # Execute recovery action
            recovery_opts = await self.auth_error_handler.execute_recovery_action(
                action, self.config.auth_config, url
            )
            
            if recovery_opts:
                # Merge with current options
                updated_opts = current_opts.copy()
                updated_opts.update(recovery_opts)
                return updated_opts
            
            return None
            
        except Exception as e:
            logger.error(f"Authentication error handling failed: {e}")
            return None

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
        Download media with real-time progress tracking and adaptive quality control.

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
        self._progress_history[task_id] = []

        # Select optimal starting format if not specified
        current_format = None
        if not format_spec and info.formats:
            try:
                current_format = self.quality_controller.get_optimal_starting_quality(
                    info.formats, self.config.format_preferences
                )
                format_spec = current_format.format_id
                logger.info(f"Selected optimal starting format: {format_spec}")
            except Exception as e:
                logger.warning(f"Failed to select optimal starting format: {e}")

        try:
            # Create task-specific options
            opts = self._yt_dlp_opts.copy()
            if format_spec:
                opts["format"] = format_spec
            
            # Set up authentication for this URL
            opts = await self._setup_authentication_for_url(info.webpage_url, opts)

            # Create download task
            download_task = asyncio.create_task(
                self._download_with_yt_dlp(info.webpage_url, opts, task_id)
            )
            self._download_tasks[task_id] = download_task

            # Yield progress updates with quality adaptation
            async for progress in self._stream_progress_with_adaptation(task_id, info, current_format):
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
            # Check if this is an authentication error
            if await self._is_authentication_error(e):
                logger.info(f"Authentication error detected for task {task_id}, attempting recovery")
                try:
                    # Attempt to handle authentication error
                    recovered_opts = await self._handle_authentication_error(e, info.webpage_url, opts)
                    if recovered_opts:
                        # Retry download with recovered authentication
                        logger.info(f"Retrying download with recovered authentication for task {task_id}")
                        download_task = asyncio.create_task(
                            self._download_with_yt_dlp(info.webpage_url, recovered_opts, task_id)
                        )
                        self._download_tasks[task_id] = download_task
                        await download_task
                        return  # Success after recovery
                except Exception as recovery_error:
                    logger.error(f"Authentication recovery failed for task {task_id}: {recovery_error}")
            
            logger.error(f"Download failed for task {task_id}: {e}")
            self._download_states[task_id] = DownloadStatus.ERROR
            raise RuntimeError(f"Media download failed: {e}") from e
        finally:
            # Cleanup
            self._cleanup_task(task_id)
            # Clean up quality controller data
            self.quality_controller.cleanup_download_data(task_id)

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

    async def _stream_progress_with_adaptation(
        self, 
        task_id: str, 
        info: MediaInfo, 
        current_format: MediaFormat | None
    ) -> AsyncIterator[DownloadProgress]:
        """
        Stream progress updates with adaptive quality control.

        Args:
            task_id: Task identifier
            info: Media information with available formats
            current_format: Currently selected format

        Yields:
            DownloadProgress updates
        """
        last_progress = None
        adaptation_check_counter = 0
        
        while True:
            try:
                # Wait for progress update with timeout
                progress = await asyncio.wait_for(
                    self.progress_queue.get(),
                    timeout=1.0
                )

                # Store progress in history for analysis
                self._progress_history[task_id].append(progress)
                
                # Check for quality adaptation every 10 progress updates
                adaptation_check_counter += 1
                if (adaptation_check_counter >= 10 and 
                    current_format and 
                    info.formats and 
                    self.config.format_preferences.adaptive_quality):
                    
                    should_adapt, trigger = self.quality_controller.should_adapt_quality(
                        task_id, progress, current_format, info.formats
                    )
                    
                    if should_adapt and trigger:
                        logger.info(f"Quality adaptation triggered for {task_id}: {trigger.value}")
                        
                        # Get new format
                        new_format = self.quality_controller.adapt_quality(
                            task_id, current_format, info.formats, 
                            self.config.format_preferences, trigger
                        )
                        
                        if new_format and new_format.format_id != current_format.format_id:
                            logger.info(f"Adapting quality from {current_format.format_id} "
                                       f"to {new_format.format_id} for {task_id}")
                            
                            # Cancel current download
                            if task_id in self._download_tasks:
                                self._download_tasks[task_id].cancel()
                            
                            # Start new download with adapted format
                            await self._restart_download_with_new_format(
                                task_id, info.webpage_url, new_format.format_id
                            )
                            
                            current_format = new_format
                    
                    adaptation_check_counter = 0

                # Only yield if progress has changed significantly
                if self._should_yield_progress(progress, last_progress):
                    yield progress
                    last_progress = progress

                # Check for completion
                if progress.status in (DownloadStatus.FINISHED, DownloadStatus.ERROR):
                    # Analyze performance for future optimizations
                    if self._progress_history[task_id]:
                        self.quality_controller.analyze_download_performance(
                            task_id, self._progress_history[task_id]
                        )
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

    async def _restart_download_with_new_format(
        self, 
        task_id: str, 
        url: str, 
        format_spec: str
    ) -> None:
        """
        Restart download with a new format specification.
        
        Args:
            task_id: Task identifier
            url: URL to download
            format_spec: New format specification
        """
        logger.info(f"Restarting download {task_id} with format {format_spec}")
        
        try:
            # Create new options with adapted format
            opts = self._yt_dlp_opts.copy()
            opts["format"] = format_spec
            
            # Set up authentication for this URL
            opts = await self._setup_authentication_for_url(url, opts)
            
            # Create new download task
            download_task = asyncio.create_task(
                self._download_with_yt_dlp(url, opts, task_id)
            )
            self._download_tasks[task_id] = download_task
            
            logger.info(f"Download restarted successfully for {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to restart download {task_id} with new format: {e}")
            self._download_states[task_id] = DownloadStatus.ERROR

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

    async def download_audio_only(
        self,
        info: MediaInfo,
        task_id: str,
        target_bitrate: int | None = None
    ) -> AsyncIterator[DownloadProgress]:
        """
        Download audio-only format with quality optimization.

        Args:
            info: Media information from extraction
            task_id: Unique task identifier
            target_bitrate: Target audio bitrate in kbps

        Yields:
            DownloadProgress updates

        Raises:
            RuntimeError: If download fails
            ValueError: If no audio formats available
        """
        logger.info(f"Starting audio-only download for task {task_id}: {info.title}")

        if not info.formats:
            raise ValueError("No formats available for audio extraction")

        # Select optimal audio format
        try:
            from .format_extractor import FormatExtractor
            extractor = FormatExtractor(self.config)
            
            audio_format = extractor.select_audio_only_format(
                info.formats, self.config.format_preferences, target_bitrate
            )
            
            logger.info(f"Selected audio format: {audio_format.format_id} "
                       f"({audio_format.ext}, {audio_format.abr or 'unknown'} kbps)")
            
        except Exception as e:
            logger.error(f"Failed to select audio format: {e}")
            raise ValueError(f"No suitable audio format found: {e}") from e

        # Configure for audio extraction
        opts = self._yt_dlp_opts.copy()
        opts["format"] = audio_format.format_id
        
        # Force audio extraction post-processing
        opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": self.config.audio_format or "mp3",
            "preferredquality": str(target_bitrate) if target_bitrate else self.config.audio_quality,
        }]

        # Use regular download method with audio-specific options
        self._yt_dlp_opts_backup = self._yt_dlp_opts.copy()
        self._yt_dlp_opts = opts
        
        try:
            async for progress in self.download_media(info, task_id):
                yield progress
        finally:
            # Restore original options
            self._yt_dlp_opts = self._yt_dlp_opts_backup

    def get_recommended_audio_bitrate(
        self,
        formats: list[MediaFormat],
        preferences: FormatPreferences | None = None
    ) -> int:
        """
        Get recommended audio bitrate based on available formats.

        Args:
            formats: Available formats
            preferences: User preferences

        Returns:
            Recommended bitrate in kbps
        """
        prefs = preferences or self.config.format_preferences
        
        # If user specified target bitrate, use it
        if prefs.target_audio_bitrate:
            return prefs.target_audio_bitrate
        
        # Find audio formats and their bitrates
        audio_bitrates = []
        for fmt in formats:
            if fmt.abr and (fmt.vcodec in (None, "none") or fmt.acodec):
                audio_bitrates.append(fmt.abr)
        
        if not audio_bitrates:
            return 192  # Default fallback
        
        # Recommend based on available quality
        max_bitrate = max(audio_bitrates)
        
        if max_bitrate >= 320:
            return 320  # High quality
        elif max_bitrate >= 256:
            return 256  # Good quality
        elif max_bitrate >= 192:
            return 192  # Standard quality
        else:
            return int(max_bitrate)  # Use best available

    def analyze_format_quality_distribution(
        self,
        formats: list[MediaFormat]
    ) -> dict[str, any]:
        """
        Analyze quality distribution of available formats.

        Args:
            formats: List of formats to analyze

        Returns:
            Analysis dictionary with quality metrics
        """
        return self.format_selector.analyze_format_distribution(formats)

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
        
        # Clean up authentication resources
        try:
            await self.auth_error_handler.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up authentication resources: {e}")

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
