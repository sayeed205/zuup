"""Playlist and batch download management for media downloads."""

import asyncio
from collections.abc import AsyncIterator
import logging
from pathlib import Path
from typing import Any

import yt_dlp  # type: ignore[import-untyped]

from .format_extractor import FormatExtractor
from .media_downloader import MediaDownloader
from .media_models import (
    BatchDownloadConfig,
    BatchProgress,
    DownloadStatus,
    MediaConfig,
    MediaInfo,
    PlaylistInfo,
)

logger = logging.getLogger(__name__)


class PlaylistManager:
    """Manages playlist extraction and batch downloads with concurrency control."""

    def __init__(self, config: MediaConfig) -> None:
        """
        Initialize PlaylistManager with configuration.

        Args:
            config: Media configuration for downloads
        """
        self.config = config
        self.extractor = FormatExtractor(config)
        self.downloader = MediaDownloader(config)
        self._active_downloads: dict[str, asyncio.Task[None]] = {}
        self._download_semaphore = asyncio.Semaphore(config.concurrent_downloads)
        self._archive_entries: set[str] = set()
        self._batch_stats = {
            "total_items": 0,
            "completed_items": 0,
            "failed_items": 0,
            "skipped_items": 0,
            "overall_downloaded_bytes": 0,
            "overall_total_bytes": 0,
        }
        logger.info("PlaylistManager initialized")

    async def extract_playlist(self, url: str) -> PlaylistInfo:
        """
        Extract playlist information from URL.

        Args:
            url: Playlist URL to extract

        Returns:
            PlaylistInfo object with all entries

        Raises:
            ValueError: If URL is not a playlist or extraction fails
            RuntimeError: If yt-dlp extraction fails
        """
        logger.info(f"Extracting playlist info for URL: {url}")

        try:
            # Create yt-dlp options for playlist extraction
            opts = {
                "quiet": True,
                "no_warnings": False,
                "extract_flat": True,  # Extract playlist entries without full info
                "playliststart": self.config.playlist_start,
                "playlistend": self.config.playlist_end,
                "socket_timeout": self.config.socket_timeout,
                "retries": self.config.retries,
                "proxy": self.config.proxy,
                "geo_bypass": self.config.geo_bypass,
                "geo_bypass_country": self.config.geo_bypass_country,
                "extractor_args": self.config.extractor_args,
            }

            # Add authentication if configured
            if self.config.auth_config.username:
                opts["username"] = self.config.auth_config.username
            if self.config.auth_config.password:
                opts["password"] = self.config.auth_config.password
            if self.config.auth_config.cookies_file:
                opts["cookiefile"] = str(self.config.auth_config.cookies_file)

            # Run extraction in thread pool
            loop = asyncio.get_event_loop()
            info_dict = await loop.run_in_executor(
                None, self._extract_playlist_sync, url, opts
            )

            # Parse playlist information
            playlist_info = await self._parse_playlist_dict(info_dict)

            logger.info(f"Successfully extracted playlist: {playlist_info.title} ({playlist_info.entry_count} entries)")
            return playlist_info

        except yt_dlp.DownloadError as e:
            logger.error(f"yt-dlp playlist extraction failed for {url}: {e}")
            raise RuntimeError(f"Failed to extract playlist info: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error extracting playlist for {url}: {e}")
            raise RuntimeError(f"Unexpected playlist extraction error: {e}") from e

    def _extract_playlist_sync(self, url: str, opts: dict[str, Any]) -> dict[str, Any]:
        """
        Synchronous yt-dlp playlist extraction.

        Args:
            url: URL to extract playlist from
            opts: yt-dlp options

        Returns:
            Raw yt-dlp info dictionary
        """
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)  # type: ignore[no-any-return]

    async def _parse_playlist_dict(self, info_dict: dict[str, Any]) -> PlaylistInfo:
        """
        Parse yt-dlp playlist dictionary into PlaylistInfo model.

        Args:
            info_dict: Raw yt-dlp playlist dictionary

        Returns:
            Parsed PlaylistInfo object
        """
        # Check if this is actually a playlist
        if info_dict.get("_type") != "playlist":
            # Single video, create a playlist with one entry
            media_info = await self._parse_single_entry(info_dict)
            return PlaylistInfo(
                id=info_dict.get("id", "single"),
                title=info_dict.get("title", "Single Video"),
                description=info_dict.get("description"),
                uploader=info_dict.get("uploader"),
                entry_count=1,
                entries=[media_info],
                webpage_url=info_dict.get("webpage_url", ""),
            )

        # Extract playlist metadata
        playlist_id = info_dict.get("id", "")
        title = info_dict.get("title", "Unknown Playlist")
        description = info_dict.get("description")
        uploader = info_dict.get("uploader")
        webpage_url = info_dict.get("webpage_url", "")

        # Extract entries
        entries_data = info_dict.get("entries", [])
        entries = []

        for entry_data in entries_data:
            if entry_data is None:
                continue

            try:
                # For flat extraction, we need to get full info for each entry
                if entry_data.get("_type") == "url":
                    # Extract full info for this entry
                    entry_url = entry_data.get("url") or entry_data.get("webpage_url")
                    if entry_url:
                        media_info = await self.extractor.extract_info(entry_url)
                        entries.append(media_info)
                else:
                    # Already has full info
                    media_info = await self._parse_single_entry(entry_data)
                    entries.append(media_info)
            except Exception as e:
                logger.warning(f"Failed to parse playlist entry: {e}")
                continue

        return PlaylistInfo(
            id=playlist_id,
            title=title,
            description=description,
            uploader=uploader,
            entry_count=len(entries),
            entries=entries,
            webpage_url=webpage_url,
        )

    async def _parse_single_entry(self, entry_data: dict[str, Any]) -> MediaInfo:
        """
        Parse a single entry from playlist data.

        Args:
            entry_data: Single entry data from yt-dlp

        Returns:
            MediaInfo object
        """
        # If this is just a URL reference, extract full info
        if entry_data.get("_type") == "url":
            entry_url = entry_data.get("url") or entry_data.get("webpage_url")
            if entry_url:
                return await self.extractor.extract_info(entry_url)

        # Parse as MediaInfo directly
        temp_extractor = FormatExtractor(self.config)
        return temp_extractor._parse_info_dict(entry_data)

    async def download_playlist(
        self,
        playlist_info: PlaylistInfo,
        batch_config: BatchDownloadConfig | None = None
    ) -> AsyncIterator[BatchProgress]:
        """
        Download all entries in a playlist with concurrent management.

        Args:
            playlist_info: Playlist information to download
            batch_config: Batch download configuration (uses defaults if None)

        Yields:
            BatchProgress updates during download
        """
        if batch_config is None:
            batch_config = BatchDownloadConfig(
                concurrent_downloads=self.config.concurrent_downloads,
                delay_between_downloads=self.config.delay_between_downloads,
            )

        logger.info(f"Starting playlist download: {playlist_info.title} ({playlist_info.entry_count} entries)")

        # Initialize batch statistics
        self._batch_stats = {
            "total_items": playlist_info.entry_count,
            "completed_items": 0,
            "failed_items": 0,
            "skipped_items": 0,
            "overall_downloaded_bytes": 0,
            "overall_total_bytes": 0,
        }

        # Load download archive if configured
        if batch_config.archive_file:
            await self._load_download_archive(batch_config.archive_file)

        # Update semaphore for concurrent downloads
        self._download_semaphore = asyncio.Semaphore(batch_config.concurrent_downloads)

        try:
            # Create download tasks for all entries
            download_tasks = []
            for i, entry in enumerate(playlist_info.entries):
                # Check if already downloaded (archive support)
                if self._is_already_downloaded(entry):
                    logger.info(f"Skipping already downloaded entry: {entry.title}")
                    self._batch_stats["skipped_items"] += 1
                    continue

                # Create download task
                task_id = f"playlist_{playlist_info.id}_{i}_{entry.id}"
                task = asyncio.create_task(
                    self._download_single_entry(entry, task_id, batch_config)
                )
                download_tasks.append((task, entry, task_id))

            # Process downloads with progress reporting
            async for progress in self._process_batch_downloads(download_tasks, batch_config):
                yield progress

        except Exception as e:
            logger.error(f"Playlist download failed: {e}")
            raise
        finally:
            # Cleanup active downloads
            await self._cleanup_active_downloads()

    async def _download_single_entry(
        self,
        entry: MediaInfo,
        task_id: str,
        batch_config: BatchDownloadConfig
    ) -> None:
        """
        Download a single playlist entry with concurrency control.

        Args:
            entry: Media entry to download
            task_id: Unique task identifier
            batch_config: Batch configuration
        """
        async with self._download_semaphore:
            try:
                logger.info(f"Starting download for entry: {entry.title}")

                # Add delay between downloads if configured
                if batch_config.delay_between_downloads > 0:
                    await asyncio.sleep(batch_config.delay_between_downloads)

                # Download the entry
                downloaded_bytes = 0

                async for progress in self.downloader.download_media(entry, task_id):
                    # Update overall statistics
                    downloaded_bytes = progress.downloaded_bytes

                    # Check for completion
                    if progress.status == DownloadStatus.FINISHED:
                        self._batch_stats["completed_items"] += 1
                        self._batch_stats["overall_downloaded_bytes"] += downloaded_bytes

                        # Add to archive if configured
                        if batch_config.archive_file:
                            await self._add_to_archive(entry, batch_config.archive_file)

                        logger.info(f"Completed download for entry: {entry.title}")
                        break
                    elif progress.status == DownloadStatus.ERROR:
                        self._batch_stats["failed_items"] += 1
                        logger.error(f"Failed to download entry: {entry.title}")

                        if not batch_config.continue_on_error:
                            raise RuntimeError(f"Download failed for {entry.title}")
                        break

            except Exception as e:
                self._batch_stats["failed_items"] += 1
                logger.error(f"Error downloading entry {entry.title}: {e}")

                if not batch_config.continue_on_error:
                    raise
            finally:
                # Remove from active downloads
                self._active_downloads.pop(task_id, None)

    async def _process_batch_downloads(
        self,
        download_tasks: list[tuple[asyncio.Task[None], MediaInfo, str]],
        batch_config: BatchDownloadConfig
    ) -> AsyncIterator[BatchProgress]:
        """
        Process batch downloads and yield progress updates.

        Args:
            download_tasks: List of download tasks with metadata
            batch_config: Batch configuration

        Yields:
            BatchProgress updates
        """
        completed_tasks = 0
        current_entry = None

        # Store active tasks
        for task, _entry, task_id in download_tasks:
            self._active_downloads[task_id] = task

        # Initial progress
        yield BatchProgress(
            total_items=self._batch_stats["total_items"],
            completed_items=self._batch_stats["completed_items"],
            failed_items=self._batch_stats["failed_items"],
            current_item=None,
            overall_downloaded_bytes=self._batch_stats["overall_downloaded_bytes"],
            overall_total_bytes=self._batch_stats["overall_total_bytes"],
        )

        # Process tasks as they complete
        pending_tasks = {task: (entry, task_id) for task, entry, task_id in download_tasks}

        while pending_tasks:
            # Wait for at least one task to complete
            done, _pending = await asyncio.wait(
                pending_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=1.0  # Regular progress updates
            )

            # Process completed tasks
            for task in done:
                _entry, task_id = pending_tasks.pop(task)
                completed_tasks += 1

                try:
                    await task  # Get result or exception
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")

            # Update current entry (first pending task)
            if pending_tasks:
                current_entry = next(iter(pending_tasks.values()))[0].title
            else:
                current_entry = None

            # Yield progress update
            progress = BatchProgress(
                total_items=self._batch_stats["total_items"],
                completed_items=self._batch_stats["completed_items"],
                failed_items=self._batch_stats["failed_items"],
                current_item=current_entry,
                overall_downloaded_bytes=self._batch_stats["overall_downloaded_bytes"],
                overall_total_bytes=self._batch_stats["overall_total_bytes"],
            )
            yield progress

            # Check failure threshold
            if (batch_config.max_failures > 0 and
                self._batch_stats["failed_items"] >= batch_config.max_failures):
                logger.error(f"Maximum failures ({batch_config.max_failures}) reached, stopping batch download")
                # Cancel remaining tasks
                for task in pending_tasks:
                    task.cancel()
                break

        # Final progress update
        yield BatchProgress(
            total_items=self._batch_stats["total_items"],
            completed_items=self._batch_stats["completed_items"],
            failed_items=self._batch_stats["failed_items"],
            current_item=None,
            overall_downloaded_bytes=self._batch_stats["overall_downloaded_bytes"],
            overall_total_bytes=self._batch_stats["overall_total_bytes"],
        )

        logger.info(f"Batch download completed: {self._batch_stats['completed_items']} completed, "
                   f"{self._batch_stats['failed_items']} failed, {self._batch_stats['skipped_items']} skipped")

    async def _load_download_archive(self, archive_file: Path) -> None:
        """
        Load download archive for duplicate detection.

        Args:
            archive_file: Path to archive file
        """
        try:
            if archive_file.exists():
                content = archive_file.read_text(encoding="utf-8")
                self._archive_entries = {line.strip() for line in content.splitlines() if line.strip()}
                logger.info(f"Loaded {len(self._archive_entries)} entries from download archive")
            else:
                self._archive_entries = set()
                logger.info("Download archive file does not exist, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load download archive: {e}")
            self._archive_entries = set()

    async def _add_to_archive(self, entry: MediaInfo, archive_file: Path) -> None:
        """
        Add entry to download archive.

        Args:
            entry: Media entry to add
            archive_file: Path to archive file
        """
        try:
            # Create archive entry (extractor:id format)
            archive_entry = f"{entry.extractor_key}:{entry.id}"

            if archive_entry not in self._archive_entries:
                self._archive_entries.add(archive_entry)

                # Append to file
                archive_file.parent.mkdir(parents=True, exist_ok=True)
                with archive_file.open("a", encoding="utf-8") as f:
                    f.write(f"{archive_entry}\n")

                logger.debug(f"Added to archive: {archive_entry}")
        except Exception as e:
            logger.error(f"Failed to add entry to archive: {e}")

    def _is_already_downloaded(self, entry: MediaInfo) -> bool:
        """
        Check if entry is already downloaded (in archive).

        Args:
            entry: Media entry to check

        Returns:
            True if already downloaded, False otherwise
        """
        archive_entry = f"{entry.extractor_key}:{entry.id}"
        return archive_entry in self._archive_entries

    async def _cleanup_active_downloads(self) -> None:
        """Clean up active download tasks."""
        logger.info("Cleaning up active downloads")

        # Cancel all active downloads
        for task_id, task in list(self._active_downloads.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error cleaning up task {task_id}: {e}")

        self._active_downloads.clear()

    async def cancel_batch_download(self) -> None:
        """Cancel all active batch downloads."""
        logger.info("Cancelling batch download")
        await self._cleanup_active_downloads()

    def get_batch_statistics(self) -> dict[str, Any]:
        """
        Get current batch download statistics.

        Returns:
            Dictionary with batch statistics
        """
        return self._batch_stats.copy()

    async def cleanup(self) -> None:
        """Clean up playlist manager resources."""
        logger.info("Cleaning up PlaylistManager")

        await self._cleanup_active_downloads()
        await self.downloader.cleanup()

        logger.info("PlaylistManager cleanup completed")
