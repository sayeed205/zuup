"""Base download engine implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage.models import DownloadTask, ProgressInfo

logger = logging.getLogger(__name__)


class EngineError(Exception):
    """Base exception for engine-related errors."""

    def __init__(self, message: str, task_id: str | None = None) -> None:
        """Initialize engine error."""
        super().__init__(message)
        self.task_id = task_id
        self.timestamp = datetime.now()


class DownloadError(EngineError):
    """Exception raised during download operations."""

    pass


class NetworkError(EngineError):
    """Exception raised for network-related issues."""

    pass


class ValidationError(EngineError):
    """Exception raised for validation failures."""

    pass


class BaseDownloadEngine(ABC):
    """Base class for all download engines with common functionality."""

    def __init__(self) -> None:
        """Initialize base download engine."""
        self.name = self.__class__.__name__
        self._active_tasks: dict[str, asyncio.Task[None]] = {}
        self._task_progress: dict[str, ProgressInfo] = {}
        self._task_start_times: dict[str, float] = {}
        logger.info(f"Initialized {self.name}")

    @abstractmethod
    def download(self, task: DownloadTask) -> AsyncIterator[ProgressInfo]:
        """
        Start downloading a task and yield progress updates.

        Args:
            task: Download task to execute

        Yields:
            Progress information updates

        Raises:
            DownloadError: If download fails
            NetworkError: If network issues occur
            ValidationError: If task validation fails
        """
        pass

    @abstractmethod
    async def pause(self, task_id: str) -> None:
        """
        Pause a download task.

        Args:
            task_id: ID of task to pause

        Raises:
            EngineError: If pause operation fails
        """
        pass

    @abstractmethod
    async def resume(self, task_id: str) -> None:
        """
        Resume a paused download task.

        Args:
            task_id: ID of task to resume

        Raises:
            EngineError: If resume operation fails
        """
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> None:
        """
        Cancel a download task.

        Args:
            task_id: ID of task to cancel

        Raises:
            EngineError: If cancel operation fails
        """
        pass

    @abstractmethod
    def get_progress(self, task_id: str) -> ProgressInfo:
        """
        Get current progress for a task.

        Args:
            task_id: Task ID

        Returns:
            Current progress information

        Raises:
            KeyError: If task ID is not found
        """
        pass

    @abstractmethod
    def supports_protocol(self, url: str) -> bool:
        """
        Check if this engine supports the given URL protocol.

        Args:
            url: URL to check

        Returns:
            True if supported, False otherwise
        """
        pass

    # Common functionality methods

    def _validate_task(self, task: DownloadTask) -> None:
        """
        Validate a download task before processing.

        Args:
            task: Task to validate

        Raises:
            ValidationError: If task validation fails
        """
        if not task.url:
            raise ValidationError("Task URL cannot be empty", task.id)

        if not self.supports_protocol(task.url):
            raise ValidationError(
                f"Engine {self.name} does not support URL: {task.url}", task.id
            )

        if not task.destination:
            raise ValidationError("Task destination cannot be empty", task.id)

    def _update_progress(self, task_id: str, progress: ProgressInfo) -> None:
        """
        Update progress information for a task.

        Args:
            task_id: Task ID
            progress: New progress information
        """
        self._task_progress[task_id] = progress
        logger.debug(
            f"Updated progress for task {task_id}: {progress.downloaded_bytes} bytes"
        )

    def _calculate_eta(
        self,
        task_id: str,
        downloaded_bytes: int,
        total_bytes: int | None,
        current_speed: float,
    ) -> timedelta | None:
        """
        Calculate estimated time of arrival for a download.

        Args:
            task_id: Task ID
            downloaded_bytes: Bytes downloaded so far
            total_bytes: Total bytes to download (if known)
            current_speed: Current download speed in bytes per second

        Returns:
            Estimated time remaining, or None if cannot be calculated
        """
        if not total_bytes or current_speed <= 0:
            return None

        remaining_bytes = total_bytes - downloaded_bytes
        if remaining_bytes <= 0:
            return timedelta(0)

        eta_seconds = remaining_bytes / current_speed
        return timedelta(seconds=eta_seconds)

    def _calculate_average_speed(self, task_id: str, downloaded_bytes: int) -> float:
        """
        Calculate average download speed for a task.

        Args:
            task_id: Task ID
            downloaded_bytes: Total bytes downloaded

        Returns:
            Average speed in bytes per second
        """
        start_time = self._task_start_times.get(task_id)
        if not start_time:
            return 0.0

        elapsed_time = time.time() - start_time
        if elapsed_time <= 0:
            return 0.0

        return downloaded_bytes / elapsed_time

    def _register_task_start(self, task_id: str) -> None:
        """
        Register the start time for a task.

        Args:
            task_id: Task ID
        """
        self._task_start_times[task_id] = time.time()
        logger.debug(f"Registered start time for task {task_id}")

    def _unregister_task(self, task_id: str) -> None:
        """
        Clean up task tracking data.

        Args:
            task_id: Task ID to clean up
        """
        self._task_progress.pop(task_id, None)
        self._task_start_times.pop(task_id, None)
        self._active_tasks.pop(task_id, None)
        logger.debug(f"Cleaned up tracking data for task {task_id}")

    def _is_task_active(self, task_id: str) -> bool:
        """
        Check if a task is currently active.

        Args:
            task_id: Task ID to check

        Returns:
            True if task is active, False otherwise
        """
        task = self._active_tasks.get(task_id)
        return task is not None and not task.done()

    def get_active_tasks(self) -> list[str]:
        """
        Get list of currently active task IDs.

        Returns:
            List of active task IDs
        """
        return [
            task_id for task_id, task in self._active_tasks.items() if not task.done()
        ]

    def get_engine_stats(self) -> dict[str, int | float]:
        """
        Get engine statistics.

        Returns:
            Dictionary with engine statistics
        """
        active_count = len(self.get_active_tasks())
        total_tasks = len(self._task_progress)

        # Calculate total downloaded bytes across all tasks
        total_downloaded = sum(
            progress.downloaded_bytes for progress in self._task_progress.values()
        )

        # Calculate average speed across active tasks
        active_speeds = [
            progress.download_speed
            for task_id, progress in self._task_progress.items()
            if self._is_task_active(task_id) and progress.download_speed > 0
        ]
        avg_speed = sum(active_speeds) / len(active_speeds) if active_speeds else 0.0

        return {
            "active_tasks": active_count,
            "total_tasks": total_tasks,
            "total_downloaded_bytes": total_downloaded,
            "average_speed_bps": avg_speed,
        }

    async def cleanup(self) -> None:
        """Clean up engine resources and cancel active tasks."""
        logger.info(f"Cleaning up {self.name} engine")

        # Cancel all active tasks
        for task_id in list(self._active_tasks.keys()):
            try:
                await self.cancel(task_id)
            except Exception as e:
                logger.error(f"Error cancelling task {task_id} during cleanup: {e}")

        # Clear all tracking data
        self._active_tasks.clear()
        self._task_progress.clear()
        self._task_start_times.clear()

        logger.info(f"{self.name} engine cleanup completed")
