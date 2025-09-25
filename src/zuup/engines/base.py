"""Base download engine implementation."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
import logging

from ..storage.models import DownloadTask, ProgressInfo

logger = logging.getLogger(__name__)


class BaseDownloadEngine(ABC):
    """Base class for all download engines."""

    def __init__(self) -> None:
        """Initialize base download engine."""
        self.name = self.__class__.__name__
        logger.info(f"Initialized {self.name}")

    @abstractmethod
    async def download(self, task: DownloadTask) -> AsyncIterator[ProgressInfo]:
        """
        Start downloading a task and yield progress updates.

        Args:
            task: Download task to execute

        Yields:
            Progress information updates
        """
        pass

    @abstractmethod
    async def pause(self, task_id: str) -> None:
        """
        Pause a download task.

        Args:
            task_id: ID of task to pause
        """
        pass

    @abstractmethod
    async def resume(self, task_id: str) -> None:
        """
        Resume a paused download task.

        Args:
            task_id: ID of task to resume
        """
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> None:
        """
        Cancel a download task.

        Args:
            task_id: ID of task to cancel
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
