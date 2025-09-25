"""Core interfaces and protocols for the download manager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ..storage.models import DownloadTask, GlobalConfig, ProgressInfo, TaskConfig


class DownloadEngine(Protocol):
    """Protocol for download engine implementations."""

    async def download(self, task: DownloadTask) -> AsyncIterator[ProgressInfo]:
        """
        Start downloading a task and yield progress updates.

        Args:
            task: Download task to execute

        Yields:
            Progress information updates
        """
        ...

    async def pause(self, task_id: str) -> None:
        """
        Pause a download task.

        Args:
            task_id: ID of task to pause
        """
        ...

    async def resume(self, task_id: str) -> None:
        """
        Resume a paused download task.

        Args:
            task_id: ID of task to resume
        """
        ...

    async def cancel(self, task_id: str) -> None:
        """
        Cancel a download task.

        Args:
            task_id: ID of task to cancel
        """
        ...

    def get_progress(self, task_id: str) -> ProgressInfo:
        """
        Get current progress for a task.

        Args:
            task_id: ID of task

        Returns:
            Current progress information
        """
        ...

    def supports_protocol(self, url: str) -> bool:
        """
        Check if this engine supports the given URL protocol.

        Args:
            url: URL to check

        Returns:
            True if supported, False otherwise
        """
        ...


class ConfigManager(Protocol):
    """Protocol for configuration management."""

    def get_global_config(self) -> GlobalConfig:
        """Get global configuration."""
        ...

    def get_task_config(self, task_id: str) -> TaskConfig:
        """Get task-specific configuration."""
        ...

    def update_global_config(self, config: GlobalConfig) -> None:
        """Update global configuration."""
        ...

    def update_task_config(self, task_id: str, config: TaskConfig) -> None:
        """Update task-specific configuration."""
        ...


class TaskManager(Protocol):
    """Protocol for task management."""

    async def create_task(self, url: str, destination: str) -> DownloadTask:
        """Create a new download task."""
        ...

    async def start_task(self, task_id: str) -> None:
        """Start a download task."""
        ...

    async def pause_task(self, task_id: str) -> None:
        """Pause a download task."""
        ...

    async def resume_task(self, task_id: str) -> None:
        """Resume a download task."""
        ...

    async def cancel_task(self, task_id: str) -> None:
        """Cancel a download task."""
        ...

    def get_task(self, task_id: str) -> DownloadTask:
        """Get task by ID."""
        ...

    def list_tasks(self) -> list[DownloadTask]:
        """List all tasks."""
        ...
