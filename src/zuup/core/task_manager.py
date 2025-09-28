"""Task management system."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from ..engines.registry import get_registry
from ..storage.models import (
    DownloadTask,
    ProgressInfo,
    TaskConfig,
    TaskStatus,
)
from ..storage.validation import get_engine_for_url, validate_download_task_data
from .queue import DownloadQueue, QueuedTask, QueuePriority

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..core.interfaces import DownloadEngine
    from ..storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class TaskManagerError(Exception):
    """Base exception for task manager errors."""

    pass


class TaskNotFoundError(TaskManagerError):
    """Exception raised when a task is not found."""

    pass


class TaskValidationError(TaskManagerError):
    """Exception raised when task validation fails."""

    pass


class TaskManager:
    """Manages download tasks across different engines."""

    def __init__(
        self,
        database: DatabaseManager,
        max_concurrent: int = 3,
    ) -> None:
        """
        Initialize the task manager.

        Args:
            database: Database manager for persistence
            max_concurrent: Maximum concurrent downloads
        """
        self.database = database
        self.queue = DownloadQueue(max_concurrent)
        self.engine_registry = get_registry()

        # Task tracking
        self._tasks: dict[str, DownloadTask] = {}
        self._task_engines: dict[str, DownloadEngine] = {}
        self._task_processors: dict[str, asyncio.Task[None]] = {}
        self._progress_callbacks: dict[
            str, list[Callable[[str, ProgressInfo], None]]
        ] = {}

        # State management
        self._running = False
        self._manager_task: asyncio.Task[None] | None = None

        logger.info(f"TaskManager initialized with max_concurrent={max_concurrent}")

    async def initialize(self) -> None:
        """Initialize the task manager and load existing tasks."""
        try:
            # Initialize default engines first
            from ..engines.registry import initialize_default_engines

            initialize_default_engines()
            logger.info("Default engines initialized")

            # Load existing tasks from database
            existing_tasks = await self.database.list_tasks()
            for task in existing_tasks:
                self._tasks[task.id] = task

                # Re-queue incomplete tasks
                if task.status in (TaskStatus.PENDING, TaskStatus.PAUSED):
                    priority = (
                        QueuePriority.HIGH
                        if task.status == TaskStatus.PAUSED
                        else QueuePriority.NORMAL
                    )
                    await self.queue.add_task(task, priority)

            logger.info(f"Loaded {len(existing_tasks)} existing tasks from database")

            # Start queue processing
            await self.queue.start_processing()
            self._running = True
            self._manager_task = asyncio.create_task(self._process_tasks())

            logger.info("Task manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize task manager: {e}")
            raise TaskManagerError(f"Initialization failed: {e}") from e

    async def create_task(
        self,
        url: str,
        destination: str,
        filename: str | None = None,
        config: TaskConfig | None = None,
        auto_start: bool = True,
    ) -> DownloadTask:
        """
        Create a new download task.

        Args:
            url: URL to download
            destination: Destination directory path
            filename: Optional custom filename
            config: Optional task configuration
            auto_start: Whether to automatically start the task

        Returns:
            Created download task

        Raises:
            TaskValidationError: If task validation fails
            TaskManagerError: If task creation fails
        """
        try:
            # Validate URL and determine engine type
            engine_type_str = get_engine_for_url(url)
            if not engine_type_str:
                raise TaskValidationError(f"Unsupported URL: {url}")

            # Convert string to EngineType enum
            from ..storage.models import EngineType

            try:
                engine_type = EngineType(engine_type_str)
            except ValueError:
                raise TaskValidationError(f"Invalid engine type: {engine_type_str}")

            # Create destination path
            dest_path = Path(destination)
            if filename:
                dest_path = dest_path / filename
            elif not dest_path.suffix:
                # Auto-generate filename from URL
                parsed_url = urlparse(url)
                auto_filename = Path(parsed_url.path).name or "download"
                dest_path = dest_path / auto_filename

            # Create task
            task = DownloadTask(
                url=url,
                destination=dest_path,
                filename=filename,
                engine_type=engine_type,
                config=config or TaskConfig(),
            )

            # Validate task data
            validation_result = validate_download_task_data(task.model_dump())
            if not validation_result.is_valid:
                raise TaskValidationError(
                    f"Task validation failed: {validation_result.errors}"
                )

            # Store task
            self._tasks[task.id] = task
            await self.database.save_task(task)

            logger.info(f"Created task {task.id} for URL: {url}")

            # Add to queue if auto-start is enabled
            if auto_start:
                await self.queue.add_task(task, QueuePriority.NORMAL)
                logger.info(f"Added task {task.id} to download queue")

            return task

        except Exception as e:
            logger.error(f"Failed to create task for URL {url}: {e}")
            if isinstance(e, (TaskValidationError, TaskManagerError)):
                raise
            raise TaskManagerError(f"Task creation failed: {e}") from e

    async def start_task(self, task_id: str) -> None:
        """
        Start a download task.

        Args:
            task_id: ID of task to start

        Raises:
            TaskNotFoundError: If task is not found
            TaskManagerError: If task cannot be started
        """
        try:
            task = self.get_task(task_id)

            if task.status == TaskStatus.DOWNLOADING:
                logger.warning(f"Task {task_id} is already downloading")
                return

            if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
                raise TaskManagerError(
                    f"Cannot start task {task_id}: status is {task.status.value}"
                )

            # Update task status
            task.status = TaskStatus.PENDING
            await self._update_task(task)

            # Add to queue
            priority = (
                QueuePriority.HIGH
                if task.status == TaskStatus.PAUSED
                else QueuePriority.NORMAL
            )
            await self.queue.add_task(task, priority)

            logger.info(f"Started task {task_id}")

        except TaskNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to start task {task_id}: {e}")
            raise TaskManagerError(f"Failed to start task: {e}") from e

    async def pause_task(self, task_id: str) -> None:
        """
        Pause a download task.

        Args:
            task_id: ID of task to pause

        Raises:
            TaskNotFoundError: If task is not found
            TaskManagerError: If task cannot be paused
        """
        try:
            task = self.get_task(task_id)

            if task.status != TaskStatus.DOWNLOADING:
                raise TaskManagerError(f"Cannot pause task {task_id}: not downloading")

            # Pause in queue
            if not await self.queue.pause_task(task_id):
                logger.warning(f"Task {task_id} not found in queue for pausing")

            # Pause in engine
            engine = self._task_engines.get(task_id)
            if engine:
                await engine.pause(task_id)

            # Update task status
            task.status = TaskStatus.PAUSED
            await self._update_task(task)

            logger.info(f"Paused task {task_id}")

        except TaskNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to pause task {task_id}: {e}")
            raise TaskManagerError(f"Failed to pause task: {e}") from e

    async def resume_task(self, task_id: str) -> None:
        """
        Resume a download task.

        Args:
            task_id: ID of task to resume

        Raises:
            TaskNotFoundError: If task is not found
            TaskManagerError: If task cannot be resumed
        """
        try:
            task = self.get_task(task_id)

            if task.status != TaskStatus.PAUSED:
                raise TaskManagerError(f"Cannot resume task {task_id}: not paused")

            # Resume in queue
            if not await self.queue.resume_task(task_id):
                # If not in queue, add it back
                await self.queue.add_task(task, QueuePriority.HIGH)

            # Update task status
            task.status = TaskStatus.PENDING
            await self._update_task(task)

            logger.info(f"Resumed task {task_id}")

        except TaskNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to resume task {task_id}: {e}")
            raise TaskManagerError(f"Failed to resume task: {e}") from e

    async def cancel_task(self, task_id: str) -> None:
        """
        Cancel a download task.

        Args:
            task_id: ID of task to cancel

        Raises:
            TaskNotFoundError: If task is not found
            TaskManagerError: If task cannot be cancelled
        """
        try:
            task = self.get_task(task_id)

            if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
                logger.warning(f"Task {task_id} is already {task.status.value}")
                return

            # Remove from queue
            await self.queue.remove_task(task_id)

            # Cancel in engine
            engine = self._task_engines.get(task_id)
            if engine:
                await engine.cancel(task_id)

            # Cancel processor task
            processor = self._task_processors.get(task_id)
            if processor and not processor.done():
                processor.cancel()
                try:
                    await processor
                except asyncio.CancelledError:
                    pass

            # Clean up tracking
            self._task_engines.pop(task_id, None)
            self._task_processors.pop(task_id, None)

            # Update task status
            task.status = TaskStatus.CANCELLED
            await self._update_task(task)

            logger.info(f"Cancelled task {task_id}")

        except TaskNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            raise TaskManagerError(f"Failed to cancel task: {e}") from e

    def get_task(self, task_id: str) -> DownloadTask:
        """
        Get task by ID.

        Args:
            task_id: Task ID

        Returns:
            Download task

        Raises:
            TaskNotFoundError: If task is not found
        """
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Task {task_id} not found")
        return self._tasks[task_id]

    def list_tasks(self) -> list[DownloadTask]:
        """
        List all tasks.

        Returns:
            List of all download tasks
        """
        return list(self._tasks.values())

    def get_tasks_by_status(self, status: TaskStatus) -> list[DownloadTask]:
        """
        Get tasks filtered by status.

        Args:
            status: Task status to filter by

        Returns:
            List of tasks with the specified status
        """
        return [task for task in self._tasks.values() if task.status == status]

    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a task completely.

        Args:
            task_id: ID of task to delete

        Returns:
            True if task was deleted, False if not found

        Raises:
            TaskManagerError: If task cannot be deleted
        """
        try:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]

            # Cancel task if it's active
            if task.status in (
                TaskStatus.DOWNLOADING,
                TaskStatus.PENDING,
                TaskStatus.PAUSED,
            ):
                await self.cancel_task(task_id)

            # Remove from database
            await self.database.delete_task(task_id)

            # Remove from memory
            del self._tasks[task_id]

            logger.info(f"Deleted task {task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            raise TaskManagerError(f"Failed to delete task: {e}") from e

    def add_progress_callback(
        self,
        task_id: str,
        callback: Callable[[str, ProgressInfo], None],
    ) -> None:
        """
        Add a progress callback for a task.

        Args:
            task_id: Task ID
            callback: Progress callback function
        """
        if task_id not in self._progress_callbacks:
            self._progress_callbacks[task_id] = []
        self._progress_callbacks[task_id].append(callback)

    def remove_progress_callback(
        self,
        task_id: str,
        callback: Callable[[str, ProgressInfo], None],
    ) -> None:
        """
        Remove a progress callback for a task.

        Args:
            task_id: Task ID
            callback: Progress callback function to remove
        """
        if task_id in self._progress_callbacks:
            try:
                self._progress_callbacks[task_id].remove(callback)
                if not self._progress_callbacks[task_id]:
                    del self._progress_callbacks[task_id]
            except ValueError:
                pass

    def get_manager_stats(self) -> dict[str, int | dict[str, int]]:
        """
        Get task manager statistics.

        Returns:
            Dictionary with manager statistics
        """
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = len(self.get_tasks_by_status(status))

        return {
            "total_tasks": len(self._tasks),
            "active_processors": len(self._task_processors),
            "status_counts": status_counts,
            "queue_status": self.queue.get_queue_status(),
        }

    async def _process_tasks(self) -> None:
        """Process tasks from the queue continuously."""
        logger.info("Task processor started")

        while self._running:
            try:
                # Check for tasks ready to process
                queued_task = await self.queue.get_next_task()
                if not queued_task:
                    # Wait for queue changes before checking again
                    try:
                        await asyncio.wait_for(
                            self.queue._queue_changed.wait(), timeout=1.0
                        )
                        self.queue._queue_changed.clear()
                    except asyncio.TimeoutError:
                        pass
                    continue

                # Start processing the task
                task_id = queued_task.task.id
                processor = asyncio.create_task(self._process_single_task(queued_task))
                self._task_processors[task_id] = processor

                logger.info(f"Started processor for task {task_id}")

            except Exception as e:
                logger.error(f"Error in task processor: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying

        logger.info("Task processor stopped")

    async def _process_single_task(self, queued_task: QueuedTask) -> None:
        """
        Process a single download task.

        Args:
            queued_task: Queued task to process
        """
        task = queued_task.task
        task_id = task.id

        try:
            # Get appropriate engine
            engine = self.engine_registry.get_engine_for_url(task.url)
            if not engine:
                raise TaskManagerError(f"No engine available for URL: {task.url}")

            self._task_engines[task_id] = engine

            # Update task status
            task.status = TaskStatus.DOWNLOADING
            await self._update_task(task)

            logger.info(f"Starting download for task {task_id}")

            # Process download with progress updates
            async for progress in engine.download(task):
                # Update task progress
                task.update_progress(progress)
                await self._update_task(task)

                # Call progress callbacks
                await self._call_progress_callbacks(task_id, progress)

                # Check if task was cancelled
                if task.status == TaskStatus.CANCELLED:
                    break

            # Mark as completed if successful
            if task.status == TaskStatus.DOWNLOADING:
                await self.queue.mark_task_completed(task_id)
                task.mark_completed()
                await self._update_task(task)
                logger.info(f"Task {task_id} completed successfully")

        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was cancelled")
            await self.queue.mark_task_failed(task_id, "Task cancelled")
        except Exception as e:
            error_msg = f"Download failed: {e}"
            logger.error(f"Task {task_id} failed: {error_msg}")
            await self.queue.mark_task_failed(task_id, error_msg)
            task.mark_failed(error_msg)
            await self._update_task(task)
        finally:
            # Clean up
            self._task_engines.pop(task_id, None)
            self._task_processors.pop(task_id, None)

    async def _update_task(self, task: DownloadTask) -> None:
        """
        Update task in memory and database.

        Args:
            task: Task to update
        """
        self._tasks[task.id] = task
        await self.database.save_task(task)

    async def _call_progress_callbacks(
        self, task_id: str, progress: ProgressInfo
    ) -> None:
        """
        Call all progress callbacks for a task.

        Args:
            task_id: Task ID
            progress: Progress information
        """
        callbacks = self._progress_callbacks.get(task_id, [])
        for callback in callbacks:
            try:
                callback(task_id, progress)
            except Exception as e:
                logger.error(f"Error in progress callback for task {task_id}: {e}")

    async def shutdown(self) -> None:
        """Shutdown the task manager gracefully."""
        logger.info("Shutting down task manager")

        self._running = False

        # Stop queue processing
        await self.queue.stop_processing()

        # Cancel manager task
        if self._manager_task and not self._manager_task.done():
            self._manager_task.cancel()
            try:
                await self._manager_task
            except asyncio.CancelledError:
                pass

        # Cancel all active processors
        for task_id, processor in list(self._task_processors.items()):
            if not processor.done():
                processor.cancel()
                try:
                    await processor
                except asyncio.CancelledError:
                    pass

        # Clean up queue
        await self.queue.cleanup()

        logger.info("Task manager shutdown complete")
