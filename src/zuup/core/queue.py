"""Download queue management module.

This module provides the download queue implementation for managing
concurrent downloads across different engines.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from enum import Enum
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..storage.models import DownloadTask, ProgressInfo

logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Download queue priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class QueuedTask:
    """Represents a task in the download queue."""

    def __init__(
        self,
        task: DownloadTask,
        priority: QueuePriority = QueuePriority.NORMAL,
        callback: Callable[[str, ProgressInfo], None] | None = None,
    ) -> None:
        """
        Initialize a queued task.

        Args:
            task: Download task
            priority: Queue priority
            callback: Optional progress callback function
        """
        self.task = task
        self.priority = priority
        self.callback = callback
        self.queued_at = datetime.now()
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None

    def __lt__(self, other: QueuedTask) -> bool:
        """Compare tasks for priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value  # Higher priority first
        return (
            self.queued_at < other.queued_at
        )  # Earlier queued first for same priority


class DownloadQueue:
    """Manages download queue with priority and concurrency control."""

    def __init__(self, max_concurrent: int = 3) -> None:
        """
        Initialize download queue.

        Args:
            max_concurrent: Maximum number of concurrent downloads
        """
        self.max_concurrent = max_concurrent
        self._queue: deque[QueuedTask] = deque()
        self._active_tasks: dict[str, QueuedTask] = {}
        self._paused_tasks: dict[str, QueuedTask] = {}
        self._completed_tasks: dict[str, QueuedTask] = {}
        self._failed_tasks: dict[str, QueuedTask] = {}
        self._lock = asyncio.Lock()
        self._queue_changed = asyncio.Event()
        self._running = False
        self._processor_task: asyncio.Task[None] | None = None

        logger.info(f"Download queue initialized with max_concurrent={max_concurrent}")

    async def add_task(
        self,
        task: DownloadTask,
        priority: QueuePriority = QueuePriority.NORMAL,
        callback: Callable[[str, ProgressInfo], None] | None = None,
    ) -> None:
        """
        Add a task to the download queue.

        Args:
            task: Download task to add
            priority: Queue priority
            callback: Optional progress callback function
        """
        async with self._lock:
            # Check if task is already in queue or active
            if self._is_task_tracked(task.id):
                logger.warning(f"Task {task.id} is already in queue or active")
                return

            queued_task = QueuedTask(task, priority, callback)

            # Insert task in priority order
            inserted = False
            for i, existing_task in enumerate(self._queue):
                if queued_task < existing_task:
                    self._queue.insert(i, queued_task)
                    inserted = True
                    break

            if not inserted:
                self._queue.append(queued_task)

            logger.info(f"Added task {task.id} to queue with priority {priority.name}")
            self._queue_changed.set()

    async def remove_task(self, task_id: str) -> bool:
        """
        Remove a task from the queue.

        Args:
            task_id: ID of task to remove

        Returns:
            True if task was removed, False if not found
        """
        async with self._lock:
            # Try to remove from queue
            for i, queued_task in enumerate(self._queue):
                if queued_task.task.id == task_id:
                    del self._queue[i]
                    logger.info(f"Removed task {task_id} from queue")
                    return True

            # Check if task is in other states
            if task_id in self._paused_tasks:
                del self._paused_tasks[task_id]
                logger.info(f"Removed paused task {task_id}")
                return True

            logger.warning(f"Task {task_id} not found in queue")
            return False

    async def pause_task(self, task_id: str) -> bool:
        """
        Pause a task (move from active to paused).

        Args:
            task_id: ID of task to pause

        Returns:
            True if task was paused, False if not found or not active
        """
        async with self._lock:
            if task_id in self._active_tasks:
                queued_task = self._active_tasks.pop(task_id)
                self._paused_tasks[task_id] = queued_task
                logger.info(f"Paused task {task_id}")
                self._queue_changed.set()  # May allow new tasks to start
                return True

            logger.warning(f"Cannot pause task {task_id}: not active")
            return False

    async def resume_task(self, task_id: str) -> bool:
        """
        Resume a paused task (move back to queue with high priority).

        Args:
            task_id: ID of task to resume

        Returns:
            True if task was resumed, False if not found or not paused
        """
        async with self._lock:
            if task_id in self._paused_tasks:
                queued_task = self._paused_tasks.pop(task_id)
                # Add back to queue with high priority
                queued_task.priority = QueuePriority.HIGH
                self._queue.appendleft(queued_task)  # Add to front
                logger.info(f"Resumed task {task_id}")
                self._queue_changed.set()
                return True

            logger.warning(f"Cannot resume task {task_id}: not paused")
            return False

    async def get_next_task(self) -> QueuedTask | None:
        """
        Get the next task to process from the queue.

        Returns:
            Next queued task or None if queue is empty or at capacity
        """
        async with self._lock:
            if len(self._active_tasks) >= self.max_concurrent:
                return None

            if not self._queue:
                return None

            queued_task = self._queue.popleft()
            queued_task.started_at = datetime.now()
            self._active_tasks[queued_task.task.id] = queued_task

            logger.info(f"Dequeued task {queued_task.task.id} for processing")
            return queued_task

    async def mark_task_completed(self, task_id: str) -> None:
        """
        Mark a task as completed.

        Args:
            task_id: ID of completed task
        """
        async with self._lock:
            if task_id in self._active_tasks:
                queued_task = self._active_tasks.pop(task_id)
                queued_task.completed_at = datetime.now()
                self._completed_tasks[task_id] = queued_task
                logger.info(f"Marked task {task_id} as completed")
                self._queue_changed.set()  # May allow new tasks to start

    async def mark_task_failed(self, task_id: str, error: str) -> None:
        """
        Mark a task as failed.

        Args:
            task_id: ID of failed task
            error: Error message
        """
        async with self._lock:
            if task_id in self._active_tasks:
                queued_task = self._active_tasks.pop(task_id)
                queued_task.task.mark_failed(error)
                self._failed_tasks[task_id] = queued_task
                logger.error(f"Marked task {task_id} as failed: {error}")
                self._queue_changed.set()  # May allow new tasks to start

    def get_queue_status(self) -> dict[str, int]:
        """
        Get current queue status.

        Returns:
            Dictionary with queue statistics
        """
        return {
            "queued": len(self._queue),
            "active": len(self._active_tasks),
            "paused": len(self._paused_tasks),
            "completed": len(self._completed_tasks),
            "failed": len(self._failed_tasks),
            "max_concurrent": self.max_concurrent,
        }

    def get_queued_tasks(self) -> list[DownloadTask]:
        """
        Get list of queued tasks.

        Returns:
            List of tasks in queue
        """
        return [qt.task for qt in self._queue]

    def get_active_tasks(self) -> list[DownloadTask]:
        """
        Get list of active tasks.

        Returns:
            List of active tasks
        """
        return [qt.task for qt in self._active_tasks.values()]

    def get_paused_tasks(self) -> list[DownloadTask]:
        """
        Get list of paused tasks.

        Returns:
            List of paused tasks
        """
        return [qt.task for qt in self._paused_tasks.values()]

    def get_task_position(self, task_id: str) -> int | None:
        """
        Get position of task in queue.

        Args:
            task_id: Task ID

        Returns:
            Position in queue (0-based) or None if not in queue
        """
        for i, queued_task in enumerate(self._queue):
            if queued_task.task.id == task_id:
                return i
        return None

    def set_max_concurrent(self, max_concurrent: int) -> None:
        """
        Update maximum concurrent downloads.

        Args:
            max_concurrent: New maximum concurrent downloads
        """
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")

        old_max = self.max_concurrent
        self.max_concurrent = max_concurrent
        logger.info(f"Updated max_concurrent from {old_max} to {max_concurrent}")

        if max_concurrent > old_max:
            # May allow more tasks to start
            self._queue_changed.set()

    def _is_task_tracked(self, task_id: str) -> bool:
        """
        Check if task is tracked in any state.

        Args:
            task_id: Task ID to check

        Returns:
            True if task is tracked, False otherwise
        """
        return (
            any(qt.task.id == task_id for qt in self._queue)
            or task_id in self._active_tasks
            or task_id in self._paused_tasks
            or task_id in self._completed_tasks
            or task_id in self._failed_tasks
        )

    async def start_processing(self) -> None:
        """Start the queue processor."""
        if self._running:
            logger.warning("Queue processor is already running")
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())
        logger.info("Started queue processor")

    async def stop_processing(self) -> None:
        """Stop the queue processor."""
        if not self._running:
            return

        self._running = False
        self._queue_changed.set()  # Wake up processor

        if self._processor_task:
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

        logger.info("Stopped queue processor")

    async def _process_queue(self) -> None:
        """Process the download queue continuously."""
        logger.info("Queue processor started")

        while self._running:
            try:
                # Wait for queue changes or timeout
                try:
                    await asyncio.wait_for(self._queue_changed.wait(), timeout=1.0)
                    self._queue_changed.clear()
                except asyncio.TimeoutError:
                    continue

                # Process available tasks
                while self._running and len(self._active_tasks) < self.max_concurrent:
                    queued_task = await self.get_next_task()
                    if not queued_task:
                        break

                    # Start processing the task
                    # Note: Actual task execution will be handled by TaskManager
                    logger.debug(f"Task {queued_task.task.id} ready for processing")

            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying

        logger.info("Queue processor stopped")

    async def clear_completed(self) -> int:
        """
        Clear completed and failed tasks from tracking.

        Returns:
            Number of tasks cleared
        """
        async with self._lock:
            completed_count = len(self._completed_tasks)
            failed_count = len(self._failed_tasks)

            self._completed_tasks.clear()
            self._failed_tasks.clear()

            total_cleared = completed_count + failed_count
            logger.info(f"Cleared {total_cleared} completed/failed tasks")
            return total_cleared

    async def cleanup(self) -> None:
        """Clean up queue resources."""
        await self.stop_processing()

        async with self._lock:
            self._queue.clear()
            self._active_tasks.clear()
            self._paused_tasks.clear()
            self._completed_tasks.clear()
            self._failed_tasks.clear()

        logger.info("Download queue cleaned up")
