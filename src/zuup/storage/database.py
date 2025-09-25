"""Database management for task persistence."""

import logging
from pathlib import Path

from .models import DownloadTask

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database for task persistence."""

    def __init__(self, db_path: Path | None = None) -> None:
        """
        Initialize database manager.

        Args:
            db_path: Optional custom database path
        """
        if db_path is None:
            db_path = Path.home() / ".config" / "zuup" / "tasks.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"DatabaseManager initialized with db: {db_path}")

    async def save_task(self, task: DownloadTask) -> None:
        """
        Save a download task to the database.

        Args:
            task: Download task to save
        """
        # Implementation will be added in task 5
        logger.info(f"Saving task {task.id} to database")
        raise NotImplementedError("Database operations will be implemented in task 5")

    async def load_task(self, task_id: str) -> DownloadTask | None:
        """
        Load a download task from the database.

        Args:
            task_id: Task ID to load

        Returns:
            Download task if found, None otherwise
        """
        # Implementation will be added in task 5
        logger.info(f"Loading task {task_id} from database")
        raise NotImplementedError("Database operations will be implemented in task 5")

    async def list_tasks(self) -> list[DownloadTask]:
        """
        List all tasks from the database.

        Returns:
            List of all download tasks
        """
        # Implementation will be added in task 5
        logger.info("Listing all tasks from database")
        raise NotImplementedError("Database operations will be implemented in task 5")

    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from the database.

        Args:
            task_id: Task ID to delete

        Returns:
            True if deleted, False if not found
        """
        # Implementation will be added in task 5
        logger.info(f"Deleting task {task_id} from database")
        raise NotImplementedError("Database operations will be implemented in task 5")
