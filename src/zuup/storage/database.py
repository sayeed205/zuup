"""Database management for task persistence."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import DownloadTask

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""

    pass


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
        self._lock = asyncio.Lock()

        logger.info(f"DatabaseManager initialized with db: {db_path}")

    async def initialize(self) -> None:
        """Initialize the database schema."""
        async with self._lock:
            await asyncio.to_thread(self._create_tables)
        logger.info("Database schema initialized")

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")

            # Create tasks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    filename TEXT,
                    engine_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    file_size INTEGER,
                    mime_type TEXT,
                    checksum TEXT,
                    config_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL
                )
            """)

            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_engine_type ON tasks(engine_type)")

            conn.commit()

    async def save_task(self, task: DownloadTask) -> None:
        """
        Save a download task to the database.

        Args:
            task: Download task to save

        Raises:
            DatabaseError: If save operation fails
        """
        try:
            async with self._lock:
                await asyncio.to_thread(self._save_task_sync, task)
            logger.debug(f"Saved task {task.id} to database")
        except Exception as e:
            logger.error(f"Failed to save task {task.id}: {e}")
            raise DatabaseError(f"Failed to save task: {e}") from e

    def _save_task_sync(self, task: DownloadTask) -> None:
        """Synchronous task save operation."""
        with sqlite3.connect(self.db_path) as conn:
            # Serialize complex objects to JSON
            config_json = task.config.model_dump_json()
            progress_json = task.progress.model_dump_json()

            conn.execute("""
                INSERT OR REPLACE INTO tasks (
                    id, url, destination, filename, engine_type, status,
                    created_at, updated_at, file_size, mime_type, checksum,
                    config_json, progress_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id,
                task.url,
                str(task.destination),
                task.filename,
                task.engine_type.value,
                task.status.value,
                task.created_at.isoformat(),
                task.updated_at.isoformat(),
                task.file_size,
                task.mime_type,
                task.checksum,
                config_json,
                progress_json,
            ))
            conn.commit()

    async def load_task(self, task_id: str) -> DownloadTask | None:
        """
        Load a download task from the database.

        Args:
            task_id: Task ID to load

        Returns:
            Download task if found, None otherwise

        Raises:
            DatabaseError: If load operation fails
        """
        try:
            async with self._lock:
                return await asyncio.to_thread(self._load_task_sync, task_id)
        except Exception as e:
            logger.error(f"Failed to load task {task_id}: {e}")
            raise DatabaseError(f"Failed to load task: {e}") from e

    def _load_task_sync(self, task_id: str) -> DownloadTask | None:
        """Synchronous task load operation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (task_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_task(row)

    async def list_tasks(
        self,
        status_filter: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[DownloadTask]:
        """
        List tasks from the database.

        Args:
            status_filter: Optional status to filter by
            limit: Optional limit on number of results
            offset: Offset for pagination

        Returns:
            List of download tasks

        Raises:
            DatabaseError: If list operation fails
        """
        try:
            async with self._lock:
                return await asyncio.to_thread(
                    self._list_tasks_sync, status_filter, limit, offset
                )
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            raise DatabaseError(f"Failed to list tasks: {e}") from e

    def _list_tasks_sync(
        self,
        status_filter: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[DownloadTask]:
        """Synchronous task list operation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM tasks"
            params = []

            if status_filter:
                query += " WHERE status = ?"
                params.append(status_filter)

            query += " ORDER BY created_at DESC"

            if limit:
                query += " LIMIT ? OFFSET ?"
                params.append(str(limit))
                params.append(str(offset))

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_task(row) for row in rows]

    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a task from the database.

        Args:
            task_id: Task ID to delete

        Returns:
            True if deleted, False if not found

        Raises:
            DatabaseError: If delete operation fails
        """
        try:
            async with self._lock:
                return await asyncio.to_thread(self._delete_task_sync, task_id)
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            raise DatabaseError(f"Failed to delete task: {e}") from e

    def _delete_task_sync(self, task_id: str) -> bool:
        """Synchronous task delete operation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()
            return cursor.rowcount > 0

    async def get_task_count(self, status_filter: str | None = None) -> int:
        """
        Get count of tasks in database.

        Args:
            status_filter: Optional status to filter by

        Returns:
            Number of tasks

        Raises:
            DatabaseError: If count operation fails
        """
        try:
            async with self._lock:
                return await asyncio.to_thread(self._get_task_count_sync, status_filter)
        except Exception as e:
            logger.error(f"Failed to get task count: {e}")
            raise DatabaseError(f"Failed to get task count: {e}") from e

    def _get_task_count_sync(self, status_filter: str | None = None) -> int:
        """Synchronous task count operation."""
        with sqlite3.connect(self.db_path) as conn:
            if status_filter:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE status = ?",
                    (status_filter,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM tasks")

            result = cursor.fetchone()
            return int(result[0]) if result else 0

    async def cleanup_completed_tasks(self, older_than_days: int = 30) -> int:
        """
        Clean up completed tasks older than specified days.

        Args:
            older_than_days: Remove completed tasks older than this many days

        Returns:
            Number of tasks removed

        Raises:
            DatabaseError: If cleanup operation fails
        """
        try:
            async with self._lock:
                return await asyncio.to_thread(self._cleanup_completed_tasks_sync, older_than_days)
        except Exception as e:
            logger.error(f"Failed to cleanup completed tasks: {e}")
            raise DatabaseError(f"Failed to cleanup completed tasks: {e}") from e

    def _cleanup_completed_tasks_sync(self, older_than_days: int) -> int:
        """Synchronous cleanup operation."""
        cutoff_date = datetime.now().replace(microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - older_than_days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM tasks 
                WHERE status IN ('completed', 'failed', 'cancelled') 
                AND updated_at < ?
            """, (cutoff_date.isoformat(),))
            conn.commit()
            return cursor.rowcount

    async def get_database_stats(self) -> dict[str, int | str]:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics

        Raises:
            DatabaseError: If stats operation fails
        """
        try:
            async with self._lock:
                return await asyncio.to_thread(self._get_database_stats_sync)
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            raise DatabaseError(f"Failed to get database stats: {e}") from e

    def _get_database_stats_sync(self) -> dict[str, int | str]:
        """Synchronous database stats operation."""
        with sqlite3.connect(self.db_path) as conn:
            # Get total task count
            total_cursor = conn.execute("SELECT COUNT(*) FROM tasks")
            total_tasks = total_cursor.fetchone()[0]

            # Get status counts
            status_cursor = conn.execute("""
                SELECT status, COUNT(*) 
                FROM tasks 
                GROUP BY status
            """)
            status_counts = dict(status_cursor.fetchall())

            # Get database file size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "total_tasks": total_tasks,
                "database_size_bytes": db_size,
                "database_path": str(self.db_path),
                "status_counts": status_counts,
            }

    def _row_to_task(self, row: sqlite3.Row) -> DownloadTask:
        """
        Convert database row to DownloadTask object.

        Args:
            row: SQLite row object

        Returns:
            DownloadTask object
        """
        from .models import (
            DownloadTask,
            EngineType,
            ProgressInfo,
            TaskConfig,
            TaskStatus,
        )

        # Parse JSON fields
        config_data = json.loads(row["config_json"])
        progress_data = json.loads(row["progress_json"])

        # Parse datetime fields
        created_at = datetime.fromisoformat(row["created_at"])
        updated_at = datetime.fromisoformat(row["updated_at"])

        return DownloadTask(
            id=row["id"],
            url=row["url"],
            destination=Path(row["destination"]),
            filename=row["filename"],
            engine_type=EngineType(row["engine_type"]),
            config=TaskConfig(**config_data),
            status=TaskStatus(row["status"]),
            progress=ProgressInfo(**progress_data),
            created_at=created_at,
            updated_at=updated_at,
            file_size=row["file_size"],
            mime_type=row["mime_type"],
            checksum=row["checksum"],
        )

    async def backup_database(self, backup_path: Path) -> None:
        """
        Create a backup of the database.

        Args:
            backup_path: Path for the backup file

        Raises:
            DatabaseError: If backup operation fails
        """
        try:
            async with self._lock:
                await asyncio.to_thread(self._backup_database_sync, backup_path)
            logger.info(f"Database backed up to {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            raise DatabaseError(f"Failed to backup database: {e}") from e

    def _backup_database_sync(self, backup_path: Path) -> None:
        """Synchronous database backup operation."""
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as source:
            with sqlite3.connect(backup_path) as backup:
                source.backup(backup)

    async def close(self) -> None:
        """Close database connections and clean up resources."""
        # SQLite connections are automatically closed when context managers exit
        # This method is provided for consistency with async patterns
        logger.info("Database manager closed")
