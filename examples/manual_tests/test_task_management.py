#!/usr/bin/env python3
"""Manual test script for task management system."""

import asyncio
import logging
from pathlib import Path

from zuup.core import TaskManager
from zuup.storage import DatabaseManager, TaskConfig
from zuup.engines.registry import initialize_default_engines


async def test_task_management():
    """Test the task management system."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting task management test")

    try:
        # Initialize engines
        initialize_default_engines()
        logger.info("Engines initialized")

        # Create database manager
        test_db_path = Path("test_tasks.db")
        if test_db_path.exists():
            test_db_path.unlink()  # Clean slate for testing

        db_manager = DatabaseManager(test_db_path)
        await db_manager.initialize()
        logger.info("Database initialized")

        # Create task manager
        task_manager = TaskManager(db_manager, max_concurrent=2)
        await task_manager.initialize()
        logger.info("Task manager initialized")

        # Test task creation
        logger.info("Creating test tasks...")

        # Create HTTP download task
        task1 = await task_manager.create_task(
            url="https://httpbin.org/bytes/1024",
            destination=str(Path.home() / "Downloads"),
            filename="test_file_1.bin",
            auto_start=False,
        )
        logger.info(f"Created task 1: {task1.id}")

        # Create another HTTP task with custom config
        config = TaskConfig(max_connections=4, retry_attempts=5, timeout=60)
        task2 = await task_manager.create_task(
            url="https://httpbin.org/bytes/2048",
            destination=str(Path.home() / "Downloads"),
            filename="test_file_2.bin",
            config=config,
            auto_start=False,
        )
        logger.info(f"Created task 2: {task2.id}")

        # Test task listing
        all_tasks = task_manager.list_tasks()
        logger.info(f"Total tasks: {len(all_tasks)}")

        # Test task starting
        logger.info("Starting tasks...")
        await task_manager.start_task(task1.id)
        await task_manager.start_task(task2.id)

        # Let tasks run for a bit
        await asyncio.sleep(2)

        # Test task pausing (only if task is actually downloading)
        task1_current = task_manager.get_task(task1.id)
        if task1_current.status.value == "downloading":
            logger.info("Pausing task 1...")
            await task_manager.pause_task(task1.id)

            # Let task 2 continue
            await asyncio.sleep(2)

            # Test task resuming
            logger.info("Resuming task 1...")
            await task_manager.resume_task(task1.id)
        else:
            logger.info(
                f"Task 1 status is {task1_current.status.value}, skipping pause/resume test"
            )

        # Wait for tasks to complete or timeout
        timeout = 30
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            stats = task_manager.get_manager_stats()
            logger.info(f"Manager stats: {stats}")

            # Check if all tasks are done
            active_count = stats["queue_status"]["active"]
            if active_count == 0:
                logger.info("All tasks completed")
                break

            await asyncio.sleep(1)

        # Final status check
        final_tasks = task_manager.list_tasks()
        for task in final_tasks:
            logger.info(
                f"Task {task.id}: {task.status.value} - {task.progress.downloaded_bytes} bytes"
            )

        # Test database persistence
        logger.info("Testing database persistence...")
        db_stats = await db_manager.get_database_stats()
        logger.info(f"Database stats: {db_stats}")

        # Test task deletion
        logger.info("Testing task deletion...")
        deleted = await task_manager.delete_task(task1.id)
        logger.info(f"Task 1 deleted: {deleted}")

        # Cleanup
        await task_manager.shutdown()
        await db_manager.close()

        logger.info("Task management test completed successfully")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Clean up test database
        if test_db_path.exists():
            test_db_path.unlink()


if __name__ == "__main__":
    asyncio.run(test_task_management())
