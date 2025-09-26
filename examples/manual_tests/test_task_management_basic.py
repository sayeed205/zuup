#!/usr/bin/env python3
"""Basic manual test script for task management system without actual downloads."""

import asyncio
import logging
from pathlib import Path

from zuup.core import TaskManager
from zuup.storage import DatabaseManager, TaskConfig, TaskStatus


async def test_basic_task_management():
    """Test the basic task management system functionality."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting basic task management test")
    
    try:
        # Create database manager
        test_db_path = Path("test_basic_tasks.db")
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
            auto_start=False
        )
        logger.info(f"Created task 1: {task1.id}")
        
        # Create another HTTP task with custom config
        config = TaskConfig(
            max_connections=4,
            retry_attempts=5,
            timeout=60
        )
        task2 = await task_manager.create_task(
            url="https://httpbin.org/bytes/2048", 
            destination=str(Path.home() / "Downloads"),
            filename="test_file_2.bin",
            config=config,
            auto_start=False
        )
        logger.info(f"Created task 2: {task2.id}")
        
        # Test task listing
        all_tasks = task_manager.list_tasks()
        logger.info(f"Total tasks: {len(all_tasks)}")
        
        # Test filtering by status
        pending_tasks = task_manager.get_tasks_by_status(TaskStatus.PENDING)
        logger.info(f"Pending tasks: {len(pending_tasks)}")
        
        # Test task retrieval
        retrieved_task = task_manager.get_task(task1.id)
        logger.info(f"Retrieved task: {retrieved_task.id} - {retrieved_task.url}")
        
        # Test queue operations without starting downloads
        logger.info("Testing queue operations...")
        
        # Add tasks to queue manually
        await task_manager.queue.add_task(task1)
        await task_manager.queue.add_task(task2)
        
        # Check queue status
        queue_status = task_manager.queue.get_queue_status()
        logger.info(f"Queue status: {queue_status}")
        
        # Test queue priority and ordering
        from zuup.core.queue import QueuePriority
        
        # Create a high priority task
        task3 = await task_manager.create_task(
            url="https://httpbin.org/bytes/512",
            destination=str(Path.home() / "Downloads"),
            filename="urgent_file.bin",
            auto_start=False
        )
        
        # Add with high priority
        await task_manager.queue.add_task(task3, QueuePriority.HIGH)
        
        # Check queue order
        queued_tasks = task_manager.queue.get_queued_tasks()
        logger.info(f"Queued tasks order: {[t.id for t in queued_tasks]}")
        
        # Test task removal from queue
        removed = await task_manager.queue.remove_task(task2.id)
        logger.info(f"Task 2 removed from queue: {removed}")
        
        # Test database operations
        logger.info("Testing database operations...")
        
        # Save tasks to database
        await db_manager.save_task(task1)
        await db_manager.save_task(task2)
        await db_manager.save_task(task3)
        
        # Load task from database
        loaded_task = await db_manager.load_task(task1.id)
        logger.info(f"Loaded task from DB: {loaded_task.id if loaded_task else 'None'}")
        
        # List tasks from database
        db_tasks = await db_manager.list_tasks()
        logger.info(f"Tasks in database: {len(db_tasks)}")
        
        # Test database filtering
        pending_db_tasks = await db_manager.list_tasks(status_filter="pending")
        logger.info(f"Pending tasks in DB: {len(pending_db_tasks)}")
        
        # Test task count
        total_count = await db_manager.get_task_count()
        logger.info(f"Total task count in DB: {total_count}")
        
        # Test database stats
        db_stats = await db_manager.get_database_stats()
        logger.info(f"Database stats: {db_stats}")
        
        # Test task manager stats
        manager_stats = task_manager.get_manager_stats()
        logger.info(f"Manager stats: {manager_stats}")
        
        # Test task deletion
        logger.info("Testing task deletion...")
        deleted = await task_manager.delete_task(task3.id)
        logger.info(f"Task 3 deleted: {deleted}")
        
        # Verify deletion
        try:
            task_manager.get_task(task3.id)
            logger.error("Task 3 should have been deleted!")
        except Exception:
            logger.info("Task 3 correctly deleted from memory")
        
        # Test database deletion
        db_deleted = await db_manager.delete_task(task2.id)
        logger.info(f"Task 2 deleted from DB: {db_deleted}")
        
        # Test queue cleanup
        await task_manager.queue.clear_completed()
        
        # Final status
        final_tasks = task_manager.list_tasks()
        logger.info(f"Final task count: {len(final_tasks)}")
        
        for task in final_tasks:
            logger.info(f"Final task {task.id}: {task.status.value}")
        
        # Cleanup
        await task_manager.shutdown()
        await db_manager.close()
        
        logger.info("Basic task management test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Clean up test database
        if test_db_path.exists():
            test_db_path.unlink()


if __name__ == "__main__":
    asyncio.run(test_basic_task_management())