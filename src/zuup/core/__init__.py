"""Core application logic module."""

from .app import Application
from .interfaces import DownloadEngine
from .queue import DownloadQueue, QueuedTask, QueuePriority
from .task_manager import TaskManager

__all__ = [
    "Application",
    "DownloadEngine",
    "DownloadQueue",
    "QueuePriority",
    "QueuedTask",
    "TaskManager",
]
