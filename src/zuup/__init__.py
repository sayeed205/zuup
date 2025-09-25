"""
Zuup - Unified Download Manager

A comprehensive download manager supporting HTTP/HTTPS, FTP/SFTP, BitTorrent,
and media downloads with a modern PySide6 GUI and browser extension support.
"""

__version__ = "0.1.0"
__author__ = "Zuup Team"
__email__ = "team@zuup.dev"

from .core.app import Application
from .core.interfaces import DownloadEngine
from .core.task_manager import TaskManager
from .storage.models import DownloadTask, ProgressInfo, TaskStatus

__all__ = [
    "Application",
    "DownloadEngine",
    "DownloadTask",
    "ProgressInfo",
    "TaskManager",
    "TaskStatus",
]
