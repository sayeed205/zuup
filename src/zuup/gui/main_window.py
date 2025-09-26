"""Main window for PySide6 GUI application."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ..config.manager import ConfigManager
    from ..core.task_manager import TaskManager

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window using PySide6."""

    # Signals
    shutdown_requested = Signal()

    def __init__(
        self,
        task_manager: TaskManager | None = None,
        config_manager: ConfigManager | None = None,
    ) -> None:
        """
        Initialize main window with dependency injection.

        Args:
            task_manager: Task manager instance
            config_manager: Configuration manager instance
        """
        super().__init__()

        self.task_manager = task_manager
        self.config_manager = config_manager

        # Window properties
        self.setWindowTitle("Zuup Download Manager")
        self.setMinimumSize(800, 600)

        # Setup UI components
        self._setup_ui()
        self._setup_menu_bar()
        self._setup_status_bar()

        # Setup update timer
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_status)
        self._update_timer.start(1000)  # Update every second

        logger.info("MainWindow initialized with dependency injection")

    def _setup_ui(self) -> None:
        """Setup the main UI components."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("Zuup Download Manager")
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        # Status display
        self.status_label = QLabel("Initializing...")
        header_layout.addWidget(self.status_label)

        main_layout.addLayout(header_layout)

        # Task management section
        task_layout = QVBoxLayout()

        task_header = QLabel("Download Tasks")
        task_header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        task_layout.addWidget(task_header)

        # Task controls
        controls_layout = QHBoxLayout()

        self.add_task_btn = QPushButton("Add Download")
        self.add_task_btn.clicked.connect(self._add_download)
        controls_layout.addWidget(self.add_task_btn)

        self.pause_all_btn = QPushButton("Pause All")
        self.pause_all_btn.clicked.connect(self._pause_all)
        controls_layout.addWidget(self.pause_all_btn)

        self.resume_all_btn = QPushButton("Resume All")
        self.resume_all_btn.clicked.connect(self._resume_all)
        controls_layout.addWidget(self.resume_all_btn)

        controls_layout.addStretch()

        task_layout.addLayout(controls_layout)

        # Task list placeholder
        self.task_list_label = QLabel("No active downloads")
        self.task_list_label.setStyleSheet("padding: 20px; border: 1px solid #ccc; margin: 10px;")
        task_layout.addWidget(self.task_list_label)

        main_layout.addLayout(task_layout)
        main_layout.addStretch()

    def _setup_menu_bar(self) -> None:
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # Add download action
        add_action = file_menu.addAction("Add Download")
        add_action.triggered.connect(self._add_download)

        file_menu.addSeparator()

        # Exit action
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        # View menu
        view_menu = menubar.addMenu("View")

        # Refresh action
        refresh_action = view_menu.addAction("Refresh")
        refresh_action.triggered.connect(self._refresh_view)

        # Help menu
        help_menu = menubar.addMenu("Help")

        # About action
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self._show_about)

    def _setup_status_bar(self) -> None:
        """Setup the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status labels
        self.connection_status = QLabel("Server: Disconnected")
        self.task_count_status = QLabel("Tasks: 0")

        self.status_bar.addWidget(self.connection_status)
        self.status_bar.addPermanentWidget(self.task_count_status)

    def _update_status(self) -> None:
        """Update the status display."""
        if not self.task_manager:
            self.status_label.setText("Task Manager: Not Available")
            self.connection_status.setText("Server: Not Available")
            self.task_count_status.setText("Tasks: N/A")
            return

        try:
            # Get task manager stats
            stats = self.task_manager.get_manager_stats()
            total_tasks = stats.get("total_tasks", 0)

            # Update status labels
            self.status_label.setText("Status: Running")
            self.connection_status.setText("Server: Connected")
            self.task_count_status.setText(f"Tasks: {total_tasks}")

            # Update task list display
            if total_tasks > 0:
                tasks = self.task_manager.list_tasks()
                task_info = []
                for task in tasks[:5]:  # Show first 5 tasks
                    task_info.append(f"• {task.url[:50]}... ({task.status.value})")

                if total_tasks > 5:
                    task_info.append(f"... and {total_tasks - 5} more")

                self.task_list_label.setText("\n".join(task_info))
            else:
                self.task_list_label.setText("No active downloads")

        except Exception as e:
            logger.error(f"Error updating status: {e}")
            self.status_label.setText("Status: Error")

    def _add_download(self) -> None:
        """Add a new download (placeholder)."""
        from PySide6.QtWidgets import QInputDialog, QMessageBox

        url, ok = QInputDialog.getText(self, "Add Download", "Enter URL:")
        if ok and url:
            if self.task_manager:
                try:
                    # This is a basic implementation - full GUI will be in later tasks
                    import asyncio

                    async def create_task():
                        await self.task_manager.create_task(
                            url=url,
                            destination="./downloads",
                            auto_start=True,
                        )

                    # Run in current event loop
                    loop = asyncio.get_event_loop()
                    loop.create_task(create_task())

                    QMessageBox.information(self, "Success", f"Download added: {url}")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to add download: {e}")
            else:
                QMessageBox.warning(self, "Error", "Task manager not available")

    def _pause_all(self) -> None:
        """Pause all downloads (placeholder)."""
        from PySide6.QtWidgets import QMessageBox

        if self.task_manager:
            # This is a placeholder - full implementation in later tasks
            QMessageBox.information(self, "Info", "Pause all functionality will be implemented in later tasks")
        else:
            QMessageBox.warning(self, "Error", "Task manager not available")

    def _resume_all(self) -> None:
        """Resume all downloads (placeholder)."""
        from PySide6.QtWidgets import QMessageBox

        if self.task_manager:
            # This is a placeholder - full implementation in later tasks
            QMessageBox.information(self, "Info", "Resume all functionality will be implemented in later tasks")
        else:
            QMessageBox.warning(self, "Error", "Task manager not available")

    def _refresh_view(self) -> None:
        """Refresh the view."""
        self._update_status()
        logger.info("View refreshed")

    def _show_about(self) -> None:
        """Show about dialog."""
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About Zuup Download Manager",
            "Zuup Download Manager v1.0.0\n\n"
            "A unified download manager supporting:\n"
            "• HTTP/HTTPS downloads\n"
            "• FTP/SFTP downloads\n"
            "• BitTorrent downloads\n"
            "• Media downloads (yt-dlp)\n\n"
            "Built with PySide6 and FastAPI"
        )

    def show(self) -> None:
        """Show the main window."""
        logger.info("Showing main window")
        super().show()
        self._update_status()

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        logger.info("Main window close requested")

        # Emit shutdown signal
        self.shutdown_requested.emit()

        # Stop update timer
        if self._update_timer:
            self._update_timer.stop()

        # Accept the close event
        event.accept()
        logger.info("Main window closed")
