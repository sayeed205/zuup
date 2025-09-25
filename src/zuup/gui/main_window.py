"""Main window for PySide6 GUI application."""

import logging

logger = logging.getLogger(__name__)


class MainWindow:
    """Main application window using PySide6."""

    def __init__(self) -> None:
        """Initialize main window."""
        logger.info("MainWindow initialized")

    def show(self) -> None:
        """Show the main window."""
        # Implementation will be added in task 7
        logger.info("Showing main window")
        raise NotImplementedError("GUI will be implemented in task 7")

    def close(self) -> None:
        """Close the main window."""
        # Implementation will be added in task 7
        logger.info("Closing main window")
        raise NotImplementedError("GUI will be implemented in task 7")
