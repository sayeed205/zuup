"""Main application controller."""

import logging

from ..config.manager import ConfigManager

logger = logging.getLogger(__name__)


class Application:
    """Main application controller that coordinates all components."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """
        Initialize the application.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        logger.info("Application initialized")

    def start_gui(self) -> None:
        """Start the application in GUI mode."""
        logger.info("Starting GUI mode")
        # Implementation will be added in task 7
        raise NotImplementedError("GUI mode will be implemented in task 7")

    def start_server(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """
        Start the application in server-only mode.

        Args:
            host: Server host address
            port: Server port number
        """
        logger.info(f"Starting server mode on {host}:{port}")
        # Implementation will be added in task 7
        raise NotImplementedError("Server mode will be implemented in task 7")

    def start_combined(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """
        Start the application with both GUI and server.

        Args:
            host: Server host address
            port: Server port number
        """
        logger.info(f"Starting combined mode with server on {host}:{port}")
        # Implementation will be added in task 7
        raise NotImplementedError("Combined mode will be implemented in task 7")
