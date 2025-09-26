"""Main application controller."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
from typing import TYPE_CHECKING

from ..api.server import APIServer
from ..gui.main_window import MainWindow
from ..storage.database import DatabaseManager
from .task_manager import TaskManager

if TYPE_CHECKING:
    from ..config.manager import ConfigManager

logger = logging.getLogger(__name__)


class ApplicationError(Exception):
    """Base exception for application errors."""

    pass


class Application:
    """Main application controller that coordinates all components."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """
        Initialize the application with dependency injection.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Core components (dependency injection)
        self.database: DatabaseManager | None = None
        self.task_manager: TaskManager | None = None
        self.api_server: APIServer | None = None
        self.main_window: MainWindow | None = None

        # Runtime state
        self._server_task: asyncio.Task[None] | None = None
        self._gui_thread: threading.Thread | None = None

        logger.info("Application initialized with dependency injection")

    async def _initialize_core_components(self) -> None:
        """Initialize core application components."""
        try:
            # Get global configuration
            global_config = self.config_manager.get_global_config()

            # Initialize database
            self.database = DatabaseManager(
                db_path=global_config.database_path,
            )
            await self.database.initialize()
            logger.info("Database manager initialized")

            # Initialize task manager
            self.task_manager = TaskManager(
                database=self.database,
                max_concurrent=global_config.max_concurrent_downloads,
            )
            await self.task_manager.initialize()
            logger.info("Task manager initialized")

        except Exception as e:
            logger.error(f"Failed to initialize core components: {e}")
            raise ApplicationError(f"Core component initialization failed: {e}") from e

    def start_gui(self) -> None:
        """
        Start the application in GUI mode with embedded server.
        
        Implements Requirement 3.1: GUI mode with embedded server
        """
        logger.info("Starting GUI mode with embedded server")

        try:
            # Configure logging for GUI mode
            self._configure_logging_for_mode("gui")

            # Initialize PySide6 application
            from PySide6.QtCore import QTimer
            from PySide6.QtWidgets import QApplication

            # Create Qt application
            qt_app = QApplication(sys.argv)
            qt_app.setApplicationName("Zuup Download Manager")
            qt_app.setApplicationVersion("1.0.0")

            # Initialize core components in async context
            async def init_and_run():
                await self._initialize_core_components()

                # Initialize GUI components
                self.main_window = MainWindow(
                    task_manager=self.task_manager,
                    config_manager=self.config_manager,
                )

                # Start embedded server
                global_config = self.config_manager.get_global_config()
                self.api_server = APIServer(
                    host="127.0.0.1",  # Local only for embedded mode
                    port=global_config.server_port,
                    task_manager=self.task_manager,
                    config_manager=self.config_manager,
                )

                self._server_task = asyncio.create_task(self.api_server.start())
                logger.info(f"Embedded server started on port {global_config.server_port}")

                # Show main window
                self.main_window.show()
                self._running = True

                # Setup graceful shutdown
                self._setup_signal_handlers()

                logger.info("GUI mode with embedded server started successfully")

            # Run async initialization
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Use QTimer to integrate asyncio with Qt event loop
            timer = QTimer()
            timer.timeout.connect(lambda: loop.run_until_complete(asyncio.sleep(0.01)))
            timer.start(10)  # 10ms interval

            # Initialize components
            loop.run_until_complete(init_and_run())

            # Run Qt application
            exit_code = qt_app.exec()

            # Cleanup
            loop.run_until_complete(self._shutdown())
            loop.close()

            sys.exit(exit_code)

        except Exception as e:
            logger.error(f"Failed to start GUI mode: {e}")
            raise ApplicationError(f"GUI startup failed: {e}") from e

    def start_server(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """
        Start the application in headless server mode.
        
        Implements Requirement 3.2: Headless server mode for remote access
        
        Args:
            host: Server host address
            port: Server port number
        """
        logger.info(f"Starting headless server mode on {host}:{port}")

        try:
            # Configure logging for server mode
            self._configure_logging_for_mode("server")

            # Run server in async context
            async def run_server():
                await self._initialize_core_components()

                # Initialize API server
                self.api_server = APIServer(
                    host=host,
                    port=port,
                    task_manager=self.task_manager,
                    config_manager=self.config_manager,
                )

                # Setup graceful shutdown
                self._setup_signal_handlers()

                # Start server
                self._running = True
                await self.api_server.start()

                # Wait for shutdown signal
                await self._shutdown_event.wait()

                logger.info("Headless server mode started successfully")

            # Run server
            asyncio.run(run_server())

        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Failed to start server mode: {e}")
            raise ApplicationError(f"Server startup failed: {e}") from e

    def start_combined(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """
        Start the application with both GUI and server.
        
        Implements Requirement 3.1: GUI mode with embedded server
        
        Args:
            host: Server host address  
            port: Server port number
        """
        logger.info(f"Starting combined mode with GUI and server on {host}:{port}")

        try:
            # Configure logging for combined mode
            self._configure_logging_for_mode("combined")

            # Initialize PySide6 application
            from PySide6.QtCore import QTimer
            from PySide6.QtWidgets import QApplication

            # Create Qt application
            qt_app = QApplication(sys.argv)
            qt_app.setApplicationName("Zuup Download Manager")
            qt_app.setApplicationVersion("1.0.0")

            # Initialize components in async context
            async def init_and_run():
                await self._initialize_core_components()

                # Initialize GUI components
                self.main_window = MainWindow(
                    task_manager=self.task_manager,
                    config_manager=self.config_manager,
                )

                # Initialize API server
                self.api_server = APIServer(
                    host=host,
                    port=port,
                    task_manager=self.task_manager,
                    config_manager=self.config_manager,
                )

                # Start server in background
                self._server_task = asyncio.create_task(self.api_server.start())
                logger.info(f"API server started on {host}:{port}")

                # Show main window
                self.main_window.show()
                self._running = True

                # Setup graceful shutdown
                self._setup_signal_handlers()

                logger.info("Combined mode started successfully")

            # Run async initialization
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Use QTimer to integrate asyncio with Qt event loop
            timer = QTimer()
            timer.timeout.connect(lambda: loop.run_until_complete(asyncio.sleep(0.01)))
            timer.start(10)  # 10ms interval

            # Initialize components
            loop.run_until_complete(init_and_run())

            # Run Qt application
            exit_code = qt_app.exec()

            # Cleanup
            loop.run_until_complete(self._shutdown())
            loop.close()

            sys.exit(exit_code)

        except Exception as e:
            logger.error(f"Failed to start combined mode: {e}")
            raise ApplicationError(f"Combined mode startup failed: {e}") from e

    def _configure_logging_for_mode(self, mode: str) -> None:
        """
        Configure logging based on deployment mode.
        
        Implements Requirement 3.3: Configure appropriate logging and monitoring
        
        Args:
            mode: Deployment mode (gui, server, combined)
        """
        global_config = self.config_manager.get_global_config()

        # Configure logging level based on mode
        if mode == "server":
            # More verbose logging for server mode
            logging.getLogger().setLevel(logging.INFO)
            logger.info("Configured logging for server mode")
        elif mode == "gui":
            # Less verbose for GUI mode
            logging.getLogger().setLevel(logging.WARNING)
            logger.info("Configured logging for GUI mode")
        elif mode == "combined":
            # Balanced logging for combined mode
            logging.getLogger().setLevel(logging.INFO)
            logger.info("Configured logging for combined mode")

        # Configure monitoring based on global config
        if global_config.enable_monitoring:
            logger.info("Monitoring enabled for deployment mode")
        else:
            logger.info("Monitoring disabled for deployment mode")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            def signal_handler(signum: int, frame) -> None:
                logger.info(f"Received signal {signum}, initiating shutdown")
                if self._running:
                    asyncio.create_task(self._initiate_shutdown())

            # Setup signal handlers for Unix systems (only works in main thread)
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, signal_handler)
            if hasattr(signal, 'SIGINT'):
                signal.signal(signal.SIGINT, signal_handler)

        except ValueError as e:
            # Signal handlers can only be set in the main thread
            logger.warning(f"Could not setup signal handlers: {e}")
            logger.info("Signal handling will be managed by the calling process")

    async def _initiate_shutdown(self) -> None:
        """Initiate graceful shutdown."""
        logger.info("Initiating graceful shutdown")
        self._running = False
        self._shutdown_event.set()
        await self._shutdown()

    async def _shutdown(self) -> None:
        """Perform graceful shutdown of all components."""
        logger.info("Shutting down application components")

        try:
            # Stop API server
            if self.api_server:
                await self.api_server.stop()
                logger.info("API server stopped")

            # Cancel server task
            if self._server_task and not self._server_task.done():
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass

            # Shutdown task manager
            if self.task_manager:
                await self.task_manager.shutdown()
                logger.info("Task manager stopped")

            # Close database
            if self.database:
                await self.database.close()
                logger.info("Database closed")

            # Close GUI
            if self.main_window:
                self.main_window.close()
                logger.info("GUI closed")

            logger.info("Application shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def is_running(self) -> bool:
        """Check if application is running."""
        return self._running

    def get_status(self) -> dict[str, str | bool | dict]:
        """
        Get application status information.
        
        Returns:
            Dictionary with application status
        """
        status = {
            "running": self._running,
            "components": {
                "database": self.database is not None,
                "task_manager": self.task_manager is not None,
                "api_server": self.api_server is not None,
                "gui": self.main_window is not None,
            }
        }

        # Add task manager stats if available
        if self.task_manager:
            status["task_stats"] = self.task_manager.get_manager_stats()

        return status
