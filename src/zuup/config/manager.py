"""Configuration manager implementation."""

import logging
from pathlib import Path

from .settings import GlobalConfig, TaskConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages application configuration with type safety and validation."""

    def __init__(self, config_dir: Path | None = None) -> None:
        """
        Initialize configuration manager.

        Args:
            config_dir: Optional custom configuration directory
        """
        if config_dir is None:
            config_dir = Path.home() / ".config" / "zuup"

        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._global_config: GlobalConfig | None = None
        logger.info(f"ConfigManager initialized with config dir: {config_dir}")

    def get_global_config(self) -> GlobalConfig:
        """
        Get global configuration.

        Returns:
            Global configuration object
        """
        if self._global_config is None:
            # Implementation will be added in task 3
            self._global_config = GlobalConfig()
            logger.info("Loaded default global configuration")

        return self._global_config

    def get_task_config(self, task_id: str) -> TaskConfig:
        """
        Get task-specific configuration.

        Args:
            task_id: Task ID

        Returns:
            Task configuration object
        """
        # Implementation will be added in task 3
        logger.info(f"Loading task config for {task_id}")
        return TaskConfig()

    def update_global_config(self, config: GlobalConfig) -> None:
        """
        Update global configuration.

        Args:
            config: New global configuration
        """
        # Implementation will be added in task 3
        self._global_config = config
        logger.info("Global configuration updated")

    def update_task_config(self, task_id: str, config: TaskConfig) -> None:
        """
        Update task-specific configuration.

        Args:
            task_id: Task ID
            config: New task configuration
        """
        # Implementation will be added in task 3
        logger.info(f"Task configuration updated for {task_id} with config: {config}")
