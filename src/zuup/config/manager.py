"""Configuration manager implementation."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Generic, TypeVar

from cryptography.fernet import Fernet
from pydantic import BaseModel, ValidationError

from .defaults import get_default_global_config, get_default_task_config
from .settings import GlobalConfig, TaskConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ValidationResult(Generic[T]):
    """Result of configuration validation."""

    def __init__(
        self, is_valid: bool, config: T | None = None, errors: list[str] | None = None
    ):
        self.is_valid = is_valid
        self.config = config
        self.errors = errors or []


class SecureStorage:
    """Handles secure storage of sensitive configuration data."""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.key_file = config_dir / ".encryption_key"
        self.credentials_file = config_dir / ".credentials.enc"
        self._key: bytes | None = None

    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key."""
        if self._key is not None:
            return self._key

        if self.key_file.exists():
            try:
                with self.key_file.open("rb") as f:
                    self._key = f.read()
                logger.debug("Loaded existing encryption key")
            except Exception as e:
                logger.warning(f"Failed to load encryption key: {e}")
                self._key = None

        if self._key is None:
            # Generate new key
            self._key = Fernet.generate_key()
            try:
                # Ensure only owner can read the key file
                self.key_file.touch(mode=0o600)
                with self.key_file.open("wb") as f:
                    f.write(self._key)
                logger.info("Generated new encryption key")
            except Exception as e:
                logger.error(f"Failed to save encryption key: {e}")
                raise

        return self._key

    def store_credentials(self, credentials: dict[str, Any]) -> None:
        """Store encrypted credentials."""
        try:
            key = self._get_or_create_key()
            fernet = Fernet(key)

            # Serialize and encrypt
            data = json.dumps(credentials).encode()
            encrypted_data = fernet.encrypt(data)

            # Ensure only owner can read the credentials file
            self.credentials_file.touch(mode=0o600)
            with self.credentials_file.open("wb") as f:
                f.write(encrypted_data)

            logger.debug("Stored encrypted credentials")
        except Exception as e:
            logger.error(f"Failed to store credentials: {e}")
            raise

    def load_credentials(self) -> dict[str, Any]:
        """Load and decrypt credentials."""
        if not self.credentials_file.exists():
            return {}

        try:
            key = self._get_or_create_key()
            fernet = Fernet(key)

            with self.credentials_file.open("rb") as f:
                encrypted_data = f.read()

            # Decrypt and deserialize
            decrypted_data = fernet.decrypt(encrypted_data)
            credentials: dict[str, Any] = json.loads(decrypted_data.decode())

            logger.debug("Loaded encrypted credentials")
            return credentials
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return {}


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

        # Configuration file paths
        self.global_config_file = self.config_dir / "global_config.json"
        self.task_configs_dir = self.config_dir / "tasks"
        self.task_configs_dir.mkdir(exist_ok=True)

        # Secure storage for credentials
        self.secure_storage = SecureStorage(self.config_dir)

        self._global_config: GlobalConfig | None = None
        self._task_configs: dict[str, TaskConfig] = {}

        logger.info(f"ConfigManager initialized with config dir: {config_dir}")

    def _apply_env_overrides(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_prefix = "ZUUP_"

        # Map of environment variable suffixes to config paths
        env_mappings = {
            "MAX_CONCURRENT_DOWNLOADS": "max_concurrent_downloads",
            "DEFAULT_DOWNLOAD_PATH": "default_download_path",
            "TEMP_DIRECTORY": "temp_directory",
            "MAX_CONNECTIONS_PER_DOWNLOAD": "max_connections_per_download",
            "USER_AGENT": "user_agent",
            "LOGGING_LEVEL": "logging_level",
            "SERVER_HOST": "server_host",
            "SERVER_PORT": "server_port",
            "THEME": "theme",
            "AUTO_START_DOWNLOADS": "auto_start_downloads",
            "SHOW_NOTIFICATIONS": "show_notifications",
            # Proxy settings
            "PROXY_ENABLED": "proxy_settings.enabled",
            "HTTP_PROXY": "proxy_settings.http_proxy",
            "HTTPS_PROXY": "proxy_settings.https_proxy",
            "SOCKS_PROXY": "proxy_settings.socks_proxy",
            "PROXY_USERNAME": "proxy_settings.username",
            "PROXY_PASSWORD": "proxy_settings.password",
        }

        for env_suffix, config_path in env_mappings.items():
            env_var = env_prefix + env_suffix
            env_value = os.getenv(env_var)

            if env_value is not None:
                # Handle nested config paths (e.g., "proxy_settings.enabled")
                keys = config_path.split(".")
                current = config_dict

                # Navigate to parent of target key
                for key in keys[:-1]:
                    if key not in current or current[key] is None:
                        current[key] = {}
                    current = current[key]

                # Set the value with appropriate type conversion
                final_key = keys[-1]
                try:
                    # Convert string values to appropriate types
                    if env_suffix.endswith(("_PORT", "_DOWNLOADS", "_CONNECTIONS")):
                        current[final_key] = int(env_value)
                    elif env_suffix.endswith(
                        ("_DOWNLOADS", "_NOTIFICATIONS", "_ENABLED")
                    ):
                        current[final_key] = env_value.lower() in (
                            "true",
                            "1",
                            "yes",
                            "on",
                        )
                    elif env_suffix.endswith(("_PATH", "_DIRECTORY")):
                        current[final_key] = Path(env_value)
                    else:
                        current[final_key] = env_value

                    logger.debug(f"Applied environment override: {env_var}={env_value}")
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Invalid environment variable {env_var}={env_value}: {e}"
                    )

        return config_dict

    def _load_config_file(self, file_path: Path, config_class: type[T]) -> T | None:
        """Load configuration from JSON file with validation."""
        if not file_path.exists():
            return None

        try:
            with file_path.open(encoding="utf-8") as f:
                config_dict = json.load(f)

            # Apply environment variable overrides
            config_dict = self._apply_env_overrides(config_dict)

            # Validate and create config object
            config = config_class.model_validate(config_dict)
            logger.debug(f"Loaded configuration from {file_path}")
            return config

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error loading configuration from {file_path}: {e}"
            )
            return None

    def _save_config_file(self, file_path: Path, config: BaseModel) -> bool:
        """Save configuration to JSON file."""
        try:
            # Create parent directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict and save
            config_dict = config.model_dump(mode="json")

            with file_path.open("w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved configuration to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            return False

    def get_global_config(self) -> GlobalConfig:
        """
        Get global configuration.

        Returns:
            Global configuration object
        """
        if self._global_config is None:
            # Try to load from file
            self._global_config = self._load_config_file(
                self.global_config_file, GlobalConfig
            )

            if self._global_config is None:
                # Use default configuration
                self._global_config = get_default_global_config()

                # Apply environment overrides to default config
                config_dict = self._global_config.model_dump()
                config_dict = self._apply_env_overrides(config_dict)
                self._global_config = GlobalConfig.model_validate(config_dict)

                # Save default configuration
                self._save_config_file(self.global_config_file, self._global_config)
                logger.info("Created default global configuration")
            else:
                logger.info("Loaded global configuration from file")

        return self._global_config

    def get_task_config(self, task_id: str) -> TaskConfig:
        """
        Get task-specific configuration.

        Args:
            task_id: Task ID

        Returns:
            Task configuration object
        """
        if task_id not in self._task_configs:
            # Try to load from file
            task_config_file = self.task_configs_dir / f"{task_id}.json"
            task_config = self._load_config_file(task_config_file, TaskConfig)

            if task_config is None:
                # Use default task configuration
                task_config = get_default_task_config()
                logger.debug(f"Using default task configuration for {task_id}")
            else:
                logger.debug(f"Loaded task configuration for {task_id}")

            self._task_configs[task_id] = task_config

        return self._task_configs[task_id]

    def update_global_config(self, config: GlobalConfig) -> None:
        """
        Update global configuration.

        Args:
            config: New global configuration
        """
        # Validate configuration
        validation_result = self.validate_config(config)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid configuration: {validation_result.errors}")

        # Save to file
        if self._save_config_file(self.global_config_file, config):
            self._global_config = config
            logger.info("Global configuration updated")
        else:
            raise RuntimeError("Failed to save global configuration")

    def update_task_config(self, task_id: str, config: TaskConfig) -> None:
        """
        Update task-specific configuration.

        Args:
            task_id: Task ID
            config: New task configuration
        """
        # Validate configuration
        validation_result = self.validate_config(config)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid task configuration: {validation_result.errors}")

        # Save to file
        task_config_file = self.task_configs_dir / f"{task_id}.json"
        if self._save_config_file(task_config_file, config):
            self._task_configs[task_id] = config
            logger.info(f"Task configuration updated for {task_id}")
        else:
            raise RuntimeError(f"Failed to save task configuration for {task_id}")

    def validate_config(self, config: T) -> ValidationResult[T]:
        """
        Validate configuration object.

        Args:
            config: Configuration object to validate

        Returns:
            ValidationResult with validation status and any errors
        """
        try:
            # Pydantic models automatically validate on creation
            # We can perform additional validation here if needed
            validated_config = config.model_validate(config.model_dump())
            return ValidationResult(is_valid=True, config=validated_config)
        except ValidationError as e:
            errors = [
                f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                for error in e.errors()
            ]
            return ValidationResult(is_valid=False, errors=errors)
        except Exception as e:
            return ValidationResult(is_valid=False, errors=[str(e)])

    def store_secure_credentials(self, credentials: dict[str, Any]) -> None:
        """
        Store sensitive credentials securely.

        Args:
            credentials: Dictionary of credentials to store securely
        """
        self.secure_storage.store_credentials(credentials)
        logger.info("Stored secure credentials")

    def load_secure_credentials(self) -> dict[str, Any]:
        """
        Load secure credentials.

        Returns:
            Dictionary of stored credentials
        """
        credentials = self.secure_storage.load_credentials()
        logger.debug("Loaded secure credentials")
        return credentials

    def delete_task_config(self, task_id: str) -> bool:
        """
        Delete task-specific configuration.

        Args:
            task_id: Task ID

        Returns:
            True if configuration was deleted, False otherwise
        """
        try:
            task_config_file = self.task_configs_dir / f"{task_id}.json"
            if task_config_file.exists():
                task_config_file.unlink()
                logger.info(f"Deleted task configuration for {task_id}")

            # Remove from cache
            self._task_configs.pop(task_id, None)
            return True

        except Exception as e:
            logger.error(f"Failed to delete task configuration for {task_id}: {e}")
            return False

    def reset_to_defaults(self) -> None:
        """Reset all configuration to defaults."""
        try:
            # Reset global config
            self._global_config = get_default_global_config()
            self._save_config_file(self.global_config_file, self._global_config)

            # Clear task configs
            self._task_configs.clear()

            # Remove all task config files
            for config_file in self.task_configs_dir.glob("*.json"):
                config_file.unlink()

            logger.info("Reset all configuration to defaults")

        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}")
            raise

    def export_config(self, export_path: Path) -> bool:
        """
        Export all configuration to a file.

        Args:
            export_path: Path to export configuration to

        Returns:
            True if export was successful, False otherwise
        """
        try:
            export_data = {
                "global_config": self.get_global_config().model_dump(mode="json"),
                "task_configs": {
                    task_id: config.model_dump(mode="json")
                    for task_id, config in self._task_configs.items()
                },
            }

            with export_path.open("w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported configuration to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False

    def import_config(self, import_path: Path) -> bool:
        """
        Import configuration from a file.

        Args:
            import_path: Path to import configuration from

        Returns:
            True if import was successful, False otherwise
        """
        try:
            with import_path.open(encoding="utf-8") as f:
                import_data = json.load(f)

            # Import global config
            if "global_config" in import_data:
                global_config = GlobalConfig.model_validate(
                    import_data["global_config"]
                )
                self.update_global_config(global_config)

            # Import task configs
            if "task_configs" in import_data:
                for task_id, task_config_data in import_data["task_configs"].items():
                    task_config = TaskConfig.model_validate(task_config_data)
                    self.update_task_config(task_id, task_config)

            logger.info(f"Imported configuration from {import_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False
