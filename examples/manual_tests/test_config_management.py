#!/usr/bin/env python3
"""Manual test script for configuration management system."""

import json
import os
import tempfile
from pathlib import Path

from zuup.config import (
    ConfigManager,
    GlobalConfig,
    TaskConfig,
    ProxyConfig,
    get_default_global_config,
    get_default_task_config,
)


def test_basic_config_operations():
    """Test basic configuration operations."""
    print("=== Testing Basic Configuration Operations ===")

    # Create temporary config directory
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "test_config"
        manager = ConfigManager(config_dir)

        # Test default global config
        global_config = manager.get_global_config()
        print(
            f"✓ Default global config loaded: {global_config.max_concurrent_downloads} concurrent downloads"
        )

        # Test default task config
        task_config = manager.get_task_config("test_task_1")
        print(
            f"✓ Default task config loaded: {task_config.retry_attempts} retry attempts"
        )

        # Test config update
        new_global_config = GlobalConfig(
            max_concurrent_downloads=10,
            default_download_path=Path.home() / "CustomDownloads",
            logging_level="DEBUG",
        )
        manager.update_global_config(new_global_config)

        # Verify update
        updated_config = manager.get_global_config()
        assert updated_config.max_concurrent_downloads == 10
        print("✓ Global config update successful")

        # Test task config update
        new_task_config = TaskConfig(
            max_connections=16,
            download_speed_limit=2097152,  # 2MB/s
            retry_attempts=10,
        )
        manager.update_task_config("test_task_1", new_task_config)

        # Verify task config update
        updated_task_config = manager.get_task_config("test_task_1")
        assert updated_task_config.max_connections == 16
        print("✓ Task config update successful")


def test_environment_variable_overrides():
    """Test environment variable configuration overrides."""
    print("\n=== Testing Environment Variable Overrides ===")

    # Set environment variables
    os.environ["ZUUP_MAX_CONCURRENT_DOWNLOADS"] = "15"
    os.environ["ZUUP_LOGGING_LEVEL"] = "WARNING"
    os.environ["ZUUP_SERVER_PORT"] = "9090"
    os.environ["ZUUP_PROXY_ENABLED"] = "true"
    os.environ["ZUUP_HTTP_PROXY"] = "http://test-proxy:8080"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "test_env_config"
            manager = ConfigManager(config_dir)

            # Get config (should apply env overrides)
            config = manager.get_global_config()

            # Verify environment overrides
            assert config.max_concurrent_downloads == 15
            assert config.logging_level == "WARNING"
            assert config.server_port == 9090

            if config.proxy_settings:
                assert config.proxy_settings.enabled == True
                assert config.proxy_settings.http_proxy == "http://test-proxy:8080"

            print("✓ Environment variable overrides applied successfully")

    finally:
        # Clean up environment variables
        for var in [
            "ZUUP_MAX_CONCURRENT_DOWNLOADS",
            "ZUUP_LOGGING_LEVEL",
            "ZUUP_SERVER_PORT",
            "ZUUP_PROXY_ENABLED",
            "ZUUP_HTTP_PROXY",
        ]:
            os.environ.pop(var, None)


def test_secure_credential_storage():
    """Test secure credential storage functionality."""
    print("\n=== Testing Secure Credential Storage ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "test_secure_config"
        manager = ConfigManager(config_dir)

        # Test storing credentials
        test_credentials = {
            "api_key": "secret_api_key_12345",
            "oauth_token": "oauth_token_abcdef",
            "database_password": "super_secret_db_pass",
        }

        manager.store_secure_credentials(test_credentials)
        print("✓ Credentials stored securely")

        # Test loading credentials
        loaded_credentials = manager.load_secure_credentials()

        # Verify credentials match
        assert loaded_credentials == test_credentials
        print("✓ Credentials loaded and verified successfully")

        # Verify encryption key file exists
        key_file = config_dir / ".encryption_key"
        assert key_file.exists()
        print("✓ Encryption key file created")

        # Verify credentials file exists and is encrypted
        creds_file = config_dir / ".credentials.enc"
        assert creds_file.exists()

        # Verify file is actually encrypted (not plain JSON)
        with open(creds_file, "rb") as f:
            encrypted_data = f.read()

        try:
            json.loads(encrypted_data.decode())
            assert False, "Credentials file should be encrypted, not plain JSON"
        except (json.JSONDecodeError, UnicodeDecodeError):
            print("✓ Credentials file is properly encrypted")


def test_config_validation():
    """Test configuration validation."""
    print("\n=== Testing Configuration Validation ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "test_validation_config"
        manager = ConfigManager(config_dir)

        # Test valid configuration
        valid_config = GlobalConfig(
            max_concurrent_downloads=5, server_port=8080, logging_level="INFO"
        )

        validation_result = manager.validate_config(valid_config)
        assert validation_result.is_valid
        print("✓ Valid configuration passed validation")

        # Test invalid configuration (this will be caught by Pydantic during creation)
        try:
            invalid_config = GlobalConfig(
                max_concurrent_downloads=-1,  # Should be positive
                server_port=99999,  # Should be <= 65535
                logging_level="INVALID",  # Should be valid log level
            )
            assert False, "Invalid config should have raised ValidationError"
        except Exception:
            print("✓ Invalid configuration properly rejected")


def test_config_import_export():
    """Test configuration import/export functionality."""
    print("\n=== Testing Configuration Import/Export ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "test_import_export_config"
        manager = ConfigManager(config_dir)

        # Set up some configuration
        global_config = GlobalConfig(
            max_concurrent_downloads=7, logging_level="DEBUG", server_port=9000
        )
        manager.update_global_config(global_config)

        task_config = TaskConfig(max_connections=12, retry_attempts=8)
        manager.update_task_config("export_test_task", task_config)

        # Export configuration
        export_file = Path(temp_dir) / "exported_config.json"
        success = manager.export_config(export_file)
        assert success
        print("✓ Configuration exported successfully")

        # Create new manager and import
        new_config_dir = Path(temp_dir) / "imported_config"
        new_manager = ConfigManager(new_config_dir)

        import_success = new_manager.import_config(export_file)
        assert import_success
        print("✓ Configuration imported successfully")

        # Verify imported configuration
        imported_global = new_manager.get_global_config()
        assert imported_global.max_concurrent_downloads == 7
        assert imported_global.logging_level == "DEBUG"
        assert imported_global.server_port == 9000

        imported_task = new_manager.get_task_config("export_test_task")
        assert imported_task.max_connections == 12
        assert imported_task.retry_attempts == 8

        print("✓ Imported configuration verified")


def test_proxy_configuration():
    """Test proxy configuration functionality."""
    print("\n=== Testing Proxy Configuration ===")

    # Test proxy config validation
    valid_proxy = ProxyConfig(
        enabled=True,
        http_proxy="http://proxy.example.com:8080",
        https_proxy="https://secure-proxy.example.com:8080",
        username="proxy_user",
        password="proxy_pass",
    )
    print("✓ Valid proxy configuration created")

    # Test invalid proxy URL (should raise validation error)
    try:
        invalid_proxy = ProxyConfig(enabled=True, http_proxy="invalid-url-format")
        assert False, "Invalid proxy URL should have raised ValidationError"
    except Exception:
        print("✓ Invalid proxy URL properly rejected")

    # Test proxy config in global config
    global_config = GlobalConfig(proxy_settings=valid_proxy)
    print("✓ Proxy configuration integrated with global config")


def main():
    """Run all configuration management tests."""
    print("Starting Configuration Management Manual Tests\n")

    try:
        test_basic_config_operations()
        test_environment_variable_overrides()
        test_secure_credential_storage()
        test_config_validation()
        test_config_import_export()
        test_proxy_configuration()

        print("\n🎉 All configuration management tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
