#!/usr/bin/env python3
"""
Manual test script for basic functionality.

This script provides interactive testing of core components.
"""

from pathlib import Path

from zuup.config.manager import ConfigManager
from zuup.core.app import Application
from zuup.storage.models import DownloadTask, EngineType
from zuup.utils.helpers import format_bytes, format_duration, format_speed
from zuup.utils.logging import setup_logging
from zuup.utils.validation import validate_path, validate_url


def test_data_models() -> None:
    """Test data model creation and validation."""
    print("🔄 Testing data models...")

    try:
        # Test DownloadTask creation
        task = DownloadTask(
            url="https://example.com/file.zip",
            destination=Path("/tmp/test"),
            engine_type=EngineType.HTTP,
        )

        print(f"✅ Created task with ID: {task.id}")
        print(f"   URL: {task.url}")
        print(f"   Status: {task.status}")
        print(f"   Engine: {task.engine_type}")

        # Test status changes
        task.mark_failed("Test error")
        print(f"✅ Task status changed to: {task.status}")

    except Exception as e:
        print(f"❌ Data model test failed: {e}")


def test_validation() -> None:
    """Test validation utilities."""
    print("🔄 Testing validation utilities...")

    # Test URL validation
    test_urls = [
        "https://example.com/file.zip",
        "http://test.org/download",
        "ftp://files.example.com/data.tar.gz",
        "invalid-url",
        "not-a-url-at-all",
    ]

    for url in test_urls:
        is_valid = validate_url(url)
        status = "✅" if is_valid else "❌"
        print(f"   {status} {url}: {is_valid}")

    # Test path validation
    test_paths = [
        "/tmp/test",
        "~/Downloads",
        "/invalid/path/that/does/not/exist",
        "relative/path",
    ]

    for path in test_paths:
        is_valid = validate_path(path)
        status = "✅" if is_valid else "❌"
        print(f"   {status} {path}: {is_valid}")


def test_formatting() -> None:
    """Test formatting utilities."""
    print("🔄 Testing formatting utilities...")

    # Test byte formatting
    test_bytes = [0, 1024, 1048576, 1073741824, 1099511627776]
    for bytes_val in test_bytes:
        formatted = format_bytes(bytes_val)
        print(f"   {bytes_val} bytes = {formatted}")

    # Test speed formatting
    test_speeds = [0, 1024, 1048576, 10485760]
    for speed in test_speeds:
        formatted = format_speed(speed)
        print(f"   {speed} B/s = {formatted}")

    # Test duration formatting
    test_durations = [0, 30, 90, 3600, 7200, 86400]
    for duration in test_durations:
        formatted = format_duration(duration)
        print(f"   {duration} seconds = {formatted}")


def test_configuration() -> None:
    """Test configuration management."""
    print("🔄 Testing configuration management...")

    try:
        config_manager = ConfigManager()
        global_config = config_manager.get_global_config()

        print("✅ Global config loaded:")
        print(f"   Max concurrent downloads: {global_config.max_concurrent_downloads}")
        print(f"   Default download path: {global_config.default_download_path}")
        print(f"   User agent: {global_config.user_agent}")

        # Test task config
        task_config = config_manager.get_task_config("test-task")
        print("✅ Task config loaded:")
        print(f"   Retry attempts: {task_config.retry_attempts}")
        print(f"   Timeout: {task_config.timeout}")

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")


def test_application() -> None:
    """Test application initialization."""
    print("🔄 Testing application initialization...")

    try:
        config_manager = ConfigManager()
        app = Application(config_manager=config_manager)

        print("✅ Application initialized successfully")

        # Test that methods exist (they should raise NotImplementedError)
        try:
            app.start_gui()
        except NotImplementedError:
            print("✅ GUI mode method exists (not implemented yet)")

        try:
            app.start_server()
        except NotImplementedError:
            print("✅ Server mode method exists (not implemented yet)")

    except Exception as e:
        print(f"❌ Application test failed: {e}")


def main() -> None:
    """Run all manual tests."""
    setup_logging(level="INFO")

    print("🚀 Running Zuup manual functionality tests")
    print("=" * 50)

    test_data_models()
    print()

    test_validation()
    print()

    test_formatting()
    print()

    test_configuration()
    print()

    test_application()
    print()

    print("=" * 50)
    print("✅ Manual tests completed")
    print(
        "\nNote: Some functionality is not yet implemented and will be added in later tasks."
    )


if __name__ == "__main__":
    main()
