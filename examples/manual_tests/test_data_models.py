#!/usr/bin/env python3
"""Manual test script for data models and validation."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from zuup.storage.models import (
    DownloadTask,
    EngineType,
    GlobalConfig,
    ProgressInfo,
    ProxyConfig,
    TaskConfig,
    TaskStatus,
)
from zuup.storage.validation import (
    ModelValidator,
    URLValidator,
    FileSystemValidator,
    ConfigValidator,
    validate_download_task_data,
    validate_config_data,
    is_supported_url,
    get_engine_for_url,
)


def test_task_status_enum():
    """Test TaskStatus enum."""
    print("Testing TaskStatus enum...")

    # Test all enum values
    statuses = [
        TaskStatus.PENDING,
        TaskStatus.DOWNLOADING,
        TaskStatus.PAUSED,
        TaskStatus.COMPLETED,
        TaskStatus.SEEDING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    ]

    for status in statuses:
        print(f"  ✓ {status.name} = {status.value}")

    print("TaskStatus enum test passed!\n")
    return True


def test_engine_type_enum():
    """Test EngineType enum."""
    print("Testing EngineType enum...")

    # Test all enum values
    engines = [
        EngineType.HTTP,
        EngineType.FTP,
        EngineType.TORRENT,
        EngineType.MEDIA,
    ]

    for engine in engines:
        print(f"  ✓ {engine.name} = {engine.value}")

    print("EngineType enum test passed!\n")
    return True


def test_progress_info_model():
    """Test ProgressInfo model and validation."""
    print("Testing ProgressInfo model...")

    # Valid progress info
    try:
        progress = ProgressInfo(
            downloaded_bytes=1024,
            total_bytes=2048,
            download_speed=512.5,
            status=TaskStatus.DOWNLOADING,
        )
        print(f"  ✓ Valid progress: {progress.progress_percentage}% complete")
    except Exception as e:
        print(f"  ✗ Failed to create valid progress: {e}")
        return False

    # Test validation - negative bytes
    try:
        ProgressInfo(downloaded_bytes=-100)
        print("  ✗ Should have failed with negative bytes")
        return False
    except Exception:
        print("  ✓ Correctly rejected negative bytes")

    # Test validation - downloaded > total
    try:
        ProgressInfo(downloaded_bytes=2048, total_bytes=1024)
        print("  ✗ Should have failed with downloaded > total")
        return False
    except Exception:
        print("  ✓ Correctly rejected downloaded > total")

    # Test torrent-specific fields
    try:
        torrent_progress = ProgressInfo(
            downloaded_bytes=1024,
            total_bytes=2048,
            upload_speed=256.0,
            peers_connected=5,
            peers_total=10,
            seeds_connected=2,
            seeds_total=3,
            ratio=0.5,
            is_seeding=False,
        )
        print("  ✓ Valid torrent progress created")
    except Exception as e:
        print(f"  ✗ Failed to create torrent progress: {e}")
        return False

    print("ProgressInfo model test passed!\n")
    return True


def test_proxy_config_model():
    """Test ProxyConfig model and validation."""
    print("Testing ProxyConfig model...")

    # Valid proxy config
    try:
        proxy = ProxyConfig(
            enabled=True,
            http_proxy="http://proxy.example.com:8080",
            https_proxy="https://proxy.example.com:8443",
            username="user",
            password="pass",
        )
        print("  ✓ Valid proxy config created")
    except Exception as e:
        print(f"  ✗ Failed to create valid proxy: {e}")
        return False

    # Test validation - invalid proxy URL
    try:
        ProxyConfig(enabled=True, http_proxy="invalid-url")
        print("  ✗ Should have failed with invalid proxy URL")
        return False
    except Exception:
        print("  ✓ Correctly rejected invalid proxy URL")

    # Test validation - enabled but no proxy URLs
    try:
        ProxyConfig(enabled=True)
        print("  ✗ Should have failed with enabled but no URLs")
        return False
    except Exception:
        print("  ✓ Correctly rejected enabled proxy without URLs")

    print("ProxyConfig model test passed!\n")
    return True


def test_task_config_model():
    """Test TaskConfig model and validation."""
    print("Testing TaskConfig model...")

    # Valid task config
    try:
        config = TaskConfig(
            max_connections=8,
            download_speed_limit=1048576,  # 1 MB/s
            retry_attempts=3,
            timeout=30,
            headers={"User-Agent": "Test"},
            cookies={"session": "abc123"},
        )
        print("  ✓ Valid task config created")
    except Exception as e:
        print(f"  ✗ Failed to create valid config: {e}")
        return False

    # Test validation - negative values
    try:
        TaskConfig(max_connections=-1)
        print("  ✗ Should have failed with negative max_connections")
        return False
    except Exception:
        print("  ✓ Correctly rejected negative max_connections")

    try:
        TaskConfig(timeout=0)
        print("  ✗ Should have failed with zero timeout")
        return False
    except Exception:
        print("  ✓ Correctly rejected zero timeout")

    print("TaskConfig model test passed!\n")
    return True


def test_global_config_model():
    """Test GlobalConfig model and validation."""
    print("Testing GlobalConfig model...")

    # Valid global config
    try:
        config = GlobalConfig(
            max_concurrent_downloads=5,
            default_download_path=Path.home() / "Downloads",
            max_connections_per_download=8,
            logging_level="INFO",
            server_port=8080,
            theme="dark",
        )
        print("  ✓ Valid global config created")
    except Exception as e:
        print(f"  ✗ Failed to create valid config: {e}")
        return False

    # Test validation - invalid logging level
    try:
        GlobalConfig(logging_level="INVALID")
        print("  ✗ Should have failed with invalid logging level")
        return False
    except Exception:
        print("  ✓ Correctly rejected invalid logging level")

    # Test validation - invalid port
    try:
        GlobalConfig(server_port=70000)
        print("  ✗ Should have failed with invalid port")
        return False
    except Exception:
        print("  ✓ Correctly rejected invalid port")

    print("GlobalConfig model test passed!\n")
    return True


def test_download_task_model():
    """Test DownloadTask model and validation."""
    print("Testing DownloadTask model...")

    # Valid HTTP download task
    try:
        task = DownloadTask(
            url="https://example.com/file.zip",
            destination=Path.home() / "Downloads" / "file.zip",
            engine_type=EngineType.HTTP,
        )
        print(f"  ✓ Valid HTTP task created with ID: {task.id}")
    except Exception as e:
        print(f"  ✗ Failed to create valid HTTP task: {e}")
        return False

    # Valid torrent task (magnet)
    try:
        torrent_task = DownloadTask(
            url="magnet:?xt=urn:btih:1234567890abcdef1234567890abcdef12345678",
            destination=Path.home() / "Downloads" / "torrent",
            engine_type=EngineType.TORRENT,
        )
        print(f"  ✓ Valid magnet torrent task created with ID: {torrent_task.id}")
    except Exception as e:
        print(f"  ✗ Failed to create valid magnet torrent task: {e}")
        return False

    # Valid torrent file task (remote)
    try:
        torrent_file_task = DownloadTask(
            url="https://example.com/file.torrent",
            destination=Path.home() / "Downloads" / "torrent_file",
            engine_type=EngineType.TORRENT,
        )
        print(
            f"  ✓ Valid remote .torrent file task created with ID: {torrent_file_task.id}"
        )
    except Exception as e:
        print(f"  ✗ Failed to create valid remote .torrent file task: {e}")
        return False

    # Valid local torrent file task
    try:
        local_torrent_task = DownloadTask(
            url="file:///home/user/downloads/example.torrent",
            destination=Path.home() / "Downloads" / "local_torrent",
            engine_type=EngineType.TORRENT,
        )
        print(
            f"  ✓ Valid local .torrent file task created with ID: {local_torrent_task.id}"
        )
    except Exception as e:
        print(f"  ✗ Failed to create valid local .torrent file task: {e}")
        return False

    # Test validation - invalid URL
    try:
        DownloadTask(
            url="invalid-url",
            destination=Path.home() / "Downloads" / "file",
            engine_type=EngineType.HTTP,
        )
        print("  ✗ Should have failed with invalid URL")
        return False
    except Exception:
        print("  ✓ Correctly rejected invalid URL")

    # Test validation - engine/URL mismatch
    try:
        DownloadTask(
            url="https://example.com/file.zip",
            destination=Path.home() / "Downloads" / "file.zip",
            engine_type=EngineType.TORRENT,
        )
        print("  ✗ Should have failed with engine/URL mismatch")
        return False
    except Exception:
        print("  ✓ Correctly rejected engine/URL mismatch")

    print("DownloadTask model test passed!\n")
    return True


def test_url_validator():
    """Test URL validation utilities."""
    print("Testing URL validation...")

    test_urls = [
        ("https://example.com/file.zip", True, "http"),
        ("ftp://ftp.example.com/file.txt", True, "ftp"),
        (
            "magnet:?xt=urn:btih:1234567890abcdef1234567890abcdef12345678",
            True,
            "torrent",
        ),
        ("https://example.com/file.torrent", True, "torrent"),
        ("file:///home/user/example.torrent", True, "torrent"),
        ("https://youtube.com/watch?v=example", True, "media"),
        ("invalid-url", False, None),
        ("file:///local/file.txt", True, "http"),
    ]

    for url, should_be_valid, expected_engine in test_urls:
        is_valid = is_supported_url(url)
        engine = get_engine_for_url(url)

        if is_valid == should_be_valid:
            print(f"  ✓ {url} -> valid: {is_valid}, engine: {engine}")
        else:
            print(f"  ✗ {url} -> expected valid: {should_be_valid}, got: {is_valid}")
            return False

        # Also check if engine matches expectation (when URL is valid)
        if is_valid and expected_engine and engine != expected_engine:
            print(f"  ✗ {url} -> expected engine: {expected_engine}, got: {engine}")
            return False

    print("URL validation test passed!\n")
    return True


def test_model_validation():
    """Test model validation utilities."""
    print("Testing model validation utilities...")

    # Test valid download task data
    valid_task_data = {
        "url": "https://example.com/file.zip",
        "destination": str(Path.home() / "Downloads" / "file.zip"),
        "engine_type": "http",
    }

    result = validate_download_task_data(valid_task_data)
    if result.is_valid:
        print(f"  ✓ Valid task data accepted: {result.data.id}")
    else:
        print(f"  ✗ Valid task data rejected: {result.errors}")
        return False

    # Test invalid download task data
    invalid_task_data = {
        "url": "invalid-url",
        "destination": str(Path.home() / "Downloads" / "file.zip"),
        "engine_type": "http",
    }

    result = validate_download_task_data(invalid_task_data)
    if not result.is_valid:
        print(f"  ✓ Invalid task data correctly rejected: {result.errors[0]}")
    else:
        print("  ✗ Invalid task data incorrectly accepted")
        return False

    # Test config validation
    valid_config_data = {
        "max_concurrent_downloads": 5,
        "logging_level": "INFO",
        "server_port": 8080,
    }

    result = validate_config_data(valid_config_data, "global")
    if result.is_valid:
        print("  ✓ Valid config data accepted")
    else:
        print(f"  ✗ Valid config data rejected: {result.errors}")
        return False

    print("Model validation test passed!\n")
    return True


def main():
    """Run all tests."""
    print("Running data model and validation tests...\n")

    tests = [
        test_task_status_enum,
        test_engine_type_enum,
        test_progress_info_model,
        test_proxy_config_model,
        test_task_config_model,
        test_global_config_model,
        test_download_task_model,
        test_url_validator,
        test_model_validation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}\n")
            failed += 1

    print(f"Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
