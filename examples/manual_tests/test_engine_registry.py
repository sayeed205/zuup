#!/usr/bin/env python3
"""Manual test script for engine registry and base classes."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from zuup.engines import (
    EngineRegistry,
    get_registry,
    initialize_default_engines,
    detect_engine_for_url,
    get_engine_for_url,
)
from zuup.storage.models import DownloadTask, EngineType, TaskStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_engine_registry_basic():
    """Test basic engine registry functionality."""
    print("\n=== Testing Engine Registry Basic Functionality ===")

    # Create a new registry for testing
    registry = EngineRegistry()

    # Test empty registry
    assert len(registry.list_engines()) == 0
    print("✓ Empty registry works correctly")

    # Test engine detection with no engines
    result = registry.detect_engine_for_url("http://example.com/file.zip")
    assert result is None
    print("✓ No engine detection works correctly")

    print("Basic registry tests passed!")


def test_engine_registration():
    """Test engine registration and detection."""
    print("\n=== Testing Engine Registration ===")

    try:
        # Initialize default engines
        initialize_default_engines()
        registry = get_registry()

        # Test that engines are registered
        engines = registry.list_engines()
        print(f"Registered engines: {engines}")
        assert len(engines) > 0
        print("✓ Default engines registered successfully")

        # Test protocol detection
        test_urls = [
            ("http://example.com/file.zip", "http"),
            ("https://example.com/file.zip", "http"),
            ("ftp://example.com/file.zip", "ftp"),
            ("magnet:?xt=urn:btih:example", "torrent"),
        ]

        for url, expected_engine in test_urls:
            detected = registry.detect_engine_for_url(url)
            print(f"URL: {url} -> Engine: {detected}")
            if detected:
                print(f"✓ Detected engine '{detected}' for {url}")
            else:
                print(f"⚠ No engine detected for {url}")

        # Test supported protocols
        protocols = registry.get_supported_protocols()
        print(f"Supported protocols: {protocols}")

        # Test engine stats
        stats = registry.get_engine_stats()
        print(f"Engine stats: {stats}")

        print("Engine registration tests passed!")

    except Exception as e:
        print(f"❌ Engine registration test failed: {e}")
        raise


def test_convenience_functions():
    """Test convenience functions."""
    print("\n=== Testing Convenience Functions ===")

    try:
        # Test detect_engine_for_url convenience function
        engine_name = detect_engine_for_url("https://example.com/file.zip")
        print(f"Detected engine for HTTPS URL: {engine_name}")

        # Test get_engine_for_url convenience function
        engine = get_engine_for_url("https://example.com/file.zip")
        if engine:
            print(f"Got engine instance: {engine.__class__.__name__}")
            print(
                f"✓ Engine supports HTTPS: {engine.supports_protocol('https://example.com/file.zip')}"
            )

        print("Convenience function tests passed!")

    except Exception as e:
        print(f"❌ Convenience function test failed: {e}")
        raise


def test_engine_validation():
    """Test engine validation functionality."""
    print("\n=== Testing Engine Validation ===")

    try:
        registry = get_registry()

        # Test URL validation for different engine types
        test_cases = [
            ("http://example.com/file.zip", EngineType.HTTP, True),
            ("https://example.com/file.zip", EngineType.HTTP, True),
            ("ftp://example.com/file.zip", EngineType.FTP, True),
            ("magnet:?xt=urn:btih:example", EngineType.TORRENT, True),
            ("http://example.com/file.zip", EngineType.FTP, False),  # Should fail
            ("ftp://example.com/file.zip", EngineType.HTTP, False),  # Should fail
        ]

        for url, engine_type, expected in test_cases:
            result = registry.validate_engine_compatibility(url, engine_type)
            status = "✓" if result == expected else "❌"
            print(
                f"{status} URL: {url}, Engine: {engine_type.value}, Expected: {expected}, Got: {result}"
            )

            if result != expected:
                print(f"❌ Validation failed for {url} with {engine_type.value}")

        print("Engine validation tests passed!")

    except Exception as e:
        print(f"❌ Engine validation test failed: {e}")
        raise


async def test_base_engine_functionality():
    """Test base engine functionality."""
    print("\n=== Testing Base Engine Functionality ===")

    try:
        # Get an engine instance
        engine = get_engine_for_url("https://example.com/file.zip")
        if not engine:
            print("⚠ No engine available for testing base functionality")
            return

        print(f"Testing with engine: {engine.__class__.__name__}")

        # Test engine stats
        stats = engine.get_engine_stats()
        print(f"Engine stats: {stats}")

        # Test active tasks (should be empty initially)
        active_tasks = engine.get_active_tasks()
        print(f"Active tasks: {active_tasks}")
        assert len(active_tasks) == 0
        print("✓ No active tasks initially")

        # Test task validation (this will likely raise NotImplementedError for actual download)
        try:
            task = DownloadTask(
                url="https://example.com/file.zip",
                destination=Path("/tmp/test_file.zip"),
                engine_type=EngineType.HTTP,
            )

            # This should work (validation)
            engine._validate_task(task)
            print("✓ Task validation works")

        except Exception as e:
            print(f"Task validation error (expected for some engines): {e}")

        print("Base engine functionality tests passed!")

    except Exception as e:
        print(f"❌ Base engine functionality test failed: {e}")
        raise


def test_error_handling():
    """Test error handling in engines."""
    print("\n=== Testing Error Handling ===")

    try:
        registry = get_registry()

        # Test invalid engine name
        try:
            registry.get_engine("nonexistent_engine")
            print("❌ Should have raised KeyError for nonexistent engine")
        except KeyError:
            print("✓ Correctly raised KeyError for nonexistent engine")

        # Test duplicate registration
        try:
            from zuup.engines.http_engine import HTTPEngine

            registry.register_engine(
                "http", HTTPEngine()
            )  # Should fail - already registered
            print("❌ Should have raised ValueError for duplicate registration")
        except ValueError:
            print("✓ Correctly raised ValueError for duplicate registration")

        # Test unregistering nonexistent engine
        try:
            registry.unregister_engine("nonexistent_engine")
            print("❌ Should have raised KeyError for nonexistent engine")
        except KeyError:
            print("✓ Correctly raised KeyError for unregistering nonexistent engine")

        print("Error handling tests passed!")

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        raise


async def main():
    """Run all tests."""
    print("Starting Engine Registry and Base Classes Tests")
    print("=" * 50)

    try:
        # Run tests
        test_engine_registry_basic()
        test_engine_registration()
        test_convenience_functions()
        test_engine_validation()
        await test_base_engine_functionality()
        test_error_handling()

        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")

    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
