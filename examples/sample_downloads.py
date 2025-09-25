#!/usr/bin/env python3
"""
Sample download scripts for manual testing.

This script demonstrates how to use the download manager programmatically.
"""

import asyncio

from zuup import Application
from zuup.config.manager import ConfigManager
from zuup.utils.logging import setup_logging


async def test_http_download() -> None:
    """Test HTTP download functionality."""
    print("🔄 Testing HTTP download...")

    # This will be implemented in later tasks
    print("⚠️  HTTP download will be implemented in task 4")


async def test_torrent_download() -> None:
    """Test torrent download functionality."""
    print("🔄 Testing torrent download...")

    # This will be implemented in later tasks
    print("⚠️  Torrent download will be implemented in task 4")


async def test_media_download() -> None:
    """Test media download functionality."""
    print("🔄 Testing media download...")

    # This will be implemented in later tasks
    print("⚠️  Media download will be implemented in task 4")


async def main() -> None:
    """Main testing function."""
    # Setup logging
    setup_logging(level="DEBUG")

    # Initialize application
    config_manager = ConfigManager()
    app = Application(config_manager=config_manager)

    print("🚀 Starting Zuup download manager tests")

    # Run tests
    await test_http_download()
    await test_torrent_download()
    await test_media_download()

    print("✅ All tests completed")


if __name__ == "__main__":
    asyncio.run(main())
