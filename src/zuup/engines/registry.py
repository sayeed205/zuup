"""Engine registry for protocol detection and engine selection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from ..core.interfaces import DownloadEngine
    from ..storage.models import EngineType

logger = logging.getLogger(__name__)


class EngineRegistry:
    """Registry for managing download engines and protocol detection."""

    def __init__(self) -> None:
        """Initialize the engine registry."""
        self._engines: dict[str, DownloadEngine] = {}
        self._protocol_mapping: dict[str, str] = {}
        logger.info("Engine registry initialized")

    def register_engine(self, name: str, engine: DownloadEngine) -> None:
        """
        Register a download engine.

        Args:
            name: Unique name for the engine
            engine: Engine instance to register

        Raises:
            ValueError: If engine name is already registered
        """
        if name in self._engines:
            raise ValueError(f"Engine '{name}' is already registered")

        self._engines[name] = engine
        logger.info(f"Registered engine: {name}")

    def unregister_engine(self, name: str) -> None:
        """
        Unregister a download engine.

        Args:
            name: Name of engine to unregister

        Raises:
            KeyError: If engine name is not registered
        """
        if name not in self._engines:
            raise KeyError(f"Engine '{name}' is not registered")

        del self._engines[name]
        logger.info(f"Unregistered engine: {name}")

    def get_engine(self, name: str) -> DownloadEngine:
        """
        Get a registered engine by name.

        Args:
            name: Name of the engine

        Returns:
            The requested engine instance

        Raises:
            KeyError: If engine name is not registered
        """
        if name not in self._engines:
            raise KeyError(f"Engine '{name}' is not registered")

        return self._engines[name]

    def list_engines(self) -> list[str]:
        """
        List all registered engine names.

        Returns:
            List of registered engine names
        """
        return list(self._engines.keys())

    def detect_engine_for_url(self, url: str) -> str | None:
        """
        Detect the appropriate engine for a given URL.

        Args:
            url: URL to analyze

        Returns:
            Name of the engine that supports the URL, or None if no engine supports it
        """
        try:
            parsed = urlparse(url)
            scheme = parsed.scheme.lower()

            # Check if we have a cached mapping for this scheme
            if scheme in self._protocol_mapping:
                engine_name = self._protocol_mapping[scheme]
                if engine_name in self._engines:
                    return engine_name

            # Check each engine to see if it supports the URL
            for name, engine in self._engines.items():
                if engine.supports_protocol(url):
                    # Cache the mapping for future use
                    self._protocol_mapping[scheme] = name
                    logger.debug(f"Detected engine '{name}' for URL scheme '{scheme}'")
                    return name

            logger.warning(f"No engine found for URL: {url}")
            return None

        except Exception as e:
            logger.error(f"Error detecting engine for URL '{url}': {e}")
            return None

    def get_engine_for_url(self, url: str) -> DownloadEngine | None:
        """
        Get the appropriate engine instance for a given URL.

        Args:
            url: URL to analyze

        Returns:
            Engine instance that supports the URL, or None if no engine supports it
        """
        engine_name = self.detect_engine_for_url(url)
        if engine_name:
            return self._engines[engine_name]
        return None

    def get_supported_protocols(self) -> dict[str, list[str]]:
        """
        Get a mapping of engine names to their supported protocols.

        Returns:
            Dictionary mapping engine names to lists of supported URL schemes
        """
        protocols: dict[str, list[str]] = {}

        # Common test URLs for different protocols
        test_urls = [
            "http://example.com/file.zip",
            "https://example.com/file.zip",
            "ftp://example.com/file.zip",
            "ftps://example.com/file.zip",
            "sftp://example.com/file.zip",
            "magnet:?xt=urn:btih:example",
            "file:///path/to/file.torrent",
        ]

        for name, engine in self._engines.items():
            supported = []
            for test_url in test_urls:
                if engine.supports_protocol(test_url):
                    scheme = urlparse(test_url).scheme.lower()
                    if scheme not in supported:
                        supported.append(scheme)
            protocols[name] = supported

        return protocols

    def validate_engine_compatibility(self, url: str, engine_type: EngineType) -> bool:
        """
        Validate that a specific engine type is compatible with a URL.

        Args:
            url: URL to validate
            engine_type: Engine type to check compatibility for

        Returns:
            True if the engine type is compatible with the URL, False otherwise
        """
        try:
            # Map engine types to expected engine names
            engine_type_mapping = {
                "http": "http",
                "ftp": "ftp",
                "torrent": "torrent",
                "media": "media",
            }

            expected_engine = engine_type_mapping.get(engine_type.value)
            if not expected_engine:
                logger.warning(f"Unknown engine type: {engine_type}")
                return False

            if expected_engine not in self._engines:
                logger.warning(f"Engine '{expected_engine}' not registered")
                return False

            engine = self._engines[expected_engine]
            return engine.supports_protocol(url)

        except Exception as e:
            logger.error(f"Error validating engine compatibility: {e}")
            return False

    def clear_protocol_cache(self) -> None:
        """Clear the protocol mapping cache."""
        self._protocol_mapping.clear()
        logger.debug("Protocol mapping cache cleared")

    def get_engine_stats(self) -> dict[str, dict[str, int]]:
        """
        Get statistics about registered engines.

        Returns:
            Dictionary with engine statistics
        """
        stats = {}
        supported_protocols = self.get_supported_protocols()

        for name in self._engines:
            stats[name] = {
                "supported_protocols": len(supported_protocols.get(name, [])),
                "cached_mappings": sum(
                    1
                    for engine_name in self._protocol_mapping.values()
                    if engine_name == name
                ),
            }

        return stats


# Global registry instance
_registry: EngineRegistry | None = None


def get_registry() -> EngineRegistry:
    """
    Get the global engine registry instance.

    Returns:
        Global engine registry instance
    """
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
    return _registry


def initialize_default_engines() -> None:
    """Initialize the registry with default engines."""
    from .http_ftp_engine import HttpFtpEngine
    from .media_engine import MediaEngine
    from .torrent_engine import TorrentEngine

    registry = get_registry()

    # Register default engines
    try:
        # Use the unified HttpFtpEngine for both HTTP and FTP protocols
        http_ftp_engine = HttpFtpEngine()
        registry.register_engine("http", http_ftp_engine)
        registry.register_engine("ftp", http_ftp_engine)  # Same engine handles both
        registry.register_engine("torrent", TorrentEngine())
        registry.register_engine("media", MediaEngine())
        logger.info("Default engines initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize default engines: {e}")
        raise


def detect_engine_for_url(url: str) -> str | None:
    """
    Convenience function to detect engine for URL using global registry.

    Args:
        url: URL to analyze

    Returns:
        Name of the engine that supports the URL, or None if no engine supports it
    """
    return get_registry().detect_engine_for_url(url)


def get_engine_for_url(url: str) -> DownloadEngine | None:
    """
    Convenience function to get engine for URL using global registry.

    Args:
        url: URL to analyze

    Returns:
        Engine instance that supports the URL, or None if no engine supports it
    """
    return get_registry().get_engine_for_url(url)
