"""Connection pooling for pycurl handles to improve performance and resource reuse."""

from __future__ import annotations

from collections import deque
import logging
import threading
import time
from typing import Any
from urllib.parse import urlparse

import pycurl

from .pycurl_models import HttpFtpConfig

logger = logging.getLogger(__name__)


class CurlHandlePool:
    """Pool of reusable pycurl handles for improved performance."""

    def __init__(
        self,
        max_pool_size: int = 20,
        max_idle_time: int = 300,  # 5 minutes
        cleanup_interval: int = 60,  # 1 minute
    ) -> None:
        """
        Initialize the curl handle pool.

        Args:
            max_pool_size: Maximum number of handles to keep in pool
            max_idle_time: Maximum idle time before handle is discarded (seconds)
            cleanup_interval: Interval between cleanup runs (seconds)
        """
        self.max_pool_size = max_pool_size
        self.max_idle_time = max_idle_time
        self.cleanup_interval = cleanup_interval

        # Pool storage: deque of (handle, last_used_time, host) tuples
        self._pool: deque[tuple[pycurl.Curl, float, str]] = deque()
        self._pool_lock = threading.RLock()

        # Statistics
        self._stats = {
            "created": 0,
            "reused": 0,
            "discarded": 0,
            "cleanup_runs": 0,
        }

        # Cleanup thread
        self._cleanup_thread: threading.Thread | None = None
        self._shutdown = False
        self._start_cleanup_thread()

        logger.info(
            f"Initialized CurlHandlePool with max_size={max_pool_size}, "
            f"max_idle_time={max_idle_time}s"
        )

    def _start_cleanup_thread(self) -> None:
        """Start the background cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker, daemon=True, name="CurlPoolCleanup"
        )
        self._cleanup_thread.start()

    def _cleanup_worker(self) -> None:
        """Background worker to clean up expired handles."""
        while not self._shutdown:
            try:
                time.sleep(self.cleanup_interval)
                if not self._shutdown:
                    self._cleanup_expired_handles()
            except Exception as e:
                logger.warning(f"Error in cleanup worker: {e}")

    def _cleanup_expired_handles(self) -> None:
        """Remove expired handles from the pool."""
        current_time = time.time()
        expired_count = 0

        with self._pool_lock:
            # Create new deque with non-expired handles
            new_pool = deque()

            while self._pool:
                handle, last_used, host = self._pool.popleft()

                if current_time - last_used <= self.max_idle_time:
                    # Handle is still valid
                    new_pool.append((handle, last_used, host))
                else:
                    # Handle expired, close it
                    try:
                        handle.close()
                        expired_count += 1
                        self._stats["discarded"] += 1
                    except Exception as e:
                        logger.warning(f"Error closing expired handle: {e}")

            self._pool = new_pool
            self._stats["cleanup_runs"] += 1

        if expired_count > 0:
            logger.debug(f"Cleaned up {expired_count} expired curl handles")

    def _extract_host(self, url: str) -> str:
        """Extract host from URL for connection reuse."""
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return url

    def get_handle(self, url: str, config: HttpFtpConfig) -> pycurl.Curl:
        """
        Get a curl handle from the pool or create a new one.

        Args:
            url: URL for the request
            config: Configuration for the handle

        Returns:
            Configured curl handle
        """
        host = self._extract_host(url)
        current_time = time.time()

        with self._pool_lock:
            # Try to find a reusable handle for the same host
            for i, (handle, last_used, handle_host) in enumerate(self._pool):
                if handle_host == host:
                    # Remove from pool and return
                    del self._pool[i]
                    self._stats["reused"] += 1
                    logger.debug(f"Reused curl handle for {host}")

                    # Reset handle for new use
                    self._reset_handle(handle, config)
                    return handle

        # No reusable handle found, create new one
        handle = self._create_new_handle(config)
        self._stats["created"] += 1
        logger.debug(f"Created new curl handle for {host}")
        return handle

    def return_handle(self, handle: pycurl.Curl, url: str) -> None:
        """
        Return a handle to the pool for reuse.

        Args:
            handle: Curl handle to return
            url: URL the handle was used for
        """
        host = self._extract_host(url)
        current_time = time.time()

        with self._pool_lock:
            # Check if pool is full
            if len(self._pool) >= self.max_pool_size:
                # Pool is full, discard oldest handle
                if self._pool:
                    old_handle, _, _ = self._pool.popleft()
                    try:
                        old_handle.close()
                        self._stats["discarded"] += 1
                    except Exception as e:
                        logger.warning(f"Error closing discarded handle: {e}")

            # Add handle to pool
            self._pool.append((handle, current_time, host))
            logger.debug(f"Returned curl handle to pool for {host}")

    def _create_new_handle(self, config: HttpFtpConfig) -> pycurl.Curl:
        """
        Create a new curl handle with basic configuration.

        Args:
            config: Configuration to apply

        Returns:
            New curl handle
        """
        handle = pycurl.Curl()

        # Apply basic configuration that's safe to reuse
        handle.setopt(pycurl.FOLLOWLOCATION, 1 if config.follow_redirects else 0)
        handle.setopt(pycurl.MAXREDIRS, config.max_redirects)
        handle.setopt(pycurl.TIMEOUT, config.timeout)
        handle.setopt(pycurl.CONNECTTIMEOUT, config.connect_timeout)
        handle.setopt(pycurl.USERAGENT, config.user_agent.encode("utf-8"))
        handle.setopt(pycurl.TCP_NODELAY, 1 if config.tcp_nodelay else 0)

        # SSL settings
        if config.verify_ssl and not config.ssl_development_mode:
            handle.setopt(pycurl.SSL_VERIFYPEER, 1)
            handle.setopt(pycurl.SSL_VERIFYHOST, 2)
        else:
            handle.setopt(pycurl.SSL_VERIFYPEER, 0)
            handle.setopt(pycurl.SSL_VERIFYHOST, 0)

        # Compression
        if config.enable_compression:
            handle.setopt(pycurl.ACCEPT_ENCODING, b"")

        return handle

    def _reset_handle(self, handle: pycurl.Curl, config: HttpFtpConfig) -> None:
        """
        Reset a handle for reuse with new configuration.

        Args:
            handle: Handle to reset
            config: New configuration to apply
        """
        # Reset handle to clean state
        handle.reset()

        # Reapply basic configuration
        handle.setopt(pycurl.FOLLOWLOCATION, 1 if config.follow_redirects else 0)
        handle.setopt(pycurl.MAXREDIRS, config.max_redirects)
        handle.setopt(pycurl.TIMEOUT, config.timeout)
        handle.setopt(pycurl.CONNECTTIMEOUT, config.connect_timeout)
        handle.setopt(pycurl.USERAGENT, config.user_agent.encode("utf-8"))
        handle.setopt(pycurl.TCP_NODELAY, 1 if config.tcp_nodelay else 0)

        # SSL settings
        if config.verify_ssl and not config.ssl_development_mode:
            handle.setopt(pycurl.SSL_VERIFYPEER, 1)
            handle.setopt(pycurl.SSL_VERIFYHOST, 2)
        else:
            handle.setopt(pycurl.SSL_VERIFYPEER, 0)
            handle.setopt(pycurl.SSL_VERIFYHOST, 0)

        # Compression
        if config.enable_compression:
            handle.setopt(pycurl.ACCEPT_ENCODING, b"")

    def get_stats(self) -> dict[str, Any]:
        """
        Get pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        with self._pool_lock:
            current_size = len(self._pool)

        return {
            "current_size": current_size,
            "max_size": self.max_pool_size,
            "handles_created": self._stats["created"],
            "handles_reused": self._stats["reused"],
            "handles_discarded": self._stats["discarded"],
            "cleanup_runs": self._stats["cleanup_runs"],
            "reuse_rate": (
                self._stats["reused"] / (self._stats["created"] + self._stats["reused"])
                if (self._stats["created"] + self._stats["reused"]) > 0
                else 0.0
            ),
        }

    def clear(self) -> None:
        """Clear all handles from the pool."""
        with self._pool_lock:
            while self._pool:
                handle, _, _ = self._pool.popleft()
                try:
                    handle.close()
                    self._stats["discarded"] += 1
                except Exception as e:
                    logger.warning(f"Error closing handle during clear: {e}")

        logger.info("Cleared all handles from pool")

    def shutdown(self) -> None:
        """Shutdown the pool and cleanup resources."""
        logger.info("Shutting down CurlHandlePool")

        self._shutdown = True

        # Wait for cleanup thread to finish
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)

        # Clear all handles
        self.clear()

        logger.info("CurlHandlePool shutdown complete")

    def __del__(self) -> None:
        """Cleanup when pool is destroyed."""
        if not self._shutdown:
            logger.warning("CurlHandlePool destroyed without proper shutdown")
            self.shutdown()


# Global pool instance
_global_pool: CurlHandlePool | None = None
_pool_lock = threading.Lock()


def get_global_pool() -> CurlHandlePool:
    """
    Get the global curl handle pool instance.

    Returns:
        Global pool instance
    """
    global _global_pool

    if _global_pool is None:
        with _pool_lock:
            if _global_pool is None:
                _global_pool = CurlHandlePool()

    return _global_pool


def shutdown_global_pool() -> None:
    """Shutdown the global pool."""
    global _global_pool

    if _global_pool is not None:
        with _pool_lock:
            if _global_pool is not None:
                _global_pool.shutdown()
                _global_pool = None
