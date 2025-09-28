"""Advanced network management for media downloads with proxy support and geo-bypass."""

import logging
import random
from typing import Any
from urllib.parse import urlparse

import httpx

from .media_models import GeoBypassConfig, NetworkConfig, ProxyConfig

logger = logging.getLogger(__name__)


class NetworkManager:
    """Manages network configuration, proxy rotation, and geo-bypass for media downloads."""

    def __init__(
        self,
        network_config: NetworkConfig,
        proxy_config: ProxyConfig,
        geo_bypass_config: GeoBypassConfig,
    ) -> None:
        """
        Initialize NetworkManager with advanced configuration.

        Args:
            network_config: Network timeout and retry configuration
            proxy_config: Proxy configuration and rotation settings
            geo_bypass_config: Geo-bypass configuration with country selection
        """
        self.network_config = network_config
        self.proxy_config = proxy_config
        self.geo_bypass_config = geo_bypass_config

        # Proxy rotation state
        self._current_proxy_index = 0
        self._proxy_request_count = 0
        self._tested_proxies: dict[str, bool] = {}

        # User agent rotation state
        self._current_user_agent_index = 0
        self._default_user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        ]

        logger.info(
            "NetworkManager initialized with advanced proxy and geo-bypass support"
        )

    def create_yt_dlp_options(self, url: str) -> dict[str, Any]:
        """
        Create comprehensive yt-dlp options with network configuration.

        Args:
            url: URL being processed (for platform-specific settings)

        Returns:
            Dictionary of yt-dlp network options
        """
        opts = {}

        # Basic network settings
        opts.update(self._get_basic_network_options())

        # Proxy configuration
        proxy_opts = self._get_proxy_options(url)
        opts.update(proxy_opts)

        # Geo-bypass configuration
        geo_opts = self._get_geo_bypass_options(url)
        opts.update(geo_opts)

        # User agent and headers
        header_opts = self._get_header_options(url)
        opts.update(header_opts)

        # SSL/TLS settings
        ssl_opts = self._get_ssl_options()
        opts.update(ssl_opts)

        logger.debug(f"Created yt-dlp network options for {urlparse(url).netloc}")
        return opts

    def _get_basic_network_options(self) -> dict[str, Any]:
        """Get basic network timeout and retry options."""
        opts: dict[str, Any] = {
            "socket_timeout": self.network_config.socket_timeout,
            "retries": self.network_config.retries,
            "fragment_retries": self.network_config.fragment_retries,
            "sleep_interval": self.network_config.sleep_interval,
            "max_sleep_interval": self.network_config.max_sleep_interval,
            "sleep_interval_subtitles": self.network_config.sleep_interval_subtitles,
        }
        return opts

    def _get_proxy_options(self, url: str) -> dict[str, Any]:
        """
        Get proxy configuration options with rotation support.

        Args:
            url: URL being processed

        Returns:
            Dictionary of proxy options
        """
        opts: dict[str, Any] = {}

        # Determine which proxy to use
        proxy_url = self._get_current_proxy(url)
        if proxy_url:
            opts["proxy"] = proxy_url
            logger.debug(f"Using proxy: {proxy_url}")

        return opts

    def _get_geo_bypass_options(self, url: str) -> dict[str, Any]:
        """
        Get geo-bypass configuration options.

        Args:
            url: URL being processed

        Returns:
            Dictionary of geo-bypass options
        """
        opts: dict[str, Any] = {
            "geo_bypass": self.geo_bypass_config.geo_bypass,
        }

        # Set geo-bypass country
        if self.geo_bypass_config.geo_bypass_country:
            opts["geo_bypass_country"] = self.geo_bypass_config.geo_bypass_country
        elif self.geo_bypass_config.preferred_countries:
            # Use first preferred country as default
            opts["geo_bypass_country"] = self.geo_bypass_config.preferred_countries[0]

        # Set geo-bypass IP block if specified
        if self.geo_bypass_config.geo_bypass_ip_block:
            opts["geo_bypass_ip_block"] = self.geo_bypass_config.geo_bypass_ip_block

        return opts

    def _get_header_options(self, url: str) -> dict[str, Any]:
        """
        Get user agent and header options with platform-specific support.

        Args:
            url: URL being processed

        Returns:
            Dictionary of header options
        """
        opts: dict[str, Any] = {}

        # Determine user agent
        user_agent = self._get_current_user_agent(url)
        if user_agent:
            opts["user_agent"] = user_agent

        # Set referer if specified
        if self.network_config.referer:
            opts["referer"] = self.network_config.referer

        # Add custom headers
        if self.network_config.custom_headers:
            # yt-dlp expects headers in a specific format
            headers = []
            for key, value in self.network_config.custom_headers.items():
                headers.append(f"{key}: {value}")
            if headers:
                opts["add_header"] = headers

        return opts

    def _get_ssl_options(self) -> dict[str, Any]:
        """Get SSL/TLS configuration options."""
        opts: dict[str, Any] = {}

        if self.network_config.no_check_certificate:
            opts["no_check_certificate"] = True

        if self.network_config.client_certificate:
            opts["client_certificate"] = self.network_config.client_certificate

        if self.network_config.client_certificate_key:
            opts["client_certificate_key"] = self.network_config.client_certificate_key

        if self.network_config.client_certificate_password:
            opts["client_certificate_password"] = (
                self.network_config.client_certificate_password
            )

        # IP preference settings
        if self.network_config.prefer_ipv4:
            opts["prefer_ipv4"] = True
        elif self.network_config.prefer_ipv6:
            opts["prefer_ipv6"] = True

        if self.network_config.source_address:
            opts["source_address"] = self.network_config.source_address

        return opts

    def _get_current_proxy(self, url: str) -> str | None:
        """
        Get current proxy URL with rotation and geo-bypass support.

        Args:
            url: URL being processed

        Returns:
            Proxy URL to use, or None if no proxy
        """
        # Check if geo-bypass proxy should be used
        if self.geo_bypass_config.auto_detect_geo_blocking:
            geo_proxy = self._get_geo_bypass_proxy(url)
            if geo_proxy:
                return geo_proxy

        # Use configured proxy
        if self.proxy_config.proxy_url:
            return self._format_proxy_url(self.proxy_config.proxy_url)

        # Use proxy rotation if enabled
        if self.proxy_config.enable_proxy_rotation and self.proxy_config.proxy_list:
            return self._get_rotated_proxy()

        return None

    def _get_geo_bypass_proxy(self, url: str) -> str | None:
        """
        Get geo-bypass specific proxy if needed.

        Args:
            url: URL being processed

        Returns:
            Geo-bypass proxy URL or None
        """
        if self.proxy_config.geo_bypass_proxy:
            return self._format_proxy_url(self.proxy_config.geo_bypass_proxy)

        # Could implement geo-blocking detection logic here
        # For now, return None to use regular proxy logic
        return None

    def _get_rotated_proxy(self) -> str:
        """
        Get next proxy from rotation list.

        Returns:
            Next proxy URL in rotation
        """
        if not self.proxy_config.proxy_list:
            return ""

        # Check if rotation is needed
        if self._proxy_request_count >= self.proxy_config.proxy_rotation_interval:
            self._current_proxy_index = (self._current_proxy_index + 1) % len(
                self.proxy_config.proxy_list
            )
            self._proxy_request_count = 0
            logger.info(f"Rotated to proxy index {self._current_proxy_index}")

        self._proxy_request_count += 1
        proxy_url = self.proxy_config.proxy_list[self._current_proxy_index]
        return self._format_proxy_url(proxy_url)

    def _format_proxy_url(self, proxy_url: str) -> str:
        """
        Format proxy URL with authentication if needed.

        Args:
            proxy_url: Base proxy URL

        Returns:
            Formatted proxy URL with authentication
        """
        if not proxy_url:
            return ""

        # Add authentication if provided
        if self.proxy_config.proxy_username and self.proxy_config.proxy_password:
            parsed = urlparse(proxy_url)
            if not parsed.username:  # Only add auth if not already present
                auth = f"{self.proxy_config.proxy_username}:{self.proxy_config.proxy_password}"
                proxy_url = proxy_url.replace("://", f"://{auth}@", 1)

        return proxy_url

    def _get_current_user_agent(self, url: str) -> str | None:
        """
        Get current user agent with platform-specific and rotation support.

        Args:
            url: URL being processed

        Returns:
            User agent string to use
        """
        # Check for platform-specific user agent
        if self.network_config.platform_user_agents:
            domain = urlparse(url).netloc.lower()
            for (
                platform,
                user_agent,
            ) in self.network_config.platform_user_agents.items():
                if platform.lower() in domain:
                    return user_agent

        # Use configured user agent
        if self.network_config.user_agent:
            return self.network_config.user_agent

        # Use user agent rotation if enabled
        if self.network_config.rotate_user_agents:
            return self._get_rotated_user_agent()

        return None

    def _get_rotated_user_agent(self) -> str:
        """
        Get next user agent from rotation list.

        Returns:
            Next user agent in rotation
        """
        self._current_user_agent_index = (self._current_user_agent_index + 1) % len(
            self._default_user_agents
        )
        return self._default_user_agents[self._current_user_agent_index]

    async def test_proxy_connectivity(self, proxy_url: str) -> bool:
        """
        Test proxy connectivity and performance.

        Args:
            proxy_url: Proxy URL to test

        Returns:
            True if proxy is working, False otherwise
        """
        if not self.proxy_config.test_proxy_connectivity:
            return True  # Skip testing if disabled

        if proxy_url in self._tested_proxies:
            return self._tested_proxies[proxy_url]

        logger.info(f"Testing proxy connectivity: {proxy_url}")

        try:
            # Test with a simple HTTP request
            async with httpx.AsyncClient(
                proxy=proxy_url,
                timeout=self.proxy_config.proxy_timeout,
            ) as client:
                response = await client.get("https://httpbin.org/ip")
                success = response.status_code == 200

                if success:
                    logger.info(f"Proxy test successful: {proxy_url}")
                else:
                    logger.warning(
                        f"Proxy test failed with status {response.status_code}: {proxy_url}"
                    )

                self._tested_proxies[proxy_url] = success
                return success

        except Exception as e:
            logger.warning(f"Proxy test failed with exception: {proxy_url} - {e}")
            self._tested_proxies[proxy_url] = False
            return False

    async def handle_geo_blocking_error(
        self, url: str, error: Exception
    ) -> dict[str, Any]:
        """
        Handle geo-blocking errors with automatic fallback.

        Args:
            url: URL that encountered geo-blocking
            error: The geo-blocking error

        Returns:
            Updated yt-dlp options to retry with
        """
        logger.warning(f"Geo-blocking detected for {url}: {error}")

        # Try fallback countries
        if self.geo_bypass_config.geo_bypass_fallback_countries:
            for country in self.geo_bypass_config.geo_bypass_fallback_countries:
                if country != self.geo_bypass_config.geo_bypass_country:
                    logger.info(f"Trying geo-bypass with country: {country}")

                    # Update geo-bypass country
                    self.geo_bypass_config.geo_bypass_country = country

                    # Return updated options
                    return self.create_yt_dlp_options(url)

        # Try using geo-bypass proxy if available
        if self.proxy_config.geo_bypass_proxy:
            logger.info("Trying geo-bypass proxy")

            # Temporarily switch to geo-bypass proxy
            original_proxy = self.proxy_config.proxy_url
            self.proxy_config.proxy_url = self.proxy_config.geo_bypass_proxy

            try:
                return self.create_yt_dlp_options(url)
            finally:
                # Restore original proxy
                self.proxy_config.proxy_url = original_proxy

        # No more fallback options
        logger.error(f"No more geo-bypass options available for {url}")
        raise error

    async def handle_network_error(
        self, url: str, error: Exception, attempt: int
    ) -> float:
        """
        Handle network errors with retry delay calculation.

        Args:
            url: URL that encountered network error
            error: The network error
            attempt: Current attempt number

        Returns:
            Delay in seconds before retry
        """
        logger.warning(f"Network error for {url} (attempt {attempt}): {error}")

        # Calculate exponential backoff delay
        base_delay = self.network_config.retry_sleep
        max_delay = self.network_config.max_retry_sleep
        backoff_factor = self.network_config.retry_backoff_factor

        delay = min(base_delay * (backoff_factor**attempt), max_delay)

        # Add some jitter to avoid thundering herd
        jitter = random.uniform(0.1, 0.3) * delay
        final_delay = delay + jitter

        logger.info(f"Retrying {url} after {final_delay:.1f}s delay")
        return final_delay

    def get_platform_specific_options(self, url: str) -> dict[str, Any]:
        """
        Get platform-specific options for known sites.

        Args:
            url: URL being processed

        Returns:
            Dictionary of platform-specific options
        """
        domain = urlparse(url).netloc.lower()
        opts = {}

        # YouTube specific optimizations
        if "youtube.com" in domain or "youtu.be" in domain:
            opts.update(
                {
                    "sleep_interval": max(
                        self.network_config.sleep_interval, 1.0
                    ),  # Respect rate limits
                    "max_sleep_interval": 10.0,
                }
            )

        # Twitter/X specific settings
        elif "twitter.com" in domain or "x.com" in domain:
            opts.update(
                {
                    "sleep_interval": max(
                        self.network_config.sleep_interval, 2.0
                    ),  # Higher rate limit
                }
            )

        # TikTok specific settings
        elif "tiktok.com" in domain:
            opts.update(
                {
                    "sleep_interval": max(self.network_config.sleep_interval, 1.5),
                }
            )

        # Instagram specific settings
        elif "instagram.com" in domain:
            opts.update(
                {
                    "sleep_interval": max(self.network_config.sleep_interval, 2.0),
                }
            )

        return opts

    def reset_proxy_rotation(self) -> None:
        """Reset proxy rotation state."""
        self._current_proxy_index = 0
        self._proxy_request_count = 0
        logger.info("Proxy rotation state reset")

    def reset_user_agent_rotation(self) -> None:
        """Reset user agent rotation state."""
        self._current_user_agent_index = 0
        logger.info("User agent rotation state reset")

    def get_network_stats(self) -> dict[str, Any]:
        """
        Get network manager statistics.

        Returns:
            Dictionary with network statistics
        """
        return {
            "current_proxy_index": self._current_proxy_index,
            "proxy_request_count": self._proxy_request_count,
            "tested_proxies": len(self._tested_proxies),
            "working_proxies": sum(self._tested_proxies.values()),
            "current_user_agent_index": self._current_user_agent_index,
        }
