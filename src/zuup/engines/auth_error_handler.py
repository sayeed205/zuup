"""Authentication error handling for media downloads."""

import asyncio
from enum import Enum
import logging
from typing import Any

from .cookie_manager import AuthenticationManager
from .media_models import AuthConfig, AuthMethod

logger = logging.getLogger(__name__)


class AuthErrorType(Enum):
    """Types of authentication errors."""

    INVALID_CREDENTIALS = "invalid_credentials"
    EXPIRED_SESSION = "expired_session"
    MISSING_COOKIES = "missing_cookies"
    OAUTH_TOKEN_EXPIRED = "oauth_token_expired"
    NETWORK_ERROR = "network_error"
    RATE_LIMITED = "rate_limited"
    CAPTCHA_REQUIRED = "captcha_required"
    TWO_FACTOR_REQUIRED = "two_factor_required"
    UNKNOWN = "unknown"


class AuthErrorAction(Enum):
    """Actions to take when authentication errors occur."""

    RETRY_WITH_REFRESH = "retry_with_refresh"
    RETRY_WITH_FALLBACK = "retry_with_fallback"
    PROMPT_FOR_CREDENTIALS = "prompt_for_credentials"
    SWITCH_AUTH_METHOD = "switch_auth_method"
    FAIL_PERMANENTLY = "fail_permanently"
    WAIT_AND_RETRY = "wait_and_retry"


class AuthenticationErrorHandler:
    """
    Handles authentication errors with automatic recovery strategies.
    """

    def __init__(self, auth_manager: AuthenticationManager) -> None:
        """
        Initialize authentication error handler.

        Args:
            auth_manager: Authentication manager instance
        """
        self.auth_manager = auth_manager
        self.logger = logging.getLogger(__name__)

        # Error pattern matching
        self._error_patterns = {
            AuthErrorType.INVALID_CREDENTIALS: [
                "invalid username",
                "invalid password",
                "login failed",
                "authentication failed",
                "unauthorized",
                "403",
                "401",
            ],
            AuthErrorType.EXPIRED_SESSION: [
                "session expired",
                "session invalid",
                "cookie expired",
                "please log in",
                "login required",
            ],
            AuthErrorType.MISSING_COOKIES: [
                "no cookies",
                "cookie required",
                "session required",
                "please enable cookies",
            ],
            AuthErrorType.OAUTH_TOKEN_EXPIRED: [
                "token expired",
                "invalid token",
                "token revoked",
                "oauth error",
                "access denied",
            ],
            AuthErrorType.RATE_LIMITED: [
                "rate limit",
                "too many requests",
                "429",
                "slow down",
                "quota exceeded",
            ],
            AuthErrorType.CAPTCHA_REQUIRED: [
                "captcha",
                "verify you are human",
                "robot check",
                "security check",
            ],
            AuthErrorType.TWO_FACTOR_REQUIRED: [
                "two factor",
                "2fa",
                "verification code",
                "authentication code",
                "verify your identity",
            ],
            AuthErrorType.NETWORK_ERROR: [
                "network error",
                "connection failed",
                "timeout",
                "dns error",
                "connection refused",
            ],
        }

    def classify_error(
        self, error: Exception, auth_config: AuthConfig
    ) -> AuthErrorType:
        """
        Classify authentication error type.

        Args:
            error: Authentication error
            auth_config: Authentication configuration

        Returns:
            Classified error type
        """
        error_message = str(error).lower()

        # Check each error type pattern
        for error_type, patterns in self._error_patterns.items():
            if any(pattern in error_message for pattern in patterns):
                return error_type

        # Check for method-specific errors
        if auth_config.method == AuthMethod.COOKIES and "cookie" in error_message:
            return AuthErrorType.MISSING_COOKIES
        elif auth_config.method == AuthMethod.OAUTH and "token" in error_message:
            return AuthErrorType.OAUTH_TOKEN_EXPIRED
        elif (
            auth_config.method == AuthMethod.USERNAME_PASSWORD
            and "password" in error_message
        ):
            return AuthErrorType.INVALID_CREDENTIALS

        return AuthErrorType.UNKNOWN

    async def handle_auth_error(
        self, error: Exception, auth_config: AuthConfig, url: str, attempt: int = 0
    ) -> AuthErrorAction:
        """
        Handle authentication error and determine recovery action.

        Args:
            error: Authentication error
            auth_config: Authentication configuration
            url: URL that failed authentication
            attempt: Current attempt number

        Returns:
            Recommended action to take
        """
        error_type = self.classify_error(error, auth_config)

        self.logger.warning(
            f"Authentication error (attempt {attempt + 1}): {error_type.value} - {error}"
        )

        # Determine action based on error type and attempt count
        if error_type == AuthErrorType.EXPIRED_SESSION:
            if attempt < 2:
                return AuthErrorAction.RETRY_WITH_REFRESH
            else:
                return AuthErrorAction.PROMPT_FOR_CREDENTIALS

        elif error_type == AuthErrorType.MISSING_COOKIES:
            if attempt < 1:
                return AuthErrorAction.RETRY_WITH_REFRESH
            else:
                return AuthErrorAction.SWITCH_AUTH_METHOD

        elif error_type == AuthErrorType.INVALID_CREDENTIALS:
            if attempt < 1 and auth_config.method == AuthMethod.USERNAME_PASSWORD:
                return AuthErrorAction.RETRY_WITH_FALLBACK
            else:
                return AuthErrorAction.PROMPT_FOR_CREDENTIALS

        elif error_type == AuthErrorType.OAUTH_TOKEN_EXPIRED:
            return AuthErrorAction.PROMPT_FOR_CREDENTIALS  # Manual token refresh needed

        elif error_type == AuthErrorType.RATE_LIMITED:
            if attempt < 3:
                return AuthErrorAction.WAIT_AND_RETRY
            else:
                return AuthErrorAction.FAIL_PERMANENTLY

        elif error_type == AuthErrorType.CAPTCHA_REQUIRED:
            return AuthErrorAction.PROMPT_FOR_CREDENTIALS  # Manual intervention needed

        elif error_type == AuthErrorType.TWO_FACTOR_REQUIRED:
            return AuthErrorAction.PROMPT_FOR_CREDENTIALS  # Manual 2FA needed

        elif error_type == AuthErrorType.NETWORK_ERROR:
            if attempt < 3:
                return AuthErrorAction.WAIT_AND_RETRY
            else:
                return AuthErrorAction.FAIL_PERMANENTLY

        elif attempt < 2:
            return AuthErrorAction.RETRY_WITH_REFRESH
        else:
            return AuthErrorAction.FAIL_PERMANENTLY

    async def execute_recovery_action(
        self,
        action: AuthErrorAction,
        auth_config: AuthConfig,
        url: str,
        attempt: int = 0,
    ) -> dict[str, Any] | None:
        """
        Execute recovery action for authentication error.

        Args:
            action: Recovery action to execute
            auth_config: Authentication configuration
            url: URL that failed authentication
            attempt: Current attempt number

        Returns:
            Updated authentication options, or None if action failed
        """
        try:
            if action == AuthErrorAction.RETRY_WITH_REFRESH:
                self.logger.info("Attempting to refresh authentication")
                auth_dict = self._auth_config_to_dict(auth_config)
                return await self.auth_manager.refresh_authentication(auth_dict, url)

            elif action == AuthErrorAction.RETRY_WITH_FALLBACK:
                self.logger.info("Attempting fallback authentication method")
                return await self._try_fallback_auth(auth_config, url)

            elif action == AuthErrorAction.SWITCH_AUTH_METHOD:
                self.logger.info("Switching to alternative authentication method")
                return await self._switch_auth_method(auth_config, url)

            elif action == AuthErrorAction.WAIT_AND_RETRY:
                # Calculate exponential backoff delay
                delay = min(2**attempt, 60)  # Max 60 seconds
                self.logger.info(f"Waiting {delay} seconds before retry")
                await asyncio.sleep(delay)

                # Return current config to retry
                auth_dict = self._auth_config_to_dict(auth_config)
                return await self.auth_manager.setup_authentication(auth_dict, url)

            elif action == AuthErrorAction.PROMPT_FOR_CREDENTIALS:
                self.logger.warning("Manual credential update required")
                # This would typically trigger a UI prompt or callback
                # For now, return None to indicate manual intervention needed
                return None

            elif action == AuthErrorAction.FAIL_PERMANENTLY:
                self.logger.error("Authentication failed permanently")
                return None

            else:
                self.logger.error(f"Unknown recovery action: {action}")
                return None

        except Exception as e:
            self.logger.error(f"Recovery action {action.value} failed: {e}")
            return None

    async def _try_fallback_auth(
        self, auth_config: AuthConfig, url: str
    ) -> dict[str, Any] | None:
        """
        Try fallback authentication methods.

        Args:
            auth_config: Current authentication configuration
            url: URL being accessed

        Returns:
            Fallback authentication options, or None if no fallback available
        """
        from urllib.parse import urlparse

        domain = urlparse(url).netloc

        # Try stored credentials first
        if auth_config.method == AuthMethod.USERNAME_PASSWORD:
            stored_creds = (
                await self.auth_manager.credential_manager.retrieve_credentials(domain)
            )
            if stored_creds:
                fallback_config = auth_config.model_copy()
                fallback_config.username = stored_creds["username"]
                fallback_config.password = stored_creds["password"]

                auth_dict = self._auth_config_to_dict(fallback_config)
                return await self.auth_manager.setup_authentication(auth_dict, url)

        # Try browser cookies as fallback
        if auth_config.method != AuthMethod.COOKIES:
            browser_cookies = (
                await self.auth_manager.cookie_manager.get_cookies_for_domain(domain)
            )
            if browser_cookies:
                fallback_config = AuthConfig(method=AuthMethod.COOKIES)
                # Save cookies to temporary file
                temp_cookies_file = self.auth_manager.cookie_manager.cookies_file
                if temp_cookies_file:
                    await self.auth_manager.cookie_manager.save_cookies(
                        browser_cookies, domain
                    )
                    fallback_config.cookies_file = temp_cookies_file

                    auth_dict = self._auth_config_to_dict(fallback_config)
                    return await self.auth_manager.setup_authentication(auth_dict, url)

        return None

    async def _switch_auth_method(
        self, auth_config: AuthConfig, url: str
    ) -> dict[str, Any] | None:
        """
        Switch to an alternative authentication method.

        Args:
            auth_config: Current authentication configuration
            url: URL being accessed

        Returns:
            Alternative authentication options, or None if no alternative available
        """
        from urllib.parse import urlparse

        domain = urlparse(url).netloc

        # Define method priority order
        method_priority = [
            AuthMethod.COOKIES,
            AuthMethod.USERNAME_PASSWORD,
            AuthMethod.NETRC,
            AuthMethod.OAUTH,
        ]

        # Try methods in priority order, skipping current method
        for method in method_priority:
            if method == auth_config.method:
                continue

            try:
                if method == AuthMethod.COOKIES:
                    browser_cookies = (
                        await self.auth_manager.cookie_manager.get_cookies_for_domain(
                            domain
                        )
                    )
                    if browser_cookies:
                        alt_config = AuthConfig(method=method)
                        auth_dict = self._auth_config_to_dict(alt_config)
                        return await self.auth_manager.setup_authentication(
                            auth_dict, url
                        )

                elif method == AuthMethod.USERNAME_PASSWORD:
                    stored_creds = (
                        await self.auth_manager.credential_manager.retrieve_credentials(
                            domain
                        )
                    )
                    if stored_creds:
                        alt_config = AuthConfig(
                            method=method,
                            username=stored_creds["username"],
                            password=stored_creds["password"],
                        )
                        auth_dict = self._auth_config_to_dict(alt_config)
                        return await self.auth_manager.setup_authentication(
                            auth_dict, url
                        )

                elif method == AuthMethod.NETRC:
                    from pathlib import Path

                    if (Path.home() / ".netrc").exists():
                        alt_config = AuthConfig(method=method)
                        auth_dict = self._auth_config_to_dict(alt_config)
                        return await self.auth_manager.setup_authentication(
                            auth_dict, url
                        )

            except Exception as e:
                self.logger.debug(f"Failed to switch to {method.value}: {e}")
                continue

        return None

    def _auth_config_to_dict(self, auth_config: AuthConfig) -> dict[str, Any]:
        """
        Convert AuthConfig to dictionary format.

        Args:
            auth_config: Authentication configuration

        Returns:
            Dictionary representation
        """
        return {
            "method": auth_config.method.value,
            "username": auth_config.username,
            "password": auth_config.password,
            "cookies_file": str(auth_config.cookies_file)
            if auth_config.cookies_file
            else None,
            "netrc_file": str(auth_config.netrc_file)
            if auth_config.netrc_file
            else None,
            "oauth_token": auth_config.oauth_token,
        }

    async def cleanup(self) -> None:
        """Clean up error handler resources."""
        await self.auth_manager.cleanup()
        self.logger.info("Authentication error handler cleaned up")
