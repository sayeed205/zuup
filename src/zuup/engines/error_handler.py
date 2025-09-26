"""Comprehensive error handling and recovery system for pycurl downloads."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Dict, Optional

import pycurl

from .pycurl_models import (
    CurlError,
    ErrorAction,
    ErrorCategory,
    ErrorContext,
    WorkerStatus,
)
from ..utils.logging import DownloadLoggerAdapter


class ErrorHandler:
    """
    Handles categorized error processing, retry logic, and recovery strategies.
    
    This class provides comprehensive error handling for pycurl-based downloads,
    including automatic retry with exponential backoff, connection reduction
    strategies, and detailed error logging.
    """

    def __init__(
        self,
        max_retry_attempts: int = 3,
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter_factor: float = 0.1,
        connection_reduction_threshold: int = 3,
        min_connections: int = 1,
    ) -> None:
        """
        Initialize the error handler.

        Args:
            max_retry_attempts: Maximum number of retry attempts per error
            base_retry_delay: Base delay in seconds for retry backoff
            max_retry_delay: Maximum delay in seconds for retry backoff
            backoff_factor: Exponential backoff multiplier
            jitter_factor: Random jitter factor (0.0 to 1.0)
            connection_reduction_threshold: Number of errors before reducing connections
            min_connections: Minimum number of connections to maintain
        """
        self.max_retry_attempts = max_retry_attempts
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay
        self.backoff_factor = backoff_factor
        self.jitter_factor = jitter_factor
        self.connection_reduction_threshold = connection_reduction_threshold
        self.min_connections = min_connections

        # Error tracking for connection reduction decisions
        self._error_counts: Dict[str, int] = {}  # task_id -> error_count
        self._connection_errors: Dict[str, int] = {}  # task_id -> connection_error_count
        self._last_error_times: Dict[str, float] = {}  # task_id -> timestamp
        
        # Connection reduction tracking
        self._reduced_connections: Dict[str, int] = {}  # task_id -> reduced_connection_count
        
        self.logger = logging.getLogger(__name__)

    async def handle_error(
        self,
        error: CurlError,
        context: ErrorContext,
        logger: Optional[DownloadLoggerAdapter] = None,
    ) -> ErrorAction:
        """
        Handle a curl error and determine the appropriate action.

        Args:
            error: The curl error that occurred
            context: Additional context about the error
            logger: Optional logger adapter for structured logging

        Returns:
            The action to take in response to the error
        """
        task_id = context.get("task_id", "unknown")
        worker_id = context.get("worker_id", "unknown")
        attempt = context.get("attempt", 0)
        
        # Log the error with full context
        self._log_error(error, context, logger)
        
        # Update error tracking
        self._update_error_tracking(task_id, error)
        
        # Determine action based on error category and context
        action = self._determine_error_action(error, context)
        
        # Log the determined action
        if logger:
            logger.info(
                f"Error action determined: {action.value} for worker {worker_id}",
                extra={
                    "error_code": error.curl_code,
                    "error_category": error.category.value,
                    "attempt": attempt,
                    "action": action.value,
                }
            )
        
        return action

    def should_retry(self, error: CurlError, attempt: int, context: ErrorContext) -> bool:
        """
        Determine if an error should be retried.

        Args:
            error: The curl error that occurred
            attempt: Current attempt number (0-based)
            context: Additional context about the error

        Returns:
            True if the error should be retried, False otherwise
        """
        # Check if we've exceeded max retry attempts
        if attempt >= self.max_retry_attempts:
            return False
        
        # Check if error category is retryable
        retryable_categories = {
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.CURL,
        }
        
        if error.category not in retryable_categories:
            return False
        
        # Check specific curl error codes that should not be retried
        non_retryable_codes = {
            pycurl.E_LOGIN_DENIED,
            pycurl.E_REMOTE_ACCESS_DENIED,
            pycurl.E_SSL_PEER_CERTIFICATE,
            pycurl.E_SSL_CACERT,
            pycurl.E_FILE_COULDNT_READ_FILE,
            pycurl.E_WRITE_ERROR,
        }
        
        if error.curl_code in non_retryable_codes:
            return False
        
        # Check if we're in a rapid error loop
        task_id = context.get("task_id", "")
        if self._is_rapid_error_loop(task_id):
            return False
        
        return True

    async def calculate_retry_delay(self, attempt: int, error: CurlError) -> float:
        """
        Calculate the delay before retrying based on exponential backoff with jitter.

        Args:
            attempt: Current attempt number (0-based)
            error: The curl error that occurred

        Returns:
            Delay in seconds before retrying
        """
        # Base exponential backoff
        delay = self.base_retry_delay * (self.backoff_factor ** attempt)
        
        # Cap at maximum delay
        delay = min(delay, self.max_retry_delay)
        
        # Add jitter to prevent thundering herd
        jitter = delay * self.jitter_factor * random.random()
        delay += jitter
        
        # Adjust delay based on error category
        if error.category == ErrorCategory.NETWORK:
            # Network errors might benefit from longer delays
            delay *= 1.5
        elif error.category == ErrorCategory.TIMEOUT:
            # Timeout errors might need even longer delays
            delay *= 2.0
        
        return delay

    def should_reduce_connections(self, task_id: str, error: CurlError) -> bool:
        """
        Determine if connections should be reduced for a task.

        Args:
            task_id: Task identifier
            error: The curl error that occurred

        Returns:
            True if connections should be reduced, False otherwise
        """
        # Check if error indicates server overload
        overload_indicators = {
            pycurl.E_TOO_MANY_REDIRECTS,
            pycurl.E_COULDNT_CONNECT,
            pycurl.E_OPERATION_TIMEDOUT,
        }
        
        if error.curl_code not in overload_indicators:
            return False
        
        # Check if we've hit the error threshold
        connection_errors = self._connection_errors.get(task_id, 0)
        return connection_errors >= self.connection_reduction_threshold

    def get_reduced_connection_count(self, task_id: str, current_connections: int) -> int:
        """
        Calculate the reduced number of connections for a task.

        Args:
            task_id: Task identifier
            current_connections: Current number of connections

        Returns:
            Reduced number of connections
        """
        # Reduce by half, but maintain minimum
        reduced = max(current_connections // 2, self.min_connections)
        
        # Track the reduction
        self._reduced_connections[task_id] = reduced
        
        self.logger.info(
            f"Reducing connections for task {task_id} from {current_connections} to {reduced}"
        )
        
        return reduced

    def get_user_friendly_message(self, error: CurlError, context: ErrorContext) -> str:
        """
        Generate a user-friendly error message.

        Args:
            error: The curl error that occurred
            context: Additional context about the error

        Returns:
            User-friendly error message
        """
        url = context.get("url", "")
        worker_id = context.get("worker_id", "")
        
        # Base message templates by category
        category_messages = {
            ErrorCategory.NETWORK: "Network connection failed",
            ErrorCategory.PROTOCOL: "Server communication error",
            ErrorCategory.AUTHENTICATION: "Authentication failed",
            ErrorCategory.FILESYSTEM: "File system error",
            ErrorCategory.TIMEOUT: "Connection timed out",
            ErrorCategory.CURL: "Download library error",
            ErrorCategory.CONFIGURATION: "Configuration error",
            ErrorCategory.CANCELLED: "Download was cancelled",
        }
        
        base_message = category_messages.get(error.category, "Unknown error occurred")
        
        # Specific error code messages
        specific_messages = {
            pycurl.E_COULDNT_RESOLVE_HOST: "Could not resolve hostname",
            pycurl.E_COULDNT_CONNECT: "Could not connect to server",
            pycurl.E_OPERATION_TIMEDOUT: "Connection timed out",
            pycurl.E_HTTP_RETURNED_ERROR: "Server returned an error",
            pycurl.E_LOGIN_DENIED: "Login credentials were rejected",
            pycurl.E_REMOTE_ACCESS_DENIED: "Access denied by server",
            pycurl.E_SSL_CONNECT_ERROR: "SSL connection failed",
            pycurl.E_SSL_PEER_CERTIFICATE: "SSL certificate verification failed",
            pycurl.E_WRITE_ERROR: "Could not write to file",
            pycurl.E_PARTIAL_FILE: "Incomplete file received",
            pycurl.E_TOO_MANY_REDIRECTS: "Too many redirects",
        }
        
        if error.curl_code in specific_messages:
            base_message = specific_messages[error.curl_code]
        
        # Add context information
        if url:
            base_message += f" for {url}"
        
        if worker_id:
            base_message += f" (worker: {worker_id})"
        
        # Add retry information if applicable
        attempt = context.get("attempt", 0)
        if attempt > 0:
            base_message += f" (attempt {attempt + 1})"
        
        return base_message

    def reset_error_tracking(self, task_id: str) -> None:
        """
        Reset error tracking for a task.

        Args:
            task_id: Task identifier
        """
        self._error_counts.pop(task_id, None)
        self._connection_errors.pop(task_id, None)
        self._last_error_times.pop(task_id, None)
        self._reduced_connections.pop(task_id, None)

    def get_error_statistics(self, task_id: str) -> Dict[str, Any]:
        """
        Get error statistics for a task.

        Args:
            task_id: Task identifier

        Returns:
            Dictionary containing error statistics
        """
        return {
            "total_errors": self._error_counts.get(task_id, 0),
            "connection_errors": self._connection_errors.get(task_id, 0),
            "last_error_time": self._last_error_times.get(task_id),
            "reduced_connections": self._reduced_connections.get(task_id),
        }

    # Private methods

    def _log_error(
        self,
        error: CurlError,
        context: ErrorContext,
        logger: Optional[DownloadLoggerAdapter],
    ) -> None:
        """Log error with full context and structured information."""
        task_id = context.get("task_id", "unknown")
        worker_id = context.get("worker_id", "unknown")
        url = context.get("url", "")
        attempt = context.get("attempt", 0)
        
        # Create user-friendly message
        user_message = self.get_user_friendly_message(error, context)
        
        # Log with structured data
        log_extra = {
            "curl_code": error.curl_code,
            "curl_message": error.curl_message,
            "error_category": error.category.value,
            "error_action": error.action.value,
            "worker_id": worker_id,
            "attempt": attempt,
            "url": url,
            "timestamp": error.timestamp,
        }
        
        if logger:
            logger.error(user_message, extra=log_extra)
        else:
            self.logger.error(
                f"Task {task_id}: {user_message}",
                extra=log_extra,
            )

    def _update_error_tracking(self, task_id: str, error: CurlError) -> None:
        """Update error tracking counters and timestamps."""
        current_time = time.time()
        
        # Update total error count
        self._error_counts[task_id] = self._error_counts.get(task_id, 0) + 1
        
        # Update connection-specific error count
        if error.category in (ErrorCategory.NETWORK, ErrorCategory.TIMEOUT, ErrorCategory.PROTOCOL):
            # Count protocol errors that indicate server overload as connection errors
            if error.category == ErrorCategory.PROTOCOL and error.curl_code == pycurl.E_TOO_MANY_REDIRECTS:
                self._connection_errors[task_id] = self._connection_errors.get(task_id, 0) + 1
            elif error.category in (ErrorCategory.NETWORK, ErrorCategory.TIMEOUT):
                self._connection_errors[task_id] = self._connection_errors.get(task_id, 0) + 1
        
        # Update last error time
        self._last_error_times[task_id] = current_time

    def _determine_error_action(self, error: CurlError, context: ErrorContext) -> ErrorAction:
        """Determine the appropriate action based on error and context."""
        task_id = context.get("task_id", "")
        attempt = context.get("attempt", 0)
        
        # Use the action suggested by the error categorization
        suggested_action = error.action
        
        # Override based on retry logic
        if suggested_action == ErrorAction.RETRY:
            if not self.should_retry(error, attempt, context):
                suggested_action = ErrorAction.FAIL_SEGMENT
        
        # Override based on connection reduction logic
        if self.should_reduce_connections(task_id, error):
            suggested_action = ErrorAction.REDUCE_CONNECTIONS
        
        return suggested_action

    def _is_rapid_error_loop(self, task_id: str, time_window: float = 10.0) -> bool:
        """
        Check if we're in a rapid error loop.

        Args:
            task_id: Task identifier
            time_window: Time window in seconds to check for rapid errors

        Returns:
            True if in rapid error loop, False otherwise
        """
        current_time = time.time()
        last_error_time = self._last_error_times.get(task_id)
        
        if not last_error_time:
            return False
        
        # Check if errors are happening too frequently
        time_since_last = current_time - last_error_time
        error_count = self._error_counts.get(task_id, 0)
        
        # If we have many errors in a short time window, it's a rapid loop
        return time_since_last < time_window and error_count > 5


class ConnectionReductionStrategy:
    """
    Manages connection reduction strategies for server limitations.
    
    This class implements various strategies for reducing connections
    when servers indicate they are overloaded or have limitations.
    """

    def __init__(self, min_connections: int = 1, max_connections: int = 8) -> None:
        """
        Initialize connection reduction strategy.

        Args:
            min_connections: Minimum number of connections to maintain
            max_connections: Maximum number of connections allowed
        """
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.logger = logging.getLogger(__name__)

    def calculate_optimal_connections(
        self,
        current_connections: int,
        error_rate: float,
        success_rate: float,
        server_response_time: float,
    ) -> int:
        """
        Calculate optimal number of connections based on performance metrics.

        Args:
            current_connections: Current number of connections
            error_rate: Error rate (0.0 to 1.0)
            success_rate: Success rate (0.0 to 1.0)
            server_response_time: Average server response time in seconds

        Returns:
            Optimal number of connections
        """
        # Start with current connections
        optimal = current_connections
        
        # Reduce if error rate is high
        if error_rate > 0.3:  # More than 30% errors
            optimal = max(optimal // 2, self.min_connections)
        elif error_rate > 0.1:  # More than 10% errors
            optimal = max(optimal - 1, self.min_connections)
        
        # Reduce if server response time is high
        if server_response_time > 5.0:  # More than 5 seconds
            optimal = max(optimal // 2, self.min_connections)
        elif server_response_time > 2.0:  # More than 2 seconds
            optimal = max(optimal - 1, self.min_connections)
        
        # Increase if performance is good
        if error_rate < 0.05 and success_rate > 0.9 and server_response_time < 1.0:
            optimal = min(optimal + 1, self.max_connections)
        
        return optimal

    def apply_server_hints(self, current_connections: int, server_hints: Dict[str, Any]) -> int:
        """
        Apply server-provided hints to adjust connection count.

        Args:
            current_connections: Current number of connections
            server_hints: Dictionary of server hints (e.g., from headers)

        Returns:
            Adjusted number of connections
        """
        optimal = current_connections
        
        # Check for retry-after header
        retry_after = server_hints.get("retry_after")
        if retry_after:
            # Server is asking us to back off
            optimal = max(optimal // 2, self.min_connections)
        
        # Check for connection limits in headers
        connection_limit = server_hints.get("connection_limit")
        if connection_limit:
            optimal = min(optimal, connection_limit)
        
        # Check for rate limiting indicators
        rate_limited = server_hints.get("rate_limited", False)
        if rate_limited:
            optimal = max(optimal - 2, self.min_connections)
        
        return optimal

    def get_backoff_strategy(self, consecutive_failures: int) -> Dict[str, Any]:
        """
        Get backoff strategy based on consecutive failures.

        Args:
            consecutive_failures: Number of consecutive failures

        Returns:
            Dictionary with backoff strategy parameters
        """
        if consecutive_failures <= 2:
            return {
                "connections": max(self.max_connections // 2, self.min_connections),
                "delay": 1.0,
                "strategy": "mild_reduction",
            }
        elif consecutive_failures <= 5:
            return {
                "connections": max(self.max_connections // 4, self.min_connections),
                "delay": 5.0,
                "strategy": "moderate_reduction",
            }
        else:
            return {
                "connections": self.min_connections,
                "delay": 15.0,
                "strategy": "aggressive_reduction",
            }


# Utility functions for error handling

def categorize_curl_error(curl_code: int) -> tuple[ErrorCategory, ErrorAction]:
    """
    Categorize a curl error code and suggest an action.

    Args:
        curl_code: Curl error code

    Returns:
        Tuple of (error_category, suggested_action)
    """
    # Network errors - usually retryable
    network_errors = {
        pycurl.E_COULDNT_RESOLVE_HOST,
        pycurl.E_COULDNT_CONNECT,
        pycurl.E_RECV_ERROR,
        pycurl.E_SEND_ERROR,
        pycurl.E_GOT_NOTHING,
    }
    
    # Timeout errors - retryable with backoff
    timeout_errors = {
        pycurl.E_OPERATION_TIMEDOUT,
        pycurl.E_OPERATION_TIMEOUTED,  # Alternative spelling in some versions
    }
    
    # Protocol errors - some retryable, some not
    protocol_errors = {
        pycurl.E_HTTP_RETURNED_ERROR,
        pycurl.E_FTP_WEIRD_SERVER_REPLY,
        pycurl.E_PARTIAL_FILE,
        pycurl.E_TOO_MANY_REDIRECTS,
    }
    
    # Authentication errors - not retryable
    auth_errors = {
        pycurl.E_LOGIN_DENIED,
        pycurl.E_REMOTE_ACCESS_DENIED,
    }
    
    # SSL errors - not retryable
    ssl_errors = {
        pycurl.E_SSL_CONNECT_ERROR,
        pycurl.E_SSL_PEER_CERTIFICATE,
        pycurl.E_SSL_CACERT,
    }
    
    # File system errors - not retryable
    fs_errors = {
        pycurl.E_WRITE_ERROR,
        pycurl.E_FILE_COULDNT_READ_FILE,
    }
    
    if curl_code in network_errors:
        return ErrorCategory.NETWORK, ErrorAction.RETRY
    elif curl_code in timeout_errors:
        return ErrorCategory.TIMEOUT, ErrorAction.RETRY
    elif curl_code in protocol_errors:
        if curl_code == pycurl.E_TOO_MANY_REDIRECTS:
            return ErrorCategory.PROTOCOL, ErrorAction.REDUCE_CONNECTIONS
        else:
            return ErrorCategory.PROTOCOL, ErrorAction.RETRY
    elif curl_code in auth_errors:
        return ErrorCategory.AUTHENTICATION, ErrorAction.FAIL_DOWNLOAD
    elif curl_code in ssl_errors:
        return ErrorCategory.PROTOCOL, ErrorAction.FAIL_DOWNLOAD
    elif curl_code in fs_errors:
        return ErrorCategory.FILESYSTEM, ErrorAction.FAIL_DOWNLOAD
    else:
        return ErrorCategory.CURL, ErrorAction.RETRY


def create_error_context(
    task_id: str,
    worker_id: str,
    url: str,
    attempt: int = 0,
    **kwargs: Any,
) -> ErrorContext:
    """
    Create an error context dictionary.

    Args:
        task_id: Task identifier
        worker_id: Worker identifier
        url: URL being downloaded
        attempt: Current attempt number
        **kwargs: Additional context data

    Returns:
        Error context dictionary
    """
    context = {
        "task_id": task_id,
        "worker_id": worker_id,
        "url": url,
        "attempt": attempt,
        "timestamp": time.time(),
    }
    context.update(kwargs)
    return context