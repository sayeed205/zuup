"""Comprehensive logging and debugging support for pycurl engine."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

import pycurl

from ..storage.models import EngineType
from ..utils.logging import get_download_logger
from .pycurl_models import (
    CurlError,
    DownloadSegment,
    HttpFtpConfig,
    WorkerProgress,
)


class LogLevel(Enum):
    """Logging verbosity levels for curl operations."""

    MINIMAL = "minimal"  # Only errors and completion
    BASIC = "basic"  # Basic progress and status
    DETAILED = "detailed"  # Detailed curl operations
    VERBOSE = "verbose"  # All curl debug information
    DEBUG = "debug"  # Maximum verbosity with internal state


class CurlDebugType(Enum):
    """Types of curl debug information."""

    TEXT = 0  # Informational text
    HEADER_IN = 1  # Header data received from peer
    HEADER_OUT = 2  # Header data sent to peer
    DATA_IN = 3  # Data received from peer
    DATA_OUT = 4  # Data sent to peer
    SSL_DATA_IN = 5  # SSL/TLS data received
    SSL_DATA_OUT = 6  # SSL/TLS data sent


@dataclass
class CurlDebugInfo:
    """Container for curl debug information."""

    debug_type: CurlDebugType
    data: bytes
    timestamp: float = field(default_factory=time.time)
    worker_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "debug_type": self.debug_type.name,
            "data_size": len(self.data),
            "data_preview": self.data[:200].decode("utf-8", errors="replace"),
            "timestamp": self.timestamp,
            "worker_id": self.worker_id,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for download analysis."""

    # Timing metrics
    start_time: float = 0.0
    end_time: float = 0.0
    connect_time: float = 0.0
    pretransfer_time: float = 0.0
    starttransfer_time: float = 0.0
    total_time: float = 0.0

    # Transfer metrics
    total_bytes: int = 0
    downloaded_bytes: int = 0
    upload_bytes: int = 0
    download_speed: float = 0.0
    upload_speed: float = 0.0

    # Connection metrics
    num_connects: int = 0
    num_redirects: int = 0
    response_code: int = 0

    # SSL metrics
    ssl_verify_result: int = 0
    ssl_engines: list[str] = field(default_factory=list)

    # Network metrics
    namelookup_time: float = 0.0
    appconnect_time: float = 0.0
    redirect_time: float = 0.0

    # Additional info
    effective_url: str = ""
    content_type: str = ""
    primary_ip: str = ""
    primary_port: int = 0
    local_ip: str = ""
    local_port: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timing": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration": self.end_time - self.start_time if self.end_time > 0 else 0,
                "connect_time": self.connect_time,
                "pretransfer_time": self.pretransfer_time,
                "starttransfer_time": self.starttransfer_time,
                "total_time": self.total_time,
                "namelookup_time": self.namelookup_time,
                "appconnect_time": self.appconnect_time,
                "redirect_time": self.redirect_time,
            },
            "transfer": {
                "total_bytes": self.total_bytes,
                "downloaded_bytes": self.downloaded_bytes,
                "upload_bytes": self.upload_bytes,
                "download_speed": self.download_speed,
                "upload_speed": self.upload_speed,
                "efficiency": (self.downloaded_bytes / self.total_bytes * 100)
                if self.total_bytes > 0
                else 0,
            },
            "connection": {
                "num_connects": self.num_connects,
                "num_redirects": self.num_redirects,
                "response_code": self.response_code,
                "effective_url": self.effective_url,
                "content_type": self.content_type,
                "primary_ip": self.primary_ip,
                "primary_port": self.primary_port,
                "local_ip": self.local_ip,
                "local_port": self.local_port,
            },
            "ssl": {
                "ssl_verify_result": self.ssl_verify_result,
                "ssl_engines": self.ssl_engines,
            },
        }


class CurlLogger:
    """Comprehensive logging for curl operations."""

    def __init__(
        self,
        task_id: str,
        worker_id: str,
        log_level: LogLevel = LogLevel.BASIC,
        debug_file: Path | None = None,
        metrics_file: Path | None = None,
    ):
        """
        Initialize curl logger.

        Args:
            task_id: Download task identifier
            worker_id: Worker identifier
            log_level: Logging verbosity level
            debug_file: Optional file for debug output
            metrics_file: Optional file for metrics output
        """
        self.task_id = task_id
        self.worker_id = worker_id
        self.log_level = log_level
        self.debug_file = debug_file
        self.metrics_file = metrics_file

        # Get logger adapter with context
        self.logger = get_download_logger(task_id, EngineType.HTTP)

        # Debug information storage
        self.debug_info: list[CurlDebugInfo] = []
        self.max_debug_entries = 1000  # Limit memory usage

        # Performance metrics
        self.metrics = PerformanceMetrics()

        # Speed tracking for analysis
        self.speed_history: deque = deque(maxlen=100)  # Last 100 speed samples
        self.progress_history: deque = deque(maxlen=100)  # Last 100 progress samples

        # Error tracking
        self.error_count = 0
        self.retry_count = 0
        self.last_error: Exception | None = None

        # Connection tracking
        self.connection_attempts = 0
        self.successful_connections = 0
        self.failed_connections = 0

        self.logger.debug(
            f"Initialized CurlLogger for worker {worker_id} with level {log_level.value}"
        )

    def set_log_level(self, level: LogLevel) -> None:
        """Update logging level."""
        self.log_level = level
        self.logger.info(
            f"Updated log level to {level.value} for worker {self.worker_id}"
        )

    def create_debug_callback(self) -> Callable[[int, bytes], int]:
        """
        Create curl debug callback function.

        Returns:
            Debug callback function for pycurl
        """

        def debug_callback(debug_type: int, debug_data: bytes) -> int:
            """Curl debug callback function."""
            try:
                curl_debug_type = CurlDebugType(debug_type)
                debug_info = CurlDebugInfo(
                    debug_type=curl_debug_type,
                    data=debug_data,
                    worker_id=self.worker_id,
                )

                # Store debug info (with size limit)
                if len(self.debug_info) >= self.max_debug_entries:
                    self.debug_info.pop(0)  # Remove oldest entry
                self.debug_info.append(debug_info)

                # Log based on verbosity level
                if self.log_level in (LogLevel.VERBOSE, LogLevel.DEBUG):
                    self._log_debug_info(debug_info)

                # Write to debug file if specified
                if self.debug_file and self.log_level == LogLevel.DEBUG:
                    self._write_debug_to_file(debug_info)

                return 0  # Continue

            except Exception as e:
                # Don't let debug callback errors break the download
                self.logger.error(f"Error in debug callback: {e}")
                return 0

        return debug_callback

    def _log_debug_info(self, debug_info: CurlDebugInfo) -> None:
        """Log debug information based on type."""
        debug_type = debug_info.debug_type
        data_preview = debug_info.data[:100].decode("utf-8", errors="replace")

        if debug_type == CurlDebugType.TEXT:
            self.logger.debug(f"CURL INFO: {data_preview.strip()}")
        elif debug_type == CurlDebugType.HEADER_IN:
            self.logger.debug(f"RECV HEADER: {data_preview.strip()}")
        elif debug_type == CurlDebugType.HEADER_OUT:
            self.logger.debug(f"SEND HEADER: {data_preview.strip()}")
        elif debug_type in (CurlDebugType.DATA_IN, CurlDebugType.DATA_OUT):
            self.logger.debug(f"DATA {debug_type.name}: {len(debug_info.data)} bytes")
        elif debug_type in (CurlDebugType.SSL_DATA_IN, CurlDebugType.SSL_DATA_OUT):
            self.logger.debug(f"SSL {debug_type.name}: {len(debug_info.data)} bytes")

    def _write_debug_to_file(self, debug_info: CurlDebugInfo) -> None:
        """Write debug information to file."""
        if not self.debug_file:
            return

        try:
            self.debug_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.debug_file, "a", encoding="utf-8") as f:
                timestamp = datetime.fromtimestamp(debug_info.timestamp).isoformat()
                f.write(
                    f"[{timestamp}] {self.worker_id} {debug_info.debug_type.name}: "
                )

                if (
                    debug_info.debug_type == CurlDebugType.TEXT
                    or debug_info.debug_type
                    in (CurlDebugType.HEADER_IN, CurlDebugType.HEADER_OUT)
                ):
                    f.write(debug_info.data.decode("utf-8", errors="replace").strip())
                else:
                    f.write(f"{len(debug_info.data)} bytes")

                f.write("\n")

        except Exception as e:
            self.logger.error(f"Failed to write debug info to file: {e}")

    def log_connection_attempt(self, url: str, segment: DownloadSegment) -> None:
        """Log connection attempt."""
        self.connection_attempts += 1

        if self.log_level in (LogLevel.DETAILED, LogLevel.VERBOSE, LogLevel.DEBUG):
            parsed_url = urlparse(url)
            self.logger.info(
                f"Attempting connection to {parsed_url.scheme}://{parsed_url.netloc} "
                f"for segment {segment.id} (attempt {self.connection_attempts})",
                extra={
                    "url": url,
                    "segment_id": segment.id,
                    "connection_attempt": self.connection_attempts,
                    "segment_range": f"{segment.start_byte}-{segment.end_byte}",
                },
            )

    def log_connection_success(self, curl_handle: pycurl.Curl) -> None:
        """Log successful connection with details."""
        self.successful_connections += 1

        try:
            # Extract connection information
            effective_url = curl_handle.getinfo(pycurl.EFFECTIVE_URL)
            primary_ip = curl_handle.getinfo(pycurl.PRIMARY_IP)
            primary_port = curl_handle.getinfo(pycurl.PRIMARY_PORT)
            local_ip = curl_handle.getinfo(pycurl.LOCAL_IP)
            local_port = curl_handle.getinfo(pycurl.LOCAL_PORT)

            if self.log_level in (LogLevel.DETAILED, LogLevel.VERBOSE, LogLevel.DEBUG):
                self.logger.info(
                    f"Connection established: {primary_ip}:{primary_port} -> {local_ip}:{local_port}",
                    extra={
                        "effective_url": effective_url.decode("utf-8")
                        if isinstance(effective_url, bytes)
                        else (effective_url or ""),
                        "primary_ip": primary_ip.decode("utf-8")
                        if isinstance(primary_ip, bytes)
                        else (primary_ip or ""),
                        "primary_port": primary_port,
                        "local_ip": local_ip.decode("utf-8")
                        if isinstance(local_ip, bytes)
                        else (local_ip or ""),
                        "local_port": local_port,
                        "successful_connections": self.successful_connections,
                    },
                )

        except Exception as e:
            self.logger.warning(f"Failed to extract connection info: {e}")

    def log_connection_failure(
        self, error: Exception, curl_code: int | None = None
    ) -> None:
        """Log connection failure."""
        self.failed_connections += 1
        self.error_count += 1
        self.last_error = error

        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "failed_connections": self.failed_connections,
            "total_attempts": self.connection_attempts,
        }

        if curl_code is not None:
            error_info["curl_code"] = curl_code

        self.logger.error(
            f"Connection failed: {error} (attempt {self.connection_attempts})",
            extra=error_info,
        )

    def log_retry_attempt(self, attempt: int, delay: float, reason: str) -> None:
        """Log retry attempt."""
        self.retry_count += 1

        if self.log_level != LogLevel.MINIMAL:
            self.logger.info(
                f"Retrying in {delay:.1f}s (attempt {attempt}): {reason}",
                extra={
                    "retry_attempt": attempt,
                    "retry_delay": delay,
                    "retry_reason": reason,
                    "total_retries": self.retry_count,
                },
            )

    def log_progress_update(self, progress: WorkerProgress) -> None:
        """Log progress update with speed tracking."""
        current_time = time.time()

        # Store progress for analysis
        self.progress_history.append(
            {
                "timestamp": current_time,
                "downloaded_bytes": progress.downloaded_bytes,
                "speed": progress.download_speed,
                "status": progress.status.value,
            }
        )

        # Store speed sample
        if progress.download_speed > 0:
            self.speed_history.append(
                {
                    "timestamp": current_time,
                    "speed": progress.download_speed,
                }
            )

        # Log based on verbosity
        if self.log_level in (
            LogLevel.BASIC,
            LogLevel.DETAILED,
            LogLevel.VERBOSE,
            LogLevel.DEBUG,
        ):
            percentage = (
                (progress.downloaded_bytes / progress.total_bytes * 100)
                if progress.total_bytes > 0
                else 0
            )
            speed_mb = progress.download_speed / (1024 * 1024)

            self.logger.info(
                f"Progress: {percentage:.1f}% ({progress.downloaded_bytes}/{progress.total_bytes} bytes) "
                f"Speed: {speed_mb:.2f} MB/s",
                extra={
                    "progress_percentage": percentage,
                    "downloaded_bytes": progress.downloaded_bytes,
                    "total_bytes": progress.total_bytes,
                    "download_speed": progress.download_speed,
                    "status": progress.status.value,
                },
            )

    def collect_performance_metrics(
        self, curl_handle: pycurl.Curl
    ) -> PerformanceMetrics:
        """
        Collect comprehensive performance metrics from curl handle.

        Args:
            curl_handle: Curl handle to extract metrics from

        Returns:
            Performance metrics object
        """
        try:
            self.metrics.end_time = time.time()

            # Timing information
            self.metrics.total_time = curl_handle.getinfo(pycurl.TOTAL_TIME)
            self.metrics.namelookup_time = curl_handle.getinfo(pycurl.NAMELOOKUP_TIME)
            self.metrics.connect_time = curl_handle.getinfo(pycurl.CONNECT_TIME)
            self.metrics.appconnect_time = curl_handle.getinfo(pycurl.APPCONNECT_TIME)
            self.metrics.pretransfer_time = curl_handle.getinfo(pycurl.PRETRANSFER_TIME)
            self.metrics.starttransfer_time = curl_handle.getinfo(
                pycurl.STARTTRANSFER_TIME
            )
            self.metrics.redirect_time = curl_handle.getinfo(pycurl.REDIRECT_TIME)

            # Transfer information
            self.metrics.downloaded_bytes = curl_handle.getinfo(pycurl.SIZE_DOWNLOAD)
            self.metrics.upload_bytes = curl_handle.getinfo(pycurl.SIZE_UPLOAD)
            self.metrics.download_speed = curl_handle.getinfo(pycurl.SPEED_DOWNLOAD)
            self.metrics.upload_speed = curl_handle.getinfo(pycurl.SPEED_UPLOAD)

            # Connection information
            self.metrics.num_connects = curl_handle.getinfo(pycurl.NUM_CONNECTS)
            self.metrics.num_redirects = curl_handle.getinfo(pycurl.REDIRECT_COUNT)
            self.metrics.response_code = curl_handle.getinfo(pycurl.RESPONSE_CODE)

            # Helper function to safely decode bytes or return string
            def safe_decode(value):
                if isinstance(value, bytes):
                    return value.decode("utf-8")
                return value if value is not None else ""

            # URL and content information
            effective_url = curl_handle.getinfo(pycurl.EFFECTIVE_URL)
            self.metrics.effective_url = safe_decode(effective_url)

            content_type = curl_handle.getinfo(pycurl.CONTENT_TYPE)
            self.metrics.content_type = safe_decode(content_type)

            # Network information
            primary_ip = curl_handle.getinfo(pycurl.PRIMARY_IP)
            self.metrics.primary_ip = safe_decode(primary_ip)
            self.metrics.primary_port = curl_handle.getinfo(pycurl.PRIMARY_PORT)

            local_ip = curl_handle.getinfo(pycurl.LOCAL_IP)
            self.metrics.local_ip = safe_decode(local_ip)
            self.metrics.local_port = curl_handle.getinfo(pycurl.LOCAL_PORT)

            # SSL information
            self.metrics.ssl_verify_result = curl_handle.getinfo(
                pycurl.SSL_VERIFYRESULT
            )

            # Log metrics if detailed logging is enabled
            if self.log_level in (LogLevel.DETAILED, LogLevel.VERBOSE, LogLevel.DEBUG):
                self._log_performance_metrics()

            # Write metrics to file if specified
            if self.metrics_file:
                self._write_metrics_to_file()

            return self.metrics

        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}")
            return self.metrics

    def _log_performance_metrics(self) -> None:
        """Log performance metrics summary."""
        metrics_dict = self.metrics.to_dict()

        self.logger.info(
            f"Performance Summary - Duration: {metrics_dict['timing']['duration']:.2f}s, "
            f"Speed: {metrics_dict['transfer']['download_speed'] / (1024 * 1024):.2f} MB/s, "
            f"Efficiency: {metrics_dict['transfer']['efficiency']:.1f}%",
            extra=metrics_dict,
        )

    def _write_metrics_to_file(self) -> None:
        """Write performance metrics to JSON file."""
        if not self.metrics_file:
            return

        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "task_id": self.task_id,
                "worker_id": self.worker_id,
                "metrics": self.metrics.to_dict(),
                "debug_summary": {
                    "debug_entries": len(self.debug_info),
                    "error_count": self.error_count,
                    "retry_count": self.retry_count,
                    "connection_attempts": self.connection_attempts,
                    "successful_connections": self.successful_connections,
                    "failed_connections": self.failed_connections,
                },
                "speed_analysis": self._analyze_speed_history(),
            }

            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to write metrics to file: {e}")

    def _analyze_speed_history(self) -> dict[str, Any]:
        """Analyze speed history for performance insights."""
        if not self.speed_history:
            return {}

        speeds = [sample["speed"] for sample in self.speed_history]

        return {
            "sample_count": len(speeds),
            "average_speed": sum(speeds) / len(speeds),
            "max_speed": max(speeds),
            "min_speed": min(speeds),
            "speed_variance": self._calculate_variance(speeds),
            "speed_stability": self._calculate_stability(speeds),
        }

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _calculate_stability(self, speeds: list[float]) -> float:
        """Calculate speed stability (0-1, higher is more stable)."""
        if len(speeds) < 2:
            return 1.0

        avg_speed = sum(speeds) / len(speeds)
        if avg_speed == 0:
            return 0.0

        # Calculate coefficient of variation (lower is more stable)
        variance = self._calculate_variance(speeds)
        std_dev = variance**0.5
        cv = std_dev / avg_speed

        # Convert to stability score (0-1)
        stability = max(0.0, 1.0 - min(cv, 1.0))
        return stability

    def log_completion(self, success: bool, final_size: int, duration: float) -> None:
        """Log download completion."""
        if success:
            avg_speed = final_size / duration if duration > 0 else 0
            avg_speed_mb = avg_speed / (1024 * 1024)

            self.logger.info(
                f"Download completed successfully: {final_size} bytes in {duration:.2f}s "
                f"(avg: {avg_speed_mb:.2f} MB/s)",
                extra={
                    "success": True,
                    "final_size": final_size,
                    "duration": duration,
                    "average_speed": avg_speed,
                    "connection_attempts": self.connection_attempts,
                    "successful_connections": self.successful_connections,
                    "retry_count": self.retry_count,
                },
            )
        else:
            self.logger.error(
                f"Download failed after {duration:.2f}s, {self.retry_count} retries, "
                f"{self.connection_attempts} connection attempts",
                extra={
                    "success": False,
                    "duration": duration,
                    "error_count": self.error_count,
                    "retry_count": self.retry_count,
                    "connection_attempts": self.connection_attempts,
                    "failed_connections": self.failed_connections,
                    "last_error": str(self.last_error) if self.last_error else None,
                },
            )

    def get_debug_summary(self) -> dict[str, Any]:
        """Get summary of debug information."""
        debug_types = defaultdict(int)
        for debug_info in self.debug_info:
            debug_types[debug_info.debug_type.name] += 1

        return {
            "total_debug_entries": len(self.debug_info),
            "debug_types": dict(debug_types),
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "connection_stats": {
                "attempts": self.connection_attempts,
                "successful": self.successful_connections,
                "failed": self.failed_connections,
                "success_rate": (
                    self.successful_connections / self.connection_attempts * 100
                )
                if self.connection_attempts > 0
                else 0,
            },
            "last_error": str(self.last_error) if self.last_error else None,
        }


class CurlDebugUtilities:
    """Debug utilities for connection and protocol troubleshooting."""

    @staticmethod
    def diagnose_connection_error(curl_code: int, url: str) -> dict[str, Any]:
        """
        Diagnose connection errors and provide troubleshooting suggestions.

        Args:
            curl_code: Curl error code
            url: URL that failed

        Returns:
            Diagnosis information with suggestions
        """
        parsed_url = urlparse(url)
        protocol = parsed_url.scheme.lower()

        # Create a CurlError to get the error message
        curl_error = CurlError.from_curl_code(curl_code)

        diagnosis = {
            "curl_code": curl_code,
            "error_name": curl_error.curl_message,
            "error_category": curl_error.category.value,
            "recommended_action": curl_error.action.value,
            "protocol": protocol,
            "host": parsed_url.netloc,
            "suggestions": [],
            "checks": [],
        }

        # Common connection issues
        if curl_code == pycurl.E_COULDNT_CONNECT:
            diagnosis["suggestions"].extend(
                [
                    "Check if the server is running and accessible",
                    "Verify firewall settings allow outbound connections",
                    "Try connecting with a different tool (curl, wget, browser)",
                    "Check if proxy settings are correct",
                ]
            )
            diagnosis["checks"].extend(
                [
                    f"ping {parsed_url.hostname}",
                    f"telnet {parsed_url.hostname} {parsed_url.port or (443 if protocol == 'https' else 80)}",
                    "Check network connectivity",
                ]
            )

        elif curl_code == pycurl.E_COULDNT_RESOLVE_HOST:
            diagnosis["suggestions"].extend(
                [
                    "Check DNS settings",
                    "Verify hostname spelling",
                    "Try using IP address instead of hostname",
                    "Check /etc/hosts file for conflicts",
                ]
            )
            diagnosis["checks"].extend(
                [
                    f"nslookup {parsed_url.hostname}",
                    f"dig {parsed_url.hostname}",
                    "Check DNS server configuration",
                ]
            )

        elif curl_code == pycurl.E_OPERATION_TIMEDOUT:
            diagnosis["suggestions"].extend(
                [
                    "Increase timeout values",
                    "Check network latency",
                    "Verify server is not overloaded",
                    "Try reducing number of concurrent connections",
                ]
            )
            diagnosis["checks"].extend(
                [
                    f"ping -c 4 {parsed_url.hostname}",
                    "Check network latency and packet loss",
                    "Monitor server response times",
                ]
            )

        elif curl_code == pycurl.E_SSL_CONNECT_ERROR:
            diagnosis["suggestions"].extend(
                [
                    "Check SSL certificate validity",
                    "Verify SSL/TLS version compatibility",
                    "Check system time accuracy",
                    "Try disabling SSL verification for testing",
                ]
            )
            diagnosis["checks"].extend(
                [
                    f"openssl s_client -connect {parsed_url.hostname}:{parsed_url.port or 443}",
                    "Check certificate expiration",
                    "Verify CA certificate bundle",
                ]
            )

        elif curl_code == pycurl.E_PEER_FAILED_VERIFICATION:
            diagnosis["suggestions"].extend(
                [
                    "Check SSL certificate chain",
                    "Verify hostname matches certificate",
                    "Update CA certificate bundle",
                    "Check for self-signed certificates",
                ]
            )

        elif curl_code == pycurl.E_HTTP_RETURNED_ERROR:
            diagnosis["suggestions"].extend(
                [
                    "Check HTTP response code",
                    "Verify authentication credentials",
                    "Check request headers and parameters",
                    "Review server logs for details",
                ]
            )

        # Protocol-specific suggestions
        if protocol == "ftp":
            diagnosis["suggestions"].extend(
                [
                    "Check FTP server supports passive mode",
                    "Verify FTP credentials",
                    "Try switching between EPSV and PASV modes",
                    "Check firewall allows FTP data connections",
                ]
            )
        elif protocol == "sftp":
            diagnosis["suggestions"].extend(
                [
                    "Verify SSH key permissions (600 for private key)",
                    "Check SSH server configuration",
                    "Verify known_hosts file",
                    "Test SSH connection manually",
                ]
            )
            diagnosis["checks"].extend(
                [
                    f"ssh -T {parsed_url.hostname}",
                    "Check SSH key authentication",
                ]
            )

        return diagnosis

    @staticmethod
    def analyze_curl_info(curl_handle: pycurl.Curl) -> dict[str, Any]:
        """
        Analyze curl handle information for debugging.

        Args:
            curl_handle: Curl handle to analyze

        Returns:
            Analysis information
        """
        try:
            info = {}

            # Basic information
            effective_url = curl_handle.getinfo(pycurl.EFFECTIVE_URL)
            info["effective_url"] = (
                effective_url.decode("utf-8")
                if isinstance(effective_url, bytes)
                else effective_url
            )
            info["response_code"] = curl_handle.getinfo(pycurl.RESPONSE_CODE)
            info["total_time"] = curl_handle.getinfo(pycurl.TOTAL_TIME)

            # Timing breakdown
            info["timing"] = {
                "namelookup": curl_handle.getinfo(pycurl.NAMELOOKUP_TIME),
                "connect": curl_handle.getinfo(pycurl.CONNECT_TIME),
                "appconnect": curl_handle.getinfo(pycurl.APPCONNECT_TIME),
                "pretransfer": curl_handle.getinfo(pycurl.PRETRANSFER_TIME),
                "starttransfer": curl_handle.getinfo(pycurl.STARTTRANSFER_TIME),
                "redirect": curl_handle.getinfo(pycurl.REDIRECT_TIME),
            }

            # Transfer information
            info["transfer"] = {
                "size_download": curl_handle.getinfo(pycurl.SIZE_DOWNLOAD),
                "size_upload": curl_handle.getinfo(pycurl.SIZE_UPLOAD),
                "speed_download": curl_handle.getinfo(pycurl.SPEED_DOWNLOAD),
                "speed_upload": curl_handle.getinfo(pycurl.SPEED_UPLOAD),
            }

            # Connection information
            primary_ip = curl_handle.getinfo(pycurl.PRIMARY_IP)
            local_ip = curl_handle.getinfo(pycurl.LOCAL_IP)
            info["connection"] = {
                "primary_ip": primary_ip.decode("utf-8")
                if isinstance(primary_ip, bytes)
                else primary_ip,
                "primary_port": curl_handle.getinfo(pycurl.PRIMARY_PORT),
                "local_ip": local_ip.decode("utf-8")
                if isinstance(local_ip, bytes)
                else local_ip,
                "local_port": curl_handle.getinfo(pycurl.LOCAL_PORT),
                "num_connects": curl_handle.getinfo(pycurl.NUM_CONNECTS),
                "redirect_count": curl_handle.getinfo(pycurl.REDIRECT_COUNT),
            }

            # SSL information
            info["ssl"] = {
                "ssl_verify_result": curl_handle.getinfo(pycurl.SSL_VERIFYRESULT),
            }

            # Content information
            content_type = curl_handle.getinfo(pycurl.CONTENT_TYPE)
            if content_type:
                info["content_type"] = (
                    content_type.decode("utf-8")
                    if isinstance(content_type, bytes)
                    else content_type
                )

            return info

        except Exception as e:
            return {"error": f"Failed to analyze curl info: {e}"}

    @staticmethod
    def generate_curl_command(
        url: str, config: HttpFtpConfig, segment: DownloadSegment | None = None
    ) -> str:
        """
        Generate equivalent curl command for debugging.

        Args:
            url: URL to download
            config: Configuration to use
            segment: Optional segment information

        Returns:
            Curl command string
        """
        cmd_parts = ["curl"]

        # Basic options
        cmd_parts.extend(["-L", "-v"])  # Follow redirects, verbose

        # Timeouts
        cmd_parts.extend(["--connect-timeout", str(config.connect_timeout)])
        cmd_parts.extend(["--max-time", str(config.timeout)])

        # User agent
        cmd_parts.extend(["--user-agent", f'"{config.user_agent}"'])

        # Range request
        if segment:
            cmd_parts.extend(["--range", f"{segment.start_byte}-{segment.end_byte}"])

        # Authentication
        if config.auth.method.value != "none":
            username = config.auth.get_username()
            password = config.auth.get_password()

            if config.auth.method.value == "basic":
                cmd_parts.extend(["--basic", "--user", f"{username}:{password}"])
            elif config.auth.method.value == "digest":
                cmd_parts.extend(["--digest", "--user", f"{username}:{password}"])
            elif config.auth.method.value == "bearer":
                token = config.auth.get_token()
                cmd_parts.extend(["--header", f'"Authorization: Bearer {token}"'])

        # Headers
        if config.custom_headers:
            for key, value in config.custom_headers.items():
                cmd_parts.extend(["--header", f'"{key}: {value}"'])

        # Cookies
        if config.cookies:
            cookie_string = "; ".join([f"{k}={v}" for k, v in config.cookies.items()])
            cmd_parts.extend(["--cookie", f'"{cookie_string}"'])

        # SSL options
        if not config.verify_ssl:
            cmd_parts.extend(["--insecure"])

        if config.ca_cert_path:
            cmd_parts.extend(["--cacert", str(config.ca_cert_path)])

        # Proxy
        if config.proxy.enabled and config.proxy.host:
            cmd_parts.extend(["--proxy", config.proxy.proxy_url])
            if config.proxy.username and config.proxy.password:
                cmd_parts.extend(
                    ["--proxy-user", f"{config.proxy.username}:{config.proxy.password}"]
                )

        # Output
        if segment:
            cmd_parts.extend(["--output", str(segment.temp_file_path)])
        else:
            cmd_parts.extend(["--remote-name"])

        # URL (last)
        cmd_parts.append(f'"{url}"')

        return " ".join(cmd_parts)


def setup_curl_logging(
    task_id: str,
    worker_id: str,
    log_level: LogLevel = LogLevel.BASIC,
    log_dir: Path | None = None,
) -> CurlLogger:
    """
    Set up curl logging for a worker.

    Args:
        task_id: Download task identifier
        worker_id: Worker identifier
        log_level: Logging verbosity level
        log_dir: Optional directory for log files

    Returns:
        Configured curl logger
    """
    debug_file = None
    metrics_file = None

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_level == LogLevel.DEBUG:
            debug_file = log_dir / f"{worker_id}_debug.log"

        if log_level in (LogLevel.DETAILED, LogLevel.VERBOSE, LogLevel.DEBUG):
            metrics_file = log_dir / f"{worker_id}_metrics.json"

    return CurlLogger(
        task_id=task_id,
        worker_id=worker_id,
        log_level=log_level,
        debug_file=debug_file,
        metrics_file=metrics_file,
    )
