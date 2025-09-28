"""HttpFtpEngine implementation for HTTP/HTTPS and FTP/SFTP downloads using pycurl."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import logging
import math
from pathlib import Path
import re
import time
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import pycurl

from ..storage.models import DownloadTask, ProgressInfo, TaskStatus, TaskConfig, GlobalConfig
from ..utils.logging import get_download_logger, EngineType
from .base import BaseDownloadEngine, DownloadError, NetworkError, ValidationError
from .connection_manager import ConnectionManager
from .pycurl_models import (
    DownloadSegment,
    HttpFtpConfig,
    SegmentStatus,
    WorkerProgress,
    WorkerStatus,
    CompletedSegment,
)
from .segment_merger import SegmentMerger
from .config_integration import ConfigurationManager, ConfigurationError
from .pycurl_logging import LogLevel, CurlDebugUtilities

logger = logging.getLogger(__name__)


class HttpFtpEngine(BaseDownloadEngine):
    """Main engine implementing HTTP/HTTPS and FTP/SFTP downloads using pycurl."""

    def __init__(
        self, 
        config: HttpFtpConfig | None = None,
        global_config: GlobalConfig | None = None,
        config_manager: ConfigurationManager | None = None,
        log_level: LogLevel = LogLevel.BASIC,
        log_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize HttpFtpEngine.

        Args:
            config: Direct configuration for HTTP/FTP downloads (optional)
            global_config: Global application configuration (optional)
            config_manager: Configuration manager for advanced config handling (optional)
            log_level: Default logging level for workers
            log_dir: Optional directory for debug logs
        """
        super().__init__()
        
        # Configuration management
        self._config_manager = config_manager or ConfigurationManager()
        self._global_config = global_config
        self._base_config = config or HttpFtpConfig()

        # Logging configuration
        self._log_level = log_level
        self._log_dir = log_dir

        # Active downloads tracking
        self._active_downloads: dict[str, dict[str, Any]] = {}
        self._download_locks: dict[str, asyncio.Lock] = {}

        logger.info(
            f"Initialized HttpFtpEngine with max {self._base_config.max_connections} connections "
            f"and {log_level.value} logging"
        )
    
    def _get_effective_config(self, task: DownloadTask) -> HttpFtpConfig:
        """
        Get the effective configuration for a download task.
        
        Args:
            task: Download task
            
        Returns:
            Effective HttpFtpConfig for the task
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # If task has specific config, use configuration manager to map it
            if hasattr(task, 'config') and task.config:
                return self._config_manager.create_engine_config(
                    task.config,
                    self._global_config,
                    validate=True
                )
            
            # Otherwise use base config
            return self._base_config
            
        except Exception as e:
            logger.error(f"Failed to get effective configuration for task {task.id}: {e}")
            raise ConfigurationError(f"Configuration error: {e}") from e

    def supports_protocol(self, url: str) -> bool:
        """
        Check if this engine supports the given URL protocol.

        Args:
            url: URL to check

        Returns:
            True if supported, False otherwise
        """
        try:
            parsed = urlparse(url)
            supported_schemes = {"http", "https", "ftp", "ftps", "sftp"}
            return parsed.scheme.lower() in supported_schemes
        except Exception:
            return False

    async def download(self, task: DownloadTask) -> AsyncIterator[ProgressInfo]:
        """
        Start downloading a task and yield progress updates.

        Args:
            task: Download task to execute

        Yields:
            Progress information updates

        Raises:
            DownloadError: If download fails
            NetworkError: If network issues occur
            ValidationError: If task validation fails
        """
        # Validate task before starting
        self._validate_task(task)

        # Register task start
        self._register_task_start(task.id)

        # Create download lock for this task
        if task.id not in self._download_locks:
            self._download_locks[task.id] = asyncio.Lock()

        async with self._download_locks[task.id]:
            try:
                # Get effective configuration for this task
                effective_config = self._get_effective_config(task)
                
                # Store effective config in download context for later use
                if task.id not in self._active_downloads:
                    self._active_downloads[task.id] = {}
                self._active_downloads[task.id]["effective_config"] = effective_config
                
                # Initialize download context
                await self._initialize_download(task)

                # Stream progress updates
                async for progress in self._perform_download(task):
                    # Update internal progress tracking
                    self._update_progress(task.id, progress)
                    yield progress

            except Exception as e:
                logger.error(f"Download failed for task {task.id}: {e}")

                # Create error progress info
                error_progress = ProgressInfo(
                    downloaded_bytes=self._task_progress.get(
                        task.id, ProgressInfo()
                    ).downloaded_bytes,
                    total_bytes=task.file_size,
                    download_speed=0.0,
                    status=TaskStatus.FAILED,
                    error_message=str(e),
                )

                self._update_progress(task.id, error_progress)
                yield error_progress

                # Re-raise the exception
                if isinstance(e, (DownloadError, NetworkError, ValidationError)):
                    raise
                else:
                    raise DownloadError(str(e), task.id) from e

            finally:
                # Cleanup download context
                await self._cleanup_download(task.id)

    async def _initialize_download(self, task: DownloadTask) -> None:
        """
        Initialize download context and detect file information.

        Args:
            task: Download task to initialize

        Raises:
            NetworkError: If unable to get file information
            ValidationError: If file information is invalid
        """
        logger.info(f"Initializing download for task {task.id}")

        try:
            # Get effective configuration for this task
            effective_config = self._active_downloads[task.id]["effective_config"]
            
            # Get file information from server
            file_info = await self._get_file_info(task.url, effective_config)

            # Update task with file information
            if not task.file_size and file_info.get("content_length"):
                task.file_size = file_info["content_length"]

            if not task.filename:
                task.filename = file_info.get(
                    "filename"
                ) or self._extract_filename_from_url(task.url)

            if not task.mime_type and file_info.get("content_type"):
                task.mime_type = file_info["content_type"]

            # Determine if server supports range requests
            supports_ranges = file_info.get("accept_ranges", False)

            # Calculate segments for multi-connection download
            segments = await self._calculate_segments(task, supports_ranges, effective_config)

            # Create temporary directory for segments
            temp_dir = Path(task.destination).parent / f".{task.id}_temp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Initialize segment merger
            segment_merger = SegmentMerger(task.destination, temp_dir)

            # Try to load resume data
            resume_segments = segment_merger.load_resume_data(task.id)
            if resume_segments:
                logger.info(f"Loaded resume data for {len(resume_segments)} segments")
                segments = resume_segments
            else:
                # Save initial segment data for resume capability
                segment_merger.save_resume_data(task.id, segments)

            # Store download context
            self._active_downloads[task.id] = {
                "task": task,
                "segments": segments,
                "segment_merger": segment_merger,
                "temp_dir": temp_dir,
                "supports_ranges": supports_ranges,
                "file_info": file_info,
                "start_time": time.time(),
            }

            logger.info(
                f"Initialized download: {len(segments)} segments, "
                f"ranges_supported={supports_ranges}, file_size={task.file_size}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize download for task {task.id}: {e}")
            raise NetworkError(f"Download initialization failed: {e}", task.id) from e

    async def _get_file_info(self, url: str, config: HttpFtpConfig) -> dict[str, Any]:
        """
        Get file information from server using HEAD request.

        Args:
            url: URL to check

        Returns:
            Dictionary with file information

        Raises:
            NetworkError: If unable to get file information
        """
        logger.debug(f"Getting file info for URL: {url}")

        curl_handle = None
        try:
            curl_handle = pycurl.Curl()

            # Configure for HEAD request
            curl_handle.setopt(pycurl.URL, url.encode("utf-8"))
            curl_handle.setopt(pycurl.NOBODY, 1)  # HEAD request
            curl_handle.setopt(pycurl.FOLLOWLOCATION, 1)
            curl_handle.setopt(pycurl.MAXREDIRS, config.max_redirects)
            # Automatically referer on redirects
            curl_handle.setopt(pycurl.AUTOREFERER, 1)
            curl_handle.setopt(pycurl.TIMEOUT, config.connect_timeout)
            curl_handle.setopt(pycurl.USERAGENT, config.user_agent.encode("utf-8"))

            # SSL/TLS settings
            self._setup_curl_ssl_options(curl_handle, config)

            # Authentication
            if config.auth.method.value != "none":
                self._setup_curl_auth(curl_handle, config)

            # Custom headers (including bearer token if applicable)
            self._setup_curl_headers(curl_handle, config)

            # Cookies
            self._setup_curl_cookies(curl_handle, config)

            # Proxy settings
            self._setup_curl_proxy(curl_handle, config)

            # Perform HEAD request
            await asyncio.get_event_loop().run_in_executor(None, curl_handle.perform)

            # Get response information
            response_code = curl_handle.getinfo(pycurl.RESPONSE_CODE)
            content_length = curl_handle.getinfo(pycurl.CONTENT_LENGTH_DOWNLOAD)
            content_type = curl_handle.getinfo(pycurl.CONTENT_TYPE)
            effective_url = curl_handle.getinfo(pycurl.EFFECTIVE_URL)

            if response_code not in (200, 206):
                error_msg = self._get_http_error_message(response_code)
                raise NetworkError(error_msg)

            # Check for range support
            accept_ranges = False
            # Note: pycurl doesn't provide direct access to response headers in this simple setup
            # In a full implementation, we'd use HEADERFUNCTION to capture headers

            # Extract filename from effective URL or Content-Disposition header
            filename = self._extract_filename_from_url(effective_url)

            file_info = {
                "content_length": int(content_length) if content_length > 0 else None,
                "content_type": content_type.decode("utf-8") if content_type else None,
                "filename": filename,
                "accept_ranges": accept_ranges,  # Would be determined from headers
                "effective_url": effective_url.decode("utf-8")
                if effective_url
                else url,
            }

            logger.debug(f"File info: {file_info}")
            return file_info

        except pycurl.error as e:
            raise NetworkError(f"Failed to get file info: {e}") from e
        except Exception as e:
            raise NetworkError(f"Unexpected error getting file info: {e}") from e
        finally:
            if curl_handle:
                curl_handle.close()

    def _setup_curl_auth(self, curl_handle: pycurl.Curl, config: HttpFtpConfig) -> None:
        """Setup authentication for curl handle."""
        auth = config.auth

        if auth.method.value == "basic":
            curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_BASIC)
        elif auth.method.value == "digest":
            curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_DIGEST)
        elif auth.method.value == "bearer":
            # Bearer token authentication via Authorization header
            token = auth.get_token()
            if token:
                headers = config.custom_headers.copy()
                headers["Authorization"] = f"Bearer {token}"
                header_list = [f"{k}: {v}" for k, v in headers.items()]
                curl_handle.setopt(pycurl.HTTPHEADER, header_list)
        elif auth.method.value == "ntlm":
            curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_NTLM)
        elif auth.method.value == "negotiate":
            curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_GSSNEGOTIATE)
        elif auth.method.value == "auto":
            curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_ANY)

        # Use secure credential retrieval
        username = auth.get_username()
        password = auth.get_password()
        
        if username and password:
            curl_handle.setopt(pycurl.USERPWD, f"{username}:{password}")

    def _setup_curl_cookies(self, curl_handle: pycurl.Curl, config: HttpFtpConfig) -> None:
        """Setup cookies for curl handle."""
        if config.cookies:
            # Convert cookies dict to cookie string format
            cookie_string = "; ".join(
                [f"{k}={v}" for k, v in config.cookies.items()]
            )
            curl_handle.setopt(pycurl.COOKIE, cookie_string.encode("utf-8"))

    def _setup_curl_proxy(self, curl_handle: pycurl.Curl, config: HttpFtpConfig) -> None:
        """Setup proxy configuration for curl handle."""
        proxy = config.proxy

        if not proxy.enabled or not proxy.host:
            return

        # Set proxy URL
        proxy_url = proxy.proxy_url
        curl_handle.setopt(pycurl.PROXY, proxy_url.encode("utf-8"))

        # Set proxy type
        proxy_type_map = {
            "http": pycurl.PROXYTYPE_HTTP,
            "https": pycurl.PROXYTYPE_HTTP,  # HTTPS proxy uses HTTP CONNECT
            "socks4": pycurl.PROXYTYPE_SOCKS4,
            "socks4a": pycurl.PROXYTYPE_SOCKS4A,
            "socks5": pycurl.PROXYTYPE_SOCKS5,
            "socks5h": pycurl.PROXYTYPE_SOCKS5_HOSTNAME,
        }

        if proxy.proxy_type.value in proxy_type_map:
            curl_handle.setopt(pycurl.PROXYTYPE, proxy_type_map[proxy.proxy_type.value])

        # Set proxy authentication if provided
        if proxy.username and proxy.password:
            curl_handle.setopt(
                pycurl.PROXYUSERPWD, f"{proxy.username}:{proxy.password}"
            )

    def _setup_curl_headers(self, curl_handle: pycurl.Curl, config: HttpFtpConfig) -> None:
        """Setup custom headers for curl handle."""
        headers = []

        # Add custom headers
        if config.custom_headers:
            headers.extend([f"{k}: {v}" for k, v in config.custom_headers.items()])

        # Add bearer token if using bearer authentication
        if (
            config.auth.method.value == "bearer"
            and "Authorization" not in config.custom_headers
        ):
            token = config.auth.get_token()
            if token:
                headers.append(f"Authorization: Bearer {token}")

        if headers:
            curl_handle.setopt(pycurl.HTTPHEADER, headers)

    def _setup_curl_ssl_options(self, curl_handle: pycurl.Curl, config: HttpFtpConfig) -> None:
        """Setup SSL/TLS options for curl handle."""
        # Basic SSL verification
        if config.verify_ssl and not config.ssl_development_mode:
            curl_handle.setopt(pycurl.SSL_VERIFYPEER, 1)
            curl_handle.setopt(pycurl.SSL_VERIFYHOST, 2)
        else:
            curl_handle.setopt(pycurl.SSL_VERIFYPEER, 0)
            curl_handle.setopt(pycurl.SSL_VERIFYHOST, 0)
            
            # Log warning for development mode
            if config.ssl_development_mode:
                logger.warning("SSL verification disabled for development mode - NOT SECURE")

        # SSL/TLS version specification
        if config.ssl_version:
            ssl_version_map = {
                "TLSv1": pycurl.SSLVERSION_TLSv1,
                "TLSv1.0": pycurl.SSLVERSION_TLSv1_0,
                "TLSv1.1": pycurl.SSLVERSION_TLSv1_1,
                "TLSv1.2": pycurl.SSLVERSION_TLSv1_2,
                "TLSv1.3": pycurl.SSLVERSION_TLSv1_3,
                "SSLv2": pycurl.SSLVERSION_SSLv2,
                "SSLv3": pycurl.SSLVERSION_SSLv3,
            }
            if config.ssl_version in ssl_version_map:
                curl_handle.setopt(pycurl.SSLVERSION, ssl_version_map[config.ssl_version])

        # Custom CA certificate bundle
        if config.ca_cert_path and config.ca_cert_path.exists():
            curl_handle.setopt(pycurl.CAINFO, str(config.ca_cert_path))

        # CA certificate directory
        if config.ssl_ca_cert_dir and config.ssl_ca_cert_dir.exists():
            curl_handle.setopt(pycurl.CAPATH, str(config.ssl_ca_cert_dir))

        # Certificate Revocation List
        if config.ssl_crl_file and config.ssl_crl_file.exists():
            curl_handle.setopt(pycurl.CRLFILE, str(config.ssl_crl_file))

        # Client certificate authentication
        if config.client_cert_path and config.client_cert_path.exists():
            curl_handle.setopt(pycurl.SSLCERT, str(config.client_cert_path))
            curl_handle.setopt(pycurl.SSLCERTTYPE, config.ssl_cert_type.encode("utf-8"))

        if config.client_key_path and config.client_key_path.exists():
            curl_handle.setopt(pycurl.SSLKEY, str(config.client_key_path))
            curl_handle.setopt(pycurl.SSLKEYTYPE, config.ssl_key_type.encode("utf-8"))
            
            # Private key password
            if config.ssl_key_password:
                curl_handle.setopt(pycurl.KEYPASSWD, config.ssl_key_password.encode("utf-8"))

        # SSL cipher list
        if config.ssl_cipher_list:
            curl_handle.setopt(
                pycurl.SSL_CIPHER_LIST, config.ssl_cipher_list.encode("utf-8")
            )

        # Public key pinning
        if config.ssl_pinned_public_key:
            # Format the pinned key for curl
            pinned_key = config.ssl_pinned_public_key
            if not pinned_key.startswith("sha256//"):
                pinned_key = f"sha256//{pinned_key}"
            curl_handle.setopt(pycurl.PINNEDPUBLICKEY, pinned_key.encode("utf-8"))

        # OCSP stapling verification
        if config.ssl_verify_status:
            curl_handle.setopt(pycurl.SSL_VERIFYSTATUS, 1)

        # SSL session ID caching
        if config.ssl_session_id_cache:
            curl_handle.setopt(pycurl.SSL_SESSIONID_CACHE, 1)
        else:
            curl_handle.setopt(pycurl.SSL_SESSIONID_CACHE, 0)

        # SSL False Start (performance optimization)
        if config.ssl_falsestart:
            curl_handle.setopt(pycurl.SSL_FALSESTART, 1)

        # ALPN (Application-Layer Protocol Negotiation)
        if config.ssl_enable_alpn:
            curl_handle.setopt(pycurl.SSL_ENABLE_ALPN, 1)
        else:
            curl_handle.setopt(pycurl.SSL_ENABLE_ALPN, 0)

        # NPN (Next Protocol Negotiation) - deprecated but still supported
        if config.ssl_enable_npn:
            curl_handle.setopt(pycurl.SSL_ENABLE_NPN, 1)
        else:
            curl_handle.setopt(pycurl.SSL_ENABLE_NPN, 0)

        # SSL debug level
        if config.ssl_debug_level > 0:
            curl_handle.setopt(pycurl.VERBOSE, 1)
            # Note: More detailed SSL debugging would require custom debug callback

        # Additional SSL options
        for ssl_option in config.ssl_options:
            # Parse and apply additional SSL options
            # Format: "OPTION_NAME:value" or just "OPTION_NAME" for boolean flags
            if ":" in ssl_option:
                option_name, option_value = ssl_option.split(":", 1)
                # This would require mapping option names to pycurl constants
                # For now, we'll log the option for debugging
                logger.debug(f"SSL option requested: {option_name}={option_value}")
            else:
                logger.debug(f"SSL flag requested: {ssl_option}")

        # Enable compression if configured
        if config.enable_compression:
            # Accept all supported encodings
            curl_handle.setopt(pycurl.ACCEPT_ENCODING, b"")

    def _extract_filename_from_url(self, url: str) -> str:
        """
        Extract filename from URL.

        Args:
            url: URL to extract filename from

        Returns:
            Extracted filename or default name
        """
        try:
            parsed = urlparse(url)
            path = parsed.path

            if path and "/" in path:
                filename = path.split("/")[-1]
                if filename and "." in filename:
                    return filename

            # Fallback to generic name
            return "download"

        except Exception:
            return "download"

    def _get_http_error_message(self, response_code: int) -> str:
        """
        Get a descriptive error message for HTTP status codes.

        Args:
            response_code: HTTP response code

        Returns:
            Descriptive error message
        """
        http_status_messages = {
            # 3xx Redirection
            300: "Multiple Choices",
            301: "Moved Permanently",
            302: "Found",
            303: "See Other",
            304: "Not Modified",
            305: "Use Proxy",
            307: "Temporary Redirect",
            308: "Permanent Redirect",
            # 4xx Client Error
            400: "Bad Request",
            401: "Unauthorized - Authentication required",
            402: "Payment Required",
            403: "Forbidden - Access denied",
            404: "Not Found - File does not exist",
            405: "Method Not Allowed",
            406: "Not Acceptable",
            407: "Proxy Authentication Required",
            408: "Request Timeout",
            409: "Conflict",
            410: "Gone - File no longer available",
            411: "Length Required",
            412: "Precondition Failed",
            413: "Payload Too Large",
            414: "URI Too Long",
            415: "Unsupported Media Type",
            416: "Range Not Satisfiable",
            417: "Expectation Failed",
            418: "I'm a teapot",
            421: "Misdirected Request",
            422: "Unprocessable Entity",
            423: "Locked",
            424: "Failed Dependency",
            425: "Too Early",
            426: "Upgrade Required",
            428: "Precondition Required",
            429: "Too Many Requests - Rate limited",
            431: "Request Header Fields Too Large",
            451: "Unavailable For Legal Reasons",
            # 5xx Server Error
            500: "Internal Server Error",
            501: "Not Implemented",
            502: "Bad Gateway",
            503: "Service Unavailable - Server temporarily overloaded",
            504: "Gateway Timeout",
            505: "HTTP Version Not Supported",
            506: "Variant Also Negotiates",
            507: "Insufficient Storage",
            508: "Loop Detected",
            510: "Not Extended",
            511: "Network Authentication Required",
        }

        status_message = http_status_messages.get(response_code, "Unknown Error")
        return f"HTTP {response_code}: {status_message}"

    async def _calculate_segments(
        self, task: DownloadTask, supports_ranges: bool, config: HttpFtpConfig
    ) -> list[DownloadSegment]:
        """
        Calculate download segments for multi-connection downloads.

        Args:
            task: Download task
            supports_ranges: Whether server supports range requests

        Returns:
            List of download segments
        """
        segments = []

        # If server doesn't support ranges or file size is unknown, use single segment
        if not supports_ranges or not task.file_size:
            logger.info(
                "Using single segment download (no range support or unknown file size)"
            )

            temp_file_path = (
                Path(task.destination).parent / f".{task.id}_temp" / "segment_0.tmp"
            )

            segment = DownloadSegment(
                task_id=task.id,
                url=task.url,
                start_byte=0,
                end_byte=task.file_size - 1 if task.file_size else 0,
                temp_file_path=temp_file_path,
            )
            segments.append(segment)
            return segments

        # Calculate optimal number of segments
        max_segments = min(config.max_connections, 8)  # Reasonable limit
        min_segment_size = config.segment_size

        # Don't create more segments than needed
        optimal_segments = min(
            max_segments, math.ceil(task.file_size / min_segment_size)
        )
        optimal_segments = max(1, optimal_segments)  # At least 1 segment

        # Calculate segment size
        segment_size = task.file_size // optimal_segments

        logger.info(
            f"Creating {optimal_segments} segments of ~{segment_size} bytes each"
        )

        # Create segments
        temp_dir = Path(task.destination).parent / f".{task.id}_temp"

        for i in range(optimal_segments):
            start_byte = i * segment_size

            # Last segment gets any remaining bytes
            if i == optimal_segments - 1:
                end_byte = task.file_size - 1
            else:
                end_byte = start_byte + segment_size - 1

            temp_file_path = temp_dir / f"segment_{i}.tmp"

            segment = DownloadSegment(
                task_id=task.id,
                url=task.url,
                start_byte=start_byte,
                end_byte=end_byte,
                temp_file_path=temp_file_path,
            )

            segments.append(segment)

        return segments

    async def _perform_download(
        self, task: DownloadTask
    ) -> AsyncIterator[ProgressInfo]:
        """
        Perform the actual download with progress streaming.

        Args:
            task: Download task to execute

        Yields:
            Progress information updates
        """
        download_context = self._active_downloads[task.id]
        segments = download_context["segments"]
        segment_merger = download_context["segment_merger"]
        effective_config = download_context["effective_config"]

        logger.info(f"Starting download with {len(segments)} segments")

        # Initialize progress tracking
        total_bytes = task.file_size or sum(seg.segment_size for seg in segments)

        # Create connection manager
        async with ConnectionManager(effective_config) as conn_manager:
            # Create workers for segments with logging configuration
            workers = await conn_manager.create_workers(
                segments, 
                log_level=self._log_level,
                log_dir=self._log_dir,
            )

            # Monitor download progress
            async for worker_progress_dict in conn_manager.monitor_workers(workers):
                # Aggregate progress from all workers
                aggregated_progress = self._aggregate_worker_progress(
                    worker_progress_dict, total_bytes, task.id
                )

                # Update segment merger with progress
                segment_merger.update_merge_info(len(segments), total_bytes)

                yield aggregated_progress

                # Check if all workers completed
                all_completed = all(
                    progress.status in (WorkerStatus.COMPLETED, WorkerStatus.FAILED)
                    for progress in worker_progress_dict.values()
                )

                if all_completed:
                    break

        # Merge completed segments
        logger.info("Download completed, merging segments")

        completed_segments = []
        failed_segments = []

        for segment in segments:
            if segment.status == SegmentStatus.COMPLETED:
                completed_segment = CompletedSegment(
                    segment=segment,
                    temp_file_path=segment.temp_file_path,
                )
                completed_segments.append(completed_segment)
            else:
                failed_segments.append(segment)

        if failed_segments:
            error_msg = f"{len(failed_segments)} segments failed to download"
            raise DownloadError(error_msg, task.id)

        # Merge all segments
        for completed_segment in completed_segments:
            merge_result = await segment_merger.merge_segment(completed_segment)
            if not merge_result["success"]:
                raise DownloadError(
                    f"Failed to merge segment: {merge_result.get('error')}", task.id
                )

        # Finalize download
        finalize_result = await segment_merger.finalize_download(completed_segments)
        if not finalize_result["success"]:
            raise DownloadError(
                f"Failed to finalize download: {finalize_result.get('error')}", task.id
            )

        # Final progress update
        final_progress = ProgressInfo(
            downloaded_bytes=total_bytes,
            total_bytes=total_bytes,
            download_speed=0.0,
            status=TaskStatus.COMPLETED,
        )

        yield final_progress

    def _aggregate_worker_progress(
        self,
        worker_progress_dict: dict[str, WorkerProgress],
        total_bytes: int,
        task_id: str,
    ) -> ProgressInfo:
        """
        Aggregate progress from multiple workers into a single ProgressInfo.

        Args:
            worker_progress_dict: Dictionary of worker progress
            total_bytes: Total bytes to download
            task_id: Task identifier

        Returns:
            Aggregated progress information
        """
        # Sum up downloaded bytes and speeds
        total_downloaded = sum(
            progress.downloaded_bytes for progress in worker_progress_dict.values()
        )
        total_speed = sum(
            progress.download_speed for progress in worker_progress_dict.values()
        )

        # Determine overall status
        worker_statuses = [
            progress.status for progress in worker_progress_dict.values()
        ]

        if any(status == WorkerStatus.FAILED for status in worker_statuses):
            overall_status = TaskStatus.FAILED
        elif any(status == WorkerStatus.DOWNLOADING for status in worker_statuses):
            overall_status = TaskStatus.DOWNLOADING
        elif all(status == WorkerStatus.COMPLETED for status in worker_statuses):
            overall_status = TaskStatus.COMPLETED
        elif any(status == WorkerStatus.PAUSED for status in worker_statuses):
            overall_status = TaskStatus.PAUSED
        else:
            overall_status = TaskStatus.PENDING

        # Calculate ETA
        eta = None
        if total_speed > 0 and total_bytes > total_downloaded:
            remaining_bytes = total_bytes - total_downloaded
            eta_seconds = remaining_bytes / total_speed
            from datetime import timedelta

            eta = timedelta(seconds=eta_seconds)

        # Collect error messages
        error_messages = [
            progress.error
            for progress in worker_progress_dict.values()
            if progress.error and progress.status == WorkerStatus.FAILED
        ]
        error_message = "; ".join(error_messages) if error_messages else None

        return ProgressInfo(
            downloaded_bytes=total_downloaded,
            total_bytes=total_bytes,
            download_speed=total_speed,
            eta=eta,
            status=overall_status,
            error_message=error_message,
        )

    async def pause(self, task_id: str) -> None:
        """
        Pause a download task.

        Args:
            task_id: ID of task to pause

        Raises:
            EngineError: If pause operation fails
        """
        if task_id not in self._active_downloads:
            logger.warning(f"Cannot pause task {task_id}: not active")
            return

        logger.info(f"Pausing download task {task_id}")

        download_context = self._active_downloads[task_id]
        segments = download_context["segments"]

        # Update segment status to paused
        for segment in segments:
            if segment.status == SegmentStatus.DOWNLOADING:
                segment.status = SegmentStatus.PAUSED

        # Save resume data
        segment_merger = download_context["segment_merger"]
        segment_merger.save_resume_data(task_id, segments)

        # Update progress
        if task_id in self._task_progress:
            progress = self._task_progress[task_id]
            progress.status = TaskStatus.PAUSED
            self._update_progress(task_id, progress)

    async def resume(self, task_id: str) -> None:
        """
        Resume a paused download task.

        Args:
            task_id: ID of task to resume

        Raises:
            EngineError: If resume operation fails
        """
        if task_id not in self._active_downloads:
            logger.warning(f"Cannot resume task {task_id}: not active")
            return

        logger.info(f"Resuming download task {task_id}")

        download_context = self._active_downloads[task_id]
        segments = download_context["segments"]

        # Update segment status to downloading for paused segments
        for segment in segments:
            if segment.status == SegmentStatus.PAUSED:
                segment.status = SegmentStatus.DOWNLOADING

        # Update progress
        if task_id in self._task_progress:
            progress = self._task_progress[task_id]
            progress.status = TaskStatus.DOWNLOADING
            self._update_progress(task_id, progress)

    async def cancel(self, task_id: str) -> None:
        """
        Cancel a download task.

        Args:
            task_id: ID of task to cancel

        Raises:
            EngineError: If cancel operation fails
        """
        logger.info(f"Cancelling download task {task_id}")

        if task_id in self._active_downloads:
            download_context = self._active_downloads[task_id]
            segments = download_context["segments"]

            # Update segment status to failed
            for segment in segments:
                if segment.status in (SegmentStatus.DOWNLOADING, SegmentStatus.PAUSED):
                    segment.status = SegmentStatus.FAILED
                    segment.error_message = "Cancelled by user"

        # Update progress
        if task_id in self._task_progress:
            progress = self._task_progress[task_id]
            progress.status = TaskStatus.CANCELLED
            progress.error_message = "Cancelled by user"
            self._update_progress(task_id, progress)

        # Cleanup will be handled by the download context manager

    async def list_ftp_directory(self, url: str) -> list[dict[str, Any]]:
        """
        List files in an FTP directory for batch downloads.

        Args:
            url: FTP directory URL to list

        Returns:
            List of file information dictionaries

        Raises:
            NetworkError: If directory listing fails
            ValidationError: If URL is not FTP
        """
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("ftp", "ftps"):
            raise ValidationError(f"URL is not FTP/FTPS: {url}")

        logger.info(f"Listing FTP directory: {url}")

        curl_handle = None
        try:
            curl_handle = pycurl.Curl()

            # Configure for directory listing
            curl_handle.setopt(pycurl.URL, url.encode("utf-8"))
            curl_handle.setopt(pycurl.FTPLISTONLY, 1)  # List filenames only
            curl_handle.setopt(pycurl.TIMEOUT, self.config.connect_timeout)

            # SSL/TLS settings
            self._setup_curl_ssl_options(curl_handle)

            # Authentication
            if self.config.auth.method.value != "none":
                self._setup_curl_auth(curl_handle)

            # FTP specific settings
            if self.config.ftp_use_epsv:
                curl_handle.setopt(pycurl.FTP_USE_EPSV, 1)
            else:
                curl_handle.setopt(pycurl.FTP_USE_EPSV, 0)

            if self.config.ftp_use_eprt:
                curl_handle.setopt(pycurl.FTP_USE_EPRT, 1)
            else:
                curl_handle.setopt(pycurl.FTP_USE_EPRT, 0)

            # Proxy settings
            self._setup_curl_proxy(curl_handle)

            # Capture directory listing output
            listing_data = []

            def write_callback(data: bytes) -> int:
                listing_data.append(data.decode("utf-8", errors="ignore"))
                return len(data)

            curl_handle.setopt(pycurl.WRITEFUNCTION, write_callback)

            # Perform directory listing
            await asyncio.get_event_loop().run_in_executor(None, curl_handle.perform)

            # Get response information
            response_code = curl_handle.getinfo(pycurl.RESPONSE_CODE)

            if response_code not in (200, 226, 250):  # FTP success codes
                error_msg = f"FTP directory listing failed with code {response_code}"
                raise NetworkError(error_msg)

            # Parse directory listing
            files = []
            listing_text = "".join(listing_data).strip()

            if listing_text:
                for line in listing_text.split("\n"):
                    line = line.strip()
                    if line and not line.startswith(
                        "."
                    ):  # Skip hidden files and empty lines
                        # Create file URL by joining with directory URL
                        file_url = urljoin(url.rstrip("/") + "/", line)
                        files.append(
                            {
                                "name": line,
                                "url": file_url,
                                "type": "file",  # Assume file for now
                                "size": None,  # Size not available in simple listing
                            }
                        )

            logger.info(f"Found {len(files)} files in FTP directory")
            return files

        except pycurl.error as e:
            raise NetworkError(f"Failed to list FTP directory: {e}") from e
        except Exception as e:
            raise NetworkError(f"Unexpected error listing FTP directory: {e}") from e
        finally:
            if curl_handle:
                curl_handle.close()

    async def list_ftp_directory_detailed(self, url: str) -> list[dict[str, Any]]:
        """
        List files in an FTP directory with detailed information.

        Args:
            url: FTP directory URL to list

        Returns:
            List of detailed file information dictionaries

        Raises:
            NetworkError: If directory listing fails
            ValidationError: If URL is not FTP
        """
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("ftp", "ftps"):
            raise ValidationError(f"URL is not FTP/FTPS: {url}")

        logger.info(f"Listing FTP directory with details: {url}")

        curl_handle = None
        try:
            curl_handle = pycurl.Curl()

            # Configure for detailed directory listing
            curl_handle.setopt(pycurl.URL, url.encode("utf-8"))
            # Don't use FTPLISTONLY for detailed listing
            curl_handle.setopt(pycurl.TIMEOUT, self.config.connect_timeout)

            # SSL/TLS settings
            self._setup_curl_ssl_options(curl_handle)

            # Authentication
            if self.config.auth.method.value != "none":
                self._setup_curl_auth(curl_handle)

            # FTP specific settings
            if self.config.ftp_use_epsv:
                curl_handle.setopt(pycurl.FTP_USE_EPSV, 1)
            else:
                curl_handle.setopt(pycurl.FTP_USE_EPSV, 0)

            if self.config.ftp_use_eprt:
                curl_handle.setopt(pycurl.FTP_USE_EPRT, 1)
            else:
                curl_handle.setopt(pycurl.FTP_USE_EPRT, 0)

            # Proxy settings
            self._setup_curl_proxy(curl_handle)

            # Capture directory listing output
            listing_data = []

            def write_callback(data: bytes) -> int:
                listing_data.append(data.decode("utf-8", errors="ignore"))
                return len(data)

            curl_handle.setopt(pycurl.WRITEFUNCTION, write_callback)

            # Perform directory listing
            await asyncio.get_event_loop().run_in_executor(None, curl_handle.perform)

            # Get response information
            response_code = curl_handle.getinfo(pycurl.RESPONSE_CODE)

            if response_code not in (200, 226, 250):  # FTP success codes
                error_msg = f"FTP directory listing failed with code {response_code}"
                raise NetworkError(error_msg)

            # Parse detailed directory listing
            files = []
            listing_text = "".join(listing_data).strip()

            if listing_text:
                files = self._parse_ftp_listing(listing_text, url)

            logger.info(f"Found {len(files)} items in FTP directory")
            return files

        except pycurl.error as e:
            raise NetworkError(f"Failed to list FTP directory: {e}") from e
        except Exception as e:
            raise NetworkError(f"Unexpected error listing FTP directory: {e}") from e
        finally:
            if curl_handle:
                curl_handle.close()

    def _parse_ftp_listing(
        self, listing_text: str, base_url: str
    ) -> list[dict[str, Any]]:
        """
        Parse FTP directory listing text into structured data.

        Args:
            listing_text: Raw FTP listing text
            base_url: Base URL for constructing file URLs

        Returns:
            List of parsed file information
        """
        files = []

        for line in listing_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("total"):
                continue

            # Try to parse Unix-style listing (most common)
            # Format: drwxrwxrwx 1 owner group size date time name
            unix_pattern = r"^([d-])([rwx-]{9})\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(\S+\s+\S+\s+\S+)\s+(.+)$"
            match = re.match(unix_pattern, line)

            if match:
                file_type = "directory" if match.group(1) == "d" else "file"
                permissions = match.group(2)
                size = int(match.group(3)) if match.group(3).isdigit() else None
                date_str = match.group(4)
                name = match.group(5)

                # Skip current and parent directory entries
                if name in (".", ".."):
                    continue

                # Create file URL
                file_url = urljoin(base_url.rstrip("/") + "/", name)

                files.append(
                    {
                        "name": name,
                        "url": file_url,
                        "type": file_type,
                        "size": size,
                        "permissions": permissions,
                        "date": date_str,
                    }
                )
            else:
                # Try DOS-style listing
                # Format: MM-DD-YY HH:MM[AP]M <DIR> name
                # Format: MM-DD-YY HH:MM[AP]M size name
                dos_pattern = r"^(\d{2}-\d{2}-\d{2})\s+(\d{1,2}:\d{2}[AP]M)\s+(?:(<DIR>)|(\d+))\s+(.+)$"
                dos_match = re.match(dos_pattern, line)

                if dos_match:
                    date = dos_match.group(1)
                    time = dos_match.group(2)
                    is_dir = dos_match.group(3) == "<DIR>"
                    size = int(dos_match.group(4)) if dos_match.group(4) else None
                    name = dos_match.group(5)

                    file_type = "directory" if is_dir else "file"
                    file_url = urljoin(base_url.rstrip("/") + "/", name)

                    files.append(
                        {
                            "name": name,
                            "url": file_url,
                            "type": file_type,
                            "size": size,
                            "date": f"{date} {time}",
                        }
                    )
                # Fallback: treat as simple filename
                elif line and not line.startswith("."):
                    file_url = urljoin(base_url.rstrip("/") + "/", line)
                    files.append(
                        {
                            "name": line,
                            "url": file_url,
                            "type": "file",
                            "size": None,
                        }
                    )

        return files

    async def create_batch_download_tasks(
        self, directory_url: str, file_filter: str | None = None
    ) -> list[DownloadTask]:
        """
        Create download tasks for all files in an FTP directory.

        Args:
            directory_url: FTP directory URL
            file_filter: Optional regex pattern to filter files

        Returns:
            List of download tasks for files in the directory

        Raises:
            NetworkError: If directory listing fails
            ValidationError: If URL is invalid
        """
        logger.info(f"Creating batch download tasks for directory: {directory_url}")

        # List directory contents
        files = await self.list_ftp_directory_detailed(directory_url)

        # Filter files only (not directories)
        file_items = [f for f in files if f["type"] == "file"]

        # Apply file filter if provided
        if file_filter:
            try:
                pattern = re.compile(file_filter, re.IGNORECASE)
                file_items = [f for f in file_items if pattern.search(f["name"])]
            except re.error as e:
                raise ValidationError(f"Invalid file filter regex: {e}")

        # Create download tasks
        tasks = []
        for file_info in file_items:
            from cuid import cuid

            task = DownloadTask(
                id=cuid(),
                url=file_info["url"],
                filename=file_info["name"],
                file_size=file_info.get("size"),
                destination=Path.cwd()
                / file_info["name"],  # Default to current directory
                status=TaskStatus.PENDING,
            )
            tasks.append(task)

        logger.info(f"Created {len(tasks)} download tasks from directory listing")
        return tasks

    async def is_ftp_directory(self, url: str) -> bool:
        """
        Check if an FTP URL points to a directory.

        Args:
            url: FTP URL to check

        Returns:
            True if URL is a directory, False if it's a file

        Raises:
            NetworkError: If unable to determine URL type
            ValidationError: If URL is not FTP
        """
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("ftp", "ftps"):
            raise ValidationError(f"URL is not FTP/FTPS: {url}")

        logger.debug(f"Checking if FTP URL is directory: {url}")

        curl_handle = None
        try:
            curl_handle = pycurl.Curl()

            # Configure for HEAD-like request to check if it's a directory
            curl_handle.setopt(pycurl.URL, url.encode("utf-8"))
            curl_handle.setopt(pycurl.NOBODY, 1)  # Don't download body
            curl_handle.setopt(pycurl.TIMEOUT, self.config.connect_timeout)

            # SSL/TLS settings
            self._setup_curl_ssl_options(curl_handle)

            # Authentication
            if self.config.auth.method.value != "none":
                self._setup_curl_auth(curl_handle)

            # FTP specific settings
            if self.config.ftp_use_epsv:
                curl_handle.setopt(pycurl.FTP_USE_EPSV, 1)
            else:
                curl_handle.setopt(pycurl.FTP_USE_EPSV, 0)

            # Proxy settings
            self._setup_curl_proxy(curl_handle)

            # Perform request
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, curl_handle.perform
                )

                # If we get here without error, it's likely a file
                response_code = curl_handle.getinfo(pycurl.RESPONSE_CODE)

                if response_code in (200, 213, 226):  # File exists
                    return False
                elif response_code in (550,):  # Not a file, might be directory
                    return True
                else:
                    # Try as directory by appending / and doing a list
                    return await self._try_directory_listing(url)

            except pycurl.error as e:
                # If HEAD fails, try directory listing
                if e.args[0] in (
                    pycurl.E_FTP_WEIRD_SERVER_REPLY,
                    pycurl.E_REMOTE_FILE_NOT_FOUND,
                ):
                    return await self._try_directory_listing(url)
                else:
                    raise NetworkError(f"Failed to check FTP URL type: {e}") from e

        except Exception as e:
            if not isinstance(e, (NetworkError, ValidationError)):
                raise NetworkError(
                    f"Unexpected error checking FTP URL type: {e}"
                ) from e
            raise
        finally:
            if curl_handle:
                curl_handle.close()

    async def _try_directory_listing(self, url: str) -> bool:
        """
        Try to perform a directory listing to determine if URL is a directory.

        Args:
            url: URL to test

        Returns:
            True if directory listing succeeds, False otherwise
        """
        try:
            # Ensure URL ends with / for directory listing
            test_url = url.rstrip("/") + "/"

            # Try a simple directory listing
            files = await self.list_ftp_directory(test_url)

            # If we got results, it's a directory
            return True

        except Exception:
            # If directory listing fails, assume it's a file
            return False

    def get_progress(self, task_id: str) -> ProgressInfo:
        """
        Get current progress for a task.

        Args:
            task_id: Task ID

        Returns:
            Current progress information

        Raises:
            KeyError: If task ID is not found
        """
        if task_id not in self._task_progress:
            raise KeyError(f"Task {task_id} not found")

        return self._task_progress[task_id]

    async def _cleanup_download(self, task_id: str) -> None:
        """
        Clean up download context and resources.

        Args:
            task_id: Task ID to clean up
        """
        logger.debug(f"Cleaning up download context for task {task_id}")

        # Remove from active downloads
        download_context = self._active_downloads.pop(task_id, None)

        if download_context:
            # Clean up temporary files if download was cancelled or failed
            progress = self._task_progress.get(task_id)
            if progress and progress.status in (
                TaskStatus.CANCELLED,
                TaskStatus.FAILED,
            ):
                try:
                    segment_merger = download_context["segment_merger"]
                    segments = download_context["segments"]
                    segment_merger.cleanup_temp_files(segments)
                except Exception as e:
                    logger.warning(
                        f"Error cleaning up temp files for task {task_id}: {e}"
                    )

        # Remove download lock
        self._download_locks.pop(task_id, None)

        # Clean up base class tracking
        self._unregister_task(task_id)
    def set_log_level(self, log_level: LogLevel) -> None:
        """
        Set logging level for the engine and all active downloads.
        
        Args:
            log_level: New logging level
        """
        self._log_level = log_level
        
        # Update log level for all active downloads
        for task_id, download_context in self._active_downloads.items():
            connection_manager = download_context.get("connection_manager")
            if connection_manager:
                connection_manager.set_worker_log_level(log_level)
        
        logger.info(f"Updated HttpFtpEngine log level to {log_level.value}")
    
    def set_log_directory(self, log_dir: Path) -> None:
        """
        Set directory for debug logs.
        
        Args:
            log_dir: Directory for debug logs
        """
        self._log_dir = log_dir
        logger.info(f"Updated HttpFtpEngine log directory to {log_dir}")
    
    def get_debug_summary(self, task_id: Optional[str] = None) -> dict[str, Any]:
        """
        Get comprehensive debug summary for downloads.
        
        Args:
            task_id: Optional specific task ID (None for all active downloads)
            
        Returns:
            Debug summary information
        """
        summary = {
            "engine_info": {
                "engine_type": "HttpFtpEngine",
                "log_level": self._log_level.value,
                "log_directory": str(self._log_dir) if self._log_dir else None,
                "active_downloads": len(self._active_downloads),
                "supported_protocols": ["http", "https", "ftp", "ftps", "sftp"],
            },
            "downloads": {},
        }
        
        # Get debug info for specific task or all tasks
        target_tasks = {}
        if task_id and task_id in self._active_downloads:
            target_tasks[task_id] = self._active_downloads[task_id]
        else:
            target_tasks = self._active_downloads
        
        for tid, download_context in target_tasks.items():
            connection_manager = download_context.get("connection_manager")
            if connection_manager:
                summary["downloads"][tid] = connection_manager.get_debug_summary()
            else:
                summary["downloads"][tid] = {"status": "no_connection_manager"}
        
        return summary
    
    def get_performance_metrics(self, task_id: Optional[str] = None) -> dict[str, Any]:
        """
        Get performance metrics for downloads.
        
        Args:
            task_id: Optional specific task ID (None for all active downloads)
            
        Returns:
            Performance metrics information
        """
        metrics = {
            "engine_metrics": {
                "total_active_downloads": len(self._active_downloads),
                "engine_type": "HttpFtpEngine",
            },
            "downloads": {},
        }
        
        # Get metrics for specific task or all tasks
        target_tasks = {}
        if task_id and task_id in self._active_downloads:
            target_tasks[task_id] = self._active_downloads[task_id]
        else:
            target_tasks = self._active_downloads
        
        for tid, download_context in target_tasks.items():
            connection_manager = download_context.get("connection_manager")
            if connection_manager:
                metrics["downloads"][tid] = connection_manager.get_performance_metrics()
            else:
                metrics["downloads"][tid] = {"status": "no_connection_manager"}
        
        return metrics
    
    def generate_debug_commands(self, task_id: Optional[str] = None) -> dict[str, Any]:
        """
        Generate equivalent curl commands for debugging.
        
        Args:
            task_id: Optional specific task ID (None for all active downloads)
            
        Returns:
            Dictionary with curl commands for debugging
        """
        commands = {
            "downloads": {},
        }
        
        # Get commands for specific task or all tasks
        target_tasks = {}
        if task_id and task_id in self._active_downloads:
            target_tasks[task_id] = self._active_downloads[task_id]
        else:
            target_tasks = self._active_downloads
        
        for tid, download_context in target_tasks.items():
            connection_manager = download_context.get("connection_manager")
            if connection_manager:
                commands["downloads"][tid] = connection_manager.generate_debug_commands()
            else:
                commands["downloads"][tid] = {"error": "no_connection_manager"}
        
        return commands
    
    def diagnose_connection_issues(self, task_id: Optional[str] = None) -> dict[str, Any]:
        """
        Diagnose connection issues for downloads.
        
        Args:
            task_id: Optional specific task ID (None for all active downloads)
            
        Returns:
            Diagnosis information with recommendations
        """
        diagnosis = {
            "engine_diagnosis": {
                "engine_type": "HttpFtpEngine",
                "active_downloads": len(self._active_downloads),
                "overall_health": "healthy",
            },
            "downloads": {},
        }
        
        # Get diagnosis for specific task or all tasks
        target_tasks = {}
        if task_id and task_id in self._active_downloads:
            target_tasks[task_id] = self._active_downloads[task_id]
        else:
            target_tasks = self._active_downloads
        
        failed_downloads = 0
        
        for tid, download_context in target_tasks.items():
            connection_manager = download_context.get("connection_manager")
            if connection_manager:
                download_diagnosis = connection_manager.diagnose_connection_issues()
                diagnosis["downloads"][tid] = download_diagnosis
                
                # Check if download has issues
                if download_diagnosis.get("overall_health") not in ("healthy", "minor_issues"):
                    failed_downloads += 1
            else:
                diagnosis["downloads"][tid] = {"error": "no_connection_manager"}
                failed_downloads += 1
        
        # Determine overall engine health
        total_downloads = len(target_tasks)
        if total_downloads > 0:
            failure_rate = failed_downloads / total_downloads
            
            if failure_rate == 0:
                diagnosis["engine_diagnosis"]["overall_health"] = "healthy"
            elif failure_rate < 0.25:
                diagnosis["engine_diagnosis"]["overall_health"] = "minor_issues"
            elif failure_rate < 0.75:
                diagnosis["engine_diagnosis"]["overall_health"] = "degraded"
            else:
                diagnosis["engine_diagnosis"]["overall_health"] = "critical"
        
        return diagnosis
    
    def get_curl_version_info(self) -> dict[str, Any]:
        """
        Get curl version and feature information for debugging.
        
        Returns:
            Curl version and feature information
        """
        try:
            version_info = pycurl.version_info()
            
            # Extract version information safely
            curl_version = version_info[1] if len(version_info) > 1 else "unknown"
            libcurl_version = version_info[0] if len(version_info) > 0 else "unknown"
            ssl_version = version_info[2] if len(version_info) > 2 else "unknown"
            features_bitmask = version_info[3] if len(version_info) > 3 else 0
            host = version_info[4] if len(version_info) > 4 else "unknown"
            protocols = list(version_info[8]) if len(version_info) > 8 else []
            
            return {
                "curl_version": curl_version,
                "libcurl_version": libcurl_version,
                "ssl_version": ssl_version,
                "protocols": protocols,
                "features": {
                    "ipv6": bool(features_bitmask & pycurl.VERSION_IPV6) if hasattr(pycurl, 'VERSION_IPV6') else False,
                    "ssl": bool(features_bitmask & pycurl.VERSION_SSL) if hasattr(pycurl, 'VERSION_SSL') else False,
                    "libz": bool(features_bitmask & pycurl.VERSION_LIBZ) if hasattr(pycurl, 'VERSION_LIBZ') else False,
                    "ntlm": bool(features_bitmask & pycurl.VERSION_NTLM) if hasattr(pycurl, 'VERSION_NTLM') else False,
                    "gssnegotiate": bool(features_bitmask & pycurl.VERSION_GSSNEGOTIATE) if hasattr(pycurl, 'VERSION_GSSNEGOTIATE') else False,
                    "debug": bool(features_bitmask & pycurl.VERSION_DEBUG) if hasattr(pycurl, 'VERSION_DEBUG') else False,
                    "asynchdns": bool(features_bitmask & pycurl.VERSION_ASYNCHDNS) if hasattr(pycurl, 'VERSION_ASYNCHDNS') else False,
                    "spnego": bool(features_bitmask & pycurl.VERSION_SPNEGO) if hasattr(pycurl, 'VERSION_SPNEGO') else False,
                    "largefile": bool(features_bitmask & pycurl.VERSION_LARGEFILE) if hasattr(pycurl, 'VERSION_LARGEFILE') else False,
                    "idn": bool(features_bitmask & pycurl.VERSION_IDN) if hasattr(pycurl, 'VERSION_IDN') else False,
                    "sspi": bool(features_bitmask & pycurl.VERSION_SSPI) if hasattr(pycurl, 'VERSION_SSPI') else False,
                    "conv": bool(features_bitmask & pycurl.VERSION_CONV) if hasattr(pycurl, 'VERSION_CONV') else False,
                },
                "host": host,
                "raw_version_info": str(version_info),  # For debugging
            }
        except Exception as e:
            return {"error": f"Failed to get curl version info: {e}"}
    
    async def test_connection(self, url: str, config: Optional[HttpFtpConfig] = None) -> dict[str, Any]:
        """
        Test connection to a URL for debugging purposes.
        
        Args:
            url: URL to test
            config: Optional configuration to use
            
        Returns:
            Connection test results
        """
        test_config = config or self._base_config
        
        # Create a logger for this test
        test_logger = get_download_logger("connection_test", EngineType.HTTP)
        
        result = {
            "url": url,
            "timestamp": time.time(),
            "success": False,
            "response_code": None,
            "error": None,
            "timing": {},
            "connection_info": {},
            "diagnosis": {},
        }
        
        curl_handle = None
        try:
            curl_handle = pycurl.Curl()
            
            # Basic setup
            curl_handle.setopt(pycurl.URL, url.encode("utf-8"))
            curl_handle.setopt(pycurl.NOBODY, 1)  # HEAD request only
            curl_handle.setopt(pycurl.TIMEOUT, test_config.connect_timeout)
            curl_handle.setopt(pycurl.CONNECTTIMEOUT, test_config.connect_timeout)
            
            # Setup authentication, proxy, SSL, etc.
            self._setup_curl_auth(curl_handle, test_config)
            self._setup_curl_proxy(curl_handle, test_config)
            self._setup_curl_ssl_options(curl_handle, test_config)
            
            # Perform the test
            start_time = time.time()
            curl_handle.perform()
            end_time = time.time()
            
            # Extract results
            result["success"] = True
            result["response_code"] = curl_handle.getinfo(pycurl.RESPONSE_CODE)
            result["timing"] = {
                "total_time": curl_handle.getinfo(pycurl.TOTAL_TIME),
                "namelookup_time": curl_handle.getinfo(pycurl.NAMELOOKUP_TIME),
                "connect_time": curl_handle.getinfo(pycurl.CONNECT_TIME),
                "appconnect_time": curl_handle.getinfo(pycurl.APPCONNECT_TIME),
                "pretransfer_time": curl_handle.getinfo(pycurl.PRETRANSFER_TIME),
                "starttransfer_time": curl_handle.getinfo(pycurl.STARTTRANSFER_TIME),
                "test_duration": end_time - start_time,
            }
            
            # Helper function to safely decode curl info
            def safe_decode(value):
                if isinstance(value, bytes):
                    return value.decode("utf-8")
                return str(value) if value is not None else ""
            
            result["connection_info"] = {
                "effective_url": safe_decode(curl_handle.getinfo(pycurl.EFFECTIVE_URL)),
                "primary_ip": safe_decode(curl_handle.getinfo(pycurl.PRIMARY_IP)),
                "primary_port": curl_handle.getinfo(pycurl.PRIMARY_PORT),
                "local_ip": safe_decode(curl_handle.getinfo(pycurl.LOCAL_IP)),
                "local_port": curl_handle.getinfo(pycurl.LOCAL_PORT),
            }
            
            test_logger.info(f"Connection test successful for {url}")
            
        except pycurl.error as e:
            result["error"] = f"Curl error {e.args[0]}: {e.args[1]}"
            result["diagnosis"] = CurlDebugUtilities.diagnose_connection_error(e.args[0], url)
            test_logger.error(f"Connection test failed for {url}: {result['error']}")
            
        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
            test_logger.error(f"Connection test failed for {url}: {result['error']}")
            
        finally:
            if curl_handle:
                curl_handle.close()
        
        return result