"""HttpFtpEngine implementation for HTTP/HTTPS and FTP/SFTP downloads using pycurl."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import logging
import math
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

import pycurl

from ..storage.models import DownloadTask, ProgressInfo, TaskStatus
from .base import BaseDownloadEngine, DownloadError, NetworkError, ValidationError
from .connection_manager import ConnectionManager
from .pycurl_models import (
    DownloadSegment,
    HttpFtpConfig,
    SegmentStatus,
    WorkerProgress,
    WorkerStatus,
)
from .segment_merger import SegmentMerger

logger = logging.getLogger(__name__)


class HttpFtpEngine(BaseDownloadEngine):
    """Main engine implementing HTTP/HTTPS and FTP/SFTP downloads using pycurl."""

    def __init__(self, config: HttpFtpConfig | None = None) -> None:
        """
        Initialize HttpFtpEngine.

        Args:
            config: Configuration for HTTP/FTP downloads
        """
        super().__init__()
        self.config = config or HttpFtpConfig()

        # Active downloads tracking
        self._active_downloads: dict[str, dict[str, Any]] = {}
        self._download_locks: dict[str, asyncio.Lock] = {}

        logger.info(
            f"Initialized HttpFtpEngine with max {self.config.max_connections} connections"
        )

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
            # Get file information from server
            file_info = await self._get_file_info(task.url)

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
            segments = await self._calculate_segments(task, supports_ranges)

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

    async def _get_file_info(self, url: str) -> dict[str, Any]:
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
            curl_handle.setopt(pycurl.MAXREDIRS, self.config.max_redirects)
            # Automatically referer on redirects
            curl_handle.setopt(pycurl.AUTOREFERER, 1)
            curl_handle.setopt(pycurl.TIMEOUT, self.config.connect_timeout)
            curl_handle.setopt(pycurl.USERAGENT, self.config.user_agent.encode("utf-8"))

            # SSL/TLS settings
            self._setup_curl_ssl_options(curl_handle)

            # Authentication
            if self.config.auth.method.value != "none":
                self._setup_curl_auth(curl_handle)

            # Custom headers (including bearer token if applicable)
            self._setup_curl_headers(curl_handle)

            # Cookies
            self._setup_curl_cookies(curl_handle)

            # Proxy settings
            self._setup_curl_proxy(curl_handle)

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

    def _setup_curl_auth(self, curl_handle: pycurl.Curl) -> None:
        """Setup authentication for curl handle."""
        auth = self.config.auth

        if auth.method.value == "basic":
            curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_BASIC)
        elif auth.method.value == "digest":
            curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_DIGEST)
        elif auth.method.value == "bearer":
            # Bearer token authentication via Authorization header
            if auth.token:
                headers = self.config.custom_headers.copy()
                headers["Authorization"] = f"Bearer {auth.token}"
                header_list = [f"{k}: {v}" for k, v in headers.items()]
                curl_handle.setopt(pycurl.HTTPHEADER, header_list)
        elif auth.method.value == "ntlm":
            curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_NTLM)
        elif auth.method.value == "negotiate":
            curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_GSSNEGOTIATE)
        elif auth.method.value == "auto":
            curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_ANY)

        if auth.username and auth.password:
            curl_handle.setopt(pycurl.USERPWD, f"{auth.username}:{auth.password}")

    def _setup_curl_cookies(self, curl_handle: pycurl.Curl) -> None:
        """Setup cookies for curl handle."""
        if self.config.cookies:
            # Convert cookies dict to cookie string format
            cookie_string = "; ".join(
                [f"{k}={v}" for k, v in self.config.cookies.items()]
            )
            curl_handle.setopt(pycurl.COOKIE, cookie_string.encode("utf-8"))

    def _setup_curl_proxy(self, curl_handle: pycurl.Curl) -> None:
        """Setup proxy configuration for curl handle."""
        proxy = self.config.proxy

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

    def _setup_curl_headers(self, curl_handle: pycurl.Curl) -> None:
        """Setup custom headers for curl handle."""
        headers = []

        # Add custom headers
        if self.config.custom_headers:
            headers.extend([f"{k}: {v}" for k, v in self.config.custom_headers.items()])

        # Add bearer token if using bearer authentication
        if (
            self.config.auth.method.value == "bearer"
            and self.config.auth.token
            and "Authorization" not in self.config.custom_headers
        ):
            headers.append(f"Authorization: Bearer {self.config.auth.token}")

        if headers:
            curl_handle.setopt(pycurl.HTTPHEADER, headers)

    def _setup_curl_ssl_options(self, curl_handle: pycurl.Curl) -> None:
        """Setup SSL/TLS options for curl handle."""
        # Basic SSL verification
        if self.config.verify_ssl:
            curl_handle.setopt(pycurl.SSL_VERIFYPEER, 1)
            curl_handle.setopt(pycurl.SSL_VERIFYHOST, 2)
        else:
            curl_handle.setopt(pycurl.SSL_VERIFYPEER, 0)
            curl_handle.setopt(pycurl.SSL_VERIFYHOST, 0)

        # Custom CA certificate bundle
        if self.config.ca_cert_path and self.config.ca_cert_path.exists():
            curl_handle.setopt(pycurl.CAINFO, str(self.config.ca_cert_path))

        # Client certificate authentication
        if self.config.client_cert_path and self.config.client_cert_path.exists():
            curl_handle.setopt(pycurl.SSLCERT, str(self.config.client_cert_path))

        if self.config.client_key_path and self.config.client_key_path.exists():
            curl_handle.setopt(pycurl.SSLKEY, str(self.config.client_key_path))

        # SSL cipher list
        if self.config.ssl_cipher_list:
            curl_handle.setopt(
                pycurl.SSL_CIPHER_LIST, self.config.ssl_cipher_list.encode("utf-8")
            )

        # Enable compression if configured
        if self.config.enable_compression:
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
        self, task: DownloadTask, supports_ranges: bool
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
        max_segments = min(self.config.max_connections, 8)  # Reasonable limit
        min_segment_size = self.config.segment_size

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

        logger.info(f"Starting download with {len(segments)} segments")

        # Initialize progress tracking
        total_bytes = task.file_size or sum(seg.segment_size for seg in segments)

        # Create connection manager
        async with ConnectionManager(self.config) as conn_manager:
            # Create workers for segments
            workers = await conn_manager.create_workers(segments)

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
                from .pycurl_models import CompletedSegment

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
