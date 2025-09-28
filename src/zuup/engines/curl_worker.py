"""CurlWorker implementation for individual segment downloads using pycurl."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
import time
from typing import Any, Optional
from pathlib import Path

import pycurl

from .pycurl_models import (
    CurlError,
    DownloadSegment,
    HttpFtpConfig,
    SegmentResult,
    SegmentStatus,
    WorkerProgress,
    WorkerStatus,
)
from .pycurl_logging import CurlLogger, LogLevel, setup_curl_logging

logger = logging.getLogger(__name__)


class CurlWorker:
    """Individual curl worker handling a single segment download."""

    def __init__(
        self,
        segment: DownloadSegment,
        config: HttpFtpConfig,
        curl_share: pycurl.CurlShare | None = None,
        log_level: LogLevel = LogLevel.BASIC,
        log_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize CurlWorker.

        Args:
            segment: Download segment to handle
            config: Configuration for the worker
            curl_share: Shared curl handle for cookies, DNS cache, etc.
            log_level: Logging verbosity level
            log_dir: Optional directory for debug logs
        """
        self.segment = segment
        self.config = config
        self.curl_share = curl_share

        # Worker state
        self.worker_id = f"worker_{segment.id}"
        self.status = WorkerStatus.INITIALIZING
        self.curl_handle: pycurl.Curl | None = None
        self.temp_file: Any | None = None  # File handle
        self.downloaded_bytes = 0
        self.start_time = 0.0
        self.last_progress_time = 0.0
        self.current_speed = 0.0
        self.error_message: str | None = None

        # Control flags
        self._paused = False
        self._cancelled = False
        self._should_stop = False

        # Progress tracking
        self._progress_callback: Callable[[WorkerProgress], None] | None = None
        self._last_reported_bytes = 0
        self._speed_samples: list[tuple[float, int]] = []  # (timestamp, bytes)

        # Initialize comprehensive logging
        self.curl_logger = setup_curl_logging(
            task_id=segment.task_id,
            worker_id=self.worker_id,
            log_level=log_level,
            log_dir=log_dir,
        )

        logger.debug(
            f"Initialized CurlWorker {self.worker_id} for segment {segment.id} with log level {log_level.value}"
        )

    def set_progress_callback(self, callback: Callable[[WorkerProgress], None]) -> None:
        """Set callback function for progress updates."""
        self._progress_callback = callback

    def set_log_level(self, level: LogLevel) -> None:
        """Update logging level for this worker."""
        self.curl_logger.set_log_level(level)

    def get_debug_summary(self) -> dict[str, Any]:
        """Get debug summary for troubleshooting."""
        return self.curl_logger.get_debug_summary()

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics if available."""
        return self.curl_logger.metrics.to_dict()

    def generate_debug_curl_command(self) -> str:
        """Generate equivalent curl command for debugging."""
        from .pycurl_logging import CurlDebugUtilities
        return CurlDebugUtilities.generate_curl_command(
            self.segment.url, self.config, self.segment
        )

    async def download_segment(self) -> SegmentResult:
        """
        Download the assigned segment.

        Returns:
            Dictionary containing download result information

        Raises:
            CurlError: If download fails after all retries
        """
        logger.info(f"Starting download for segment {self.segment.id}")

        retry_count = 0
        max_retries = self.config.retry_attempts

        while retry_count <= max_retries:
            try:
                if self._cancelled:
                    return self._create_result(
                        success=False, error="Download cancelled"
                    )

                # Initialize curl handle for this attempt
                await self._initialize_curl()

                # Perform the download
                result = await self._perform_download()

                if result["success"]:
                    logger.info(f"Successfully downloaded segment {self.segment.id}")
                    return result

                # Handle failure
                error = result.get("error", "Unknown error")
                logger.warning(
                    f"Download attempt {retry_count + 1} failed for segment {self.segment.id}: {error}"
                )

                # Check if we should retry
                if retry_count < max_retries and not self._cancelled:
                    retry_count += 1
                    delay = self._calculate_retry_delay(retry_count)
                    
                    # Log retry attempt
                    self.curl_logger.log_retry_attempt(retry_count, delay, error)
                    
                    logger.info(
                        f"Retrying segment {self.segment.id} in {delay:.1f} seconds (attempt {retry_count + 1}/{max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                else:
                    # Max retries exceeded or cancelled
                    self.status = WorkerStatus.FAILED
                    self.error_message = error
                    return result

            except Exception as e:
                error_msg = (
                    f"Unexpected error in download attempt {retry_count + 1}: {e!s}"
                )
                logger.error(f"Segment {self.segment.id}: {error_msg}")

                if retry_count < max_retries and not self._cancelled:
                    retry_count += 1
                    delay = self._calculate_retry_delay(retry_count)
                    await asyncio.sleep(delay)
                else:
                    self.status = WorkerStatus.FAILED
                    self.error_message = error_msg
                    return self._create_result(success=False, error=error_msg)
            finally:
                # Clean up curl handle after each attempt
                self._cleanup_curl()

        # Should not reach here, but just in case
        return self._create_result(success=False, error="Max retries exceeded")

    async def _initialize_curl(self) -> None:
        """Initialize curl handle with configuration."""
        if self.curl_handle:
            self._cleanup_curl()

        self.curl_handle = pycurl.Curl()

        # Log connection attempt
        self.curl_logger.log_connection_attempt(self.segment.url, self.segment)

        # Basic URL setup
        self.curl_handle.setopt(pycurl.URL, self.segment.url.encode("utf-8"))

        # Detect protocol and setup protocol-specific options
        from urllib.parse import urlparse

        parsed_url = urlparse(self.segment.url)
        protocol = parsed_url.scheme.lower()

        if protocol in ("ftp", "ftps"):
            self._setup_ftp_options()
        elif protocol == "sftp":
            self._setup_sftp_options()
        else:
            # HTTP/HTTPS setup
            self._setup_http_options()

        # Set range request for segment (works for HTTP and FTP)
        if protocol in ("http", "https", "ftp", "ftps"):
            range_header = f"{self.segment.start_byte}-{self.segment.end_byte}"
            self.curl_handle.setopt(pycurl.RANGE, range_header.encode("utf-8"))

        # Timeout settings
        if self.config.timeout > 0:
            self.curl_handle.setopt(pycurl.TIMEOUT, self.config.timeout)
        self.curl_handle.setopt(pycurl.CONNECTTIMEOUT, self.config.connect_timeout)

        # Low speed settings for stalled connections
        self.curl_handle.setopt(pycurl.LOW_SPEED_LIMIT, self.config.low_speed_limit)
        self.curl_handle.setopt(pycurl.LOW_SPEED_TIME, self.config.low_speed_time)

        # User agent (for HTTP/HTTPS)
        if protocol in ("http", "https"):
            self.curl_handle.setopt(
                pycurl.USERAGENT, self.config.user_agent.encode("utf-8")
            )

        # SSL/TLS settings (only for protocols that use SSL/TLS)
        # IMPORTANT: Never call SSL setup for SFTP as it uses SSH, not SSL
        if protocol in ("https", "ftps"):
            self._setup_ssl_options()
        elif protocol == "sftp":
            # Ensure SSL is completely disabled for SFTP - this is redundant with
            # _setup_sftp_options but provides extra safety
            self.curl_handle.setopt(pycurl.SSL_VERIFYPEER, 0)
            self.curl_handle.setopt(pycurl.SSL_VERIFYHOST, 0)
            # Also ensure no CA bundle is used for SFTP
            self.curl_handle.setopt(pycurl.CAINFO, "")
            self.curl_handle.setopt(pycurl.CAPATH, "")

        # Authentication
        if self.config.auth.method.value != "none":
            self._setup_authentication()

        # Proxy settings
        if self.config.proxy.enabled:
            self._setup_proxy()

        # Share handle for cookies, DNS cache, SSL sessions
        # IMPORTANT: Do not use shared handle for SFTP as it may inherit SSL settings
        if self.curl_share and protocol != "sftp":
            self.curl_handle.setopt(pycurl.SHARE, self.curl_share)
        elif protocol == "sftp":
            logger.debug(f"Skipping curl share for SFTP connection in worker {self.worker_id}")

        # Performance settings
        self.curl_handle.setopt(pycurl.BUFFERSIZE, self.config.buffer_size)
        if self.config.tcp_nodelay:
            self.curl_handle.setopt(pycurl.TCP_NODELAY, 1)

        # Setup callbacks
        self.curl_handle.setopt(pycurl.WRITEFUNCTION, self._write_callback)
        self.curl_handle.setopt(pycurl.PROGRESSFUNCTION, self._progress_callback_curl)
        self.curl_handle.setopt(pycurl.NOPROGRESS, 0)  # Enable progress callback

        # Setup debug callback for detailed logging
        if self.curl_logger.log_level in (LogLevel.VERBOSE, LogLevel.DEBUG):
            debug_callback = self.curl_logger.create_debug_callback()
            self.curl_handle.setopt(pycurl.DEBUGFUNCTION, debug_callback)
            self.curl_handle.setopt(pycurl.VERBOSE, 1)

        # Open temp file for writing
        self._open_temp_file()

        # Initialize performance metrics
        self.curl_logger.metrics.start_time = time.time()

        logger.debug(f"Initialized curl handle for worker {self.worker_id}")

    def _setup_http_options(self) -> None:
        """Setup HTTP/HTTPS specific options."""
        # Follow redirects with enhanced configuration
        if self.config.follow_redirects:
            self.curl_handle.setopt(pycurl.FOLLOWLOCATION, 1)
            self.curl_handle.setopt(pycurl.MAXREDIRS, self.config.max_redirects)
            # Automatically referer on redirects
            self.curl_handle.setopt(pycurl.AUTOREFERER, 1)

        # Setup HTTP/HTTPS specific features
        self._setup_headers()
        self._setup_cookies()

    def _setup_ftp_options(self) -> None:
        """Setup FTP/FTPS specific options."""
        # FTP specific settings
        if self.config.ftp_use_epsv:
            self.curl_handle.setopt(pycurl.FTP_USE_EPSV, 1)
        else:
            self.curl_handle.setopt(pycurl.FTP_USE_EPSV, 0)

        if self.config.ftp_use_eprt:
            self.curl_handle.setopt(pycurl.FTP_USE_EPRT, 1)
        else:
            self.curl_handle.setopt(pycurl.FTP_USE_EPRT, 0)

        # Create missing directories if configured
        if self.config.ftp_create_missing_dirs:
            self.curl_handle.setopt(pycurl.FTP_CREATE_MISSING_DIRS, 1)

        # Skip PASV IP address for NAT/firewall issues (helps with NAT/firewall)
        if self.config.ftp_skip_pasv_ip:
            self.curl_handle.setopt(pycurl.FTP_SKIP_PASV_IP, 1)

        # Use PRET command for some FTP servers that require it
        if self.config.ftp_use_pret:
            self.curl_handle.setopt(pycurl.FTP_USE_PRET, 1)

        # Enable EPSV for better passive mode support
        if self.config.ftp_use_epsv:
            self.curl_handle.setopt(pycurl.FTP_USE_EPSV, 1)

        # Configure EPRT usage
        if self.config.ftp_use_eprt:
            self.curl_handle.setopt(pycurl.FTP_USE_EPRT, 1)

        # Enable FTP resume support
        if self.config.timeout > 0:
            self.curl_handle.setopt(pycurl.FTP_RESPONSE_TIMEOUT, self.config.timeout)

        # Set transfer mode to binary (important for file integrity)
        self.curl_handle.setopt(pycurl.TRANSFERTEXT, 0)

        # Set FTP file method for better compatibility
        self.curl_handle.setopt(pycurl.FTP_FILEMETHOD, pycurl.FTPMETHOD_SINGLECWD)

        logger.debug(f"Configured FTP options for worker {self.worker_id}")

    def _setup_sftp_options(self) -> None:
        """Setup SFTP specific options."""
        ssh_config = self.config.ssh

        # CRITICAL: SFTP uses SSH, not SSL/TLS - completely disable SSL
        self.curl_handle.setopt(pycurl.SSL_VERIFYPEER, 0)
        self.curl_handle.setopt(pycurl.SSL_VERIFYHOST, 0)
        self.curl_handle.setopt(pycurl.SSL_SESSIONID_CACHE, 0)
        
        # Explicitly disable CA bundle usage for SFTP
        # This prevents curl from trying to verify SSL certificates for SSH connections
        try:
            self.curl_handle.setopt(pycurl.CAINFO, None)
        except Exception:
            # If setting to None fails, try empty string
            self.curl_handle.setopt(pycurl.CAINFO, "")
        
        try:
            self.curl_handle.setopt(pycurl.CAPATH, None)
        except Exception:
            # If setting to None fails, try empty string
            self.curl_handle.setopt(pycurl.CAPATH, "")
        
        # Force SSH protocol settings and disable any SSL/TLS negotiation
        try:
            # Explicitly set SSH options to override any SSL defaults
            self.curl_handle.setopt(pycurl.PROTOCOLS, pycurl.PROTO_SFTP)
            self.curl_handle.setopt(pycurl.REDIR_PROTOCOLS, pycurl.PROTO_SFTP)
        except Exception as e:
            logger.debug(f"Could not set protocol restrictions: {e}")
        
        logger.debug(f"SFTP configured with SSH protocol only for worker {self.worker_id}")

        # SSH host key verification
        if ssh_config.known_hosts_path and ssh_config.known_hosts_path.exists():
            self.curl_handle.setopt(
                pycurl.SSH_KNOWNHOSTS, str(ssh_config.known_hosts_path)
            )
            logger.debug(f"Using known_hosts file: {ssh_config.known_hosts_path}")
        else:
            # Disable host key verification by NOT setting SSH_KNOWNHOSTS at all
            # WARNING: This is insecure and should only be used for development
            # When SSH_KNOWNHOSTS is not set, pycurl skips host key verification
            logger.debug("SSH host key verification disabled (no known_hosts file configured)")

        # SSH key-based authentication
        if ssh_config.private_key_path and ssh_config.private_key_path.exists():
            self.curl_handle.setopt(
                pycurl.SSH_PRIVATE_KEYFILE, str(ssh_config.private_key_path)
            )

            if ssh_config.public_key_path and ssh_config.public_key_path.exists():
                self.curl_handle.setopt(
                    pycurl.SSH_PUBLIC_KEYFILE, str(ssh_config.public_key_path)
                )

            # Set passphrase if provided
            if ssh_config.passphrase:
                self.curl_handle.setopt(
                    pycurl.KEYPASSWD, ssh_config.passphrase.encode("utf-8")
                )

        # SFTP specific settings
        # Enable compression for SFTP (if supported by pycurl version)
        if self.config.enable_compression:
            try:
                # SSH_COMPRESSION might not be available in all pycurl versions
                if hasattr(pycurl, 'SSH_COMPRESSION'):
                    self.curl_handle.setopt(pycurl.SSH_COMPRESSION, 1)
                else:
                    logger.debug("SSH_COMPRESSION not available in this pycurl version")
            except AttributeError:
                logger.debug("SSH_COMPRESSION not supported by this pycurl version")

        logger.debug(f"Configured SFTP options for worker {self.worker_id} (SSL verification disabled)")

    def _setup_headers(self) -> None:
        """Setup custom headers for curl handle."""
        headers = []

        # Add custom headers
        if self.config.custom_headers:
            headers.extend([f"{k}: {v}" for k, v in self.config.custom_headers.items()])

        # Add bearer token if using bearer authentication and not already in headers
        if (
            self.config.auth.method.value == "bearer"
            and not any(h.lower().startswith("authorization:") for h in headers)
        ):
            token = self.config.auth.get_token()
            if token:
                headers.append(f"Authorization: Bearer {token}")

        if headers:
            self.curl_handle.setopt(pycurl.HTTPHEADER, headers)

    def _setup_cookies(self) -> None:
        """Setup cookies for curl handle."""
        if self.config.cookies:
            # Convert cookies dict to cookie string format
            cookie_string = "; ".join(
                [f"{k}={v}" for k, v in self.config.cookies.items()]
            )
            self.curl_handle.setopt(pycurl.COOKIE, cookie_string.encode("utf-8"))

    def _setup_authentication(self) -> None:
        """Setup authentication for curl handle."""
        auth = self.config.auth

        # Detect protocol for authentication setup
        from urllib.parse import urlparse

        parsed_url = urlparse(self.segment.url)
        protocol = parsed_url.scheme.lower()

        if protocol in ("http", "https"):
            # HTTP authentication methods
            if auth.method.value == "basic":
                self.curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_BASIC)
            elif auth.method.value == "digest":
                self.curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_DIGEST)
            elif auth.method.value == "bearer":
                # Bearer token is handled in _setup_headers()
                pass
            elif auth.method.value == "ntlm":
                self.curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_NTLM)
            elif auth.method.value == "negotiate":
                self.curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_GSSNEGOTIATE)
            elif auth.method.value == "auto":
                self.curl_handle.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_ANY)

            # Set username and password for non-bearer authentication using secure retrieval
            username = auth.get_username()
            password = auth.get_password()
            
            if username and password and auth.method.value != "bearer":
                self.curl_handle.setopt(pycurl.USERPWD, f"{username}:{password}")

        elif protocol in ("ftp", "ftps"):
            # FTP authentication - always uses username/password
            username = auth.get_username()
            password = auth.get_password()
            
            if username and password:
                self.curl_handle.setopt(pycurl.USERPWD, f"{username}:{password}")
            elif username:
                # Anonymous FTP with custom username
                self.curl_handle.setopt(pycurl.USERPWD, f"{username}:")
            else:
                # Anonymous FTP - curl handles this automatically
                pass

        elif protocol == "sftp":
            # SFTP authentication - can use password or key-based
            username = auth.get_username()
            password = auth.get_password()
            
            if username:
                if password and not self.config.ssh.private_key_path:
                    # Password-based authentication (only if no SSH key is configured)
                    self.curl_handle.setopt(pycurl.USERPWD, f"{username}:{password}")
                else:
                    # Key-based authentication or username-only (keys configured in _setup_sftp_options)
                    self.curl_handle.setopt(pycurl.USERNAME, username.encode("utf-8"))

            # SSH key authentication is handled in _setup_sftp_options()

        logger.debug(
            f"Configured {protocol} authentication for worker {self.worker_id}"
        )

    def _setup_proxy(self) -> None:
        """Setup proxy configuration for curl handle."""
        proxy = self.config.proxy

        if not proxy.enabled or not proxy.host:
            return

        # Set proxy URL
        proxy_url = proxy.proxy_url
        self.curl_handle.setopt(pycurl.PROXY, proxy_url.encode("utf-8"))

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
            self.curl_handle.setopt(
                pycurl.PROXYTYPE, proxy_type_map[proxy.proxy_type.value]
            )

        # Set proxy authentication if provided
        if proxy.username and proxy.password:
            self.curl_handle.setopt(
                pycurl.PROXYUSERPWD, f"{proxy.username}:{proxy.password}"
            )

    def _setup_ssl_options(self) -> None:
        """Setup SSL/TLS options for curl handle."""
        # Basic SSL verification
        if self.config.verify_ssl and not self.config.ssl_development_mode:
            self.curl_handle.setopt(pycurl.SSL_VERIFYPEER, 1)
            self.curl_handle.setopt(pycurl.SSL_VERIFYHOST, 2)
        else:
            self.curl_handle.setopt(pycurl.SSL_VERIFYPEER, 0)
            self.curl_handle.setopt(pycurl.SSL_VERIFYHOST, 0)
            
            # Log warning for development mode
            if self.config.ssl_development_mode:
                logger.warning(f"Worker {self.worker_id}: SSL verification disabled for development mode - NOT SECURE")

        # SSL/TLS version specification
        if self.config.ssl_version:
            ssl_version_map = {
                "TLSv1": pycurl.SSLVERSION_TLSv1,
                "TLSv1.0": pycurl.SSLVERSION_TLSv1_0,
                "TLSv1.1": pycurl.SSLVERSION_TLSv1_1,
                "TLSv1.2": pycurl.SSLVERSION_TLSv1_2,
                "TLSv1.3": pycurl.SSLVERSION_TLSv1_3,
                "SSLv2": pycurl.SSLVERSION_SSLv2,
                "SSLv3": pycurl.SSLVERSION_SSLv3,
            }
            if self.config.ssl_version in ssl_version_map:
                self.curl_handle.setopt(pycurl.SSLVERSION, ssl_version_map[self.config.ssl_version])

        # Custom CA certificate bundle
        if self.config.ca_cert_path and self.config.ca_cert_path.exists():
            self.curl_handle.setopt(pycurl.CAINFO, str(self.config.ca_cert_path))

        # CA certificate directory
        if self.config.ssl_ca_cert_dir and self.config.ssl_ca_cert_dir.exists():
            self.curl_handle.setopt(pycurl.CAPATH, str(self.config.ssl_ca_cert_dir))

        # Certificate Revocation List
        if self.config.ssl_crl_file and self.config.ssl_crl_file.exists():
            self.curl_handle.setopt(pycurl.CRLFILE, str(self.config.ssl_crl_file))

        # Client certificate authentication
        if self.config.client_cert_path and self.config.client_cert_path.exists():
            self.curl_handle.setopt(pycurl.SSLCERT, str(self.config.client_cert_path))
            self.curl_handle.setopt(pycurl.SSLCERTTYPE, self.config.ssl_cert_type.encode("utf-8"))

        if self.config.client_key_path and self.config.client_key_path.exists():
            self.curl_handle.setopt(pycurl.SSLKEY, str(self.config.client_key_path))
            self.curl_handle.setopt(pycurl.SSLKEYTYPE, self.config.ssl_key_type.encode("utf-8"))
            
            # Private key password
            if self.config.ssl_key_password:
                self.curl_handle.setopt(pycurl.KEYPASSWD, self.config.ssl_key_password.encode("utf-8"))

        # SSL cipher list
        if self.config.ssl_cipher_list:
            self.curl_handle.setopt(
                pycurl.SSL_CIPHER_LIST, self.config.ssl_cipher_list.encode("utf-8")
            )

        # Public key pinning
        if self.config.ssl_pinned_public_key:
            # Format the pinned key for curl
            pinned_key = self.config.ssl_pinned_public_key
            if not pinned_key.startswith("sha256//"):
                pinned_key = f"sha256//{pinned_key}"
            self.curl_handle.setopt(pycurl.PINNEDPUBLICKEY, pinned_key.encode("utf-8"))

        # OCSP stapling verification
        if self.config.ssl_verify_status:
            self.curl_handle.setopt(pycurl.SSL_VERIFYSTATUS, 1)

        # SSL session ID caching
        if self.config.ssl_session_id_cache:
            self.curl_handle.setopt(pycurl.SSL_SESSIONID_CACHE, 1)
        else:
            self.curl_handle.setopt(pycurl.SSL_SESSIONID_CACHE, 0)

        # SSL False Start (performance optimization)
        if self.config.ssl_falsestart:
            self.curl_handle.setopt(pycurl.SSL_FALSESTART, 1)

        # ALPN (Application-Layer Protocol Negotiation)
        if self.config.ssl_enable_alpn:
            self.curl_handle.setopt(pycurl.SSL_ENABLE_ALPN, 1)
        else:
            self.curl_handle.setopt(pycurl.SSL_ENABLE_ALPN, 0)

        # NPN (Next Protocol Negotiation) - deprecated but still supported
        if self.config.ssl_enable_npn:
            self.curl_handle.setopt(pycurl.SSL_ENABLE_NPN, 1)
        else:
            self.curl_handle.setopt(pycurl.SSL_ENABLE_NPN, 0)

        # SSL debug level
        if self.config.ssl_debug_level > 0:
            self.curl_handle.setopt(pycurl.VERBOSE, 1)
            # Note: More detailed SSL debugging would require custom debug callback

        # Enable compression if configured
        if self.config.enable_compression:
            # Accept all supported encodings
            self.curl_handle.setopt(pycurl.ACCEPT_ENCODING, b"")

    def _open_temp_file(self) -> None:
        """Open temporary file for writing segment data."""
        try:
            # Ensure parent directory exists
            self.segment.temp_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Open file in binary append mode to support resume
            mode = "ab" if self.segment.downloaded_bytes > 0 else "wb"
            self.temp_file = open(self.segment.temp_file_path, mode)

            # If resuming, seek to the end
            if self.segment.downloaded_bytes > 0:
                self.temp_file.seek(0, 2)  # Seek to end
                self.downloaded_bytes = self.segment.downloaded_bytes

            logger.debug(
                f"Opened temp file {self.segment.temp_file_path} for worker {self.worker_id}"
            )

        except Exception as e:
            raise CurlError.from_curl_code(
                pycurl.E_WRITE_ERROR,
                context={
                    "error": str(e),
                    "file_path": str(self.segment.temp_file_path),
                },
            )

    def _write_callback(self, data: bytes) -> int:
        """
        Callback function for writing downloaded data.

        Args:
            data: Downloaded data chunk

        Returns:
            Number of bytes written
        """
        if self._paused or self._cancelled or self._should_stop:
            return pycurl.READFUNC_PAUSE if self._paused else pycurl.READFUNC_ABORT

        try:
            if self.temp_file:
                bytes_written = self.temp_file.write(data)
                self.temp_file.flush()  # Ensure data is written to disk
                self.downloaded_bytes += len(data)
                return bytes_written
            return 0
        except Exception as e:
            logger.error(f"Write error in worker {self.worker_id}: {e}")
            return pycurl.READFUNC_ABORT

    def _progress_callback_curl(
        self, download_total: int, downloaded: int, upload_total: int, uploaded: int
    ) -> int:
        """
        Curl progress callback function.

        Args:
            download_total: Total bytes to download
            downloaded: Bytes downloaded so far
            upload_total: Total bytes to upload (unused)
            uploaded: Bytes uploaded so far (unused)

        Returns:
            0 to continue, non-zero to abort
        """
        if self._cancelled or self._should_stop:
            return 1  # Abort download

        if self._paused:
            return 0  # Continue but don't update progress

        current_time = time.time()

        # Update speed calculation
        self._update_speed_calculation(current_time, self.downloaded_bytes)

        # Report progress at most once per second
        if current_time - self.last_progress_time >= 1.0:
            self._report_progress()
            self.last_progress_time = current_time

        return 0  # Continue download

    def _update_speed_calculation(
        self, current_time: float, current_bytes: int
    ) -> None:
        """Update download speed calculation with current data."""
        # Add current sample
        self._speed_samples.append((current_time, current_bytes))

        # Keep only samples from the last 5 seconds for accurate speed calculation
        cutoff_time = current_time - 5.0
        self._speed_samples = [
            (t, b) for t, b in self._speed_samples if t >= cutoff_time
        ]

        # Calculate speed from samples
        if len(self._speed_samples) >= 2:
            oldest_time, oldest_bytes = self._speed_samples[0]
            time_diff = current_time - oldest_time
            bytes_diff = current_bytes - oldest_bytes

            if time_diff > 0:
                self.current_speed = bytes_diff / time_diff
            else:
                self.current_speed = 0.0
        else:
            self.current_speed = 0.0

    def _report_progress(self) -> None:
        """Report current progress to callback if set."""
        if self._progress_callback:
            progress = WorkerProgress(
                worker_id=self.worker_id,
                segment_id=self.segment.id,
                downloaded_bytes=self.downloaded_bytes,
                total_bytes=self.segment.segment_size,
                download_speed=self.current_speed,
                status=self.status,
                error=self.error_message,
            )

            # Log progress update
            self.curl_logger.log_progress_update(progress)

            try:
                self._progress_callback(progress)
            except Exception as e:
                logger.error(
                    f"Error in progress callback for worker {self.worker_id}: {e}"
                )

    async def _perform_download(self) -> SegmentResult:
        """
        Perform the actual download using curl.

        Returns:
            Dictionary containing download result
        """
        if not self.curl_handle:
            return self._create_result(
                success=False, error="Curl handle not initialized"
            )

        self.status = WorkerStatus.DOWNLOADING
        self.start_time = time.time()
        self.last_progress_time = self.start_time

        try:
            # Perform the download
            await asyncio.get_event_loop().run_in_executor(
                None, self.curl_handle.perform
            )

            # Check if download was successful
            response_code = self.curl_handle.getinfo(pycurl.RESPONSE_CODE)
            effective_url = self.curl_handle.getinfo(pycurl.EFFECTIVE_URL)

            # Log connection success and collect metrics
            self.curl_logger.log_connection_success(self.curl_handle)
            metrics = self.curl_logger.collect_performance_metrics(self.curl_handle)

            # Helper function to safely decode bytes or return string
            def safe_decode(value):
                if isinstance(value, bytes):
                    return value.decode("utf-8")
                return value if value is not None else None

            # Determine success based on protocol
            from urllib.parse import urlparse
            parsed_url = urlparse(self.segment.url)
            protocol = parsed_url.scheme.lower()
            
            is_success = False
            if protocol in ("http", "https"):
                # HTTP/HTTPS: Check for 200 (OK) or 206 (Partial Content)
                is_success = response_code in (200, 206)
            elif protocol in ("ftp", "ftps"):
                # FTP/FTPS: Response code 0 means success, or 200-299 range
                is_success = response_code == 0 or (200 <= response_code < 300)
            elif protocol == "sftp":
                # SFTP: Response code 0 means success (SFTP doesn't use HTTP codes)
                is_success = response_code == 0
            else:
                # Default: assume HTTP-like behavior
                is_success = response_code in (200, 206)

            if is_success:
                self.status = WorkerStatus.COMPLETED
                self.segment.status = SegmentStatus.COMPLETED
                self.segment.downloaded_bytes = self.downloaded_bytes

                # Log successful completion
                duration = time.time() - self.start_time
                self.curl_logger.log_completion(True, self.downloaded_bytes, duration)

                # Final progress report
                self._report_progress()

                return self._create_result(
                    success=True,
                    downloaded_bytes=self.downloaded_bytes,
                    response_code=response_code,
                    effective_url=safe_decode(effective_url),
                    performance_metrics=metrics.to_dict(),
                )
            else:
                error_msg = self._get_protocol_error_message(protocol, response_code)
                self.status = WorkerStatus.FAILED
                self.error_message = error_msg
                return self._create_result(
                    success=False,
                    error=error_msg,
                    response_code=response_code,
                    effective_url=safe_decode(effective_url),
                )

        except pycurl.error as e:
            # Handle pycurl specific errors
            curl_error = CurlError.from_curl_code(
                e.args[0],
                context={
                    "segment_id": self.segment.id,
                    "worker_id": self.worker_id,
                    "url": self.segment.url,
                },
            )

            # Log connection failure with diagnostic information
            self.curl_logger.log_connection_failure(curl_error, e.args[0])
            
            # Log failed completion
            duration = time.time() - self.start_time
            self.curl_logger.log_completion(False, self.downloaded_bytes, duration)

            self.status = WorkerStatus.FAILED
            self.error_message = curl_error.curl_message

            return self._create_result(
                success=False, 
                error=curl_error.curl_message, 
                curl_error=curl_error,
                debug_summary=self.curl_logger.get_debug_summary(),
            )

        except Exception as e:
            error_msg = f"Unexpected error: {e!s}"
            self.status = WorkerStatus.FAILED
            self.error_message = error_msg
            return self._create_result(success=False, error=error_msg)

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate delay before retry attempt.

        Args:
            attempt: Current retry attempt number (1-based)

        Returns:
            Delay in seconds
        """
        base_delay = self.config.retry_delay
        backoff_factor = self.config.retry_backoff_factor

        # Exponential backoff with jitter
        delay = base_delay * (backoff_factor ** (attempt - 1))

        # Add some jitter to avoid thundering herd
        import random

        jitter = random.uniform(0.1, 0.3) * delay

        return delay + jitter

    def _create_result(self, success: bool, **kwargs) -> SegmentResult:
        """
        Create a standardized result dictionary.

        Args:
            success: Whether the operation was successful
            **kwargs: Additional result data

        Returns:
            Result dictionary
        """
        result = {
            "success": success,
            "worker_id": self.worker_id,
            "segment_id": self.segment.id,
            "downloaded_bytes": self.downloaded_bytes,
            "status": self.status.value,
            "timestamp": time.time(),
            **kwargs,
        }

        if not success and self.error_message:
            result["error"] = self.error_message

        return result

    def _get_protocol_error_message(self, protocol: str, response_code: int) -> str:
        """
        Get a descriptive error message for different protocol response codes.

        Args:
            protocol: Protocol scheme (http, https, ftp, ftps, sftp)
            response_code: Protocol-specific response code

        Returns:
            Descriptive error message
        """
        if protocol in ("http", "https"):
            return self._get_http_error_message(response_code)
        elif protocol in ("ftp", "ftps"):
            return self._get_ftp_error_message(response_code)
        elif protocol == "sftp":
            return self._get_sftp_error_message(response_code)
        else:
            return f"Protocol {protocol.upper()} error: Code {response_code}"

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

    def _get_ftp_error_message(self, response_code: int) -> str:
        """
        Get a descriptive error message for FTP response codes.

        Args:
            response_code: FTP response code

        Returns:
            Descriptive error message
        """
        ftp_status_messages = {
            0: "Success",
            421: "Service not available, closing control connection",
            425: "Can't open data connection",
            426: "Connection closed; transfer aborted",
            450: "Requested file action not taken",
            451: "Requested action aborted: local error in processing",
            452: "Requested action not taken: insufficient system storage",
            500: "Syntax error, command unrecognized",
            501: "Syntax error in parameters or arguments",
            502: "Command not implemented",
            503: "Bad sequence of commands",
            504: "Command not implemented for that parameter",
            530: "Not logged in",
            532: "Need account for storing files",
            550: "Requested action not taken: file unavailable",
            551: "Requested action aborted: page type unknown",
            552: "Requested file action aborted: exceeded storage allocation",
            553: "Requested action not taken: file name not allowed",
        }
        
        status_message = ftp_status_messages.get(response_code, "Unknown FTP Error")
        return f"FTP {response_code}: {status_message}"

    def _get_sftp_error_message(self, response_code: int) -> str:
        """
        Get a descriptive error message for SFTP response codes.

        Args:
            response_code: SFTP response code (usually 0 for success)

        Returns:
            Descriptive error message
        """
        if response_code == 0:
            return "SFTP Success"
        else:
            # For SFTP, non-zero response codes are typically curl errors, not SFTP protocol errors
            return f"SFTP Error: Code {response_code} (curl error or connection issue)"

    def pause(self) -> None:
        """Pause the download."""
        logger.info(f"Pausing worker {self.worker_id}")
        self._paused = True
        self.status = WorkerStatus.PAUSED

        # Update segment status
        self.segment.status = SegmentStatus.PAUSED
        self.segment.downloaded_bytes = self.downloaded_bytes

    def resume(self) -> None:
        """Resume the download."""
        logger.info(f"Resuming worker {self.worker_id}")
        self._paused = False

        if self.status == WorkerStatus.PAUSED:
            self.status = WorkerStatus.DOWNLOADING
            self.segment.status = SegmentStatus.DOWNLOADING

    def cancel(self) -> None:
        """Cancel the download."""
        logger.info(f"Cancelling worker {self.worker_id}")
        self._cancelled = True
        self._should_stop = True
        self.status = WorkerStatus.CANCELLED

        # Update segment status
        self.segment.status = SegmentStatus.FAILED
        self.segment.error_message = "Cancelled by user"

    def get_progress(self) -> WorkerProgress:
        """
        Get current progress information.

        Returns:
            Current worker progress
        """
        return WorkerProgress(
            worker_id=self.worker_id,
            segment_id=self.segment.id,
            downloaded_bytes=self.downloaded_bytes,
            total_bytes=self.segment.segment_size,
            download_speed=self.current_speed,
            status=self.status,
            error=self.error_message,
        )

    def _cleanup_curl(self) -> None:
        """Clean up curl handle and file resources."""
        if hasattr(self, 'curl_handle') and self.curl_handle:
            try:
                # First try to pause/abort any ongoing transfer
                try:
                    # Set a flag to stop the transfer
                    self._should_stop = True
                    # Give a brief moment for the transfer to stop
                    import time
                    time.sleep(0.1)
                except Exception:
                    pass
                
                # Now close the handle
                self.curl_handle.close()
            except Exception as e:
                # Log the specific error but don't raise it
                error_msg = str(e).lower()
                if "perform() is currently running" in error_msg:
                    logger.warning(
                        f"Curl handle for worker {self.worker_id} was closed while transfer was active"
                    )
                else:
                    logger.warning(
                        f"Error closing curl handle for worker {self.worker_id}: {e}"
                    )
            finally:
                self.curl_handle = None

        if hasattr(self, 'temp_file') and self.temp_file:
            try:
                self.temp_file.close()
            except Exception as e:
                logger.warning(
                    f"Error closing temp file for worker {self.worker_id}: {e}"
                )
            finally:
                self.temp_file = None

    def __del__(self) -> None:
        """Cleanup when worker is destroyed."""
        try:
            self._cleanup_curl()
        except AttributeError:
            # Handle case where initialization failed before setting attributes
            pass
