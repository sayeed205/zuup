"""Data models and type definitions for the pycurl HTTP/FTP engine."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from cuid import cuid
import pycurl
from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================================
# Segment and Progress Models
# ============================================================================


class SegmentStatus(Enum):
    """Status of a download segment."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    MERGED = "merged"


class DownloadSegment(BaseModel):
    """Represents a single download segment for multi-connection downloads."""

    id: str = Field(default_factory=cuid)
    task_id: str
    url: str
    start_byte: int
    end_byte: int
    temp_file_path: Path
    status: SegmentStatus = SegmentStatus.PENDING
    downloaded_bytes: int = 0
    retry_count: int = 0
    error_message: str | None = None

    @field_validator("start_byte", "end_byte", "downloaded_bytes")
    @classmethod
    def validate_byte_values(cls, v: int) -> int:
        """Validate byte values are non-negative."""
        if v < 0:
            raise ValueError("Byte values must be non-negative")
        return v

    @field_validator("retry_count")
    @classmethod
    def validate_retry_count(cls, v: int) -> int:
        """Validate retry count is non-negative."""
        if v < 0:
            raise ValueError("Retry count must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_byte_range(self) -> DownloadSegment:
        """Validate byte range consistency."""
        if self.start_byte > self.end_byte:
            raise ValueError("start_byte cannot be greater than end_byte")

        segment_size = self.end_byte - self.start_byte + 1
        if self.downloaded_bytes > segment_size:
            raise ValueError("downloaded_bytes cannot exceed segment size")

        return self

    @property
    def segment_size(self) -> int:
        """Calculate the total size of this segment."""
        return self.end_byte - self.start_byte + 1

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage for this segment."""
        if self.segment_size == 0:
            return 100.0
        return (self.downloaded_bytes / self.segment_size) * 100.0


class WorkerStatus(Enum):
    """Status of a curl worker."""

    INITIALIZING = "initializing"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerProgress(BaseModel):
    """Progress information for a single curl worker."""

    worker_id: str
    segment_id: str
    downloaded_bytes: int = 0
    total_bytes: int = 0
    download_speed: float = 0.0  # bytes per second
    status: WorkerStatus = WorkerStatus.INITIALIZING
    error: str | None = None

    @field_validator("downloaded_bytes", "total_bytes")
    @classmethod
    def validate_bytes(cls, v: int) -> int:
        """Validate byte counts are non-negative."""
        if v < 0:
            raise ValueError("Byte counts must be non-negative")
        return v

    @field_validator("download_speed")
    @classmethod
    def validate_speed(cls, v: float) -> float:
        """Validate download speed is non-negative."""
        if v < 0:
            raise ValueError("Download speed must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_progress_consistency(self) -> WorkerProgress:
        """Validate progress consistency."""
        if self.total_bytes > 0 and self.downloaded_bytes > self.total_bytes:
            raise ValueError("Downloaded bytes cannot exceed total bytes")
        return self

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100.0


# ============================================================================
# Authentication and Configuration Models
# ============================================================================


class AuthMethod(Enum):
    """Authentication methods supported by pycurl."""

    NONE = "none"
    BASIC = "basic"
    DIGEST = "digest"
    BEARER = "bearer"
    NTLM = "ntlm"
    NEGOTIATE = "negotiate"
    AUTO = "auto"  # Let curl choose the best method


class AuthConfig(BaseModel):
    """Authentication configuration for HTTP/FTP downloads."""

    method: AuthMethod = AuthMethod.NONE
    username: str | None = None
    password: str | None = None
    token: str | None = None  # For bearer token authentication

    @model_validator(mode="after")
    def validate_auth_config(self) -> AuthConfig:
        """Validate authentication configuration consistency."""
        if self.method == AuthMethod.NONE:
            return self

        if self.method == AuthMethod.BEARER:
            if not self.token:
                raise ValueError("Bearer authentication requires a token")
        else:
            if not self.username:
                raise ValueError(
                    f"{self.method.value} authentication requires a username"
                )
            if self.method in (
                AuthMethod.BASIC,
                AuthMethod.DIGEST,
                AuthMethod.NTLM,
                AuthMethod.NEGOTIATE,
                AuthMethod.AUTO,
            ):
                if not self.password:
                    raise ValueError(
                        f"{self.method.value} authentication requires a password"
                    )

        return self


class SshConfig(BaseModel):
    """SSH configuration for SFTP downloads."""

    private_key_path: Path | None = None
    public_key_path: Path | None = None
    known_hosts_path: Path | None = None
    passphrase: str | None = None

    @field_validator("private_key_path", "public_key_path", "known_hosts_path")
    @classmethod
    def validate_file_paths(cls, v: Path | None) -> Path | None:
        """Validate SSH file paths exist and are readable."""
        if v is None:
            return v

        if not v.exists():
            raise ValueError(f"SSH file does not exist: {v}")

        if not v.is_file():
            raise ValueError(f"SSH path is not a file: {v}")

        return v

    @model_validator(mode="after")
    def validate_ssh_config(self) -> SshConfig:
        """Validate SSH configuration consistency."""
        # If public key is provided, private key should also be provided
        if self.public_key_path and not self.private_key_path:
            raise ValueError("Public key path requires private key path")

        return self


class ProxyType(Enum):
    """Proxy types supported by pycurl."""

    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS4A = "socks4a"
    SOCKS5 = "socks5"
    SOCKS5_HOSTNAME = "socks5h"


class ProxyConfig(BaseModel):
    """Proxy configuration for pycurl."""

    enabled: bool = False
    proxy_type: ProxyType = ProxyType.HTTP
    host: str = ""
    port: int = 8080
    username: str | None = None
    password: str | None = None

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate proxy port range."""
        if not (1 <= v <= 65535):
            raise ValueError("Proxy port must be between 1 and 65535")
        return v

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate proxy host is not empty when enabled."""
        return v.strip()

    @model_validator(mode="after")
    def validate_proxy_config(self) -> ProxyConfig:
        """Validate proxy configuration consistency."""
        if self.enabled and not self.host:
            raise ValueError("Proxy host is required when proxy is enabled")
        return self

    @property
    def proxy_url(self) -> str:
        """Generate proxy URL for pycurl."""
        if not self.enabled:
            return ""

        if self.username and self.password:
            return f"{self.proxy_type.value}://{self.username}:{self.password}@{self.host}:{self.port}"
        else:
            return f"{self.proxy_type.value}://{self.host}:{self.port}"


class HttpFtpConfig(BaseModel):
    """Configuration for HTTP/FTP downloads using pycurl."""

    # Connection settings
    max_connections: int = 8
    segment_size: int = 1024 * 1024  # 1MB default
    timeout: int = 30
    connect_timeout: int = 10
    low_speed_limit: int = 1024  # bytes per second
    low_speed_time: int = 30  # seconds

    # Retry settings
    retry_attempts: int = 3
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0

    # HTTP settings
    user_agent: str = "Zuup-PyCurl/1.0"
    follow_redirects: bool = True
    max_redirects: int = 10
    custom_headers: dict[str, str] = Field(default_factory=dict)
    cookies: dict[str, str] = Field(default_factory=dict)

    # SSL/TLS settings
    verify_ssl: bool = True
    ca_cert_path: Path | None = None
    client_cert_path: Path | None = None
    client_key_path: Path | None = None
    ssl_cipher_list: str | None = None

    # Authentication
    auth: AuthConfig = Field(default_factory=AuthConfig)

    # SSH settings (for SFTP)
    ssh: SshConfig = Field(default_factory=SshConfig)

    # Proxy settings
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)

    # FTP specific settings
    ftp_use_epsv: bool = True
    ftp_use_eprt: bool = True
    ftp_create_missing_dirs: bool = False
    ftp_skip_pasv_ip: bool = False  # Skip PASV IP address for NAT/firewall issues
    ftp_use_pret: bool = False  # Use PRET command for some FTP servers
    ftp_alternative_to_user: str = "anonymous"  # Alternative username for anonymous FTP

    # Performance settings
    buffer_size: int = 16384  # 16KB
    enable_compression: bool = True
    tcp_nodelay: bool = True

    @field_validator("max_connections")
    @classmethod
    def validate_max_connections(cls, v: int) -> int:
        """Validate max connections is positive and reasonable."""
        if v <= 0:
            raise ValueError("max_connections must be positive")
        if v > 32:
            raise ValueError("max_connections should not exceed 32 for stability")
        return v

    @field_validator("segment_size")
    @classmethod
    def validate_segment_size(cls, v: int) -> int:
        """Validate segment size is reasonable."""
        if v < 1024:  # 1KB minimum
            raise ValueError("segment_size must be at least 1024 bytes")
        if v > 100 * 1024 * 1024:  # 100MB maximum
            raise ValueError("segment_size should not exceed 100MB")
        return v

    @field_validator("timeout", "connect_timeout", "low_speed_time")
    @classmethod
    def validate_timeouts(cls, v: int) -> int:
        """Validate timeout values are positive."""
        if v <= 0:
            raise ValueError("Timeout values must be positive")
        return v

    @field_validator("retry_attempts")
    @classmethod
    def validate_retry_attempts(cls, v: int) -> int:
        """Validate retry attempts is non-negative."""
        if v < 0:
            raise ValueError("retry_attempts must be non-negative")
        return v

    @field_validator("retry_delay", "retry_backoff_factor")
    @classmethod
    def validate_retry_settings(cls, v: float) -> float:
        """Validate retry settings are positive."""
        if v <= 0:
            raise ValueError("Retry settings must be positive")
        return v

    @field_validator("max_redirects")
    @classmethod
    def validate_max_redirects(cls, v: int) -> int:
        """Validate max redirects is non-negative."""
        if v < 0:
            raise ValueError("max_redirects must be non-negative")
        return v

    @field_validator("buffer_size")
    @classmethod
    def validate_buffer_size(cls, v: int) -> int:
        """Validate buffer size is reasonable."""
        if v < 1024:  # 1KB minimum
            raise ValueError("buffer_size must be at least 1024 bytes")
        if v > 1024 * 1024:  # 1MB maximum
            raise ValueError("buffer_size should not exceed 1MB")
        return v


# ============================================================================
# Error Handling Models
# ============================================================================


class ErrorCategory(Enum):
    """Categories of errors that can occur during downloads."""

    NETWORK = "network"
    PROTOCOL = "protocol"
    AUTHENTICATION = "authentication"
    FILESYSTEM = "filesystem"
    CURL = "curl"
    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ErrorAction(Enum):
    """Actions to take when an error occurs."""

    RETRY = "retry"
    FAIL_SEGMENT = "fail_segment"
    FAIL_DOWNLOAD = "fail_download"
    REDUCE_CONNECTIONS = "reduce_connections"
    SWITCH_PROTOCOL = "switch_protocol"
    PAUSE_DOWNLOAD = "pause_download"


class CurlError(BaseModel):
    """Represents a curl-specific error with categorization and context."""

    curl_code: int
    curl_message: str
    category: ErrorCategory
    action: ErrorAction
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=lambda: __import__("time").time())

    @field_validator("curl_code")
    @classmethod
    def validate_curl_code(cls, v: int) -> int:
        """Validate curl error code is non-negative."""
        if v < 0:
            raise ValueError("Curl error code must be non-negative")
        return v

    @classmethod
    def from_curl_code(
        cls, curl_code: int, context: dict[str, Any] | None = None
    ) -> CurlError:
        """Create a CurlError from a pycurl error code."""
        # Map curl error codes to categories and actions
        error_mapping = {
            # Network errors
            pycurl.E_COULDNT_RESOLVE_HOST: (ErrorCategory.NETWORK, ErrorAction.RETRY),
            pycurl.E_COULDNT_CONNECT: (ErrorCategory.NETWORK, ErrorAction.RETRY),
            pycurl.E_OPERATION_TIMEDOUT: (ErrorCategory.TIMEOUT, ErrorAction.RETRY),
            pycurl.E_OPERATION_TIMEOUTED: (
                ErrorCategory.TIMEOUT,
                ErrorAction.RETRY,
            ),  # Alternative spelling
            pycurl.E_RECV_ERROR: (ErrorCategory.NETWORK, ErrorAction.RETRY),
            pycurl.E_SEND_ERROR: (ErrorCategory.NETWORK, ErrorAction.RETRY),
            pycurl.E_GOT_NOTHING: (ErrorCategory.NETWORK, ErrorAction.RETRY),
            # Protocol errors
            pycurl.E_HTTP_RETURNED_ERROR: (
                ErrorCategory.PROTOCOL,
                ErrorAction.FAIL_DOWNLOAD,
            ),
            pycurl.E_FTP_WEIRD_SERVER_REPLY: (
                ErrorCategory.PROTOCOL,
                ErrorAction.RETRY,
            ),
            pycurl.E_PARTIAL_FILE: (ErrorCategory.PROTOCOL, ErrorAction.RETRY),
            pycurl.E_FTP_PARTIAL_FILE: (ErrorCategory.PROTOCOL, ErrorAction.RETRY),
            # Authentication errors
            pycurl.E_LOGIN_DENIED: (
                ErrorCategory.AUTHENTICATION,
                ErrorAction.FAIL_DOWNLOAD,
            ),
            pycurl.E_REMOTE_ACCESS_DENIED: (
                ErrorCategory.AUTHENTICATION,
                ErrorAction.FAIL_DOWNLOAD,
            ),
            pycurl.E_FTP_ACCESS_DENIED: (
                ErrorCategory.AUTHENTICATION,
                ErrorAction.FAIL_DOWNLOAD,
            ),
            # SSL errors
            pycurl.E_SSL_CONNECT_ERROR: (
                ErrorCategory.PROTOCOL,
                ErrorAction.FAIL_DOWNLOAD,
            ),
            pycurl.E_SSL_PEER_CERTIFICATE: (
                ErrorCategory.PROTOCOL,
                ErrorAction.FAIL_DOWNLOAD,
            ),
            pycurl.E_SSL_CACERT: (ErrorCategory.PROTOCOL, ErrorAction.FAIL_DOWNLOAD),
            pycurl.E_PEER_FAILED_VERIFICATION: (
                ErrorCategory.PROTOCOL,
                ErrorAction.FAIL_DOWNLOAD,
            ),
            # File system errors
            pycurl.E_WRITE_ERROR: (ErrorCategory.FILESYSTEM, ErrorAction.FAIL_DOWNLOAD),
            pycurl.E_FILE_COULDNT_READ_FILE: (
                ErrorCategory.FILESYSTEM,
                ErrorAction.FAIL_DOWNLOAD,
            ),
            # Too many connections
            pycurl.E_TOO_MANY_REDIRECTS: (
                ErrorCategory.PROTOCOL,
                ErrorAction.REDUCE_CONNECTIONS,
            ),
        }

        category, action = error_mapping.get(
            curl_code, (ErrorCategory.CURL, ErrorAction.RETRY)
        )

        # Get curl error message - map common error codes to descriptive messages
        error_messages = {
            pycurl.E_OK: "No error",
            pycurl.E_UNSUPPORTED_PROTOCOL: "Unsupported protocol",
            pycurl.E_FAILED_INIT: "Failed initialization",
            pycurl.E_URL_MALFORMAT: "URL malformed",
            pycurl.E_COULDNT_RESOLVE_PROXY: "Couldn't resolve proxy",
            pycurl.E_COULDNT_RESOLVE_HOST: "Couldn't resolve host",
            pycurl.E_COULDNT_CONNECT: "Couldn't connect to server",
            pycurl.E_FTP_WEIRD_SERVER_REPLY: "FTP weird server reply",
            pycurl.E_REMOTE_ACCESS_DENIED: "Remote access denied",
            pycurl.E_HTTP_RETURNED_ERROR: "HTTP returned error",
            pycurl.E_WRITE_ERROR: "Write error",
            pycurl.E_MALFORMAT_USER: "Malformed user",
            pycurl.E_FTP_COULDNT_STOR_FILE: "FTP couldn't store file",
            pycurl.E_READ_ERROR: "Read error",
            pycurl.E_OUT_OF_MEMORY: "Out of memory",
            pycurl.E_OPERATION_TIMEDOUT: "Operation timed out",
            pycurl.E_OPERATION_TIMEOUTED: "Operation timed out",
            pycurl.E_FTP_COULDNT_SET_ASCII: "FTP couldn't set ASCII",
            pycurl.E_FTP_PORT_FAILED: "FTP port failed",
            pycurl.E_FTP_COULDNT_USE_REST: "FTP couldn't use REST",
            pycurl.E_RANGE_ERROR: "Range error",
            pycurl.E_HTTP_POST_ERROR: "HTTP POST error",
            pycurl.E_SSL_CONNECT_ERROR: "SSL connect error",
            pycurl.E_BAD_DOWNLOAD_RESUME: "Bad download resume",
            pycurl.E_FILE_COULDNT_READ_FILE: "File couldn't read file",
            pycurl.E_LDAP_CANNOT_BIND: "LDAP cannot bind",
            pycurl.E_LDAP_SEARCH_FAILED: "LDAP search failed",
            pycurl.E_FUNCTION_NOT_FOUND: "Function not found",
            pycurl.E_ABORTED_BY_CALLBACK: "Aborted by callback",
            pycurl.E_BAD_FUNCTION_ARGUMENT: "Bad function argument",
            pycurl.E_INTERFACE_FAILED: "Interface failed",
            pycurl.E_TOO_MANY_REDIRECTS: "Too many redirects",
            pycurl.E_UNKNOWN_TELNET_OPTION: "Unknown telnet option",
            pycurl.E_TELNET_OPTION_SYNTAX: "Telnet option syntax",
            pycurl.E_PEER_FAILED_VERIFICATION: "Peer failed verification",
            pycurl.E_GOT_NOTHING: "Got nothing",
            pycurl.E_SSL_ENGINE_NOTFOUND: "SSL engine not found",
            pycurl.E_SSL_ENGINE_SETFAILED: "SSL engine set failed",
            pycurl.E_SEND_ERROR: "Send error",
            pycurl.E_RECV_ERROR: "Receive error",
            pycurl.E_SSL_CERTPROBLEM: "SSL certificate problem",
            pycurl.E_SSL_CIPHER: "SSL cipher",
            pycurl.E_SSL_CACERT: "SSL CA certificate",
            pycurl.E_BAD_CONTENT_ENCODING: "Bad content encoding",
            pycurl.E_LDAP_INVALID_URL: "LDAP invalid URL",
            pycurl.E_FILESIZE_EXCEEDED: "File size exceeded",
            pycurl.E_USE_SSL_FAILED: "Use SSL failed",
            pycurl.E_SEND_FAIL_REWIND: "Send fail rewind",
            pycurl.E_SSL_ENGINE_INITFAILED: "SSL engine init failed",
            pycurl.E_LOGIN_DENIED: "Login denied",
            pycurl.E_TFTP_NOTFOUND: "TFTP not found",
            pycurl.E_TFTP_PERM: "TFTP permission",
            pycurl.E_REMOTE_DISK_FULL: "Remote disk full",
            pycurl.E_TFTP_ILLEGAL: "TFTP illegal",
            pycurl.E_TFTP_UNKNOWNID: "TFTP unknown ID",
            pycurl.E_REMOTE_FILE_EXISTS: "Remote file exists",
            pycurl.E_TFTP_NOSUCHUSER: "TFTP no such user",
            pycurl.E_CONV_FAILED: "Conversion failed",
            pycurl.E_CONV_REQD: "Conversion required",
            pycurl.E_SSL_CACERT_BADFILE: "SSL CA certificate bad file",
            pycurl.E_REMOTE_FILE_NOT_FOUND: "Remote file not found",
            pycurl.E_SSH: "SSH error",
            pycurl.E_SSL_SHUTDOWN_FAILED: "SSL shutdown failed",
            pycurl.E_AGAIN: "Again",
            pycurl.E_SSL_CRL_BADFILE: "SSL CRL bad file",
            pycurl.E_SSL_ISSUER_ERROR: "SSL issuer error",
            pycurl.E_FTP_PRET_FAILED: "FTP PRET failed",
            pycurl.E_RTSP_CSEQ_ERROR: "RTSP CSEQ error",
            pycurl.E_RTSP_SESSION_ERROR: "RTSP session error",
            pycurl.E_FTP_BAD_FILE_LIST: "FTP bad file list",
            pycurl.E_CHUNK_FAILED: "Chunk failed",
        }

        curl_message = error_messages.get(curl_code, f"Curl error {curl_code}")

        return cls(
            curl_code=curl_code,
            curl_message=curl_message,
            category=category,
            action=action,
            context=context or {},
        )


# ============================================================================
# Curl-specific Type Definitions
# ============================================================================

# Type aliases for curl callbacks
CurlWriteCallback = Callable[[bytes], int]
CurlProgressCallback = Callable[[int, int, int, int], int]
CurlHeaderCallback = Callable[[bytes], int]

# Type alias for curl options dictionary
CurlOptions = dict[int, str | int | bool | Callable]

# Type alias for segment results
SegmentResult = dict[str, Any]

# Type alias for merge results
MergeResult = dict[str, Any]

# Type alias for finalize results
FinalizeResult = dict[str, Any]

# Type alias for error context
ErrorContext = dict[str, Any]


# ============================================================================
# Utility Models
# ============================================================================


class CompletedSegment(BaseModel):
    """Represents a completed download segment ready for merging."""

    segment: DownloadSegment
    temp_file_path: Path
    checksum: str | None = None

    @field_validator("temp_file_path")
    @classmethod
    def validate_temp_file_exists(cls, v: Path) -> Path:
        """Validate that the temporary file exists."""
        if not v.exists():
            raise ValueError(f"Temporary file does not exist: {v}")
        if not v.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return v


class SegmentMergeInfo(BaseModel):
    """Information about segment merging progress."""

    total_segments: int
    merged_segments: int
    current_segment: str | None = None
    bytes_merged: int = 0
    total_bytes: int = 0

    @field_validator("total_segments", "merged_segments", "bytes_merged", "total_bytes")
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        """Validate values are non-negative."""
        if v < 0:
            raise ValueError("Values must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_merge_consistency(self) -> SegmentMergeInfo:
        """Validate merge information consistency."""
        if self.merged_segments > self.total_segments:
            raise ValueError("Merged segments cannot exceed total segments")
        if self.total_bytes > 0 and self.bytes_merged > self.total_bytes:
            raise ValueError("Merged bytes cannot exceed total bytes")
        return self

    @property
    def progress_percentage(self) -> float:
        """Calculate merge progress percentage."""
        if self.total_segments == 0:
            return 100.0
        return (self.merged_segments / self.total_segments) * 100.0
