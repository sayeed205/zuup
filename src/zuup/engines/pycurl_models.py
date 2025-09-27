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


class SecureCredentialStore(BaseModel):
    """Secure credential storage with encryption support."""
    
    encrypted: bool = False
    encryption_key_path: Path | None = None
    credentials: dict[str, str] = Field(default_factory=dict)
    
    @field_validator("encryption_key_path")
    @classmethod
    def validate_encryption_key_path(cls, v: Path | None) -> Path | None:
        """Validate encryption key file exists and is readable."""
        if v is None:
            return v
        
        if not v.exists():
            raise ValueError(f"Encryption key file does not exist: {v}")
        
        if not v.is_file():
            raise ValueError(f"Encryption key path is not a file: {v}")
        
        return v
    
    def store_credential(self, key: str, value: str) -> None:
        """Store a credential securely."""
        if self.encrypted and self.encryption_key_path:
            # In a real implementation, this would encrypt the value
            # For now, we'll store it as-is but mark it as encrypted
            self.credentials[key] = f"encrypted:{value}"
        else:
            self.credentials[key] = value
    
    def get_credential(self, key: str) -> str | None:
        """Retrieve a credential securely."""
        value = self.credentials.get(key)
        if value is None:
            return None
        
        if self.encrypted and value.startswith("encrypted:"):
            # In a real implementation, this would decrypt the value
            return value[10:]  # Remove "encrypted:" prefix
        
        return value
    
    def clear_credentials(self) -> None:
        """Clear all stored credentials."""
        self.credentials.clear()


class SSLSecurityProfile(BaseModel):
    """Predefined SSL security profiles for different use cases."""
    
    name: str
    description: str
    verify_ssl: bool = True
    ssl_version: str | None = None
    ssl_cipher_list: str | None = None
    ssl_verify_status: bool = False
    ssl_pinned_public_key: str | None = None
    ssl_development_mode: bool = False
    
    @classmethod
    def create_high_security_profile(cls) -> SSLSecurityProfile:
        """Create a high-security SSL profile."""
        return cls(
            name="high_security",
            description="High security profile with strict SSL verification",
            verify_ssl=True,
            ssl_version="TLSv1.2",  # Minimum TLS 1.2
            ssl_cipher_list="ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS",
            ssl_verify_status=True,  # Enable OCSP stapling
            ssl_development_mode=False
        )
    
    @classmethod
    def create_development_profile(cls) -> SSLSecurityProfile:
        """Create a development-friendly SSL profile."""
        return cls(
            name="development",
            description="Development profile with relaxed SSL verification",
            verify_ssl=False,
            ssl_development_mode=True
        )
    
    @classmethod
    def create_balanced_profile(cls) -> SSLSecurityProfile:
        """Create a balanced SSL profile."""
        return cls(
            name="balanced",
            description="Balanced profile with reasonable security and compatibility",
            verify_ssl=True,
            ssl_version="TLSv1.1",  # Allow TLS 1.1+
            ssl_cipher_list="HIGH:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA",
            ssl_verify_status=False
        )


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
    
    # Secure credential storage
    use_secure_storage: bool = False
    credential_store: SecureCredentialStore | None = None
    credential_key_prefix: str = "auth"  # Prefix for credential keys

    @field_validator("method", mode="before")
    @classmethod
    def validate_method(cls, v: str | AuthMethod) -> AuthMethod:
        """Convert string method to AuthMethod enum."""
        if isinstance(v, str):
            try:
                return AuthMethod(v.lower())
            except ValueError:
                raise ValueError(f"Invalid authentication method: {v}")
        return v
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to handle method conversion."""
        if name == "method" and isinstance(value, str):
            try:
                value = AuthMethod(value.lower())
            except ValueError:
                raise ValueError(f"Invalid authentication method: {value}")
        super().__setattr__(name, value)

    @model_validator(mode="after")
    def validate_auth_config(self) -> AuthConfig:
        """Validate authentication configuration consistency."""
        if self.method == AuthMethod.NONE:
            return self

        # Skip validation if using secure storage but credentials haven't been stored yet
        # This allows creating the config first, then storing credentials
        if self.use_secure_storage and self.credential_store is not None:
            # Only validate if we're not in the initial setup phase
            return self

        if self.method == AuthMethod.BEARER:
            if not self.token and not self._has_secure_token():
                raise ValueError("Bearer authentication requires a token")
        else:
            if not self.username and not self._has_secure_username():
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
                if not self.password and not self._has_secure_password():
                    raise ValueError(
                        f"{self.method.value} authentication requires a password"
                    )

        return self
    
    def _has_secure_username(self) -> bool:
        """Check if username is available in secure storage."""
        if not self.use_secure_storage or not self.credential_store:
            return False
        return self.credential_store.get_credential(f"{self.credential_key_prefix}_username") is not None
    
    def _has_secure_password(self) -> bool:
        """Check if password is available in secure storage."""
        if not self.use_secure_storage or not self.credential_store:
            return False
        return self.credential_store.get_credential(f"{self.credential_key_prefix}_password") is not None
    
    def _has_secure_token(self) -> bool:
        """Check if token is available in secure storage."""
        if not self.use_secure_storage or not self.credential_store:
            return False
        return self.credential_store.get_credential(f"{self.credential_key_prefix}_token") is not None
    
    def get_username(self) -> str | None:
        """Get username from direct field or secure storage."""
        if self.username:
            return self.username
        
        if self.use_secure_storage and self.credential_store:
            return self.credential_store.get_credential(f"{self.credential_key_prefix}_username")
        
        return None
    
    def get_password(self) -> str | None:
        """Get password from direct field or secure storage."""
        if self.password:
            return self.password
        
        if self.use_secure_storage and self.credential_store:
            return self.credential_store.get_credential(f"{self.credential_key_prefix}_password")
        
        return None
    
    def get_token(self) -> str | None:
        """Get token from direct field or secure storage."""
        if self.token:
            return self.token
        
        if self.use_secure_storage and self.credential_store:
            return self.credential_store.get_credential(f"{self.credential_key_prefix}_token")
        
        return None
    
    def store_credentials_securely(self, username: str | None = None, 
                                 password: str | None = None, 
                                 token: str | None = None) -> None:
        """Store credentials in secure storage."""
        if not self.use_secure_storage or not self.credential_store:
            raise ValueError("Secure storage not configured")
        
        if username:
            self.credential_store.store_credential(f"{self.credential_key_prefix}_username", username)
        
        if password:
            self.credential_store.store_credential(f"{self.credential_key_prefix}_password", password)
        
        if token:
            self.credential_store.store_credential(f"{self.credential_key_prefix}_token", token)
    
    def clear_credentials(self) -> None:
        """Clear all credentials from both direct fields and secure storage."""
        self.username = None
        self.password = None
        self.token = None
        
        if self.use_secure_storage and self.credential_store:
            # Clear only this auth config's credentials
            keys_to_remove = [
                f"{self.credential_key_prefix}_username",
                f"{self.credential_key_prefix}_password", 
                f"{self.credential_key_prefix}_token"
            ]
            for key in keys_to_remove:
                if key in self.credential_store.credentials:
                    del self.credential_store.credentials[key]
    
    def validate_credentials_available(self) -> None:
        """Validate that required credentials are available when needed."""
        if self.method == AuthMethod.NONE:
            return

        if self.method == AuthMethod.BEARER:
            if not self.get_token():
                raise ValueError("Bearer authentication requires a token")
        else:
            if not self.get_username():
                raise ValueError(f"{self.method.value} authentication requires a username")
            
            if self.method in (
                AuthMethod.BASIC,
                AuthMethod.DIGEST,
                AuthMethod.NTLM,
                AuthMethod.NEGOTIATE,
                AuthMethod.AUTO,
            ):
                if not self.get_password():
                    raise ValueError(f"{self.method.value} authentication requires a password")


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
    
    # Advanced SSL/TLS security features
    ssl_version: str | None = None  # Force specific SSL/TLS version
    ssl_cert_type: str = "PEM"  # Certificate format (PEM, DER, P12)
    ssl_key_type: str = "PEM"  # Private key format (PEM, DER)
    ssl_key_password: str | None = None  # Private key password
    ssl_ca_cert_dir: Path | None = None  # CA certificate directory
    ssl_crl_file: Path | None = None  # Certificate Revocation List file
    ssl_pinned_public_key: str | None = None  # Public key pinning (SHA256 hash)
    ssl_verify_status: bool = False  # Enable OCSP stapling verification
    ssl_falsestart: bool = False  # Enable SSL False Start for performance
    ssl_enable_alpn: bool = True  # Enable ALPN (Application-Layer Protocol Negotiation)
    ssl_enable_npn: bool = True  # Enable NPN (Next Protocol Negotiation)
    
    # Development and debugging options
    ssl_development_mode: bool = False  # Allow insecure connections in development
    ssl_debug_level: int = 0  # SSL debug verbosity (0-3)
    ssl_session_id_cache: bool = True  # Enable SSL session ID caching
    ssl_options: list[str] = Field(default_factory=list)  # Additional SSL options

    # Authentication
    auth: AuthConfig = Field(default_factory=AuthConfig)

    # SSH settings (for SFTP)
    ssh: SshConfig = Field(default_factory=SshConfig)

    # Proxy settings
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    
    # SSL Security Profile
    ssl_security_profile: SSLSecurityProfile | None = None

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

    @field_validator("ssl_cert_type", "ssl_key_type")
    @classmethod
    def validate_ssl_cert_key_type(cls, v: str) -> str:
        """Validate SSL certificate and key types."""
        valid_types = {"PEM", "DER", "P12"}
        if v.upper() not in valid_types:
            raise ValueError(f"SSL cert/key type must be one of: {', '.join(valid_types)}")
        return v.upper()

    @field_validator("ssl_version")
    @classmethod
    def validate_ssl_version(cls, v: str | None) -> str | None:
        """Validate SSL/TLS version specification."""
        if v is None:
            return v
        
        valid_versions = {
            "TLSv1", "TLSv1.0", "TLSv1.1", "TLSv1.2", "TLSv1.3",
            "SSLv2", "SSLv3"  # Legacy versions (not recommended)
        }
        if v not in valid_versions:
            raise ValueError(f"SSL version must be one of: {', '.join(valid_versions)}")
        return v

    @field_validator("ssl_debug_level")
    @classmethod
    def validate_ssl_debug_level(cls, v: int) -> int:
        """Validate SSL debug level."""
        if not (0 <= v <= 3):
            raise ValueError("SSL debug level must be between 0 and 3")
        return v

    @field_validator("ssl_ca_cert_dir", "ssl_crl_file")
    @classmethod
    def validate_ssl_file_paths(cls, v: Path | None) -> Path | None:
        """Validate SSL file paths exist and are readable."""
        if v is None:
            return v

        if not v.exists():
            raise ValueError(f"SSL file/directory does not exist: {v}")
        return v

    @field_validator("ssl_pinned_public_key")
    @classmethod
    def validate_ssl_pinned_public_key(cls, v: str | None) -> str | None:
        """Validate SSL public key pinning format."""
        if v is None:
            return v
        
        # Should be SHA256 hash in base64 format or hex format
        import re
        
        # Base64 format (44 characters + =)
        if re.match(r'^[A-Za-z0-9+/]{43}=$', v):
            return v
        
        # Hex format (64 characters)
        if re.match(r'^[a-fA-F0-9]{64}$', v):
            return v
        
        # SHA256:// prefix format (8 chars prefix + 64 hex chars = 72 total)
        if v.startswith('sha256//') and len(v) == 72:
            # Validate the hex part after the prefix
            hex_part = v[8:]  # Remove 'sha256//' prefix
            if re.match(r'^[a-fA-F0-9]{64}$', hex_part):
                return v
        
        # SHA256:// prefix format with base64 (8 chars prefix + 44 base64 chars = 52 total)
        if v.startswith('sha256//') and len(v) == 52:
            # Validate the base64 part after the prefix
            b64_part = v[8:]  # Remove 'sha256//' prefix
            if re.match(r'^[A-Za-z0-9+/]{43}=$', b64_part):
                return v
            
        raise ValueError(
            "SSL pinned public key must be SHA256 hash in base64 (44 chars + =) "
            "or hex format (64 chars) or 'sha256//' prefix format with base64 (52 total) "
            "or hex (72 total)"
        )
    
    def apply_ssl_security_profile(self, profile: SSLSecurityProfile) -> None:
        """Apply an SSL security profile to this configuration."""
        self.ssl_security_profile = profile
        self.verify_ssl = profile.verify_ssl
        
        if profile.ssl_version:
            self.ssl_version = profile.ssl_version
        
        if profile.ssl_cipher_list:
            self.ssl_cipher_list = profile.ssl_cipher_list
        
        self.ssl_verify_status = profile.ssl_verify_status
        
        if profile.ssl_pinned_public_key:
            self.ssl_pinned_public_key = profile.ssl_pinned_public_key
        
        self.ssl_development_mode = profile.ssl_development_mode
    
    @classmethod
    def create_with_high_security(cls, **kwargs) -> HttpFtpConfig:
        """Create configuration with high security SSL profile."""
        config = cls(**kwargs)
        config.apply_ssl_security_profile(SSLSecurityProfile.create_high_security_profile())
        return config
    
    @classmethod
    def create_with_development_profile(cls, **kwargs) -> HttpFtpConfig:
        """Create configuration with development SSL profile."""
        config = cls(**kwargs)
        config.apply_ssl_security_profile(SSLSecurityProfile.create_development_profile())
        return config
    
    @classmethod
    def create_with_balanced_profile(cls, **kwargs) -> HttpFtpConfig:
        """Create configuration with balanced SSL profile."""
        config = cls(**kwargs)
        config.apply_ssl_security_profile(SSLSecurityProfile.create_balanced_profile())
        return config


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
