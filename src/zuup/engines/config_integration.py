"""Configuration integration and validation for the pycurl HTTP/FTP engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import os
import threading
import time

# Optional watchdog import for hot-reload functionality
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..storage.models import TaskConfig, GlobalConfig, ProxyConfig as CoreProxyConfig
from .pycurl_models import (
    HttpFtpConfig,
    AuthConfig,
    AuthMethod,
    SshConfig,
    ProxyConfig,
    ProxyType,
    SSLSecurityProfile,
    SecureCredentialStore,
)

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation or mapping fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


class ConfigValidationResult(BaseModel):
    """Result of configuration validation."""
    
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    config: Optional[HttpFtpConfig] = None


class ConfigProfile(BaseModel):
    """Configuration profile for different use cases."""
    
    name: str
    description: str
    config: HttpFtpConfig
    tags: List[str] = Field(default_factory=list)
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate profile name is not empty."""
        if not v.strip():
            raise ValueError("Profile name cannot be empty")
        return v.strip()


class ConfigurationMapper:
    """Maps TaskConfig and GlobalConfig to HttpFtpConfig with validation."""
    
    def __init__(self):
        """Initialize the configuration mapper."""
        self._default_profiles = self._create_default_profiles()
        
    def map_task_config(
        self, 
        task_config: TaskConfig, 
        global_config: Optional[GlobalConfig] = None,
        profile_name: Optional[str] = None
    ) -> HttpFtpConfig:
        """
        Map TaskConfig and GlobalConfig to HttpFtpConfig.
        
        Args:
            task_config: Task-specific configuration
            global_config: Global application configuration
            profile_name: Optional profile to use as base
            
        Returns:
            Mapped HttpFtpConfig
            
        Raises:
            ConfigurationError: If mapping fails
        """
        try:
            # Start with profile if specified
            if profile_name:
                base_config = self.get_profile(profile_name).config.model_copy()
            else:
                base_config = HttpFtpConfig()
            
            # Apply global config overrides
            if global_config:
                self._apply_global_config(base_config, global_config)
            
            # Apply task-specific overrides
            self._apply_task_config(base_config, task_config)
            
            return base_config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to map configuration: {e}") from e
    
    def _apply_global_config(self, config: HttpFtpConfig, global_config: GlobalConfig) -> None:
        """Apply global configuration settings to HttpFtpConfig."""
        # Connection settings
        config.max_connections = min(
            config.max_connections, 
            global_config.max_connections_per_download
        )
        
        # User agent
        if global_config.user_agent:
            config.user_agent = global_config.user_agent
        
        # Proxy settings
        if global_config.proxy_settings:
            config.proxy = self._map_proxy_config(global_config.proxy_settings)
        
        # Timeout from global settings (if not already set)
        # Global config doesn't have timeout, so we keep the default
        
    def _apply_task_config(self, config: HttpFtpConfig, task_config: TaskConfig) -> None:
        """Apply task-specific configuration settings to HttpFtpConfig."""
        # Connection settings
        if task_config.max_connections is not None:
            config.max_connections = task_config.max_connections
        
        # Timeout
        config.timeout = task_config.timeout
        config.connect_timeout = min(task_config.timeout, config.connect_timeout)
        
        # Retry settings
        config.retry_attempts = task_config.retry_attempts
        
        # Headers and cookies
        if task_config.headers:
            config.custom_headers.update(task_config.headers)
        
        if task_config.cookies:
            config.cookies.update(task_config.cookies)
        
        # Speed limiting (convert to curl's low speed settings)
        if task_config.download_speed_limit:
            # Set low speed limit to 10% of desired speed
            config.low_speed_limit = max(1024, task_config.download_speed_limit // 10)
            config.low_speed_time = 30  # 30 seconds before considering it too slow
        
        # Proxy override
        if task_config.proxy:
            config.proxy = self._map_proxy_config(task_config.proxy)
    
    def _map_proxy_config(self, core_proxy: CoreProxyConfig) -> ProxyConfig:
        """Map core ProxyConfig to pycurl ProxyConfig."""
        # Determine proxy type and extract host/port from URLs
        proxy_type = ProxyType.HTTP  # Default
        host = ""
        port = 8080  # Default
        username = core_proxy.username
        password = core_proxy.password
        
        # Check which proxy URL is provided and extract details
        proxy_url = None
        if core_proxy.http_proxy:
            proxy_url = core_proxy.http_proxy
            proxy_type = ProxyType.HTTP
        elif core_proxy.https_proxy:
            proxy_url = core_proxy.https_proxy
            proxy_type = ProxyType.HTTPS
        elif core_proxy.socks_proxy:
            proxy_url = core_proxy.socks_proxy
            # Determine SOCKS version from URL scheme
            if proxy_url.startswith("socks4"):
                proxy_type = ProxyType.SOCKS4
            elif proxy_url.startswith("socks5"):
                proxy_type = ProxyType.SOCKS5
            else:
                proxy_type = ProxyType.SOCKS5  # Default to SOCKS5
        
        # Parse proxy URL to extract host, port, and credentials
        if proxy_url:
            from urllib.parse import urlparse
            parsed = urlparse(proxy_url)
            host = parsed.hostname or ""
            port = parsed.port or 8080
            
            # Extract credentials from URL if not provided separately
            if parsed.username and not username:
                username = parsed.username
            if parsed.password and not password:
                password = parsed.password
        
        return ProxyConfig(
            enabled=core_proxy.enabled,
            proxy_type=proxy_type,
            host=host,
            port=port,
            username=username,
            password=password,
        )
    
    def validate_config(self, config: HttpFtpConfig) -> ConfigValidationResult:
        """
        Validate HttpFtpConfig with comprehensive error checking.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with errors and warnings
        """
        errors = []
        warnings = []
        
        try:
            # Validate using Pydantic
            config.model_validate(config.model_dump())
            
            # Additional business logic validation
            self._validate_connection_settings(config, errors, warnings)
            self._validate_ssl_settings(config, errors, warnings)
            self._validate_auth_settings(config, errors, warnings)
            self._validate_proxy_settings(config, errors, warnings)
            self._validate_performance_settings(config, errors, warnings)
            
            return ConfigValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                config=config if len(errors) == 0 else None
            )
            
        except ValidationError as e:
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                errors.append(f"{field_path}: {error['msg']}")
            
            return ConfigValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                config=None
            )
    
    def _validate_connection_settings(
        self, 
        config: HttpFtpConfig, 
        errors: List[str], 
        warnings: List[str]
    ) -> None:
        """Validate connection-related settings."""
        # Check for reasonable connection limits
        if config.max_connections > 16:
            warnings.append(
                f"max_connections ({config.max_connections}) is very high, "
                "may cause server rejection or poor performance"
            )
        
        # Check timeout consistency
        if config.connect_timeout >= config.timeout:
            warnings.append(
                "connect_timeout should be less than total timeout for better error handling"
            )
        
        # Check segment size
        if config.segment_size < 64 * 1024:  # 64KB
            warnings.append(
                f"segment_size ({config.segment_size}) is very small, "
                "may cause excessive overhead"
            )
        elif config.segment_size > 50 * 1024 * 1024:  # 50MB
            warnings.append(
                f"segment_size ({config.segment_size}) is very large, "
                "may cause memory issues"
            )
    
    def _validate_ssl_settings(
        self, 
        config: HttpFtpConfig, 
        errors: List[str], 
        warnings: List[str]
    ) -> None:
        """Validate SSL/TLS settings."""
        # Check for development mode in production
        if config.ssl_development_mode and not config.verify_ssl:
            warnings.append(
                "SSL verification disabled in development mode - NOT SECURE for production"
            )
        
        # Check SSL version compatibility
        if config.ssl_version in ("SSLv2", "SSLv3"):
            errors.append(
                f"SSL version {config.ssl_version} is deprecated and insecure"
            )
        
        # Check certificate files exist
        if config.ca_cert_path and not config.ca_cert_path.exists():
            errors.append(f"CA certificate file not found: {config.ca_cert_path}")
        
        if config.client_cert_path and not config.client_cert_path.exists():
            errors.append(f"Client certificate file not found: {config.client_cert_path}")
        
        if config.client_key_path and not config.client_key_path.exists():
            errors.append(f"Client key file not found: {config.client_key_path}")
        
        # Check for client cert without key
        if config.client_cert_path and not config.client_key_path:
            errors.append("Client certificate specified without private key")
        
        # Check SSL cipher list for insecure ciphers
        if config.ssl_cipher_list:
            cipher_list_upper = config.ssl_cipher_list.upper()
            # Check for NULL ciphers that are NOT excluded (i.e., not preceded by !)
            if "NULL" in cipher_list_upper and "!NULL" not in cipher_list_upper and "!ANULL" not in cipher_list_upper and "!ENULL" not in cipher_list_upper:
                warnings.append("SSL cipher list contains NULL ciphers - may be insecure")
    
    def _validate_auth_settings(
        self, 
        config: HttpFtpConfig, 
        errors: List[str], 
        warnings: List[str]
    ) -> None:
        """Validate authentication settings."""
        try:
            config.auth.validate_credentials_available()
        except ValueError as e:
            errors.append(f"Authentication validation failed: {e}")
        
        # Check for insecure auth over HTTP
        if config.auth.method in (AuthMethod.BASIC, AuthMethod.DIGEST):
            warnings.append(
                f"{config.auth.method.value} authentication should only be used over HTTPS"
            )
    
    def _validate_proxy_settings(
        self, 
        config: HttpFtpConfig, 
        errors: List[str], 
        warnings: List[str]
    ) -> None:
        """Validate proxy settings."""
        if config.proxy.enabled:
            if not config.proxy.host:
                errors.append("Proxy enabled but no host specified")
            
            # Check for proxy auth over insecure connection
            if (config.proxy.username and config.proxy.password and 
                config.proxy.proxy_type == ProxyType.HTTP):
                warnings.append(
                    "Proxy authentication over HTTP may expose credentials"
                )
    
    def _validate_performance_settings(
        self, 
        config: HttpFtpConfig, 
        errors: List[str], 
        warnings: List[str]
    ) -> None:
        """Validate performance-related settings."""
        # Check buffer size
        if config.buffer_size > 64 * 1024:  # 64KB
            warnings.append(
                f"buffer_size ({config.buffer_size}) is large, may use excessive memory"
            )
        
        # Check retry settings
        if config.retry_attempts > 10:
            warnings.append(
                f"retry_attempts ({config.retry_attempts}) is very high, "
                "may cause long delays on persistent failures"
            )
        
        if config.retry_backoff_factor > 3.0:
            warnings.append(
                f"retry_backoff_factor ({config.retry_backoff_factor}) is high, "
                "may cause very long delays"
            )
    
    def get_profile(self, name: str) -> ConfigProfile:
        """
        Get a configuration profile by name.
        
        Args:
            name: Profile name
            
        Returns:
            Configuration profile
            
        Raises:
            ConfigurationError: If profile not found
        """
        if name not in self._default_profiles:
            raise ConfigurationError(f"Profile '{name}' not found")
        
        return self._default_profiles[name]
    
    def list_profiles(self) -> List[str]:
        """List available configuration profile names."""
        return list(self._default_profiles.keys())
    
    def _create_default_profiles(self) -> Dict[str, ConfigProfile]:
        """Create default configuration profiles."""
        profiles = {}
        
        # High Performance Profile
        high_perf_config = HttpFtpConfig(
            max_connections=8,
            segment_size=2 * 1024 * 1024,  # 2MB segments
            timeout=60,
            connect_timeout=15,
            retry_attempts=5,
            retry_delay=0.5,
            retry_backoff_factor=1.5,
            buffer_size=32 * 1024,  # 32KB buffer
            enable_compression=True,
            tcp_nodelay=True,
        )
        high_perf_config.apply_ssl_security_profile(
            SSLSecurityProfile.create_balanced_profile()
        )
        
        profiles["high_performance"] = ConfigProfile(
            name="high_performance",
            description="Optimized for maximum download speed with reasonable resource usage",
            config=high_perf_config,
            tags=["performance", "speed", "default"]
        )
        
        # Conservative Profile
        conservative_config = HttpFtpConfig(
            max_connections=2,
            segment_size=512 * 1024,  # 512KB segments
            timeout=30,
            connect_timeout=10,
            retry_attempts=3,
            retry_delay=2.0,
            retry_backoff_factor=2.0,
            buffer_size=8 * 1024,  # 8KB buffer
            enable_compression=False,
            tcp_nodelay=False,
        )
        conservative_config.apply_ssl_security_profile(
            SSLSecurityProfile.create_high_security_profile()
        )
        
        profiles["conservative"] = ConfigProfile(
            name="conservative",
            description="Conservative settings for servers that may reject aggressive connections",
            config=conservative_config,
            tags=["conservative", "compatibility", "secure"]
        )
        
        # Development Profile
        dev_config = HttpFtpConfig(
            max_connections=4,
            segment_size=1 * 1024 * 1024,  # 1MB segments
            timeout=30,
            connect_timeout=10,
            retry_attempts=2,
            retry_delay=1.0,
            ssl_development_mode=True,
            verify_ssl=False,
        )
        dev_config.apply_ssl_security_profile(
            SSLSecurityProfile.create_development_profile()
        )
        
        profiles["development"] = ConfigProfile(
            name="development",
            description="Development-friendly settings with relaxed SSL verification",
            config=dev_config,
            tags=["development", "testing", "insecure"]
        )
        
        # Mobile/Limited Bandwidth Profile
        mobile_config = HttpFtpConfig(
            max_connections=2,
            segment_size=256 * 1024,  # 256KB segments
            timeout=45,
            connect_timeout=15,
            retry_attempts=5,
            retry_delay=3.0,
            retry_backoff_factor=2.0,
            buffer_size=4 * 1024,  # 4KB buffer
            enable_compression=True,
            low_speed_limit=512,  # 512 bytes/sec minimum
            low_speed_time=60,  # 60 seconds tolerance
        )
        mobile_config.apply_ssl_security_profile(
            SSLSecurityProfile.create_balanced_profile()
        )
        
        profiles["mobile"] = ConfigProfile(
            name="mobile",
            description="Optimized for mobile or limited bandwidth connections",
            config=mobile_config,
            tags=["mobile", "bandwidth", "patient"]
        )
        
        return profiles
    
    def create_custom_profile(self, base_profile: str, overrides: Dict[str, Any]) -> ConfigProfile:
        """
        Create a custom profile based on an existing profile with overrides.
        
        Args:
            base_profile: Name of the base profile to extend
            overrides: Dictionary of configuration overrides
            
        Returns:
            New configuration profile
            
        Raises:
            ConfigurationError: If base profile not found or overrides invalid
        """
        try:
            base_config = self.get_profile(base_profile).config.model_copy()
            
            # Apply overrides
            for key, value in overrides.items():
                if hasattr(base_config, key):
                    setattr(base_config, key, value)
                else:
                    raise ConfigurationError(f"Invalid configuration key: {key}")
            
            # Validate the resulting configuration
            result = self.validate_config(base_config)
            if not result.valid:
                raise ConfigurationError(f"Custom profile validation failed: {result.errors}")
            
            return ConfigProfile(
                name=f"custom_{base_profile}",
                description=f"Custom profile based on {base_profile}",
                config=base_config,
                tags=["custom", base_profile]
            )
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create custom profile: {e}") from e


class ConfigurationHotReloader:
    """Provides hot-reload support for configuration files during development."""
    
    def __init__(self, config_path: Path, callback: Optional[callable] = None):
        """
        Initialize hot reloader.
        
        Args:
            config_path: Path to configuration file to watch
            callback: Optional callback to call when config changes
        """
        self.config_path = config_path
        self.callback = callback
        self.observer = None
        self._lock = threading.Lock()
        self._last_reload = 0
        self._reload_debounce = 1.0  # 1 second debounce
        
        if not WATCHDOG_AVAILABLE:
            logger.warning(
                "Watchdog not available - configuration hot-reload disabled. "
                "Install with: pip install watchdog"
            )
        
    def start(self) -> None:
        """Start watching for configuration file changes."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("Cannot start hot-reload: watchdog not available")
            return
            
        if self.observer is not None:
            return
        
        event_handler = ConfigFileHandler(self._on_config_changed)
        self.observer = Observer()
        self.observer.schedule(
            event_handler, 
            str(self.config_path.parent), 
            recursive=False
        )
        self.observer.start()
        logger.info(f"Started configuration hot-reload for {self.config_path}")
    
    def stop(self) -> None:
        """Stop watching for configuration file changes."""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped configuration hot-reload")
    
    def _on_config_changed(self, file_path: str) -> None:
        """Handle configuration file change event."""
        if Path(file_path) != self.config_path:
            return
        
        with self._lock:
            current_time = time.time()
            if current_time - self._last_reload < self._reload_debounce:
                return
            
            self._last_reload = current_time
        
        try:
            logger.info(f"Configuration file changed: {file_path}")
            if self.callback:
                self.callback(file_path)
        except Exception as e:
            logger.error(f"Error in configuration reload callback: {e}")


if WATCHDOG_AVAILABLE:
    class ConfigFileHandler(FileSystemEventHandler):
        """File system event handler for configuration files."""
        
        def __init__(self, callback: callable):
            """Initialize with callback function."""
            self.callback = callback
        
        def on_modified(self, event):
            """Handle file modification events."""
            if not event.is_directory:
                self.callback(event.src_path)
else:
    class ConfigFileHandler:
        """Dummy handler when watchdog is not available."""
        
        def __init__(self, callback: callable):
            """Initialize with callback function."""
            self.callback = callback


class ConfigurationManager:
    """Main configuration manager for the pycurl engine."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self.mapper = ConfigurationMapper()
        self.hot_reloader = None
        self._config_cache = {}
        self._cache_lock = threading.Lock()
        self._validation_cache = {}  # Cache validation results
    
    def create_engine_config(
        self,
        task_config: TaskConfig,
        global_config: Optional[GlobalConfig] = None,
        profile_name: Optional[str] = None,
        validate: bool = True
    ) -> HttpFtpConfig:
        """
        Create HttpFtpConfig from TaskConfig and GlobalConfig.
        
        Args:
            task_config: Task-specific configuration
            global_config: Global application configuration
            profile_name: Optional profile to use as base
            validate: Whether to validate the resulting configuration
            
        Returns:
            Configured HttpFtpConfig
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Create cache key
        cache_key = self._create_cache_key(task_config, global_config, profile_name)
        
        # Check cache
        with self._cache_lock:
            if cache_key in self._config_cache:
                return self._config_cache[cache_key].model_copy()
        
        # Map configuration
        config = self.mapper.map_task_config(task_config, global_config, profile_name)
        
        # Validate if requested
        if validate:
            result = self.mapper.validate_config(config)
            if not result.valid:
                error_msg = "Configuration validation failed:\n" + "\n".join(result.errors)
                raise ConfigurationError(error_msg)
            
            # Log warnings
            for warning in result.warnings:
                logger.warning(f"Configuration warning: {warning}")
        
        # Cache the result
        with self._cache_lock:
            self._config_cache[cache_key] = config.model_copy()
        
        return config
    
    def _create_cache_key(
        self,
        task_config: TaskConfig,
        global_config: Optional[GlobalConfig],
        profile_name: Optional[str]
    ) -> str:
        """Create a cache key for configuration."""
        import hashlib
        
        # Create a deterministic hash of the configuration
        # Use model_dump with mode='json' to handle Path serialization
        config_data = {
            "task": task_config.model_dump(mode='json'),
            "global": global_config.model_dump(mode='json') if global_config else None,
            "profile": profile_name
        }
        
        config_json = json.dumps(config_data, sort_keys=True)
        return hashlib.md5(config_json.encode()).hexdigest()
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        with self._cache_lock:
            self._config_cache.clear()
            self._validation_cache.clear()
        logger.info("Configuration cache cleared")
    
    def enable_hot_reload(self, config_file: Path) -> None:
        """
        Enable hot-reload for a configuration file.
        
        Args:
            config_file: Path to configuration file to watch
        """
        if self.hot_reloader is not None:
            self.hot_reloader.stop()
        
        def on_config_change(file_path: str):
            logger.info(f"Configuration changed, clearing cache: {file_path}")
            self.clear_cache()
        
        self.hot_reloader = ConfigurationHotReloader(config_file, on_config_change)
        self.hot_reloader.start()
    
    def disable_hot_reload(self) -> None:
        """Disable configuration hot-reload."""
        if self.hot_reloader is not None:
            self.hot_reloader.stop()
            self.hot_reloader = None
    
    def get_profile_names(self) -> List[str]:
        """Get list of available configuration profile names."""
        return self.mapper.list_profiles()
    
    def get_profile_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a configuration profile.
        
        Args:
            name: Profile name
            
        Returns:
            Profile information dictionary
        """
        profile = self.mapper.get_profile(name)
        return {
            "name": profile.name,
            "description": profile.description,
            "tags": profile.tags,
            "config_summary": {
                "max_connections": profile.config.max_connections,
                "segment_size": profile.config.segment_size,
                "timeout": profile.config.timeout,
                "ssl_security": profile.config.ssl_security_profile.name if profile.config.ssl_security_profile else "default",
                "verify_ssl": profile.config.verify_ssl,
            }
        }
    
    def validate_task_config(self, task_config: TaskConfig, global_config: Optional[GlobalConfig] = None) -> ConfigValidationResult:
        """
        Validate a TaskConfig by mapping it to HttpFtpConfig and checking for issues.
        
        Args:
            task_config: Task configuration to validate
            global_config: Optional global configuration
            
        Returns:
            Validation result
        """
        # Create cache key for validation results
        validation_key = self._create_validation_cache_key(task_config, global_config)
        
        # Check validation cache
        with self._cache_lock:
            if validation_key in self._validation_cache:
                return self._validation_cache[validation_key]
        
        try:
            mapped_config = self.mapper.map_task_config(task_config, global_config)
            result = self.mapper.validate_config(mapped_config)
            
            # Cache the validation result
            with self._cache_lock:
                self._validation_cache[validation_key] = result
            
            return result
        except Exception as e:
            result = ConfigValidationResult(
                valid=False,
                errors=[f"Configuration mapping failed: {e}"],
                warnings=[],
                config=None
            )
            
            # Cache the error result too
            with self._cache_lock:
                self._validation_cache[validation_key] = result
            
            return result
    
    def _create_validation_cache_key(self, task_config: TaskConfig, global_config: Optional[GlobalConfig]) -> str:
        """Create a cache key for validation results."""
        import hashlib
        
        validation_data = {
            "task": task_config.model_dump(mode='json'),
            "global": global_config.model_dump(mode='json') if global_config else None,
        }
        
        validation_json = json.dumps(validation_data, sort_keys=True)
        return f"validation_{hashlib.md5(validation_json.encode()).hexdigest()}"
    
    def create_profile_from_config(self, name: str, description: str, config: HttpFtpConfig, tags: Optional[List[str]] = None) -> ConfigProfile:
        """
        Create a custom configuration profile.
        
        Args:
            name: Profile name
            description: Profile description
            config: HttpFtpConfig to use as the profile
            tags: Optional tags for the profile
            
        Returns:
            Created configuration profile
        """
        return ConfigProfile(
            name=name,
            description=description,
            config=config,
            tags=tags or []
        )
    
    def export_config_to_file(self, config: HttpFtpConfig, file_path: Path) -> None:
        """
        Export HttpFtpConfig to a JSON file for persistence or sharing.
        
        Args:
            config: Configuration to export
            file_path: Path where to save the configuration
            
        Raises:
            ConfigurationError: If export fails
        """
        try:
            config_data = config.model_dump(mode='json', exclude_none=True)
            
            # Create parent directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, sort_keys=True)
            
            logger.info(f"Configuration exported to {file_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to export configuration: {e}") from e
    
    def import_config_from_file(self, file_path: Path) -> HttpFtpConfig:
        """
        Import HttpFtpConfig from a JSON file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Imported configuration
            
        Raises:
            ConfigurationError: If import fails
        """
        try:
            if not file_path.exists():
                raise ConfigurationError(f"Configuration file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            config = HttpFtpConfig.model_validate(config_data)
            
            # Validate the imported configuration
            result = self.mapper.validate_config(config)
            if not result.valid:
                logger.warning(f"Imported configuration has validation issues: {result.errors}")
            
            logger.info(f"Configuration imported from {file_path}")
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to import configuration: {e}") from e
    
    def get_configuration_summary(self, config: HttpFtpConfig) -> Dict[str, Any]:
        """
        Get a human-readable summary of configuration settings.
        
        Args:
            config: Configuration to summarize
            
        Returns:
            Configuration summary dictionary
        """
        return {
            "connection": {
                "max_connections": config.max_connections,
                "timeout": f"{config.timeout}s",
                "connect_timeout": f"{config.connect_timeout}s",
                "segment_size": f"{config.segment_size // 1024}KB",
            },
            "retry": {
                "attempts": config.retry_attempts,
                "delay": f"{config.retry_delay}s",
                "backoff_factor": config.retry_backoff_factor,
            },
            "ssl": {
                "verify": config.verify_ssl,
                "version": config.ssl_version or "auto",
                "development_mode": config.ssl_development_mode,
                "security_profile": config.ssl_security_profile.name if config.ssl_security_profile else "default",
            },
            "authentication": {
                "method": config.auth.method.value,
                "has_credentials": bool(config.auth.get_username() or config.auth.get_token()),
                "secure_storage": config.auth.use_secure_storage,
            },
            "proxy": {
                "enabled": config.proxy.enabled,
                "type": config.proxy.proxy_type.value if config.proxy.enabled else None,
                "host": config.proxy.host if config.proxy.enabled else None,
            },
            "performance": {
                "compression": config.enable_compression,
                "tcp_nodelay": config.tcp_nodelay,
                "buffer_size": f"{config.buffer_size // 1024}KB",
            }
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.disable_hot_reload()


# Convenience functions for common use cases
def create_default_config() -> HttpFtpConfig:
    """Create a default HttpFtpConfig with sensible defaults."""
    manager = ConfigurationManager()
    return manager.mapper.get_profile("high_performance").config.model_copy()


def create_config_from_task(
    task_config: TaskConfig, 
    global_config: Optional[GlobalConfig] = None,
    profile: str = "high_performance"
) -> HttpFtpConfig:
    """
    Convenience function to create HttpFtpConfig from TaskConfig.
    
    Args:
        task_config: Task configuration
        global_config: Optional global configuration
        profile: Profile name to use as base
        
    Returns:
        Configured HttpFtpConfig
    """
    manager = ConfigurationManager()
    return manager.create_engine_config(task_config, global_config, profile)


def validate_config_dict(config_dict: Dict[str, Any]) -> ConfigValidationResult:
    """
    Validate a configuration dictionary.
    
    Args:
        config_dict: Configuration as dictionary
        
    Returns:
        Validation result
    """
    try:
        config = HttpFtpConfig.model_validate(config_dict)
        manager = ConfigurationManager()
        return manager.mapper.validate_config(config)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            errors.append(f"{field_path}: {error['msg']}")
        
        return ConfigValidationResult(
            valid=False,
            errors=errors,
            warnings=[],
            config=None
        )
