# Configuration Management

This document describes the comprehensive configuration system for the Zuup download manager, including type-safe configuration loading, environment variable overrides, and secure credential storage.

## Overview

The configuration system provides:

- **Type-safe configuration** using Pydantic models with runtime validation
- **Hierarchical configuration** with global and per-task settings
- **Environment variable overrides** for deployment flexibility
- **Secure credential storage** using encryption for sensitive data
- **Configuration import/export** for backup and sharing
- **Hot-reload capability** for development

## Configuration Structure

### Global Configuration

The global configuration (`GlobalConfig`) contains application-wide settings:

```python
from zuup.config import GlobalConfig, ProxyConfig

config = GlobalConfig(
    max_concurrent_downloads=3,
    default_download_path=Path.home() / "Downloads",
    temp_directory=Path.home() / ".cache" / "zuup",
    max_connections_per_download=8,
    user_agent="Zuup/0.1.0",
    proxy_settings=ProxyConfig(enabled=False),
    logging_level="INFO",
    server_host="127.0.0.1",
    server_port=8080,
    theme="dark",
    auto_start_downloads=True,
    show_notifications=True
)
```

### Task Configuration

Task-specific configuration (`TaskConfig`) can override global settings:

```python
from zuup.config import TaskConfig, ProxyConfig

task_config = TaskConfig(
    max_connections=16,  # Override global setting
    download_speed_limit=1048576,  # 1MB/s limit
    upload_speed_limit=524288,     # 512KB/s limit (torrents)
    retry_attempts=5,
    timeout=60,
    headers={"User-Agent": "Custom Agent"},
    cookies={"session": "abc123"},
    enable_seeding=True,
    seed_ratio_limit=2.0,
    seed_time_limit=86400,  # 24 hours
    proxy=ProxyConfig(enabled=True, http_proxy="http://proxy:8080")
)
```

## Using the Configuration Manager

### Basic Usage

```python
from zuup.config import ConfigManager

# Initialize with default config directory (~/.config/zuup)
manager = ConfigManager()

# Or specify custom directory
manager = ConfigManager(Path("/custom/config/path"))

# Get global configuration
global_config = manager.get_global_config()

# Get task-specific configuration
task_config = manager.get_task_config("task_id_123")

# Update configurations
manager.update_global_config(new_global_config)
manager.update_task_config("task_id_123", new_task_config)
```

### Configuration Validation

All configurations are automatically validated using Pydantic:

```python
# Validate configuration before saving
validation_result = manager.validate_config(config)
if validation_result.is_valid:
    manager.update_global_config(config)
else:
    print(f"Validation errors: {validation_result.errors}")
```

## Environment Variable Overrides

Configuration values can be overridden using environment variables with the `ZUUP_` prefix:

### Global Configuration Overrides

| Environment Variable | Configuration Path | Type | Example |
|---------------------|-------------------|------|---------|
| `ZUUP_MAX_CONCURRENT_DOWNLOADS` | `max_concurrent_downloads` | int | `5` |
| `ZUUP_DEFAULT_DOWNLOAD_PATH` | `default_download_path` | Path | `/custom/downloads` |
| `ZUUP_TEMP_DIRECTORY` | `temp_directory` | Path | `/tmp/zuup` |
| `ZUUP_MAX_CONNECTIONS_PER_DOWNLOAD` | `max_connections_per_download` | int | `16` |
| `ZUUP_USER_AGENT` | `user_agent` | str | `"Custom Agent"` |
| `ZUUP_LOGGING_LEVEL` | `logging_level` | str | `"DEBUG"` |
| `ZUUP_SERVER_HOST` | `server_host` | str | `"0.0.0.0"` |
| `ZUUP_SERVER_PORT` | `server_port` | int | `9090` |
| `ZUUP_THEME` | `theme` | str | `"light"` |
| `ZUUP_AUTO_START_DOWNLOADS` | `auto_start_downloads` | bool | `"false"` |
| `ZUUP_SHOW_NOTIFICATIONS` | `show_notifications` | bool | `"true"` |

### Proxy Configuration Overrides

| Environment Variable | Configuration Path | Type | Example |
|---------------------|-------------------|------|---------|
| `ZUUP_PROXY_ENABLED` | `proxy_settings.enabled` | bool | `"true"` |
| `ZUUP_HTTP_PROXY` | `proxy_settings.http_proxy` | str | `"http://proxy:8080"` |
| `ZUUP_HTTPS_PROXY` | `proxy_settings.https_proxy` | str | `"https://proxy:8080"` |
| `ZUUP_SOCKS_PROXY` | `proxy_settings.socks_proxy` | str | `"socks5://proxy:1080"` |
| `ZUUP_PROXY_USERNAME` | `proxy_settings.username` | str | `"proxy_user"` |
| `ZUUP_PROXY_PASSWORD` | `proxy_settings.password` | str | `"proxy_pass"` |

### Example Usage

```bash
# Set environment variables
export ZUUP_MAX_CONCURRENT_DOWNLOADS=10
export ZUUP_LOGGING_LEVEL=DEBUG
export ZUUP_PROXY_ENABLED=true
export ZUUP_HTTP_PROXY=http://corporate-proxy:8080

# Run application (will use environment overrides)
python -m zuup
```

## Secure Credential Storage

Sensitive credentials are encrypted using the `cryptography` library:

### Storing Credentials

```python
# Store sensitive data securely
credentials = {
    "api_key": "secret_api_key_12345",
    "oauth_token": "oauth_token_abcdef",
    "database_password": "super_secret_password"
}

manager.store_secure_credentials(credentials)
```

### Loading Credentials

```python
# Load encrypted credentials
credentials = manager.load_secure_credentials()
api_key = credentials.get("api_key")
```

### Security Features

- **Encryption**: Uses Fernet symmetric encryption (AES 128 in CBC mode)
- **Key Management**: Encryption key stored separately from credentials
- **File Permissions**: Key and credential files have restricted permissions (600)
- **Automatic Key Generation**: Creates new encryption key if none exists

## Configuration Files

### File Locations

- **Global Config**: `~/.config/zuup/global_config.json`
- **Task Configs**: `~/.config/zuup/tasks/{task_id}.json`
- **Encryption Key**: `~/.config/zuup/.encryption_key`
- **Encrypted Credentials**: `~/.config/zuup/.credentials.enc`

### File Format

Configuration files use JSON format with proper type conversion:

```json
{
  "max_concurrent_downloads": 5,
  "default_download_path": "~/Downloads/zuup",
  "temp_directory": "~/.cache/zuup",
  "max_connections_per_download": 16,
  "user_agent": "Zuup/0.1.0 (Custom)",
  "proxy_settings": {
    "enabled": true,
    "http_proxy": "http://proxy.example.com:8080",
    "https_proxy": "https://proxy.example.com:8080",
    "socks_proxy": null,
    "username": "proxy_user",
    "password": "proxy_pass"
  },
  "logging_level": "DEBUG",
  "server_host": "0.0.0.0",
  "server_port": 8080,
  "theme": "dark",
  "auto_start_downloads": true,
  "show_notifications": true
}
```

## Import/Export Configuration

### Export Configuration

```python
# Export all configuration to a file
export_path = Path("backup_config.json")
success = manager.export_config(export_path)
```

### Import Configuration

```python
# Import configuration from a file
import_path = Path("backup_config.json")
success = manager.import_config(import_path)
```

### Backup Strategy

```python
# Create automated backup
from datetime import datetime

backup_name = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
backup_path = Path("backups") / backup_name
manager.export_config(backup_path)
```

## Configuration Management Operations

### Reset to Defaults

```python
# Reset all configuration to defaults
manager.reset_to_defaults()
```

### Delete Task Configuration

```python
# Remove task-specific configuration
success = manager.delete_task_config("task_id_123")
```

### Configuration Inheritance

Task configurations inherit from global configuration:

1. **Global defaults** are applied first
2. **Saved task configuration** overrides global settings
3. **Environment variables** override both global and task settings

## Error Handling

### Validation Errors

```python
try:
    manager.update_global_config(invalid_config)
except ValueError as e:
    print(f"Configuration validation failed: {e}")
```

### File System Errors

```python
try:
    config = manager.get_global_config()
except RuntimeError as e:
    print(f"Failed to load configuration: {e}")
    # Fall back to defaults
    config = get_default_global_config()
```

### Encryption Errors

```python
try:
    credentials = manager.load_secure_credentials()
except Exception as e:
    print(f"Failed to decrypt credentials: {e}")
    # Handle gracefully - maybe prompt for re-entry
```

## Best Practices

### Development

1. **Use environment variables** for development-specific settings
2. **Validate configurations** before deployment
3. **Test with different config combinations**
4. **Use type hints** for configuration parameters

### Production

1. **Backup configurations** regularly using export functionality
2. **Use secure credential storage** for sensitive data
3. **Monitor configuration changes** through logging
4. **Validate environment variables** before application start

### Security

1. **Never commit** encryption keys or credential files
2. **Use environment variables** for sensitive production settings
3. **Regularly rotate** stored credentials
4. **Restrict file permissions** on configuration directories

## Example Configurations

See the `examples/config_examples/` directory for:

- `example_global_config.json` - Sample global configuration
- `example_task_config.json` - Sample task-specific configuration
- `global_config.toml` - Alternative TOML format example

## Testing

Run the manual test suite to verify configuration functionality:

```bash
python examples/manual_tests/test_config_management.py
```

This test covers:
- Basic configuration operations
- Environment variable overrides
- Secure credential storage
- Configuration validation
- Import/export functionality
- Proxy configuration
