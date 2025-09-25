# Configuration Guide

This document describes how to configure Zuup.

## Configuration Files

Zuup uses a hierarchical configuration system:

1. **Global Configuration**: Applied to all downloads by default
2. **Task Configuration**: Overrides global settings for specific downloads

## Configuration Locations

- **Linux/macOS**: `~/.config/zuup/`
- **Windows**: `%APPDATA%\zuup\`

## Configuration Format

Configuration files use TOML format for human readability.

## Global Configuration

```toml
[global]
max_concurrent_downloads = 3
default_download_path = "~/Downloads"
temp_directory = "/tmp/zuup"
max_connections_per_download = 8
user_agent = "Zuup/0.1.0"

[global.proxy]
enabled = false
http_proxy = ""
https_proxy = ""
socks_proxy = ""

[global.logging]
level = "INFO"
file_path = "~/.config/zuup/logs/app.log"
max_file_size = "10MB"
backup_count = 5
```

## Task Configuration

Task-specific configuration will be documented in task 3.
