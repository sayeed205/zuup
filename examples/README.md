# Zuup Examples and Testing

This directory contains examples, demos, and testing utilities for the Zuup download manager's logging and monitoring infrastructure.

## Directory Structure

```
examples/
├── README.md                           # This file
├── interactive_logging_demo.py         # Interactive demo script
├── manual_tests/
│   └── test_logging_monitoring.py      # Comprehensive test suite
└── config_examples/
    └── logging_config.json             # Example configuration
```

## Quick Start

### Interactive Demo

Run the interactive demo to explore all logging and monitoring features:

```bash
python examples/interactive_logging_demo.py
```

This provides a menu-driven interface to:
- Test different logging configurations
- Demonstrate download-specific logging
- Show live monitoring of multiple downloads
- Explore debugging and profiling features
- Export and view generated log files

### Automated Tests

Run the comprehensive test suite:

```bash
python examples/manual_tests/test_logging_monitoring.py
```

This tests all components:
- ✅ Basic logging functionality
- ✅ Download-specific logging with context
- ✅ Metrics collection and monitoring
- ✅ Performance monitoring
- ✅ Debugging and error reporting
- ✅ Performance profiling

## Features Demonstrated

### 1. Structured Logging System

- **Multiple output formats**: Console (Rich), file (standard), structured JSON
- **Configurable levels**: DEBUG, INFO, WARNING, ERROR with per-handler control
- **Log rotation**: Automatic file rotation with size limits and backup retention
- **Download context**: Specialized loggers with task-specific context
- **System information**: Automatic system info logging for debugging

Example usage:
```python
from zuup.utils import setup_logging, get_download_logger

# Setup comprehensive logging
setup_logging(
    level="INFO",
    log_file=Path("logs/zuup.log"),
    structured_logging=True,
    rich_console=True
)

# Get download-specific logger
logger = get_download_logger(task_id, engine_type)
logger.log_progress(progress_info, url)
```

### 2. Metrics Collection and Monitoring

- **Real-time metrics**: Download speed, progress, connection counts
- **System monitoring**: CPU, memory, disk usage, network statistics
- **Engine-specific metrics**: HTTP, FTP, Torrent, Media download statistics
- **Historical data**: Configurable history retention with trend analysis
- **Export capabilities**: JSON export for analysis and reporting

Example usage:
```python
from zuup.utils import initialize_monitoring, PerformanceMonitor

# Initialize monitoring
metrics = initialize_monitoring(history_size=1000)

# Start performance monitoring
monitor = PerformanceMonitor(metrics, interval=5.0)
await monitor.start()

# Monitor a download task
metrics.start_task_monitoring(task)
metrics.update_task_progress(task.id, progress_info)
```

### 3. Debugging and Error Reporting

- **Function tracing**: Automatic function call logging with timing
- **Debug sessions**: Structured debugging with session management
- **Error reporting**: Centralized error collection with context
- **Performance profiling**: Operation timing and statistics
- **Task debugging**: Download-specific debugging with event tracking

Example usage:
```python
from zuup.utils import start_debug_session, debug_trace, report_error

# Start debug session
session = start_debug_session("my_debug_session")

# Trace function calls
@debug_trace
def my_function():
    # Function calls are automatically logged
    pass

# Report errors with context
try:
    risky_operation()
except Exception as e:
    report_error(e, "operation_context", task_id="123")
```

### 4. Download-Specific Features

- **Progress tracking**: Detailed progress information with speed calculations
- **Engine adaptation**: Different metrics for HTTP, FTP, Torrent, Media downloads
- **Error handling**: Retry tracking, error categorization, recovery logging
- **Performance analysis**: Speed trends, connection efficiency, resource usage

## Configuration

### Basic Configuration

```python
from zuup.utils import setup_logging

setup_logging(
    level="INFO",                    # Logging level
    log_file=Path("logs/app.log"),  # Log file path
    rich_console=True,              # Use Rich console output
    structured_logging=False,       # Use JSON structured logging
    max_log_size=10*1024*1024,     # 10MB max file size
    backup_count=5                  # Keep 5 backup files
)
```

### Advanced Configuration

See `config_examples/logging_config.json` for a complete configuration example with:
- Multiple log handlers (console, file, error file)
- Monitoring settings (metrics, performance, error reporting)
- Debug configuration (sessions, tracing, profiling)

## Output Examples

### Console Output (Rich)
```
[09/26/25 08:26:07] INFO     Progress: 50.0% (5242880/10485760 bytes) Speed: 1.00 MB/s
                    ERROR    Download error: Connection timeout
                    INFO     Download completed: 10485760 bytes in 10.50s (avg: 0.95 MB/s)
```

### Structured JSON Log
```json
{
  "timestamp": "2025-09-26T08:26:07.123456",
  "level": "INFO",
  "logger": "zuup.engines.http",
  "message": "Progress: 50.0% (5242880/10485760 bytes) Speed: 1.00 MB/s",
  "task_id": "cmg08sxyp0001zosb965vixr4",
  "engine_type": "http",
  "download_speed": 1048576,
  "progress_percentage": 50.0,
  "url": "https://example.com/file.zip"
}
```

### Metrics Export
```json
{
  "export_time": "2025-09-26T08:26:07.123456",
  "download_metrics": {
    "task_123": {
      "task_id": "task_123",
      "engine_type": "http",
      "completion_percentage": 100.0,
      "average_download_speed": 2097152,
      "peak_download_speed": 5242880,
      "duration_seconds": 45.2
    }
  },
  "system_metrics_history": [...]
}
```

## Testing and Validation

### Manual Testing
The examples include comprehensive manual tests that verify:
- All logging configurations work correctly
- Metrics are collected accurately
- Debugging features capture the right information
- Performance monitoring detects issues
- Error reporting includes proper context

### Interactive Validation
The interactive demo allows you to:
- See real-time logging output
- Monitor simulated downloads
- Explore generated log files
- Test error scenarios
- Validate configuration changes

## Integration

### In Your Download Engine
```python
from zuup.utils import get_download_logger, get_metrics_collector

class MyDownloadEngine:
    def __init__(self, task):
        self.task = task
        self.logger = get_download_logger(task.id, task.engine_type)
        self.metrics = get_metrics_collector()
        
    async def download(self):
        self.metrics.start_task_monitoring(self.task)
        
        try:
            # Download logic here
            progress = ProgressInfo(...)
            self.logger.log_progress(progress, self.task.url)
            self.metrics.update_task_progress(self.task.id, progress)
            
        except Exception as e:
            self.logger.log_error(e, "DOWNLOAD_ERROR", self.task.url)
            self.metrics.update_task_error(self.task.id, str(e))
            raise
        finally:
            self.metrics.complete_task_monitoring(self.task.id, self.task.status)
```

### In Your Application
```python
from zuup.utils import (
    setup_debug_logging, 
    initialize_monitoring, 
    initialize_error_reporting
)

# Initialize all systems
setup_debug_logging(Path("logs"))
metrics = initialize_monitoring()
initialize_error_reporting(Path("logs/errors"))

# Your application code here
```

## Requirements

- Python 3.10+
- psutil (for system monitoring)
- rich (for console output)
- All other dependencies are included in the main project

## Performance Impact

The logging and monitoring system is designed to be lightweight:
- Minimal overhead when logging is disabled
- Efficient structured logging with lazy evaluation
- Configurable detail levels to balance information vs. performance
- Automatic cleanup of old metrics and debug data

## Troubleshooting

### Common Issues

1. **Missing psutil**: Install with `uv add psutil` or `pip install psutil`
2. **Permission errors**: Ensure log directories are writable
3. **Large log files**: Configure rotation with `max_log_size` and `backup_count`
4. **Performance impact**: Reduce logging level or disable debug features

### Debug Information

When reporting issues, include:
- Log files from the `logs/` directory
- Exported metrics from `metrics.export_metrics()`
- Debug session files from debug sessions
- System information from `log_system_info()`

## Contributing

When adding new logging or monitoring features:
1. Add tests to `manual_tests/test_logging_monitoring.py`
2. Update the interactive demo if user-facing
3. Document configuration options
4. Ensure backward compatibility
5. Test performance impact

## License

Same as the main Zuup project (MIT License).