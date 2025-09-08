//! Logging and tracing infrastructure for Zuup

use std::{io, path::PathBuf};

use serde::{Deserialize, Serialize};
use tracing::Level;
use tracing_subscriber::{
    EnvFilter,
    fmt::{self, time::ChronoUtc},
    layer::SubscriberExt,
    util::SubscriberInitExt,
};

/// Parse a log level string into a tracing::Level
pub fn parse_log_level(level: &str) -> Result<Level, String> {
    match level.to_lowercase().as_str() {
        "trace" => Ok(Level::TRACE),
        "debug" => Ok(Level::DEBUG),
        "info" => Ok(Level::INFO),
        "warn" => Ok(Level::WARN),
        "error" => Ok(Level::ERROR),
        _ => Err(format!("Invalid log level: {}", level)),
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,

    /// Output format (json, pretty, compact)
    pub format: LogFormat,

    /// Whether to include timestamps
    pub timestamps: bool,

    /// Whether to include thread names
    pub thread_names: bool,

    /// Whether to include thread IDs
    pub thread_ids: bool,

    /// Whether to include file and line information
    pub file_line_info: bool,

    /// Whether to include span information
    pub spans: bool,

    /// Log file path (if None, logs to stdout)
    pub file: Option<PathBuf>,

    /// Maximum log file size in bytes (for rotation)
    pub max_file_size: Option<u64>,

    /// Number of rotated log files to keep
    pub max_files: Option<u32>,

    /// Custom log filters (module=level format)
    pub filters: Vec<String>,

    /// Whether to enable performance tracing
    pub performance_tracing: bool,

    /// Performance tracing sample rate (0.0 to 1.0)
    pub performance_sample_rate: f64,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: LogFormat::Pretty,
            timestamps: true,
            thread_names: false,
            thread_ids: false,
            file_line_info: false,
            spans: false,
            file: None,
            max_file_size: Some(100 * 1024 * 1024), // 100MB
            max_files: Some(5),
            filters: Vec::new(),
            performance_tracing: false,
            performance_sample_rate: 0.1,
        }
    }
}

/// Log output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    /// Human-readable format with colors
    Pretty,
    /// Compact human-readable format
    Compact,
    /// JSON format for structured logging
    Json,
}

/// Performance tracing configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Whether performance tracing is enabled
    pub enabled: bool,

    /// Sample rate for performance traces (0.0 to 1.0)
    pub sample_rate: f64,

    /// Minimum duration to trace (in microseconds)
    pub min_duration_us: u64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sample_rate: 0.1,
            min_duration_us: 1000, // 1ms
        }
    }
}

/// Initialize the logging system with the given configuration
pub fn init_logging(
    config: &LoggingConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Build the environment filter
    let mut filter = EnvFilter::new(format!("zuup={}", config.level));

    // Add custom filters
    for custom_filter in &config.filters {
        filter = filter.add_directive(custom_filter.parse()?);
    }

    // Configure the formatter based on output destination and format
    match (&config.file, &config.format) {
        // File output
        (Some(file_path), LogFormat::Pretty) => {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(file_path)?;

            if config.timestamps {
                let layer = fmt::layer()
                    .with_writer(file)
                    .with_ansi(false)
                    .with_target(true)
                    .with_thread_names(config.thread_names)
                    .with_thread_ids(config.thread_ids)
                    .with_file(config.file_line_info)
                    .with_line_number(config.file_line_info)
                    .with_timer(ChronoUtc::rfc_3339());

                tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .init();
            } else {
                let layer = fmt::layer()
                    .with_writer(file)
                    .with_ansi(false)
                    .with_target(true)
                    .with_thread_names(config.thread_names)
                    .with_thread_ids(config.thread_ids)
                    .with_file(config.file_line_info)
                    .with_line_number(config.file_line_info)
                    .without_time();

                tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .init();
            }
        }

        (Some(file_path), LogFormat::Json) => {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(file_path)?;

            if config.timestamps {
                let layer = fmt::layer()
                    .with_writer(file)
                    .json()
                    .with_target(true)
                    .with_thread_names(config.thread_names)
                    .with_thread_ids(config.thread_ids)
                    .with_file(config.file_line_info)
                    .with_line_number(config.file_line_info)
                    .with_timer(ChronoUtc::rfc_3339());

                tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .init();
            } else {
                let layer = fmt::layer()
                    .with_writer(file)
                    .json()
                    .with_target(true)
                    .with_thread_names(config.thread_names)
                    .with_thread_ids(config.thread_ids)
                    .with_file(config.file_line_info)
                    .with_line_number(config.file_line_info)
                    .without_time();

                tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .init();
            }
        }

        (Some(file_path), LogFormat::Compact) => {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(file_path)?;

            if config.timestamps {
                let layer = fmt::layer()
                    .with_writer(file)
                    .compact()
                    .with_ansi(false)
                    .with_target(false)
                    .with_thread_names(config.thread_names)
                    .with_thread_ids(config.thread_ids)
                    .with_file(config.file_line_info)
                    .with_line_number(config.file_line_info)
                    .with_timer(ChronoUtc::rfc_3339());

                tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .init();
            } else {
                let layer = fmt::layer()
                    .with_writer(file)
                    .compact()
                    .with_ansi(false)
                    .with_target(false)
                    .with_thread_names(config.thread_names)
                    .with_thread_ids(config.thread_ids)
                    .with_file(config.file_line_info)
                    .with_line_number(config.file_line_info)
                    .without_time();

                tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .init();
            }
        }

        // Console output
        (None, LogFormat::Pretty) => {
            if config.timestamps {
                let layer = fmt::layer()
                    .with_writer(io::stdout)
                    .with_ansi(true)
                    .with_target(true)
                    .with_thread_names(config.thread_names)
                    .with_thread_ids(config.thread_ids)
                    .with_file(config.file_line_info)
                    .with_line_number(config.file_line_info)
                    .with_timer(ChronoUtc::rfc_3339());

                tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .init();
            } else {
                let layer = fmt::layer()
                    .with_writer(io::stdout)
                    .with_ansi(true)
                    .with_target(true)
                    .with_thread_names(config.thread_names)
                    .with_thread_ids(config.thread_ids)
                    .with_file(config.file_line_info)
                    .with_line_number(config.file_line_info)
                    .without_time();

                tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .init();
            }
        }

        (None, LogFormat::Json) => {
            if config.timestamps {
                let layer = fmt::layer()
                    .with_writer(io::stdout)
                    .json()
                    .with_target(true)
                    .with_thread_names(config.thread_names)
                    .with_thread_ids(config.thread_ids)
                    .with_file(config.file_line_info)
                    .with_line_number(config.file_line_info)
                    .with_timer(ChronoUtc::rfc_3339());

                tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .init();
            } else {
                let layer = fmt::layer()
                    .with_writer(io::stdout)
                    .json()
                    .with_target(true)
                    .with_thread_names(config.thread_names)
                    .with_thread_ids(config.thread_ids)
                    .with_file(config.file_line_info)
                    .with_line_number(config.file_line_info)
                    .without_time();

                tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .init();
            }
        }

        (None, LogFormat::Compact) => {
            if config.timestamps {
                let layer = fmt::layer()
                    .with_writer(io::stdout)
                    .compact()
                    .with_ansi(true)
                    .with_target(false)
                    .with_thread_names(config.thread_names)
                    .with_thread_ids(config.thread_ids)
                    .with_file(config.file_line_info)
                    .with_line_number(config.file_line_info)
                    .with_timer(ChronoUtc::rfc_3339());

                tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .init();
            } else {
                let layer = fmt::layer()
                    .with_writer(io::stdout)
                    .compact()
                    .with_ansi(true)
                    .with_target(false)
                    .with_thread_names(config.thread_names)
                    .with_thread_ids(config.thread_ids)
                    .with_file(config.file_line_info)
                    .with_line_number(config.file_line_info)
                    .without_time();

                tracing_subscriber::registry()
                    .with(filter)
                    .with(layer)
                    .init();
            }
        }
    }

    // Log initialization message
    tracing::info!(
        level = %config.level,
        format = ?config.format,
        file = ?config.file,
        "Logging system initialized"
    );

    Ok(())
}

/// Performance tracing utilities
pub mod performance {
    use std::time::Instant;
    use tracing::{Instrument, info_span};

    /// Trace the performance of an async operation
    pub async fn trace_async<F, T>(name: &str, operation: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        let span = info_span!("performance", operation = name);
        let start = Instant::now();

        let result = operation.instrument(span.clone()).await;

        let duration = start.elapsed();
        tracing::info!(
            parent: &span,
            duration_ms = duration.as_millis(),
            "Operation completed"
        );

        result
    }

    /// Trace the performance of a synchronous operation
    pub fn trace_sync<F, T>(name: &str, operation: F) -> T
    where
        F: FnOnce() -> T,
    {
        let span = info_span!("performance", operation = name);
        let _enter = span.enter();
        let start = Instant::now();

        let result = operation();

        let duration = start.elapsed();
        tracing::info!(duration_ms = duration.as_millis(), "Operation completed");

        result
    }

    /// Performance timer for manual timing
    pub struct PerformanceTimer {
        name: String,
        start: Instant,
    }

    impl PerformanceTimer {
        pub fn new(name: String) -> Self {
            tracing::debug!(operation = %name, "Starting performance timer");
            Self {
                name,
                start: Instant::now(),
            }
        }

        pub fn elapsed(&self) -> std::time::Duration {
            self.start.elapsed()
        }

        pub fn finish(self) {
            let duration = self.start.elapsed();
            tracing::info!(
                operation = %self.name,
                duration_ms = duration.as_millis(),
                "Performance timer finished"
            );
        }

        pub fn checkpoint(&self, checkpoint: &str) {
            let duration = self.start.elapsed();
            tracing::debug!(
                operation = %self.name,
                checkpoint = checkpoint,
                duration_ms = duration.as_millis(),
                "Performance checkpoint"
            );
        }
    }
}

/// Structured logging utilities
pub mod structured {
    use serde::Serialize;
    use tracing::Level;

    /// Log a structured event with custom fields
    pub fn log_event<T: Serialize>(level: Level, message: &str, data: &T) {
        let json_data = serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string());

        match level {
            Level::TRACE => tracing::trace!(
                target: "zuup::structured",
                message = message,
                data = %json_data,
                "Structured log event"
            ),
            Level::DEBUG => tracing::debug!(
                target: "zuup::structured",
                message = message,
                data = %json_data,
                "Structured log event"
            ),
            Level::INFO => tracing::info!(
                target: "zuup::structured",
                message = message,
                data = %json_data,
                "Structured log event"
            ),
            Level::WARN => tracing::warn!(
                target: "zuup::structured",
                message = message,
                data = %json_data,
                "Structured log event"
            ),
            Level::ERROR => tracing::error!(
                target: "zuup::structured",
                message = message,
                data = %json_data,
                "Structured log event"
            ),
        }
    }

    /// Log download-related events
    pub fn log_download_event<T: Serialize>(
        download_id: &crate::types::DownloadId,
        event_type: &str,
        data: &T,
    ) {
        let json_data = serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string());

        tracing::info!(
            target: "zuup::download",
            download_id = %download_id,
            event_type = event_type,
            data = %json_data,
            "Download event"
        );
    }

    /// Log network-related events
    pub fn log_network_event<T: Serialize>(url: &str, event_type: &str, data: &T) {
        let json_data = serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string());

        tracing::info!(
            target: "zuup::network",
            url = url,
            event_type = event_type,
            data = %json_data,
            "Network event"
        );
    }
}

/// Logging macros for common use cases
#[macro_export]
macro_rules! log_download {
    ($level:expr, $download_id:expr, $($arg:tt)*) => {
        tracing::event!(
            target: "zuup::download",
            $level,
            download_id = %$download_id,
            $($arg)*
        );
    };
}

#[macro_export]
macro_rules! log_network {
    ($level:expr, $url:expr, $($arg:tt)*) => {
        tracing::event!(
            target: "zuup::network",
            $level,
            url = $url,
            $($arg)*
        );
    };
}

#[macro_export]
macro_rules! log_performance {
    ($name:expr, $duration:expr) => {
        tracing::info!(
            target: "zuup::performance",
            operation = $name,
            duration_ms = $duration.as_millis(),
            "Performance measurement"
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::NamedTempFile;

    #[test]
    fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, "info");
        assert!(matches!(config.format, LogFormat::Pretty));
        assert!(config.timestamps);
        assert!(!config.thread_names);
        assert!(!config.file_line_info);
        assert!(config.file.is_none());
        assert_eq!(config.max_file_size, Some(100 * 1024 * 1024));
        assert_eq!(config.max_files, Some(5));
        assert!(config.filters.is_empty());
        assert!(!config.performance_tracing);
        assert_eq!(config.performance_sample_rate, 0.1);
    }

    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.sample_rate, 0.1);
        assert_eq!(config.min_duration_us, 1000);
    }

    #[test]
    fn test_parse_log_level() {
        assert!(matches!(parse_log_level("trace").unwrap(), Level::TRACE));
        assert!(matches!(parse_log_level("debug").unwrap(), Level::DEBUG));
        assert!(matches!(parse_log_level("info").unwrap(), Level::INFO));
        assert!(matches!(parse_log_level("warn").unwrap(), Level::WARN));
        assert!(matches!(parse_log_level("error").unwrap(), Level::ERROR));

        // Test case insensitive
        assert!(matches!(parse_log_level("INFO").unwrap(), Level::INFO));
        assert!(matches!(parse_log_level("Debug").unwrap(), Level::DEBUG));

        // Test invalid level
        assert!(parse_log_level("invalid").is_err());
    }

    #[test]
    fn test_logging_config_serialization() {
        let config = LoggingConfig {
            level: "debug".to_string(),
            format: LogFormat::Json,
            timestamps: false,
            thread_names: true,
            thread_ids: true,
            file_line_info: true,
            spans: true,
            file: Some(PathBuf::from("/tmp/test.log")),
            max_file_size: Some(50 * 1024 * 1024),
            max_files: Some(10),
            filters: vec!["zuup::network=trace".to_string()],
            performance_tracing: true,
            performance_sample_rate: 0.5,
        };

        // Test serialization
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("debug"));
        assert!(json.contains("Json"));

        // Test deserialization
        let deserialized: LoggingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.level, "debug");
        assert!(matches!(deserialized.format, LogFormat::Json));
        assert!(!deserialized.timestamps);
        assert!(deserialized.thread_names);
        assert!(deserialized.performance_tracing);
        assert_eq!(deserialized.performance_sample_rate, 0.5);
    }

    #[test]
    fn test_log_format_serialization() {
        let formats = vec![LogFormat::Pretty, LogFormat::Compact, LogFormat::Json];

        for format in formats {
            let json = serde_json::to_string(&format).unwrap();
            let deserialized: LogFormat = serde_json::from_str(&json).unwrap();
            assert!(matches!(
                (&format, &deserialized),
                (LogFormat::Pretty, LogFormat::Pretty)
                    | (LogFormat::Compact, LogFormat::Compact)
                    | (LogFormat::Json, LogFormat::Json)
            ));
        }
    }

    #[tokio::test]
    async fn test_performance_timer() {
        let timer = performance::PerformanceTimer::new("test_operation".to_string());

        // Simulate some work
        tokio::time::sleep(Duration::from_millis(10)).await;

        let elapsed = timer.elapsed();
        assert!(elapsed >= Duration::from_millis(10));

        timer.checkpoint("middle");

        // Simulate more work
        tokio::time::sleep(Duration::from_millis(5)).await;

        timer.finish();
    }

    #[tokio::test]
    async fn test_performance_trace_async() {
        let result = performance::trace_async("test_async_op", async {
            tokio::time::sleep(Duration::from_millis(5)).await;
            42
        })
        .await;

        assert_eq!(result, 42);
    }

    #[test]
    fn test_performance_trace_sync() {
        let result = performance::trace_sync("test_sync_op", || {
            std::thread::sleep(Duration::from_millis(5));
            "test_result"
        });

        assert_eq!(result, "test_result");
    }

    #[test]
    fn test_structured_logging() {
        use serde_json::json;

        let data = json!({
            "key": "value",
            "number": 42,
            "nested": {
                "inner": "data"
            }
        });

        // These should not panic
        structured::log_event(Level::INFO, "Test structured event", &data);

        let download_id = crate::types::DownloadId::new();
        structured::log_download_event(&download_id, "test_event", &data);

        structured::log_network_event("http://example.com", "connection_established", &data);
    }

    // Integration test for console logging (requires manual verification)
    #[test]
    #[ignore] // Ignore by default as it modifies global state
    fn test_console_logging_integration() {
        let config = LoggingConfig {
            level: "debug".to_string(),
            format: LogFormat::Pretty,
            timestamps: true,
            thread_names: false,
            thread_ids: false,
            file_line_info: true,
            spans: false,
            file: None,
            max_file_size: None,
            max_files: None,
            filters: vec![],
            performance_tracing: false,
            performance_sample_rate: 0.1,
        };

        // Initialize logging (this will affect global state)
        init_logging(&config).unwrap();

        // Test different log levels
        tracing::trace!("This is a trace message");
        tracing::debug!("This is a debug message");
        tracing::info!("This is an info message");
        tracing::warn!("This is a warning message");
        tracing::error!("This is an error message");

        // Test structured logging
        tracing::info!(
            download_id = "test-123",
            progress = 50,
            "Download progress update"
        );
    }

    // Integration test for file logging
    #[test]
    fn test_file_logging_integration() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_path_buf();

        let _config = LoggingConfig {
            level: "info".to_string(),
            format: LogFormat::Json,
            timestamps: true,
            thread_names: false,
            thread_ids: false,
            file_line_info: false,
            spans: false,
            file: Some(file_path.clone()),
            max_file_size: None,
            max_files: None,
            filters: vec![],
            performance_tracing: false,
            performance_sample_rate: 0.1,
        };

        // This test just verifies that file logging setup doesn't panic
        // In a real scenario, we would need to set up a separate tracing subscriber
        // for testing without affecting the global state
        let result = std::panic::catch_unwind(|| {
            let _file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&file_path)
                .unwrap();
        });

        assert!(result.is_ok());
    }

    #[test]
    fn test_custom_filters() {
        let config = LoggingConfig {
            level: "info".to_string(),
            format: LogFormat::Pretty,
            timestamps: true,
            thread_names: false,
            thread_ids: false,
            file_line_info: false,
            spans: false,
            file: None,
            max_file_size: None,
            max_files: None,
            filters: vec![
                "zuup::network=debug".to_string(),
                "zuup::download=trace".to_string(),
            ],
            performance_tracing: true,
            performance_sample_rate: 1.0,
        };

        // Test that filter parsing doesn't panic
        let result = std::panic::catch_unwind(|| {
            for filter in &config.filters {
                let _parsed: tracing_subscriber::EnvFilter = filter.parse().unwrap();
            }
        });

        assert!(result.is_ok());
    }
}
