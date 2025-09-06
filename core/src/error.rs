//! Error types and handling for Zuup

use std::fmt;
use std::time::Duration;

use thiserror::Error;

use crate::types::{DownloadId, DownloadState};

/// Main error type for Zuup operations
#[derive(Debug, Error)]
pub enum ZuupError {
    #[error("Network error: {0}")]
    Network(#[from] NetworkError),

    #[error("Protocol error: {0}")]
    Protocol(#[from] ProtocolError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Download not found: {0}")]
    DownloadNotFound(DownloadId),

    #[error("Invalid state transition: {from:?} -> {to:?}")]
    InvalidStateTransition {
        from: DownloadState,
        to: DownloadState,
    },

    #[error("Checksum verification failed")]
    ChecksumMismatch,

    #[error("Authentication failed")]
    AuthenticationFailed,

    #[error("Permission denied")]
    PermissionDenied,

    #[error("Disk space insufficient")]
    InsufficientDiskSpace,

    #[error("Session error: {0}")]
    Session(String),

    #[error("Event system error: {0}")]
    Event(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("URL parse error: {0}")]
    UrlParse(#[from] url::ParseError),

    #[error("Task join error: {0}")]
    TaskJoin(#[from] tokio::task::JoinError),

    #[error("Channel send error")]
    ChannelSend,

    #[error("Channel receive error")]
    ChannelReceive,

    #[error("Timeout error")]
    Timeout,

    #[error("Cancelled")]
    Cancelled,

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("File exists: {0}")]
    FileExists(std::path::PathBuf),

    #[error("Invalid path: {0}")]
    InvalidPath(std::path::PathBuf),

    #[error("Too many conflicts for file: {0}")]
    TooManyConflicts(std::path::PathBuf),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Media download error: {0}")]
    MediaDownload(#[from] MediaError),
}

/// Media download specific errors
#[derive(Debug, Error, Clone)]
pub enum MediaError {
    #[error("yt-dlp not found or not executable")]
    YtDlpNotFound,

    #[error("yt-dlp execution failed: {0}")]
    YtDlpExecutionFailed(String),

    #[error("Invalid media format: {0}")]
    InvalidMediaFormat(String),

    #[error("Media extraction failed: {0}")]
    ExtractionFailed(String),

    #[error("Unsupported media URL: {0}")]
    UnsupportedUrl(String),

    #[error("Format not available: {0}")]
    FormatNotAvailable(String),

    #[error("Playlist processing failed: {0}")]
    PlaylistFailed(String),
}

/// Network-specific errors
#[derive(Debug, Error, Clone)]
pub enum NetworkError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("DNS resolution failed: {0}")]
    DnsResolutionFailed(String),

    #[error("TLS error: {0}")]
    Tls(String),

    #[error("Proxy error: {0}")]
    Proxy(String),

    #[error("Request timeout")]
    Timeout,

    #[error("Too many redirects")]
    TooManyRedirects,

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Connection reset")]
    ConnectionReset,

    #[error("Connection refused")]
    ConnectionRefused,

    #[error("Host unreachable")]
    HostUnreachable,
}

/// Protocol-specific errors
#[derive(Debug, Error, Clone)]
pub enum ProtocolError {
    #[error("Unsupported protocol: {0}")]
    UnsupportedProtocol(String),

    #[error("HTTP error: {status} - {message}")]
    Http { status: u16, message: String },

    #[error("FTP error: {0}")]
    Ftp(String),

    #[error("BitTorrent error: {0}")]
    BitTorrent(String),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Authentication required")]
    AuthenticationRequired,

    #[error("Server error: {0}")]
    Server(String),

    #[error("Protocol version not supported")]
    VersionNotSupported,

    #[error("Invalid response format")]
    InvalidResponseFormat,

    #[error("Range not satisfiable")]
    RangeNotSatisfiable,

    #[error("Initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Not initialized: {0}")]
    NotInitialized(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Unsupported URL: {0}")]
    UnsupportedUrl(String),

    #[error("Download failed: {0}")]
    DownloadFailed(String),

    #[error("Operation failed: {0}")]
    OperationFailed(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Media extraction error: {0}")]
    MediaExtraction(String),

    #[error("yt-dlp not found or not executable")]
    YtDlpNotFound,

    #[error("yt-dlp execution failed: {0}")]
    YtDlpExecutionFailed(String),

    #[error("Invalid media format: {0}")]
    InvalidMediaFormat(String),
}

/// Result type alias for Zuup operations
pub type Result<T> = std::result::Result<T, ZuupError>;

impl ZuupError {
    /// Check if the error is recoverable (can be retried)
    pub fn is_recoverable(&self) -> bool {
        match self {
            ZuupError::Network(NetworkError::Timeout) => true,
            ZuupError::Network(NetworkError::ConnectionReset) => true,
            ZuupError::Network(NetworkError::ConnectionRefused) => true,
            ZuupError::Protocol(ProtocolError::Http { status, .. }) => {
                // 5xx errors are generally recoverable
                *status >= 500 && *status < 600
            }
            ZuupError::Io(_) => true,
            ZuupError::Timeout => true,
            _ => false,
        }
    }

    /// Check if the error is a network-related error
    pub fn is_network_error(&self) -> bool {
        matches!(self, ZuupError::Network(_))
    }

    /// Check if the error is a protocol-related error
    pub fn is_protocol_error(&self) -> bool {
        matches!(self, ZuupError::Protocol(_))
    }
}

/// Error severity levels for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - operation can continue
    Low,
    /// Medium severity - operation should be retried
    Medium,
    /// High severity - operation should be aborted
    High,
    /// Critical severity - system should be shut down
    Critical,
}

/// Error category for grouping related errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Network connectivity issues
    Network,
    /// Protocol-specific errors
    Protocol,
    /// File system and I/O errors
    FileSystem,
    /// Configuration and validation errors
    Configuration,
    /// Authentication and authorization errors
    Security,
    /// Resource management errors (memory, disk space, etc.)
    Resource,
    /// Internal system errors
    System,
    /// User input validation errors
    UserInput,
}

/// Error context providing additional debugging information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that was being performed when error occurred
    pub operation: String,
    /// Download ID if applicable
    pub download_id: Option<DownloadId>,
    /// URL being processed if applicable
    pub url: Option<String>,
    /// File path if applicable
    pub file_path: Option<String>,
    /// Additional context data
    pub metadata: std::collections::HashMap<String, String>,
    /// Timestamp when error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Stack trace if available
    pub stack_trace: Option<String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            download_id: None,
            url: None,
            file_path: None,
            metadata: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            stack_trace: None,
        }
    }

    /// Add download ID to context
    pub fn with_download_id(mut self, id: DownloadId) -> Self {
        self.download_id = Some(id);
        self
    }

    /// Add URL to context
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Add file path to context
    pub fn with_file_path(mut self, path: impl Into<String>) -> Self {
        self.file_path = Some(path.into());
        self
    }

    /// Add metadata to context
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add stack trace to context
    pub fn with_stack_trace(mut self, trace: impl Into<String>) -> Self {
        self.stack_trace = Some(trace.into());
        self
    }
}

/// Recovery strategy for handling errors
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// No recovery possible, fail immediately
    None,
    /// Retry the operation with exponential backoff
    Retry {
        max_attempts: u32,
        initial_delay: Duration,
        max_delay: Duration,
        backoff_multiplier: f64,
    },
    /// Fallback to alternative method or resource
    Fallback { alternatives: Vec<String> },
    /// Skip the current operation and continue
    Skip,
    /// Reset state and restart operation
    Reset,
    /// Switch to degraded mode
    Degrade,
}

/// Enhanced error with context and recovery information
#[derive(Debug)]
pub struct EnhancedError {
    /// The underlying error
    pub error: ZuupError,
    /// Error context with debugging information
    pub context: ErrorContext,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Error category
    pub category: ErrorCategory,
    /// Suggested recovery strategy
    pub recovery_strategy: RecoveryStrategy,
    /// Chain of related errors
    pub error_chain: Vec<ZuupError>,
}

impl EnhancedError {
    /// Create a new enhanced error
    pub fn new(error: ZuupError, context: ErrorContext) -> Self {
        let severity = error.severity();
        let category = error.category();
        let recovery_strategy = error.recovery_strategy();

        Self {
            error,
            context,
            severity,
            category,
            recovery_strategy,
            error_chain: Vec::new(),
        }
    }

    /// Add an error to the chain
    pub fn with_chain_error(mut self, error: ZuupError) -> Self {
        self.error_chain.push(error);
        self
    }

    /// Check if the error should be retried
    pub fn should_retry(&self) -> bool {
        matches!(self.recovery_strategy, RecoveryStrategy::Retry { .. })
    }

    /// Get retry parameters if applicable
    pub fn retry_params(&self) -> Option<(u32, Duration, Duration, f64)> {
        match &self.recovery_strategy {
            RecoveryStrategy::Retry {
                max_attempts,
                initial_delay,
                max_delay,
                backoff_multiplier,
            } => Some((
                *max_attempts,
                *initial_delay,
                *max_delay,
                *backoff_multiplier,
            )),
            _ => None,
        }
    }

    /// Get fallback alternatives if applicable
    pub fn fallback_alternatives(&self) -> Option<&[String]> {
        match &self.recovery_strategy {
            RecoveryStrategy::Fallback { alternatives } => Some(alternatives),
            _ => None,
        }
    }
}

impl fmt::Display for EnhancedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}] {}", self.severity, self.error)?;

        if let Some(download_id) = &self.context.download_id {
            write!(f, " (Download: {})", download_id)?;
        }

        if let Some(url) = &self.context.url {
            write!(f, " (URL: {})", url)?;
        }

        if !self.error_chain.is_empty() {
            write!(f, " | Chain: ")?;
            for (i, err) in self.error_chain.iter().enumerate() {
                if i > 0 {
                    write!(f, " -> ")?;
                }
                write!(f, "{}", err)?;
            }
        }

        Ok(())
    }
}

impl std::error::Error for EnhancedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl ZuupError {
    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ZuupError::Network(NetworkError::Timeout) => ErrorSeverity::Medium,
            ZuupError::Network(NetworkError::ConnectionReset) => ErrorSeverity::Medium,
            ZuupError::Network(NetworkError::ConnectionRefused) => ErrorSeverity::Medium,
            ZuupError::Network(NetworkError::DnsResolutionFailed(_)) => ErrorSeverity::High,
            ZuupError::Network(NetworkError::TooManyRedirects) => ErrorSeverity::High,
            ZuupError::Network(_) => ErrorSeverity::Medium,

            ZuupError::Protocol(ProtocolError::Http { status, .. }) => {
                match *status {
                    400..=499 => ErrorSeverity::High,   // Client errors
                    500..=599 => ErrorSeverity::Medium, // Server errors (retryable)
                    _ => ErrorSeverity::Medium,
                }
            }
            ZuupError::Protocol(ProtocolError::UnsupportedProtocol(_)) => ErrorSeverity::High,
            ZuupError::Protocol(ProtocolError::AuthenticationRequired) => ErrorSeverity::High,
            ZuupError::Protocol(_) => ErrorSeverity::Medium,

            ZuupError::Io(_) => ErrorSeverity::Medium,
            ZuupError::Config(_) => ErrorSeverity::High,
            ZuupError::DownloadNotFound(_) => ErrorSeverity::Medium,
            ZuupError::InvalidStateTransition { .. } => ErrorSeverity::Medium,
            ZuupError::ChecksumMismatch => ErrorSeverity::High,
            ZuupError::AuthenticationFailed => ErrorSeverity::High,
            ZuupError::PermissionDenied => ErrorSeverity::High,
            ZuupError::InsufficientDiskSpace => ErrorSeverity::Critical,
            ZuupError::Session(_) => ErrorSeverity::Medium,
            ZuupError::Event(_) => ErrorSeverity::Low,
            ZuupError::Serialization(_) => ErrorSeverity::Medium,
            ZuupError::UrlParse(_) => ErrorSeverity::High,
            ZuupError::TaskJoin(_) => ErrorSeverity::Medium,
            ZuupError::ChannelSend => ErrorSeverity::Medium,
            ZuupError::ChannelReceive => ErrorSeverity::Medium,
            ZuupError::Timeout => ErrorSeverity::Medium,
            ZuupError::Cancelled => ErrorSeverity::Low,
            ZuupError::Internal(_) => ErrorSeverity::Critical,
            ZuupError::InvalidInput(_) => ErrorSeverity::High,
            ZuupError::FileExists(_) => ErrorSeverity::Medium,
            ZuupError::InvalidPath(_) => ErrorSeverity::High,
            ZuupError::TooManyConflicts(_) => ErrorSeverity::High,
            ZuupError::InvalidUrl(_) => ErrorSeverity::High,
            ZuupError::MediaDownload(MediaError::YtDlpNotFound) => ErrorSeverity::Critical,
            ZuupError::MediaDownload(MediaError::YtDlpExecutionFailed(_)) => ErrorSeverity::Medium,
            ZuupError::MediaDownload(MediaError::ExtractionFailed(_)) => ErrorSeverity::Medium,
            ZuupError::MediaDownload(MediaError::UnsupportedUrl(_)) => ErrorSeverity::High,
            ZuupError::MediaDownload(_) => ErrorSeverity::Medium,
        }
    }

    /// Get the category of this error
    pub fn category(&self) -> ErrorCategory {
        match self {
            ZuupError::Network(_) => ErrorCategory::Network,
            ZuupError::Protocol(_) => ErrorCategory::Protocol,
            ZuupError::Io(_) => ErrorCategory::FileSystem,
            ZuupError::Config(_) => ErrorCategory::Configuration,
            ZuupError::AuthenticationFailed => ErrorCategory::Security,
            ZuupError::PermissionDenied => ErrorCategory::Security,
            ZuupError::InsufficientDiskSpace => ErrorCategory::Resource,
            ZuupError::DownloadNotFound(_) => ErrorCategory::UserInput,
            ZuupError::InvalidStateTransition { .. } => ErrorCategory::System,
            ZuupError::ChecksumMismatch => ErrorCategory::FileSystem,
            ZuupError::Session(_) => ErrorCategory::System,
            ZuupError::Event(_) => ErrorCategory::System,
            ZuupError::Serialization(_) => ErrorCategory::System,
            ZuupError::UrlParse(_) => ErrorCategory::UserInput,
            ZuupError::TaskJoin(_) => ErrorCategory::System,
            ZuupError::ChannelSend => ErrorCategory::System,
            ZuupError::ChannelReceive => ErrorCategory::System,
            ZuupError::Timeout => ErrorCategory::Network,
            ZuupError::Cancelled => ErrorCategory::System,
            ZuupError::Internal(_) => ErrorCategory::System,
            ZuupError::InvalidInput(_) => ErrorCategory::UserInput,
            ZuupError::FileExists(_) => ErrorCategory::FileSystem,
            ZuupError::InvalidPath(_) => ErrorCategory::FileSystem,
            ZuupError::TooManyConflicts(_) => ErrorCategory::FileSystem,
            ZuupError::InvalidUrl(_) => ErrorCategory::UserInput,
            ZuupError::MediaDownload(_) => ErrorCategory::Protocol,
        }
    }

    /// Get the suggested recovery strategy for this error
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            ZuupError::Network(NetworkError::Timeout) => RecoveryStrategy::Retry {
                max_attempts: 3,
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
            },
            ZuupError::Network(NetworkError::ConnectionReset) => RecoveryStrategy::Retry {
                max_attempts: 5,
                initial_delay: Duration::from_millis(500),
                max_delay: Duration::from_secs(10),
                backoff_multiplier: 1.5,
            },
            ZuupError::Network(NetworkError::ConnectionRefused) => RecoveryStrategy::Retry {
                max_attempts: 3,
                initial_delay: Duration::from_secs(2),
                max_delay: Duration::from_secs(60),
                backoff_multiplier: 2.0,
            },
            ZuupError::Network(NetworkError::DnsResolutionFailed(_)) => RecoveryStrategy::Retry {
                max_attempts: 2,
                initial_delay: Duration::from_secs(5),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
            },

            ZuupError::Protocol(ProtocolError::Http { status, .. }) => {
                match *status {
                    429 => RecoveryStrategy::Retry {
                        // Too Many Requests
                        max_attempts: 3,
                        initial_delay: Duration::from_secs(10),
                        max_delay: Duration::from_secs(300),
                        backoff_multiplier: 3.0,
                    },
                    500..=599 => RecoveryStrategy::Retry {
                        // Server errors
                        max_attempts: 3,
                        initial_delay: Duration::from_secs(2),
                        max_delay: Duration::from_secs(60),
                        backoff_multiplier: 2.0,
                    },
                    404 => RecoveryStrategy::Fallback {
                        alternatives: vec!["mirror".to_string(), "alternative_url".to_string()],
                    },
                    _ => RecoveryStrategy::None,
                }
            }

            ZuupError::Io(_) => RecoveryStrategy::Retry {
                max_attempts: 2,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(5),
                backoff_multiplier: 2.0,
            },

            ZuupError::ChecksumMismatch => RecoveryStrategy::Retry {
                max_attempts: 2,
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(10),
                backoff_multiplier: 2.0,
            },

            ZuupError::Timeout => RecoveryStrategy::Retry {
                max_attempts: 3,
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
            },

            ZuupError::InvalidStateTransition { .. } => RecoveryStrategy::Reset,
            ZuupError::Event(_) => RecoveryStrategy::Skip,
            ZuupError::Cancelled => RecoveryStrategy::None,

            ZuupError::MediaDownload(MediaError::YtDlpExecutionFailed(_)) => {
                RecoveryStrategy::Retry {
                    max_attempts: 2,
                    initial_delay: Duration::from_secs(2),
                    max_delay: Duration::from_secs(30),
                    backoff_multiplier: 2.0,
                }
            }

            ZuupError::MediaDownload(MediaError::ExtractionFailed(_)) => RecoveryStrategy::Retry {
                max_attempts: 2,
                initial_delay: Duration::from_secs(5),
                max_delay: Duration::from_secs(60),
                backoff_multiplier: 2.0,
            },

            _ => RecoveryStrategy::None,
        }
    }

    /// Create an enhanced error with context
    pub fn with_context(self, context: ErrorContext) -> EnhancedError {
        EnhancedError::new(self, context)
    }

    /// Create error context for this error
    pub fn create_context(&self, operation: impl Into<String>) -> ErrorContext {
        ErrorContext::new(operation)
    }
}

/// Trait for converting errors to enhanced errors with context
pub trait ErrorExt<T> {
    /// Add context to an error result
    fn with_context(self, context: ErrorContext) -> std::result::Result<T, Box<EnhancedError>>;

    /// Add context using a closure
    fn with_context_fn<F>(self, f: F) -> std::result::Result<T, Box<EnhancedError>>
    where
        F: FnOnce() -> ErrorContext;
}

impl<T, E> ErrorExt<T> for std::result::Result<T, E>
where
    E: Into<ZuupError>,
{
    fn with_context(self, context: ErrorContext) -> std::result::Result<T, Box<EnhancedError>> {
        self.map_err(|e| Box::new(e.into().with_context(context)))
    }

    fn with_context_fn<F>(self, f: F) -> std::result::Result<T, Box<EnhancedError>>
    where
        F: FnOnce() -> ErrorContext,
    {
        self.map_err(|e| Box::new(e.into().with_context(f())))
    }
}

impl Clone for ZuupError {
    fn clone(&self) -> Self {
        match self {
            ZuupError::Network(e) => ZuupError::Network(e.clone()),
            ZuupError::Protocol(e) => ZuupError::Protocol(e.clone()),
            ZuupError::Io(e) => ZuupError::Io(std::io::Error::new(e.kind(), e.to_string())),
            ZuupError::Config(s) => ZuupError::Config(s.clone()),
            ZuupError::DownloadNotFound(id) => ZuupError::DownloadNotFound(id.clone()),
            ZuupError::InvalidStateTransition { from, to } => ZuupError::InvalidStateTransition {
                from: from.clone(),
                to: to.clone(),
            },
            ZuupError::ChecksumMismatch => ZuupError::ChecksumMismatch,
            ZuupError::AuthenticationFailed => ZuupError::AuthenticationFailed,
            ZuupError::PermissionDenied => ZuupError::PermissionDenied,
            ZuupError::InsufficientDiskSpace => ZuupError::InsufficientDiskSpace,
            ZuupError::Session(s) => ZuupError::Session(s.clone()),
            ZuupError::Event(s) => ZuupError::Event(s.clone()),
            ZuupError::Serialization(_) => {
                ZuupError::Internal("Serialization error (cloned)".to_string())
            }
            ZuupError::UrlParse(e) => ZuupError::UrlParse(*e),
            ZuupError::TaskJoin(_) => ZuupError::Internal("Task join error (cloned)".to_string()),
            ZuupError::ChannelSend => ZuupError::ChannelSend,
            ZuupError::ChannelReceive => ZuupError::ChannelReceive,
            ZuupError::Timeout => ZuupError::Timeout,
            ZuupError::Cancelled => ZuupError::Cancelled,
            ZuupError::Internal(s) => ZuupError::Internal(s.clone()),
            ZuupError::InvalidInput(s) => ZuupError::InvalidInput(s.clone()),
            ZuupError::FileExists(p) => ZuupError::FileExists(p.clone()),
            ZuupError::InvalidPath(p) => ZuupError::InvalidPath(p.clone()),
            ZuupError::TooManyConflicts(p) => ZuupError::TooManyConflicts(p.clone()),
            ZuupError::InvalidUrl(s) => ZuupError::InvalidUrl(s.clone()),
            ZuupError::MediaDownload(e) => ZuupError::MediaDownload(e.clone()),
        }
    }
}

impl From<config::ConfigError> for ZuupError {
    fn from(err: config::ConfigError) -> Self {
        ZuupError::Config(err.to_string())
    }
}
