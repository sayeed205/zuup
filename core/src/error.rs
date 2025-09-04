//! Error types and handling for Ruso

use std::fmt;
use std::time::Duration;

use thiserror::Error;

use crate::download::DownloadState;
use crate::types::DownloadId;

/// Main error type for Ruso operations
#[derive(Debug, Error)]
pub enum RusoError {
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

/// Result type alias for Ruso operations
pub type Result<T> = std::result::Result<T, RusoError>;

impl RusoError {
    /// Check if the error is recoverable (can be retried)
    pub fn is_recoverable(&self) -> bool {
        match self {
            RusoError::Network(NetworkError::Timeout) => true,
            RusoError::Network(NetworkError::ConnectionReset) => true,
            RusoError::Network(NetworkError::ConnectionRefused) => true,
            RusoError::Protocol(ProtocolError::Http { status, .. }) => {
                // 5xx errors are generally recoverable
                *status >= 500 && *status < 600
            }
            RusoError::Io(_) => true,
            RusoError::Timeout => true,
            _ => false,
        }
    }

    /// Check if the error is a network-related error
    pub fn is_network_error(&self) -> bool {
        matches!(self, RusoError::Network(_))
    }

    /// Check if the error is a protocol-related error
    pub fn is_protocol_error(&self) -> bool {
        matches!(self, RusoError::Protocol(_))
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

impl Default for RecoveryStrategy {
    fn default() -> Self {
        RecoveryStrategy::None
    }
}

/// Enhanced error with context and recovery information
#[derive(Debug)]
pub struct EnhancedError {
    /// The underlying error
    pub error: RusoError,
    /// Error context with debugging information
    pub context: ErrorContext,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Error category
    pub category: ErrorCategory,
    /// Suggested recovery strategy
    pub recovery_strategy: RecoveryStrategy,
    /// Chain of related errors
    pub error_chain: Vec<RusoError>,
}

impl EnhancedError {
    /// Create a new enhanced error
    pub fn new(error: RusoError, context: ErrorContext) -> Self {
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
    pub fn with_chain_error(mut self, error: RusoError) -> Self {
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

impl RusoError {
    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            RusoError::Network(NetworkError::Timeout) => ErrorSeverity::Medium,
            RusoError::Network(NetworkError::ConnectionReset) => ErrorSeverity::Medium,
            RusoError::Network(NetworkError::ConnectionRefused) => ErrorSeverity::Medium,
            RusoError::Network(NetworkError::DnsResolutionFailed(_)) => ErrorSeverity::High,
            RusoError::Network(NetworkError::TooManyRedirects) => ErrorSeverity::High,
            RusoError::Network(_) => ErrorSeverity::Medium,

            RusoError::Protocol(ProtocolError::Http { status, .. }) => {
                match *status {
                    400..=499 => ErrorSeverity::High,   // Client errors
                    500..=599 => ErrorSeverity::Medium, // Server errors (retryable)
                    _ => ErrorSeverity::Medium,
                }
            }
            RusoError::Protocol(ProtocolError::UnsupportedProtocol(_)) => ErrorSeverity::High,
            RusoError::Protocol(ProtocolError::AuthenticationRequired) => ErrorSeverity::High,
            RusoError::Protocol(_) => ErrorSeverity::Medium,

            RusoError::Io(_) => ErrorSeverity::Medium,
            RusoError::Config(_) => ErrorSeverity::High,
            RusoError::DownloadNotFound(_) => ErrorSeverity::Medium,
            RusoError::InvalidStateTransition { .. } => ErrorSeverity::Medium,
            RusoError::ChecksumMismatch => ErrorSeverity::High,
            RusoError::AuthenticationFailed => ErrorSeverity::High,
            RusoError::PermissionDenied => ErrorSeverity::High,
            RusoError::InsufficientDiskSpace => ErrorSeverity::Critical,
            RusoError::Session(_) => ErrorSeverity::Medium,
            RusoError::Event(_) => ErrorSeverity::Low,
            RusoError::Serialization(_) => ErrorSeverity::Medium,
            RusoError::UrlParse(_) => ErrorSeverity::High,
            RusoError::TaskJoin(_) => ErrorSeverity::Medium,
            RusoError::ChannelSend => ErrorSeverity::Medium,
            RusoError::ChannelReceive => ErrorSeverity::Medium,
            RusoError::Timeout => ErrorSeverity::Medium,
            RusoError::Cancelled => ErrorSeverity::Low,
            RusoError::Internal(_) => ErrorSeverity::Critical,
            RusoError::InvalidInput(_) => ErrorSeverity::High,
            RusoError::FileExists(_) => ErrorSeverity::Medium,
            RusoError::InvalidPath(_) => ErrorSeverity::High,
            RusoError::TooManyConflicts(_) => ErrorSeverity::High,
            RusoError::InvalidUrl(_) => ErrorSeverity::High,
            RusoError::MediaDownload(MediaError::YtDlpNotFound) => ErrorSeverity::Critical,
            RusoError::MediaDownload(MediaError::YtDlpExecutionFailed(_)) => ErrorSeverity::Medium,
            RusoError::MediaDownload(MediaError::ExtractionFailed(_)) => ErrorSeverity::Medium,
            RusoError::MediaDownload(MediaError::UnsupportedUrl(_)) => ErrorSeverity::High,
            RusoError::MediaDownload(_) => ErrorSeverity::Medium,
        }
    }

    /// Get the category of this error
    pub fn category(&self) -> ErrorCategory {
        match self {
            RusoError::Network(_) => ErrorCategory::Network,
            RusoError::Protocol(_) => ErrorCategory::Protocol,
            RusoError::Io(_) => ErrorCategory::FileSystem,
            RusoError::Config(_) => ErrorCategory::Configuration,
            RusoError::AuthenticationFailed => ErrorCategory::Security,
            RusoError::PermissionDenied => ErrorCategory::Security,
            RusoError::InsufficientDiskSpace => ErrorCategory::Resource,
            RusoError::DownloadNotFound(_) => ErrorCategory::UserInput,
            RusoError::InvalidStateTransition { .. } => ErrorCategory::System,
            RusoError::ChecksumMismatch => ErrorCategory::FileSystem,
            RusoError::Session(_) => ErrorCategory::System,
            RusoError::Event(_) => ErrorCategory::System,
            RusoError::Serialization(_) => ErrorCategory::System,
            RusoError::UrlParse(_) => ErrorCategory::UserInput,
            RusoError::TaskJoin(_) => ErrorCategory::System,
            RusoError::ChannelSend => ErrorCategory::System,
            RusoError::ChannelReceive => ErrorCategory::System,
            RusoError::Timeout => ErrorCategory::Network,
            RusoError::Cancelled => ErrorCategory::System,
            RusoError::Internal(_) => ErrorCategory::System,
            RusoError::InvalidInput(_) => ErrorCategory::UserInput,
            RusoError::FileExists(_) => ErrorCategory::FileSystem,
            RusoError::InvalidPath(_) => ErrorCategory::FileSystem,
            RusoError::TooManyConflicts(_) => ErrorCategory::FileSystem,
            RusoError::InvalidUrl(_) => ErrorCategory::UserInput,
            RusoError::MediaDownload(_) => ErrorCategory::Protocol,
        }
    }

    /// Get the suggested recovery strategy for this error
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            RusoError::Network(NetworkError::Timeout) => RecoveryStrategy::Retry {
                max_attempts: 3,
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
            },
            RusoError::Network(NetworkError::ConnectionReset) => RecoveryStrategy::Retry {
                max_attempts: 5,
                initial_delay: Duration::from_millis(500),
                max_delay: Duration::from_secs(10),
                backoff_multiplier: 1.5,
            },
            RusoError::Network(NetworkError::ConnectionRefused) => RecoveryStrategy::Retry {
                max_attempts: 3,
                initial_delay: Duration::from_secs(2),
                max_delay: Duration::from_secs(60),
                backoff_multiplier: 2.0,
            },
            RusoError::Network(NetworkError::DnsResolutionFailed(_)) => RecoveryStrategy::Retry {
                max_attempts: 2,
                initial_delay: Duration::from_secs(5),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
            },

            RusoError::Protocol(ProtocolError::Http { status, .. }) => {
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

            RusoError::Io(_) => RecoveryStrategy::Retry {
                max_attempts: 2,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(5),
                backoff_multiplier: 2.0,
            },

            RusoError::ChecksumMismatch => RecoveryStrategy::Retry {
                max_attempts: 2,
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(10),
                backoff_multiplier: 2.0,
            },

            RusoError::Timeout => RecoveryStrategy::Retry {
                max_attempts: 3,
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
            },

            RusoError::InvalidStateTransition { .. } => RecoveryStrategy::Reset,
            RusoError::Event(_) => RecoveryStrategy::Skip,
            RusoError::Cancelled => RecoveryStrategy::None,

            RusoError::MediaDownload(MediaError::YtDlpExecutionFailed(_)) => {
                RecoveryStrategy::Retry {
                    max_attempts: 2,
                    initial_delay: Duration::from_secs(2),
                    max_delay: Duration::from_secs(30),
                    backoff_multiplier: 2.0,
                }
            }

            RusoError::MediaDownload(MediaError::ExtractionFailed(_)) => RecoveryStrategy::Retry {
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
    fn with_context(self, context: ErrorContext) -> std::result::Result<T, EnhancedError>;

    /// Add context using a closure
    fn with_context_fn<F>(self, f: F) -> std::result::Result<T, EnhancedError>
    where
        F: FnOnce() -> ErrorContext;
}

impl<T, E> ErrorExt<T> for std::result::Result<T, E>
where
    E: Into<RusoError>,
{
    fn with_context(self, context: ErrorContext) -> std::result::Result<T, EnhancedError> {
        self.map_err(|e| e.into().with_context(context))
    }

    fn with_context_fn<F>(self, f: F) -> std::result::Result<T, EnhancedError>
    where
        F: FnOnce() -> ErrorContext,
    {
        self.map_err(|e| e.into().with_context(f()))
    }
}

impl Clone for RusoError {
    fn clone(&self) -> Self {
        match self {
            RusoError::Network(e) => RusoError::Network(e.clone()),
            RusoError::Protocol(e) => RusoError::Protocol(e.clone()),
            RusoError::Io(e) => RusoError::Io(std::io::Error::new(e.kind(), e.to_string())),
            RusoError::Config(s) => RusoError::Config(s.clone()),
            RusoError::DownloadNotFound(id) => RusoError::DownloadNotFound(id.clone()),
            RusoError::InvalidStateTransition { from, to } => RusoError::InvalidStateTransition {
                from: from.clone(),
                to: to.clone(),
            },
            RusoError::ChecksumMismatch => RusoError::ChecksumMismatch,
            RusoError::AuthenticationFailed => RusoError::AuthenticationFailed,
            RusoError::PermissionDenied => RusoError::PermissionDenied,
            RusoError::InsufficientDiskSpace => RusoError::InsufficientDiskSpace,
            RusoError::Session(s) => RusoError::Session(s.clone()),
            RusoError::Event(s) => RusoError::Event(s.clone()),
            RusoError::Serialization(_) => {
                RusoError::Internal("Serialization error (cloned)".to_string())
            }
            RusoError::UrlParse(e) => RusoError::UrlParse(e.clone()),
            RusoError::TaskJoin(_) => RusoError::Internal("Task join error (cloned)".to_string()),
            RusoError::ChannelSend => RusoError::ChannelSend,
            RusoError::ChannelReceive => RusoError::ChannelReceive,
            RusoError::Timeout => RusoError::Timeout,
            RusoError::Cancelled => RusoError::Cancelled,
            RusoError::Internal(s) => RusoError::Internal(s.clone()),
            RusoError::InvalidInput(s) => RusoError::InvalidInput(s.clone()),
            RusoError::FileExists(p) => RusoError::FileExists(p.clone()),
            RusoError::InvalidPath(p) => RusoError::InvalidPath(p.clone()),
            RusoError::TooManyConflicts(p) => RusoError::TooManyConflicts(p.clone()),
            RusoError::InvalidUrl(s) => RusoError::InvalidUrl(s.clone()),
            RusoError::MediaDownload(e) => RusoError::MediaDownload(e.clone()),
        }
    }
}

impl From<config::ConfigError> for RusoError {
    fn from(err: config::ConfigError) -> Self {
        RusoError::Config(err.to_string())
    }
}
