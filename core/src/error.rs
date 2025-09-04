//! Error types and handling for Zuup

use thiserror::Error;

/// Main error type for Zuup operations.
#[derive(Debug, Error)]
pub enum ZuupError {
    #[error("Session error: {0}")]
    Session(String),
}

/// Result type alias for Zuup operations
pub type Result<T> = std::result::Result<T, ZuupError>;

impl ZuupError {
    // Check if the error is recoverable (can be retried) todo))

    /// Get the severity level of the error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ZuupError::Session(_) => ErrorSeverity::Medium,
        }
    }

    /// Get the category of this error
    pub fn category(&self) -> ErrorCategory {
        match self {
            ZuupError::Session(_) => ErrorCategory::System,
        }
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

impl Clone for ZuupError {
    fn clone(&self) -> Self {
        match self {
            ZuupError::Session(s) => ZuupError::Session(s.clone()),
        }
    }
}
