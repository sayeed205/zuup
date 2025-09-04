use async_trait::async_trait;
use url::Url;

use crate::types::{DownloadProgress, DownloadRequest, DownloadState};

/// Trait for protocol-specific download handlers
#[async_trait]
pub trait ProtocolHandler: Send + Sync {
    /// Get the protocol name (e.g., "http", "https", "ftp")
    fn protocol(&self) -> &'static str;

    /// Check if this handler can handle the given URL
    fn can_handle(&self, url: &Url) -> bool;

    /// Create a new download instance for the given request
    async fn create_download(&self, request: &DownloadRequest) -> Result<Box<dyn Download>>;

    /// Resume a download from saved state
    async fn resume_download(
        &self,
        request: &DownloadRequest,
        state: &DownloadState,
    ) -> Result<Box<dyn Download>>;

    /// Get protocol-specific capabilities
    fn capabilities(&self) -> ProtocolCapabilities {
        ProtocolCapabilities::default()
    }
}

/// Capabilities supported by a protocol handler
#[derive(Debug, Clone)]
pub struct ProtocolCapabilities {
    /// Supports segmented/parallel downloading
    pub supports_segments: bool,

    /// Supports resume functionality
    pub supports_resume: bool,

    /// Supports range requests
    pub supports_ranges: bool,

    /// Supports authentication
    pub supports_auth: bool,

    /// Supports proxy connections
    pub supports_proxy: bool,

    /// Maximum number of connections per download
    pub max_connections: Option<u32>,

    /// Supports checksum verification
    pub supports_checksums: bool,

    /// Supports metadata retrieval before download
    pub supports_metadata: bool,
}

impl Default for ProtocolCapabilities {
    fn default() -> Self {
        Self {
            supports_segments: false,
            supports_resume: false,
            supports_ranges: false,
            supports_auth: false,
            supports_proxy: false,
            max_connections: Some(1),
            supports_checksums: false,
            supports_metadata: false,
        }
    }
}

/// Trait for individual download instances
#[async_trait]
pub trait Download: Send + Sync {
    /// Start the download
    async fn start(&mut self) -> Result<()>;

    /// Pause the download
    async fn pause(&mut self) -> Result<()>;

    /// Resume the download
    async fn resume(&mut self) -> Result<()>;

    /// Cancel the download
    async fn cancel(&mut self) -> Result<()>;

    /// Get current progress
    fn progress(&self) -> DownloadProgress;

    /// Get current state
    fn state(&self) -> DownloadState;

    /// Get download metadata (file size, content type, etc.)
    async fn metadata(&self) -> Result<DownloadMetadata>;

    /// Check if the download supports the given operation
    fn supports_operation(&self, operation: DownloadOperation) -> bool;
}

/// Metadata about a download
#[derive(Debug, Clone)]
pub struct DownloadMetadata {
    /// Total file size in bytes (if known)
    pub size: Option<u64>,

    /// Content type/MIME type
    pub content_type: Option<String>,

    /// Last modified timestamp
    pub last_modified: Option<chrono::DateTime<chrono::Utc>>,

    /// ETag for caching
    pub etag: Option<String>,

    /// Whether the server supports range requests
    pub supports_ranges: bool,

    /// Suggested filename from server
    pub filename: Option<String>,

    /// Additional protocol-specific metadata
    pub extra: std::collections::HashMap<String, String>,
}

impl Default for DownloadMetadata {
    fn default() -> Self {
        Self {
            size: None,
            content_type: None,
            last_modified: None,
            etag: None,
            supports_ranges: false,
            filename: None,
            extra: std::collections::HashMap::new(),
        }
    }
}

/// Operations that can be performed on downloads
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadOperation {
    Start,
    Pause,
    Resume,
    Cancel,
    GetMetadata,
    VerifyChecksum,
}

/// Registry for protocol handlers
pub struct ProtocolRegistry {
    handlers: Vec<Box<dyn ProtocolHandler>>,
}

impl ProtocolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Register a protocol handler
    pub fn register(&mut self, handler: Box<dyn ProtocolHandler>) {
        self.handlers.push(handler);
    }

    /// Find a handler for the given URL
    pub fn find_handler(&self, url: &Url) -> Option<&dyn ProtocolHandler> {
        self.handlers
            .iter()
            .find(|handler| handler.can_handle(url))
            .map(|handler| handler.as_ref())
    }

    /// Get all registered handlers
    pub fn handlers(&self) -> &[Box<dyn ProtocolHandler>] {
        &self.handlers
    }

    /// Get supported protocols
    pub fn supported_protocols(&self) -> Vec<&'static str> {
        self.handlers
            .iter()
            .map(|handler| handler.protocol())
            .collect()
    }
}

impl Default for ProtocolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
