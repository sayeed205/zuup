//! Protocol handler interfaces and traits

use async_trait::async_trait;
use url::Url;

use crate::{
    error::Result,
    types::{DownloadProgress, DownloadRequest, DownloadState},
};

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
    /// Create a new registry with default handlers based on enabled features
    pub fn new() -> Self {
        let mut registry = Self {
            handlers: Vec::new(),
        };
        registry.register_default_handlers();
        registry
    }

    /// Register default protocol handlers based on enabled features
    fn register_default_handlers(&mut self) {
        #[cfg(feature = "http")]
        {
            let handler = crate::protocols::HttpProtocolHandler::new();
            self.register(Box::new(handler));
        }

        #[cfg(feature = "ftp")]
        {
            let handler = crate::protocols::FtpProtocolHandler::new();
            self.register(Box::new(handler));
        }

        #[cfg(feature = "sftp")]
        {
            let handler = crate::protocols::SftpProtocolHandler::new();
            self.register(Box::new(handler));
        }

        #[cfg(feature = "torrent")]
        {
            let handler = crate::protocols::BitTorrentProtocolHandler::new();
            self.register(Box::new(handler));
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
    
    /// Find a handler for the given URL with detailed error information
    pub fn find_handler_with_error(&self, url: &Url) -> crate::error::Result<&dyn ProtocolHandler> {
        // Check if registry is empty first
        if self.is_empty() {
            return Err(crate::error::ZuupError::Protocol(
                crate::error::ProtocolError::NoProtocolHandlers
            ));
        }
        
        if let Some(handler) = self.find_handler(url) {
            return Ok(handler);
        }
        
        // Check if the protocol is supported but not enabled
        if let Some(required_feature) = crate::error::get_required_feature_for_scheme(url.scheme()) {
            return Err(crate::error::ZuupError::Protocol(
                crate::error::ProtocolError::UnsupportedProtocolWithFeature {
                    protocol: url.scheme().to_string(),
                    feature: required_feature.to_string(),
                }
            ));
        }
        
        // Unknown protocol
        Err(crate::error::ZuupError::Protocol(
            crate::error::ProtocolError::UnsupportedProtocol(
                format!("Unknown protocol: {}", url.scheme())
            )
        ))
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

    /// Check if the registry is empty (no handlers registered)
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }
}

impl Default for ProtocolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile-time protocol availability detection
pub struct AvailableProtocols;

impl AvailableProtocols {
    /// HTTP/HTTPS protocol availability
    pub const HTTP: bool = cfg!(feature = "http");
    
    /// FTP/FTPS protocol availability
    pub const FTP: bool = cfg!(feature = "ftp");
    
    /// SFTP protocol availability
    pub const SFTP: bool = cfg!(feature = "sftp");
    
    /// BitTorrent protocol availability
    pub const TORRENT: bool = cfg!(feature = "torrent");
    
    /// Get list of enabled protocols
    pub fn list_enabled() -> Vec<&'static str> {
        let mut protocols = Vec::new();
        
        if Self::HTTP {
            protocols.push("http");
        }
        if Self::FTP {
            protocols.push("ftp");
        }
        if Self::SFTP {
            protocols.push("sftp");
        }
        if Self::TORRENT {
            protocols.push("torrent");
        }
        
        protocols
    }
    
    /// Get all possible protocols (regardless of enabled features)
    pub fn list_all() -> Vec<&'static str> {
        vec!["http", "ftp", "sftp", "torrent"]
    }
    
    /// Check if a specific protocol is enabled
    pub fn is_enabled(protocol: &str) -> bool {
        match protocol {
            "http" | "https" => Self::HTTP,
            "ftp" | "ftps" => Self::FTP,
            "sftp" => Self::SFTP,
            "torrent" | "magnet" => Self::TORRENT,
            _ => false,
        }
    }
    
    /// Get the count of enabled protocols
    pub fn enabled_count() -> usize {
        Self::list_enabled().len()
    }
}

/// Get the required feature name for a given URL scheme
pub fn get_required_feature_for_scheme(scheme: &str) -> Option<&'static str> {
    match scheme {
        "http" | "https" => Some("http"),
        "ftp" | "ftps" => Some("ftp"),
        "sftp" => Some("sftp"),
        "magnet" => Some("torrent"),
        _ => None,
    }
}

/// Get the protocol name for a given URL scheme
pub fn get_protocol_for_scheme(scheme: &str) -> Option<&'static str> {
    match scheme {
        "http" | "https" => Some("http"),
        "ftp" | "ftps" => Some("ftp"),
        "sftp" => Some("sftp"),
        "magnet" => Some("torrent"),
        _ => None,
    }
}

/// Protocol-to-feature mapping information
#[derive(Debug, Clone)]
pub struct ProtocolFeatureMapping {
    /// Protocol name
    pub protocol: &'static str,
    /// Required feature flag
    pub feature: &'static str,
    /// Supported URL schemes
    pub schemes: Vec<&'static str>,
    /// Whether the protocol is currently enabled
    pub enabled: bool,
}

/// Get comprehensive protocol-to-feature mapping
pub fn get_protocol_feature_mappings() -> Vec<ProtocolFeatureMapping> {
    vec![
        ProtocolFeatureMapping {
            protocol: "http",
            feature: "http",
            schemes: vec!["http", "https"],
            enabled: AvailableProtocols::HTTP,
        },
        ProtocolFeatureMapping {
            protocol: "ftp",
            feature: "ftp",
            schemes: vec!["ftp", "ftps"],
            enabled: AvailableProtocols::FTP,
        },
        ProtocolFeatureMapping {
            protocol: "sftp",
            feature: "sftp",
            schemes: vec!["sftp"],
            enabled: AvailableProtocols::SFTP,
        },
        ProtocolFeatureMapping {
            protocol: "torrent",
            feature: "torrent",
            schemes: vec!["magnet"],
            enabled: AvailableProtocols::TORRENT,
        },
    ]
}
