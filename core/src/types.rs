use std::collections::HashMap;
use std::time::Duration;

use cuid2::cuid;
use serde::{Deserialize, Serialize};
use url::Url;

/// Unique identifier for downloads
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DownloadId(String);

impl DownloadId {
    /// Create a new unique download ID
    pub fn new() -> Self {
        Self(cuid())
    }

    /// Create a download ID from a string
    pub fn from_string(id: String) -> Self {
        Self(id)
    }

    /// Get the string representation of the ID
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for DownloadId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for DownloadId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Download request configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DownloadOptions {
    /// Maximum number of connections per download
    pub max_connections: Option<u32>,

    /// Maximum download speed in bytes per second
    pub max_speed: Option<u64>,

    /// Minimum size for splitting downloads into segments
    pub min_split_size: Option<u64>,

    /// Number of retry attempts for failed downloads
    pub retry_count: Option<u32>,

    /// Request timeout duration
    pub timeout: Option<Duration>,

    /// Custom HTTP headers
    pub headers: HashMap<String, String>,

    /// User agent string
    pub user_agent: Option<String>,

    /// Proxy configuration
    pub proxy: Option<ProxyConfig>,

    /// Authentication configuration
    pub auth: Option<AuthConfig>,

    /// Checksum verification configuration
    pub checksum: Option<ChecksumConfig>,

    /// Whether to resume partial downloads
    pub resume: bool,

    /// Whether to overwrite existing files
    pub overwrite: bool,

    /// File allocation method
    pub allocation: FileAllocation,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            max_connections: Some(4),
            max_speed: None,
            min_split_size: Some(1024 * 1024), // 1MB
            retry_count: Some(3),
            timeout: Some(Duration::from_secs(30)),
            headers: HashMap::new(),
            user_agent: Some(format!("Zuup/{}", env!("CARGO_PKG_VERSION"))),
            proxy: None,
            auth: None,
            checksum: None,
            resume: true,
            overwrite: false,
            allocation: FileAllocation::Prealloc,
        }
    }
}

/// Proxy configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProxyConfig {
    pub url: Url,
    pub auth: Option<ProxyAuth>,
}

/// Proxy authentication
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProxyAuth {
    pub username: String,
    pub password: String,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuthConfig {
    Basic { username: String, password: String },
    Bearer { token: String },
    Custom { header: String, value: String },
}

/// Checksum configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChecksumConfig {
    pub algorithm: ChecksumAlgorithm,
    pub expected: String,
}

/// Supported checksum algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChecksumAlgorithm {
    Md5,
    Sha1,
    Sha256,
    Sha512,
}

/// File allocation method
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FileAllocation {
    /// No preallocation
    None,
    /// Preallocate file space
    Prealloc,
    /// Truncate file to final size
    Trunc,
    /// Fallback allocation method
    Falloc,
}

/// Download segment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadSegment {
    pub id: u32,
    pub start: u64,
    pub end: u64,
    pub downloaded: u64,
    pub url: Url,
}

/// Progress information for a download segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentProgress {
    pub id: u32,
    pub downloaded: u64,
    pub total: u64,
    pub speed: u64,
}
