//! Core types and data structures

use std::time::Duration;
use std::{collections::HashMap, path::PathBuf};

use chrono::{DateTime, Utc};
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

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub bytes_downloaded: u64,
    pub bytes_uploaded: u64,
    pub download_speed: u64,
    pub upload_speed: u64,
    pub connections: u32,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Whether to verify certificates
    pub verify_certificates: bool,

    /// Custom CA certificate paths
    pub ca_certificates: Vec<PathBuf>,

    /// Client certificate for mutual TLS
    pub client_certificate: Option<ClientCertificate>,

    /// Minimum TLS version
    pub min_version: Option<TlsVersion>,

    /// Maximum TLS version
    pub max_version: Option<TlsVersion>,

    /// Allowed cipher suites
    pub cipher_suites: Vec<String>,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            verify_certificates: true,
            ca_certificates: Vec::new(),
            client_certificate: None,
            min_version: Some(TlsVersion::V1_2),
            max_version: None,
            cipher_suites: Vec::new(),
        }
    }
}

/// Client certificate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCertificate {
    pub cert_path: PathBuf,
    pub key_path: PathBuf,
    pub password: Option<String>,
}

/// TLS version
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TlsVersion {
    V1_0,
    V1_1,
    V1_2,
    V1_3,
}

/// Download request containing URLs and options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRequest {
    /// List of URLs to download from (for multi-source downloads)
    pub urls: Vec<Url>,

    /// Output directory path
    pub output_path: Option<PathBuf>,

    /// Custom filename (if not specified, derived from URL)
    pub filename: Option<String>,

    /// Download options and configuration
    pub options: DownloadOptions,

    /// Category for organization
    pub category: Option<String>,

    /// Referrer URL (for browser extension integration)
    pub referrer: Option<String>,

    /// Cookies for authentication or from browser extension
    pub cookies: Option<String>,
}

impl DownloadRequest {
    /// Create a new download request from one or more URLs
    pub fn new<T>(urls: T) -> Self
    where
        T: Into<Vec<Url>>,
    {
        Self {
            urls: urls.into(),
            output_path: None,
            filename: None,
            options: DownloadOptions::default(),
            category: None,
            referrer: None,
            cookies: None,
        }
    }

    /// Set the output path
    pub fn output_path(mut self, path: PathBuf) -> Self {
        self.output_path = Some(path);
        self
    }

    /// Set the filename
    pub fn filename(mut self, filename: String) -> Self {
        self.filename = Some(filename);
        self
    }

    /// Set the download options
    pub fn options(mut self, options: DownloadOptions) -> Self {
        self.options = options;
        self
    }
}

/// Current state of a download
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadState {
    /// Download is queued but not yet started
    Pending,

    /// Download is actively running
    Active,

    /// Download has been paused by user
    Paused,

    /// Download completed successfully
    Completed,

    /// Download failed with an error
    Failed(String), // Store error message as string for serialization

    /// Download was cancelled by user
    Cancelled,

    /// Download is waiting for resources
    Waiting,

    /// Download is being prepared (resolving metadata, etc.)
    Preparing,

    /// Download is waiting for retry after failure
    Retrying,
}

impl DownloadState {
    /// Check if the download is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            DownloadState::Completed | DownloadState::Failed(_) | DownloadState::Cancelled
        )
    }

    /// Check if the download is active (running or preparing)
    pub fn is_active(&self) -> bool {
        matches!(self, DownloadState::Active | DownloadState::Preparing)
    }

    /// Check if the download can be resumed
    pub fn can_resume(&self) -> bool {
        matches!(self, DownloadState::Paused | DownloadState::Failed(_))
    }

    /// Check if the download can be paused
    pub fn can_pause(&self) -> bool {
        matches!(self, DownloadState::Active | DownloadState::Preparing)
    }

    /// Check if the download can be started
    pub fn can_start(&self) -> bool {
        matches!(self, DownloadState::Pending | DownloadState::Waiting)
    }

    /// Check if the download is waiting for resources
    pub fn is_waiting(&self) -> bool {
        matches!(self, DownloadState::Waiting)
    }

    /// Check if the download is preparing
    pub fn is_preparing(&self) -> bool {
        matches!(self, DownloadState::Preparing)
    }

    /// Validate state transition
    pub fn can_transition_to(&self, new_state: &DownloadState) -> bool {
        use DownloadState::*;

        match (self, new_state) {
            // From Pending
            (Pending, Waiting) => true,
            (Pending, Preparing) => true,
            (Pending, Cancelled) => true,

            // From Waiting
            (Waiting, Preparing) => true,
            (Waiting, Cancelled) => true,

            // From Preparing
            (Preparing, Active) => true,
            (Preparing, Paused) => true,
            (Preparing, Failed(_)) => true,
            (Preparing, Cancelled) => true,

            // From Active
            (Active, Paused) => true,
            (Active, Completed) => true,
            (Active, Failed(_)) => true,
            (Active, Cancelled) => true,

            // From Paused
            (Paused, Active) => true,
            (Paused, Cancelled) => true,

            // From Failed
            (Failed(_), Preparing) => true,
            (Failed(_), Cancelled) => true,

            // Terminal states cannot transition
            (Completed, _) | (Cancelled, _) => false,

            // Same state is always valid
            (a, b) if a == b => true,

            // All other transitions are invalid
            _ => false,
        }
    }
}

/// Priority level for downloads
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DownloadPriority {
    /// Lowest priority
    Low = 1,
    /// Normal priority (default)
    Normal = 2,
    /// High priority
    High = 3,
    /// Highest priority
    Critical = 4,
}

impl Default for DownloadPriority {
    fn default() -> Self {
        DownloadPriority::Normal
    }
}

/// Progress information for a download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Total size of the download in bytes (None if unknown)
    pub total_size: Option<u64>,

    /// Number of bytes downloaded so far
    pub downloaded_size: u64,

    /// Current download speed in bytes per second
    pub download_speed: u64,

    /// Current upload speed in bytes per second (if available for BitTorrent)
    pub upload_speed: Option<u64>,

    /// Estimated time to completion
    pub eta: Option<Duration>,

    /// Number of active connections
    pub connections: u32,

    /// Progress of individual segments
    pub segments: Vec<SegmentProgress>,

    /// Percentage completed (0-100)
    pub percentage: u8,

    /// Time when download started
    pub started_at: Option<DateTime<Utc>>,

    /// Time of last update
    pub updated_at: DateTime<Utc>,

    /// Number of bytes uploaded (for BitTorrent)
    pub upload_size: Option<u64>,
}

impl DownloadProgress {
    /// Create a new progress tracker
    pub fn new() -> Self {
        Self {
            total_size: None,
            downloaded_size: 0,
            download_speed: 0,
            upload_speed: None,
            eta: None,
            connections: 0,
            segments: Vec::new(),
            percentage: 0,
            started_at: None,
            updated_at: Utc::now(),
            upload_size: None,
        }
    }

    /// Update the progress with new values
    pub fn update(&mut self, downloaded: u64, download_speed: u64, upload_speed: Option<u64>) {
        self.downloaded_size = downloaded;

        self.download_speed = download_speed;
        self.upload_speed = upload_speed;
        self.updated_at = Utc::now();

        // Calculate percentage if total size is known
        if let Some(total) = self.total_size {
            if total > 0 {
                self.percentage = ((downloaded as f64 / total as f64) * 100.0) as u8;
            }
        }

        // Calculate ETA if speed > 0 and total size is known
        if download_speed > 0 {
            if let Some(total) = self.total_size {
                let remaining = total.saturating_sub(downloaded);
                if remaining > 0 {
                    self.eta = Some(Duration::from_secs(remaining / download_speed));
                } else {
                    self.eta = None; // Download is complete
                }
            }
        } else {
            self.eta = None; // No speed means no ETA
        }
    }

    /// Check if the download is complete
    pub fn is_complete(&self) -> bool {
        if let Some(total) = self.total_size {
            self.downloaded_size >= total
        } else {
            false
        }
    }

    /// Set the total size
    pub fn set_total_size(&mut self, size: u64) {
        self.total_size = Some(size);
        // Recalculate percentage
        if size > 0 {
            self.percentage = ((self.downloaded_size as f64 / size as f64) * 100.0) as u8;
        }
    }

    /// Mark download as started
    pub fn start(&mut self) {
        if self.started_at.is_none() {
            self.started_at = Some(Utc::now());
        }
    }
}

impl Default for DownloadProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of download
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DownloadType {
    /// Standard HTTP/FTP download
    Standard,

    /// Media download with yt-dlp integration
    Media {
        /// Original URL provided by user
        original_url: String,
        // Extracted media information todo))
        // extracted_info: Option<crate::media::MediaInfo>,
    },

    /// BitTorrent download
    Torrent {
        /// Torrent info hash
        info_hash: Option<String>,
        /// Whether this is a magnet link
        is_magnet: bool,
    },

    /// Metalink multi-source download
    Metalink {
        /// Number of sources
        source_count: u32,
    },
}

/// Complete information about a download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadInfo {
    /// Unique identifier for this download
    pub id: DownloadId,

    /// List of URLs being downloaded from
    pub urls: Vec<Url>,

    /// Final filename of the downloaded file
    pub filename: String,

    /// Output directory path
    pub output_path: PathBuf,

    /// Current state of the download
    pub state: DownloadState,

    /// Progress information
    pub progress: DownloadProgress,

    /// Download priority
    pub priority: DownloadPriority,

    /// Type of download
    pub download_type: DownloadType,

    /// Category for organization
    pub category: Option<String>,

    /// When the download was created
    pub created_at: DateTime<Utc>,

    /// When the download was started (None if never started)
    pub started_at: Option<DateTime<Utc>>,

    /// When the download was completed (None if not completed)
    pub completed_at: Option<DateTime<Utc>>,

    /// Download options used
    pub options: DownloadOptions,

    /// Error message if download failed
    pub error_message: Option<String>,

    /// File size in bytes (None if unknown)
    pub file_size: Option<u64>,

    /// MIME type of the downloaded content
    pub content_type: Option<String>,

    /// Last modified time from server
    pub last_modified: Option<DateTime<Utc>>,

    /// Referrer URL (for browser extension integration)
    pub referrer: Option<String>,

    /// Cookies for authentication
    pub cookies: Option<String>,
}

impl DownloadInfo {
    /// Create new download info from a request
    pub fn new(
        id: DownloadId,
        request: DownloadRequest,
        filename: String,
        output_path: PathBuf,
    ) -> Self {
        Self {
            id,
            urls: request.urls,
            filename,
            output_path,
            state: DownloadState::Pending,
            progress: DownloadProgress::new(),
            priority: DownloadPriority::Normal,
            download_type: DownloadType::Standard,
            category: request.category,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            options: request.options,
            error_message: None,
            file_size: None,
            content_type: None,
            last_modified: None,
            referrer: request.referrer,
            cookies: request.cookies,
        }
    }

    /// Get the full file path
    pub fn file_path(&self) -> PathBuf {
        self.output_path.join(&self.filename)
    }

    /// Update the download state
    pub fn set_state(&mut self, state: DownloadState) {
        match &state {
            DownloadState::Active => {
                if self.started_at.is_none() {
                    self.started_at = Some(Utc::now());
                    self.progress.start();
                }
            }
            DownloadState::Completed => {
                self.completed_at = Some(Utc::now());
            }
            DownloadState::Failed(error) => {
                self.error_message = Some(error.clone());
            }
            _ => {}
        }
        self.state = state;
    }

    /// Check if download is finished (completed, failed, or cancelled)
    pub fn is_finished(&self) -> bool {
        self.state.is_terminal()
    }
}
