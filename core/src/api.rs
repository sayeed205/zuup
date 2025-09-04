use std::path::PathBuf;
use std::sync::Arc;

use url::Url;

use crate::config::ZuupConfig;
use crate::engine::ZuupEngine;
use crate::types::{DownloadId, DownloadRequest};
use crate::error::Result;

/// Main entry point for the Zuup download Manager library
///
/// This provides a high-level, easy-to-use interface for managing downloads.
/// in Rust applications. It handles all the complexity of the underlying engine
/// while providing a simple, clean async API.
///
/// # Examples
///
/// ```rust
/// // todo)) add example
/// ```
pub struct Zuup {
    engine: Arc<ZuupEngine>,
}

impl Zuup {
    /// Create a new Zuup builder for configuration
    pub fn builder() -> ZuupBuilder {
        ZuupBuilder::new()
    }

    /// Create a new Zuup instance
    pub async fn new(config: Option<ZuupConfig>) -> Result<Self> {
        let config = config.unwrap_or_default();
        let engine = Arc::new(ZuupEngine::new(config).await?);
        Ok(Self { engine })
    }

    /// Add a simple download from a URL
    ///
    /// This is the simplest way to add a download. It uses default options
    /// and downloads to the configured download directory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // todo))
    /// ```
    pub async fn download(&self, url: Url) -> Result<DownloadId> {
        // Get the default download directory from config
        let config = self.engine.config().await;
        let request = DownloadRequest::new(vec![url]).output_path(config.general.download_dir);
        let id = self.engine.add_download(request).await?;

        // Auto-start the download for better UX
        tracing::debug!("API: Attempting to auto-start download {}", id);
        if let Err(e) = self.engine.start_download(&id).await {
            tracing::warn!(error = %e, download_id = %id, "Failed to auto-start download");
        } else {
            tracing::debug!("API: Successfully auto-started download {}", id);
        }

        Ok(id)
    }
}

/// Builder for configuring and creating a Zuup instance
///
/// This provides a fluent interface for setting up the download manager
/// with custom configuration options.
///
/// # Examples
///
/// ```rust
/// // todo)) add example
/// ```
pub struct ZuupBuilder {
    config: ZuupConfig,
}

impl ZuupBuilder {
    /// Create a new ZuupBuilder with default configuration options
    pub fn new() -> Self {
        Self {
            config: ZuupConfig::default(),
        }
    }
}

impl Default for ZuupBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a completed download operation
#[derive(Debug, Clone)]
pub struct DownloadResult {
    /// Download Id
    pub id: String,

    /// Whether the download was successful
    pub success: bool,

    /// Path to the downloaded file (if successful)
    pub path: Option<PathBuf>,

    /// Total bytes downloaded
    pub bytes_downloaded: u64,

    /// Total time taken for the download
    pub duration: u64,

    /// Average download speed in bytes per second
    pub average_speed: f64,

    /// Error message (if unsuccessful)
    pub error: Option<String>,

    /// Whether the checksum verification passed (if successful)
    pub checksum_verified: Option<bool>,
}
