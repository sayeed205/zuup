use std::{path::PathBuf, sync::Arc, time::Duration};

use async_trait::async_trait;
use url::Url;

use crate::{
    config::ZuupConfig,
    engine::{EngineStats, ZuupEngine},
    error::Result,
    event::{Event, EventBus, EventSubscriber},
    media::{MediaDownloadOptions, MediaFormat},
    types::{DownloadId, DownloadInfo, DownloadRequest, DownloadState},
};

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

    /// Add a download with custom options
    ///
    /// This allows you to specify custom download options like output path,
    /// filename, and various download parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // todo)) add example
    /// ```
    pub async fn add_download(&self, request: DownloadRequest) -> Result<DownloadId> {
        let id = self.engine.add_download(request).await?;

        // Auto-start the download for better UX
        // [note] maybe will be removed to give control to user
        if let Err(e) = self.engine.start_download(&id).await {
            tracing::warn!(error = %e, download_id = %id, "Failed to auto-start download");
        }

        Ok(id)
    }

    /// Add a media download (video/audio) from a URL
    ///
    /// This uses yt-dlp integration to download media content from
    /// supported platforms like YouTube, Vimeo, etc.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // todo)) add example
    /// ```
    pub async fn download_media(
        &self,
        url: &str,
        options: MediaDownloadOptions,
    ) -> Result<DownloadId> {
        self.engine.add_media_download(url, options).await
    }

    /// Get available media formats for a URL
    ///
    /// This queries yt-dlp to get all available formats for a media URL,
    /// allowing you to choose the desired quality and format.
    pub async fn get_media_formats(&self, url: &str) -> Result<Vec<MediaFormat>> {
        self.engine.get_media_formats(url).await
    }

    /// Pause a download
    ///
    /// The download can be resumed later using `resume_download()`.
    pub async fn pause_download(&self, id: &DownloadId) -> Result<()> {
        self.engine.pause_download(id.clone()).await
    }

    /// Resume a paused download
    pub async fn resume_download(&self, id: &DownloadId) -> Result<()> {
        self.engine.resume_download(id.clone()).await
    }

    /// Cancel and remove a download
    ///
    /// If `force` is true, the download will be removed even if it's currently active.
    /// If `force` is false, active downloads will be paused first.
    pub async fn remove_download(&self, id: &DownloadId, force: bool) -> Result<()> {
        self.engine.remove_download(id.clone(), force).await
    }

    /// Get detailed information about a download
    pub async fn get_download_info(&self, id: &DownloadId) -> Result<DownloadInfo> {
        self.engine.get_download_info(id.clone()).await
    }

    /// List all downloads
    pub async fn list_downloads(&self) -> Result<Vec<DownloadInfo>> {
        self.engine.list_downloads().await
    }

    /// List downloads filtered by state
    pub async fn list_downloads_by_state(&self, state: DownloadState) -> Result<Vec<DownloadInfo>> {
        let downloads = self.engine.list_downloads().await?;
        Ok(downloads
            .into_iter()
            .filter(|d| std::mem::discriminant(&d.state) == std::mem::discriminant(&state))
            .collect())
    }

    /// Get active downloads
    pub async fn active_downloads(&self) -> Result<Vec<DownloadInfo>> {
        let downloads = self.engine.list_downloads().await?;
        Ok(downloads
            .into_iter()
            .filter(|d| d.state.is_active())
            .collect())
    }

    /// Get completed downloads
    pub async fn completed_downloads(&self) -> Result<Vec<DownloadInfo>> {
        let downloads = self.engine.list_downloads().await?;
        Ok(downloads
            .into_iter()
            .filter(|d| matches!(d.state, DownloadState::Completed))
            .collect())
    }

    /// Get failed downloads
    pub async fn failed_downloads(&self) -> Result<Vec<DownloadInfo>> {
        let downloads = self.engine.list_downloads().await?;
        Ok(downloads
            .into_iter()
            .filter(|d| matches!(d.state, DownloadState::Failed(_)))
            .collect())
    }

    /// Check if a download is finished (completed, failed, or cancelled)
    pub async fn is_finished(&self, id: &DownloadId) -> Result<bool> {
        let info = self.get_download_info(id).await?;
        Ok(matches!(
            info.state,
            DownloadState::Completed | DownloadState::Failed(_) | DownloadState::Cancelled
        ))
    }

    /// Wait for a download to complete
    ///
    /// This will block until the download reaches a terminal state
    /// (completed, failed, or cancelled).
    ///
    /// # Examples
    ///
    /// ```rust
    /// // todo)) add example
    /// ```
    pub async fn wait_for_completion(&self, id: &DownloadId) -> Result<DownloadResult> {
        loop {
            let info = self.get_download_info(id).await?;

            if matches!(
                info.state,
                DownloadState::Completed | DownloadState::Failed(_) | DownloadState::Cancelled
            ) {
                return Ok(DownloadResult {
                    id: id.to_string(),
                    // path: Some(info.output_path),
                    success: matches!(info.state, DownloadState::Completed),
                    path: if matches!(info.state, DownloadState::Completed) {
                        Some(info.output_path.join(&info.filename))
                    } else {
                        None
                    },
                    bytes_downloaded: info.progress.downloaded_size,
                    duration: info
                        .started_at
                        .and_then(|start| {
                            info.completed_at
                                .map(|end| (end - start).to_std().unwrap_or_default())
                        })
                        .unwrap_or_default()
                        .as_secs(),
                    average_speed: if let (Some(start), Some(end)) =
                        (info.started_at, info.completed_at)
                    {
                        let duration_secs = (end - start).num_seconds() as u64;
                        if duration_secs > 0 {
                            info.progress.downloaded_size / duration_secs
                        } else {
                            0
                        }
                    } else {
                        0
                    },
                    error: if let DownloadState::Failed(error) = &info.state {
                        Some(error.clone())
                    } else {
                        None
                    },
                    checksum_verified: None, // TODO: Implement checksum verification tracking
                });
            }

            // Wait a bit before checking again
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Get engine statistics
    pub async fn stats(&self) -> EngineStats {
        self.engine.stats().await
    }

    /// Get the event bus for subscribing to download events
    ///
    /// This allows you to receive real-time notifications about download
    /// progress, completion, failures, etc.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // todo)) add example
    /// ```
    pub fn event_bus(&self) -> Arc<EventBus> {
        self.engine.event_bus()
    }

    /// Subscribe to download events with a callback
    ///
    /// This is a convenience method that creates an event subscriber
    /// from a callback function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// zuup.on_event(|event| async move {
    ///     match event {
    ///         Event::DownloadCompleted { id, .. } => {
    ///             println!("Download {} completed!", id);
    ///         }
    ///         Event::DownloadFailed { id, error } => {
    ///             println!("Download {} failed: {}", id, error);
    ///         }
    ///         _ => {}
    ///     }
    ///     Ok(())
    /// }).await?;
    /// ```
    pub async fn on_event<F, Fut>(&self, callback: F) -> Result<()>
    where
        F: Fn(Event) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let subscriber = CallbackEventSubscriber::new(callback);
        self.event_bus().subscribe(Arc::new(subscriber)).await;
        Ok(())
    }

    /// Get current configuration
    pub async fn config(&self) -> ZuupConfig {
        self.engine.config().await
    }

    /// Update configuration
    ///
    /// [note]: Some configuration changes may require restarting active downloads
    /// to take effect.
    pub async fn update_config(&self, config: ZuupConfig) -> Result<()> {
        self.engine.update_config(config).await
    }

    /// Check if the engine is running
    pub async fn is_running(&self) -> bool {
        self.engine.is_running().await
    }

    /// Register a protocol handler
    ///
    /// This allows adding support for additional download protocols
    /// beyond the built-in ones.
    ///
    /// # Examples
    ///
    /// ```rust
    /// // todo)) add example
    /// ```
    pub async fn register_protocol_handler(
        &self,
        handler: Box<dyn crate::protocol::ProtocolHandler>,
    ) -> Result<()> {
        self.engine.register_protocol_handler(handler).await?;
        Ok(())
    }

    /// Get list of supported protocols
    ///
    /// Returns a list of protocol names that can be handled by registered handlers.
    pub async fn supported_protocols(&self) -> Vec<&'static str> {
        self.engine.supported_protocols().await
    }

    /// Shutdown the download manager
    ///
    /// If `force` is true, all downloads will be cancelled immediately.
    /// If `force` is false, active downloads will be paused and can be
    /// resumed when the engine is restarted.
    pub async fn shutdown(&self, force: bool) -> Result<()> {
        self.engine.shutdown(force).await
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
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ZuupConfig::default(),
        }
    }

    /// Set the download directory
    pub fn download_directory<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.config.general.download_dir = path.into();
        self
    }

    /// Set the maximum number of concurrent downloads
    pub fn max_concurrent_downloads(mut self, max: u32) -> Self {
        self.config.general.max_concurrent_downloads = max;
        self
    }

    /// Set the maximum overall download speed in bytes per second
    pub fn max_download_speed(mut self, speed: u64) -> Self {
        self.config.general.max_overall_download_speed = Some(speed);
        self
    }

    /// Set the maximum overall upload speed in bytes per second (for BitTorrent)
    pub fn max_upload_speed(mut self, speed: u64) -> Self {
        self.config.general.max_overall_upload_speed = Some(speed);
        self
    }

    /// Set the user agent string for HTTP requests
    pub fn user_agent<S: Into<String>>(mut self, user_agent: S) -> Self {
        self.config.network.user_agent = user_agent.into();
        self
    }

    /// Set the request timeout duration
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.network.timeout = timeout;
        self
    }

    /// Set the maximum number of connections per server
    pub fn max_connections_per_server(mut self, max: u32) -> Self {
        self.config.network.max_connections_per_server = max;
        self
    }

    /// Set the maximum number of retry attempts
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.config.network.max_tries = retries;
        self
    }

    /// Enable or disable session persistence
    pub fn session_file<P: Into<PathBuf>>(mut self, path: Option<P>) -> Self {
        self.config.general.session_file = path.map(|p| p.into());
        self
    }

    /// Set the auto-save interval for session data
    pub fn auto_save_interval(mut self, interval: Duration) -> Self {
        self.config.general.auto_save_interval = interval;
        self
    }

    /// Enable or disable certificate verification for HTTPS
    pub fn verify_certificates(mut self, verify: bool) -> Self {
        self.config.network.tls.verify_certificates = verify;
        self
    }

    /// Add a custom CA certificate file
    pub fn add_ca_certificate<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.config.network.tls.ca_certificates.push(path.into());
        self
    }

    /// Set a proxy for all connections
    pub fn proxy(mut self, proxy_url: Url) -> Self {
        self.config.network.proxy = Some(crate::types::ProxyConfig {
            url: proxy_url,
            auth: None,
        });
        self
    }

    /// Set a proxy with authentication
    pub fn proxy_with_auth(mut self, proxy_url: Url, username: String, password: String) -> Self {
        self.config.network.proxy = Some(crate::types::ProxyConfig {
            url: proxy_url,
            auth: Some(crate::types::ProxyAuth { username, password }),
        });
        self
    }

    /// Build the Zuup instance with the configured options
    pub async fn build(self) -> Result<Zuup> {
        Zuup::new(Some(self.config)).await
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
    /// Download ID
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
    pub average_speed: u64,

    /// Error message (if unsuccessful)
    pub error: Option<String>,

    /// Whether the checksum verification passed (if successful)
    pub checksum_verified: Option<bool>,
}

/// Event subscriber that wraps a callback function
struct CallbackEventSubscriber<F, Fut>
where
    F: Fn(Event) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    callback: F,
}

impl<F, Fut> CallbackEventSubscriber<F, Fut>
where
    F: Fn(Event) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    fn new(callback: F) -> Self {
        Self { callback }
    }
}

#[async_trait]
impl<F, Fut> EventSubscriber for CallbackEventSubscriber<F, Fut>
where
    F: Fn(Event) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    async fn handle_event(&self, event: Event) -> Result<()> {
        (self.callback)(event).await
    }
}
