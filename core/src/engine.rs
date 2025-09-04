use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use url::Url;

use crate::Zuup;
use crate::config::ZuupConfig;
use crate::download::DownloadManager;
use crate::error::ZuupError;
use crate::event::{EventBus, EventSubscriber};
use crate::protocol::ProtocolRegistry;
use crate::session::SessionManager;
use crate::types::{DownloadId, DownloadInfo, DownloadRequest, DownloadState};

pub struct ZuupEngine {
    /// Configuration for the Zuup engine.
    config: Arc<RwLock<ZuupConfig>>,

    /// Session manager for persistence
    session_manager: Arc<SessionManager>,

    /// Download manager
    download_manager: Arc<DownloadManager>,

    /// Protocol registry
    protocol_registry: Arc<RwLock<ProtocolRegistry>>,

    /// Event bus for notifications
    event_bus: Arc<EventBus>,

    /// Engine state
    state: Arc<RwLock<EngineState>>,
}

/// Engine state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineState {
    /// Engine is starting up
    Starting,

    /// Engine is running normally
    Running,

    /// Engine is shutting down
    Stopping,

    /// Engine has stopped
    Stopped,
}

impl ZuupEngine {
    /// Create a new Zuup engine with given configuration
    pub async fn new(config: ZuupConfig) -> Result<Self> {
        let session_manager = Arc::new(SessionManager::new().await?);
        let protocol_registry = Arc::new(RwLock::new(ProtocolRegistry::new()));
        let download_manager = Arc::new(DownloadManager::new(
            config.general.max_concurrent_downloads,
            protocol_registry.clone(),
        ));
        let protocol_registry = protocol_registry;
        let event_bus = Arc::new(EventBus::new(1000));

        let engine = Self {
            config: Arc::new(RwLock::new(config)),
            session_manager,
            download_manager,
            protocol_registry,
            event_bus,
            state: Arc::new(RwLock::new(EngineState::Starting)),
        };

        // Initialize the engine
        engine.initialize().await?;

        Ok(engine)
    }

    /// Initialize the engine
    async fn initialize(&self) -> Result<()> {
        // Register protocol handlers
        self.register_protocol_handlers().await?;

        // Load session if configured
        if let Some(_session_file) = &self.config.read().await.general.session_file {
            if let Err(e) = self.session_manager.load().await {
                tracing::warn!(error = %e, "Failed to load session");
            }
        }

        // Start auto-save if enabled
        self.session_manager.start_auto_save().await?;

        // Set state to running
        *self.state.write().await = EngineState::Running;

        tracing::info!("Zuup engine initialized successfully");
        Ok(())
    }

    /// Register all protocol handlers
    async fn register_protocol_handlers(&self) -> Result<()> {
        let mut registry = self.protocol_registry.write().await;

        // todo)) Protocol handlers should be registered externally via register_protocol_handler()
        // This allows for modular protocol support through separate crates like zuup-protocols

        tracing::debug!("Protocol registry initialized, handlers can be registered externally");
        Ok(())
    }

    /// Add a media download using yt-dlp
    pub async fn add_media_download(
        &self,
        url: &str,
        _options: crate::media::MediaDownloadOptions,
    ) -> Result<DownloadId> {
        // TODO: Implement media download integration
        // For now, create a regular download request
        let parsed_url =
            url::Url::parse(url).map_err(|e| ZuupError::Config(format!("Invalid URL: {}", e)))?;

        let request = DownloadRequest::new(parsed_url);
        self.add_download(request).await
    }

    /// Get available media formats for a URL
    pub async fn get_media_formats(&self, url: &str) -> Result<Vec<crate::media::MediaFormat>> {
        // TODO: Implement yt-dlp format extraction
        // For now, return empty list
        let _ = url;
        Ok(Vec::new())
    }

    /// Add a new download
    pub async fn add_download(&self, request: DownloadRequest) -> Result<DownloadId> {
        // Check if engine is running
        if *self.state.read().await != EngineState::Running {
            return Err(ZuupError::Internal("Engine is not running".to_string()));
        }

        // Validate the request
        if request.urls.is_empty() {
            return Err(ZuupError::Config("No URLs provided".to_string()));
        }

        // Check if we have a handler for at least one URL
        let protocol_registry = self.protocol_registry.read().await;
        let has_handler = request
            .urls
            .iter()
            .any(|url| protocol_registry.find_handler(url).is_some());

        if !has_handler {
            return Err(ZuupError::Protocol(
                crate::error::ProtocolError::UnsupportedProtocol(
                    request.urls[0].scheme().to_string(),
                ),
            ));
        }

        // Add to download manager
        let id = self.download_manager.add_download(request, None).await?;

        // Get download info and add to session
        let info = self.download_manager.get_download(&id).await?;
        self.session_manager.add_download(info.clone()).await?;

        // Publish event
        self.event_bus
            .publish(crate::event::Event::DownloadAdded {
                id: id.clone(),
                info,
            })
            .await?;

        Ok(id)
    }

    /// Start a download
    pub async fn start_download(&self, id: &DownloadId) -> Result<()> {
        // Check if engine is running
        if *self.state.read().await != EngineState::Running {
            return Err(ZuupError::Internal("Engine is not running".to_string()));
        }

        // Start the download using the download manager
        self.download_manager.start_download(id).await?;

        // Get download info for the event
        let info = self.download_manager.get_download(id).await?;

        // Publish event
        self.event_bus
            .publish(crate::event::Event::DownloadStarted {
                id: id.clone(),
                info,
            })
            .await?;

        Ok(())
    }

    /// Pause a download
    pub async fn pause_download(&self, id: DownloadId) -> Result<()> {
        // Check if engine is running
        if *self.state.read().await != EngineState::Running {
            return Err(ZuupError::Internal("Engine is not running".to_string()));
        }

        // Get current download info
        let info = self.download_manager.get_download(&id).await?;

        // Check if download can be paused
        if !info.state.can_pause() {
            return Err(ZuupError::InvalidStateTransition {
                from: info.state.clone(),
                to: crate::types::DownloadState::Paused,
            });
        }

        // TODO)) Implement actual pause logic
        // This would involve stopping the download task and updating state

        // Update session
        let mut updated_info = info;
        updated_info.state = crate::types::DownloadState::Paused;
        self.session_manager.update_download(updated_info).await?;

        // Publish event
        self.event_bus
            .publish(crate::event::Event::DownloadPaused { id })
            .await?;

        Ok(())
    }

    /// Resume a download
    pub async fn resume_download(&self, id: DownloadId) -> Result<()> {
        // Check if engine is running
        if *self.state.read().await != EngineState::Running {
            return Err(ZuupError::Internal("Engine is not running".to_string()));
        }

        // Get current download info
        let info = self.download_manager.get_download(&id).await?;

        // Check if download can be resumed
        if !info.state.can_resume() {
            return Err(ZuupError::InvalidStateTransition {
                from: info.state.clone(),
                to: DownloadState::Active,
            });
        }

        // TODO: Implement actual resume logic
        // This would involve starting the download task and updating state

        // Update session
        let mut updated_info = info;
        updated_info.state = DownloadState::Active;
        self.session_manager.update_download(updated_info).await?;

        // Publish event
        self.event_bus
            .publish(crate::event::Event::DownloadResumed { id })
            .await?;

        Ok(())
    }

    /// Remove a download
    pub async fn remove_download(&self, id: DownloadId, force: bool) -> Result<()> {
        // Remove from download manager
        self.download_manager.remove_download(&id, force).await?;

        // Remove from session
        self.session_manager.remove_download(&id).await?;

        // Publish event
        self.event_bus
            .publish(crate::event::Event::DownloadRemoved { id })
            .await?;

        Ok(())
    }

    /// Get download information
    pub async fn get_download_info(&self, id: DownloadId) -> Result<DownloadInfo> {
        self.download_manager.get_download(&id).await
    }

    /// List all downloads
    pub async fn list_downloads(&self) -> Result<Vec<DownloadInfo>> {
        self.download_manager.list_downloads().await
    }

    /// Get engine statistics
    pub async fn stats(&self) -> EngineStats {
        let downloads = self
            .download_manager
            .list_downloads()
            .await
            .unwrap_or_default();
        let session_stats = self.session_manager.stats().await.unwrap_or_default();

        let mut stats = EngineStats::default();
        stats.total_downloads = downloads.len();
        stats.active_downloads = downloads.iter().filter(|d| d.state.is_active()).count();
        stats.completed_downloads = session_stats.completed;
        stats.failed_downloads = session_stats.failed;
        stats.total_downloaded = session_stats.downloaded_size;

        // Calculate current download speed
        stats.current_speed = downloads.iter().map(|d| d.progress.speed).sum();

        stats
    }

    /// Get the event bus for subscribing to download events
    ///
    /// This allows you to receive real-time notifications about download
    /// progress, completion, failures, etc.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let event_bus = zuup.event_bus();
    /// let subscriber = MyEventSubscriber::new();
    /// event_bus.subscribe(subscriber).await?;
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
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        let subscriber = CallbackEventSubscriber::new(callback);
        self.event_bus().subscribe(Arc::new(subscriber)).await;
        Ok(())
    }

    /// Get current configuration
    pub async fn config(&self) -> Config {
        self.engine.config().await
    }

    /// Update configuration
    ///
    /// Note: Some configuration changes may require restarting active downloads
    /// to take effect.
    pub async fn update_config(&self, config: Config) -> Result<()> {
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
    /// use zuup_protocols::HttpProtocolHandler;
    ///
    /// zuup.register_protocol_handler(Box::new(HttpProtocolHandler::new())).await?;
    /// ```
    pub async fn register_protocol_handler(
        &self,
        handler: Box<dyn crate::protocol::ProtocolHandler>,
    ) -> Result<()> {
        self.engine.register_protocol_handler(handler).await;
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
/// let zuup = Zuup::builder()
///     .download_directory("/home/user/Downloads")
///     .max_concurrent_downloads(10)
///     .max_download_speed(1024 * 1024) // 1 MB/s
///     .user_agent("MyApp/1.0")
///     .build()
///     .await?;
/// ```
pub struct ZuupBuilder {
    config: Config,
}

impl ZuupBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: Config::default(),
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
        Zuup::new(self.config).await
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
    pub id: DownloadId,

    /// Whether the download was successful
    pub success: bool,

    /// Path to the downloaded file (if successful)
    pub file_path: Option<PathBuf>,

    /// Total bytes downloaded
    pub bytes_downloaded: u64,

    /// Total time taken for the download
    pub duration: Duration,

    /// Average download speed in bytes per second
    pub average_speed: u64,

    /// Error message (if failed)
    pub error: Option<String>,

    /// Whether checksum verification passed (if applicable)
    pub checksum_verified: Option<bool>,
}

/// Event subscriber that wraps a callback function
struct CallbackEventSubscriber<F, Fut>
where
    F: Fn(Event) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<()>> + Send + 'static,
{
    callback: F,
}

impl<F, Fut> CallbackEventSubscriber<F, Fut>
where
    F: Fn(Event) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<()>> + Send + 'static,
{
    fn new(callback: F) -> Self {
        Self { callback }
    }
}

#[async_trait::async_trait]
impl<F, Fut> EventSubscriber for CallbackEventSubscriber<F, Fut>
where
    F: Fn(Event) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<()>> + Send + 'static,
{
    async fn handle_event(&self, event: Event) -> Result<()> {
        (self.callback)(event).await
    }
}
