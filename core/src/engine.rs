use std::{sync::Arc, time::Duration};

use tokio::sync::RwLock;

use crate::{
    config::ZuupConfig,
    download::DownloadManager,
    error::{Result, ZuupError},
    event::EventBus,
    protocol::ProtocolRegistry,
    session::SessionManager,
    types::{DownloadId, DownloadInfo, DownloadRequest, DownloadState},
};

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
        let session_manager = Arc::new(SessionManager::new()?);
        let protocol_registry = Arc::new(RwLock::new(ProtocolRegistry::new()));
        let download_manager = Arc::new(DownloadManager::new(
            config.general.max_concurrent_downloads,
            protocol_registry.clone(),
        ));
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
        // Check protocol availability and register handlers
        self.check_protocol_availability().await?;
        self.register_protocol_handlers().await?;

        // Load session if configured
        if self.config.read().await.general.session_file.is_some()
            && let Err(e) = self.session_manager.load().await
        {
            tracing::warn!(error = %e, "Failed to load session");
        }

        // Start auto-save if enabled
        self.session_manager.start_auto_save().await?;

        // Set state to running
        *self.state.write().await = EngineState::Running;

        tracing::info!("Zuup engine initialized successfully");
        Ok(())
    }

    /// Check protocol availability and log information
    async fn check_protocol_availability(&self) -> Result<()> {
        use crate::protocol::AvailableProtocols;
        
        let enabled_protocols = AvailableProtocols::list_enabled();
        let enabled_count = enabled_protocols.len();
        
        if enabled_count == 0 {
            tracing::warn!("No protocol handlers are enabled. The engine will start but downloads will fail unless protocols are registered manually.");
            tracing::info!("To enable protocols, add features like 'http', 'ftp', 'sftp', or 'torrent' to your Cargo.toml");
        } else {
            tracing::info!("Enabled protocols: {}", enabled_protocols.join(", "));
            tracing::debug!("Total enabled protocols: {}", enabled_count);
        }
        
        // Log available but disabled protocols for debugging
        let all_protocols = AvailableProtocols::list_all();
        let disabled_protocols: Vec<&str> = all_protocols
            .into_iter()
            .filter(|p| !enabled_protocols.contains(p))
            .collect();
            
        if !disabled_protocols.is_empty() {
            tracing::debug!("Disabled protocols: {}", disabled_protocols.join(", "));
        }
        
        Ok(())
    }

    /// Register all protocol handlers based on enabled features
    async fn register_protocol_handlers(&self) -> Result<()> {
        let registry = self.protocol_registry.read().await;
        
        // The ProtocolRegistry::new() already registers default handlers based on features,
        // but we can add additional logging and validation here
        
        let handler_count = registry.handlers().len();
        let supported_protocols = registry.supported_protocols();
        
        if handler_count == 0 {
            tracing::warn!("No protocol handlers registered. Downloads will fail unless handlers are registered manually via register_protocol_handler()");
        } else {
            tracing::info!("Registered {} protocol handler(s): {}", handler_count, supported_protocols.join(", "));
        }
        
        // Validate that enabled features match registered handlers
        use crate::protocol::AvailableProtocols;
        let enabled_protocols = AvailableProtocols::list_enabled();
        
        for protocol in &enabled_protocols {
            let has_handler = supported_protocols.contains(protocol);
            if !has_handler {
                tracing::warn!("Protocol '{}' is enabled but no handler is registered", protocol);
            }
        }
        
        tracing::debug!("Protocol registry initialization complete");
        Ok(())
    }

    /// Add a media download using yt-dlp
    pub async fn add_media_download(
        &self,
        url: &str,
        _options: crate::media::MediaDownloadOptions,
    ) -> Result<DownloadId> {
        // TODO)) Implement media download integration
        // For now, create a regular download request
        let parsed_url =
            url::Url::parse(url).map_err(|e| ZuupError::Config(format!("Invalid URL: {}", e)))?;

        let request = DownloadRequest::new(vec![parsed_url]);
        self.add_download(request).await
    }

    /// Get available media formats for a URL
    pub async fn get_media_formats(&self, url: &str) -> Result<Vec<crate::media::MediaFormat>> {
        // TODO)) Implement yt-dlp format extraction
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
        
        // Provide graceful degradation when no protocols are available
        if protocol_registry.is_empty() {
            use crate::protocol::AvailableProtocols;
            let available_features = AvailableProtocols::list_all();
            
            tracing::error!("No protocol handlers available. Enable features: {}", available_features.join(", "));
            return Err(ZuupError::Protocol(crate::error::ProtocolError::NoProtocolHandlers));
        }
        
        // Try to find a handler for each URL and provide detailed error messages
        for url in &request.urls {
            if let Err(e) = protocol_registry.find_handler_with_error(url) {
                // Log the specific protocol issue for debugging
                tracing::debug!("Protocol handler lookup failed for URL '{}': {}", url, e);
                return Err(e);
            }
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
                to: DownloadState::Paused,
            });
        }

        // TODO)) Implement actual pause logic
        // This would involve stopping the download task and updating state

        // Update session
        let mut updated_info = info;
        updated_info.state = DownloadState::Paused;
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

        // TODO)) Implement actual resume logic
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

        EngineStats {
            total_downloads: downloads.len(),
            active_downloads: downloads.iter().filter(|d| d.state.is_active()).count(),
            completed_downloads: session_stats.completed,
            failed_downloads: session_stats.failed,
            total_downloaded: session_stats.downloaded_size,
            download_speed: downloads.iter().map(|d| d.progress.download_speed).sum(),
            upload_speed: downloads.iter().map(|d| d.progress.upload_speed).sum(),
            ..Default::default()
        }
    }

    /// Get current configuration
    pub async fn config(&self) -> ZuupConfig {
        self.config.read().await.clone()
    }

    /// Update configuration
    ///
    /// Note: Some configuration changes may require restarting active downloads
    /// to take effect.
    pub async fn update_config(&self, config: ZuupConfig) -> Result<()> {
        *self.config.write().await = config;
        Ok(())
    }

    /// Get the event bus for subscribing to download events
    pub fn event_bus(&self) -> Arc<EventBus> {
        self.event_bus.clone()
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
        let mut registry = self.protocol_registry.write().await;
        registry.register(handler);
        Ok(())
    }

    /// Get list of supported protocols
    ///
    /// Returns a list of protocol names that can be handled by registered handlers.
    pub async fn supported_protocols(&self) -> Vec<&'static str> {
        let registry = self.protocol_registry.read().await;
        registry.supported_protocols()
    }

    /// Get detailed protocol availability information
    ///
    /// Returns information about which protocols are enabled at compile-time
    /// and which handlers are actually registered.
    pub async fn protocol_availability(&self) -> ProtocolAvailabilityInfo {
        use crate::protocol::{AvailableProtocols, get_protocol_feature_mappings};
        
        let registry = self.protocol_registry.read().await;
        let registered_protocols = registry.supported_protocols();
        let enabled_protocols = AvailableProtocols::list_enabled();
        let feature_mappings = get_protocol_feature_mappings();
        
        ProtocolAvailabilityInfo {
            enabled_protocols,
            registered_protocols,
            feature_mappings,
            has_any_handlers: !registry.is_empty(),
        }
    }

    /// Shutdown the download manager
    ///
    /// If `force` is true, all downloads will be cancelled immediately.
    /// If `force` is false, active downloads will be paused and can be
    /// resumed when the engine is restarted.
    pub async fn shutdown(&self, force: bool) -> Result<()> {
        *self.state.write().await = EngineState::Stopping;

        // Publish shutdown event
        self.event_bus
            .publish(crate::event::Event::SystemShutdown)
            .await?;

        // Stop all active downloads if not forced
        if !force {
            let downloads = self.list_downloads().await?;
            for download in downloads {
                if download.state.is_active() {
                    let _ = self.pause_download(download.id).await;
                }
            }
        }

        // Save session if configured
        if self.config.read().await.general.save_session_on_exit
            && let Err(e) = self.session_manager.save().await
        {
            tracing::error!(error = %e, "Failed to save session on shutdown");
        }

        // Set state to stopped
        *self.state.write().await = EngineState::Stopped;

        tracing::info!("Zuup engine shutdown complete");
        Ok(())
    }

    /// Check if the engine is running
    pub async fn is_running(&self) -> bool {
        *self.state.read().await == EngineState::Running
    }

    /// Get current engine state
    pub async fn state(&self) -> EngineState {
        self.state.read().await.clone()
    }

    /// Check if the engine can handle a specific URL
    ///
    /// This method provides graceful degradation by checking protocol availability
    /// before attempting to create a download.
    pub async fn can_handle_url(&self, url: &str) -> Result<bool> {
        let parsed_url = url::Url::parse(url)
            .map_err(|e| ZuupError::Config(format!("Invalid URL: {}", e)))?;
            
        let protocol_registry = self.protocol_registry.read().await;
        
        // Check if registry is empty
        if protocol_registry.is_empty() {
            return Ok(false);
        }
        
        // Check if we have a handler for this URL
        Ok(protocol_registry.find_handler(&parsed_url).is_some())
    }

    /// Get suggestions for enabling support for a URL
    ///
    /// Returns the feature flag that needs to be enabled to support the given URL.
    pub fn get_url_support_suggestion(&self, url: &str) -> Option<String> {
        if let Ok(parsed_url) = url::Url::parse(url) {
            if let Some(feature) = crate::protocol::get_required_feature_for_scheme(parsed_url.scheme()) {
                return Some(format!(
                    "To support '{}' URLs, enable the '{}' feature in your Cargo.toml: zuup-core = {{ features = [\"{}\"] }}",
                    parsed_url.scheme(),
                    feature,
                    feature
                ));
            }
        }
        None
    }
}

/// Engine statistics
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    /// Total number of downloads
    pub total_downloads: usize,

    /// Number of active downloads
    pub active_downloads: usize,

    /// Number of completed downloads
    pub completed_downloads: usize,

    /// Number of failed downloads
    pub failed_downloads: usize,

    /// Total bytes downloaded
    pub total_downloaded: u64,

    /// Current download speed (bytes/sec)
    pub download_speed: u64,

    /// Current upload speed (bytes/sec)
    pub upload_speed: Option<u64>,

    /// Engine uptime
    pub uptime: Duration,
}

/// Protocol availability information
#[derive(Debug, Clone)]
pub struct ProtocolAvailabilityInfo {
    /// Protocols enabled at compile-time via features
    pub enabled_protocols: Vec<&'static str>,
    
    /// Protocols with registered handlers
    pub registered_protocols: Vec<&'static str>,
    
    /// Complete feature mapping information
    pub feature_mappings: Vec<crate::protocol::ProtocolFeatureMapping>,
    
    /// Whether any handlers are registered
    pub has_any_handlers: bool,
}

impl ProtocolAvailabilityInfo {
    /// Check if a specific protocol is available
    pub fn is_protocol_available(&self, protocol: &str) -> bool {
        self.registered_protocols.contains(&protocol)
    }
    
    /// Get protocols that are enabled but not registered
    pub fn missing_handlers(&self) -> Vec<&'static str> {
        self.enabled_protocols
            .iter()
            .filter(|p| !self.registered_protocols.contains(p))
            .copied()
            .collect()
    }
    
    /// Get protocols that are registered but not enabled (shouldn't happen normally)
    pub fn unexpected_handlers(&self) -> Vec<&'static str> {
        self.registered_protocols
            .iter()
            .filter(|p| !self.enabled_protocols.contains(p))
            .copied()
            .collect()
    }
    
    /// Get the required feature for a URL scheme
    pub fn get_required_feature_for_url(&self, url: &str) -> Option<&'static str> {
        if let Ok(parsed_url) = url::Url::parse(url) {
            self.feature_mappings
                .iter()
                .find(|mapping| mapping.schemes.contains(&parsed_url.scheme()))
                .map(|mapping| mapping.feature)
        } else {
            None
        }
    }
}
