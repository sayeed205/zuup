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

/// Engine operational mode based on available protocol handlers
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineOperationalMode {
    /// All protocol handlers are available
    Full,
    /// Some protocol handlers are available
    Partial,
    /// No protocol handlers are available (degraded mode)
    Degraded,
}

impl ZuupEngine {
    /// Create a new Zuup engine with given configuration
    pub async fn new(config: ZuupConfig) -> Result<Self> {
        let session_manager = Arc::new(SessionManager::new()?);
        
        // Create protocol registry with feature-aware initialization
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

        // Initialize the engine with enhanced protocol checking
        engine.initialize().await?;

        Ok(engine)
    }

    /// Initialize the engine
    async fn initialize(&self) -> Result<()> {
        // Perform comprehensive protocol availability checking
        self.check_protocol_availability().await?;
        
        // Register and validate protocol handlers
        self.register_protocol_handlers().await?;
        
        // Validate that the engine can function with current protocol configuration
        self.validate_engine_functionality().await?;

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

    /// Check protocol availability and log comprehensive information
    async fn check_protocol_availability(&self) -> Result<()> {
        use crate::protocol::{AvailableProtocols, get_protocol_feature_mappings};
        
        let enabled_protocols = AvailableProtocols::list_enabled();
        let enabled_count = enabled_protocols.len();
        let feature_mappings = get_protocol_feature_mappings();
        
        // Log detailed protocol availability information
        tracing::info!("Protocol availability check starting...");
        
        if enabled_count == 0 {
            tracing::warn!("⚠️  No protocol handlers are enabled at compile-time!");
            tracing::warn!("   The engine will start in degraded mode - downloads will fail unless protocols are registered manually.");
            tracing::info!("💡 To enable protocols, add features to your Cargo.toml:");
            
            for mapping in &feature_mappings {
                tracing::info!("   - For {} support: zuup-core = {{ features = [\"{}\"] }}", 
                    mapping.schemes.join("/"), mapping.feature);
            }
            
            tracing::info!("   - For all protocols: zuup-core = {{ features = [\"all-protocols\"] }}");
        } else {
            tracing::info!("✅ Enabled protocols: {}", enabled_protocols.join(", "));
            tracing::debug!("   Total enabled protocols: {}", enabled_count);
            
            // Log which URL schemes are supported
            let supported_schemes: Vec<&str> = feature_mappings
                .iter()
                .filter(|mapping| mapping.enabled)
                .flat_map(|mapping| mapping.schemes.iter())
                .copied()
                .collect();
            
            if !supported_schemes.is_empty() {
                tracing::info!("📋 Supported URL schemes: {}", supported_schemes.join(", "));
            }
        }
        
        // Log available but disabled protocols for debugging
        let all_protocols = AvailableProtocols::list_all();
        let disabled_protocols: Vec<&str> = all_protocols
            .into_iter()
            .filter(|p| !enabled_protocols.contains(p))
            .collect();
            
        if !disabled_protocols.is_empty() {
            tracing::debug!("🔒 Disabled protocols: {}", disabled_protocols.join(", "));
            tracing::debug!("   These protocols are available but not enabled via features");
        }
        
        // Provide specific guidance based on common use cases
        if enabled_count == 0 {
            tracing::info!("🚀 Quick start suggestions:");
            tracing::info!("   - For web downloads: enable 'http' feature");
            tracing::info!("   - For file servers: enable 'ftp' or 'sftp' features");
            tracing::info!("   - For torrents: enable 'torrent' feature");
        }
        
        Ok(())
    }

    /// Register and validate protocol handlers based on enabled features
    async fn register_protocol_handlers(&self) -> Result<()> {
        let registry = self.protocol_registry.read().await;
        
        // The ProtocolRegistry::new() already registers default handlers based on features,
        // but we perform comprehensive validation and logging here
        
        let handler_count = registry.handlers().len();
        let supported_protocols = registry.supported_protocols();
        
        tracing::info!("Protocol handler registration check starting...");
        
        if handler_count == 0 {
            tracing::warn!("⚠️  No protocol handlers registered!");
            tracing::warn!("   Downloads will fail unless handlers are registered manually via register_protocol_handler()");
            tracing::info!("💡 This usually means no protocol features are enabled in Cargo.toml");
        } else {
            tracing::info!("✅ Registered {} protocol handler(s): {}", handler_count, supported_protocols.join(", "));
        }
        
        // Validate that enabled features match registered handlers
        use crate::protocol::{AvailableProtocols, get_protocol_feature_mappings};
        let enabled_protocols = AvailableProtocols::list_enabled();
        let _feature_mappings = get_protocol_feature_mappings();
        
        // Check for mismatches between enabled features and registered handlers
        let mut validation_issues = Vec::new();
        
        for protocol in &enabled_protocols {
            let has_handler = supported_protocols.contains(protocol);
            if !has_handler {
                validation_issues.push(format!("Protocol '{}' is enabled but no handler is registered", protocol));
                tracing::warn!("🔧 Protocol '{}' is enabled but no handler is registered", protocol);
            }
        }
        
        // Check for unexpected handlers (shouldn't happen normally)
        for protocol in &supported_protocols {
            let is_enabled = enabled_protocols.contains(protocol);
            if !is_enabled {
                validation_issues.push(format!("Handler for '{}' is registered but feature is not enabled", protocol));
                tracing::warn!("🤔 Handler for '{}' is registered but feature is not enabled", protocol);
            }
        }
        
        // Log detailed capability information for each registered handler
        if handler_count > 0 {
            tracing::debug!("📊 Handler capabilities:");
            for handler in registry.handlers() {
                let capabilities = handler.capabilities();
                tracing::debug!("   {} - resume: {}, segments: {}, auth: {}, proxy: {}", 
                    handler.protocol(),
                    capabilities.supports_resume,
                    capabilities.supports_segments,
                    capabilities.supports_auth,
                    capabilities.supports_proxy
                );
            }
        }
        
        // Provide actionable guidance if there are validation issues
        if !validation_issues.is_empty() {
            tracing::warn!("🔍 Protocol validation found {} issue(s):", validation_issues.len());
            for issue in &validation_issues {
                tracing::warn!("   - {}", issue);
            }
            
            tracing::info!("💡 This might indicate a configuration or build issue");
            tracing::info!("   Consider checking your Cargo.toml feature configuration");
        }
        
        tracing::debug!("Protocol registry validation complete");
        Ok(())
    }

    /// Validate that the engine can function with current protocol configuration
    async fn validate_engine_functionality(&self) -> Result<()> {
        use crate::protocol::AvailableProtocols;
        
        let registry = self.protocol_registry.read().await;
        let enabled_protocols = AvailableProtocols::list_enabled();
        let handler_count = registry.handlers().len();
        
        tracing::info!("Engine functionality validation starting...");
        
        // Determine engine operational mode
        let operational_mode = if handler_count == 0 {
            EngineOperationalMode::Degraded
        } else if enabled_protocols.len() == AvailableProtocols::list_all().len() {
            EngineOperationalMode::Full
        } else {
            EngineOperationalMode::Partial
        };
        
        match operational_mode {
            EngineOperationalMode::Full => {
                tracing::info!("🚀 Engine operational mode: FULL");
                tracing::info!("   All protocol handlers are available");
            },
            EngineOperationalMode::Partial => {
                tracing::info!("⚡ Engine operational mode: PARTIAL");
                tracing::info!("   Some protocol handlers are available: {}", 
                    registry.supported_protocols().join(", "));
                
                let missing_protocols: Vec<&str> = AvailableProtocols::list_all()
                    .into_iter()
                    .filter(|p| !enabled_protocols.contains(p))
                    .collect();
                
                if !missing_protocols.is_empty() {
                    tracing::info!("   Missing protocols: {}", missing_protocols.join(", "));
                }
            },
            EngineOperationalMode::Degraded => {
                tracing::warn!("🔧 Engine operational mode: DEGRADED");
                tracing::warn!("   No protocol handlers available - downloads will fail");
                tracing::warn!("   Manual handler registration required via register_protocol_handler()");
                
                // In degraded mode, we still allow the engine to start but warn about limitations
                tracing::info!("💡 The engine will start but with limited functionality:");
                tracing::info!("   - add_download() will fail for all URLs");
                tracing::info!("   - can_handle_url() will return false for all URLs");
                tracing::info!("   - Manual protocol handler registration is required");
            }
        }
        
        // Test basic registry functionality
        if handler_count > 0 {
            // Test with common URL schemes to verify handlers work
            let test_urls = vec![
                ("http://example.com", "http"),
                ("https://example.com", "http"),
                ("ftp://example.com", "ftp"),
                ("sftp://example.com", "sftp"),
                ("magnet:?xt=urn:btih:test", "torrent"),
            ];
            
            let mut working_schemes = Vec::new();
            for (test_url, expected_protocol) in test_urls {
                if let Ok(parsed_url) = url::Url::parse(test_url) {
                    if let Some(handler) = registry.find_handler(&parsed_url) {
                        if handler.protocol() == expected_protocol {
                            working_schemes.push(parsed_url.scheme().to_string());
                        }
                    }
                }
            }
            
            if !working_schemes.is_empty() {
                tracing::debug!("✅ Verified working URL schemes: {}", working_schemes.join(", "));
            }
        }
        
        // Log final engine readiness status
        match operational_mode {
            EngineOperationalMode::Full | EngineOperationalMode::Partial => {
                tracing::info!("✅ Engine validation complete - ready for downloads");
            },
            EngineOperationalMode::Degraded => {
                tracing::warn!("⚠️  Engine validation complete - degraded mode active");
            }
        }
        
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

    /// Add a new download with enhanced protocol validation and graceful degradation
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
        
        // Provide comprehensive graceful degradation when no protocols are available
        if protocol_registry.is_empty() {
            use crate::protocol::{AvailableProtocols, get_protocol_feature_mappings};
            
            let _available_features = AvailableProtocols::list_all();
            let feature_mappings = get_protocol_feature_mappings();
            
            tracing::error!("❌ Cannot add download: No protocol handlers available");
            tracing::error!("   Engine is running in degraded mode");
            tracing::info!("💡 To enable protocol support, add features to your Cargo.toml:");
            
            // Provide specific suggestions based on the URLs in the request
            let mut suggested_features = std::collections::HashSet::new();
            for url in &request.urls {
                if let Some(mapping) = feature_mappings.iter().find(|m| m.schemes.contains(&url.scheme())) {
                    suggested_features.insert(mapping.feature);
                    tracing::info!("   - For '{}': zuup-core = {{ features = [\"{}\"] }}", 
                        url.scheme(), mapping.feature);
                }
            }
            
            if suggested_features.is_empty() {
                tracing::info!("   - For all protocols: zuup-core = {{ features = [\"all-protocols\"] }}");
            }
            
            return Err(ZuupError::Protocol(crate::error::ProtocolError::NoProtocolHandlers));
        }
        
        // Try to find a handler for each URL and provide detailed error messages
        let mut unsupported_urls = Vec::new();
        for url in &request.urls {
            if let Err(e) = protocol_registry.find_handler_with_error(url) {
                // Log the specific protocol issue for debugging
                tracing::debug!("Protocol handler lookup failed for URL '{}': {}", url, e);
                unsupported_urls.push((url.clone(), e));
            }
        }
        
        // If any URLs are unsupported, provide comprehensive error information
        if !unsupported_urls.is_empty() {
            tracing::error!("❌ Cannot add download: {} unsupported URL(s)", unsupported_urls.len());
            
            for (url, error) in &unsupported_urls {
                tracing::error!("   - {}: {}", url, error);
                
                // Provide specific feature suggestions
                if let Some(suggestion) = self.get_url_support_suggestion(&url.to_string()) {
                    tracing::info!("     💡 {}", suggestion);
                }
            }
            
            // Return the first error (they should all be similar)
            return Err(unsupported_urls.into_iter().next().unwrap().1);
        }

        // All URLs are supported, proceed with download creation
        tracing::debug!("✅ All URLs in request are supported by available handlers");

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

        tracing::info!("✅ Download added successfully: {}", id);
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

    /// Get comprehensive engine status including protocol availability
    pub async fn engine_status(&self) -> EngineStatus {
        let state = self.state().await;
        let protocol_availability = self.protocol_availability().await;
        let stats = self.stats().await;
        
        let operational_mode = if !protocol_availability.has_any_handlers {
            EngineOperationalMode::Degraded
        } else if protocol_availability.enabled_protocols.len() == crate::protocol::AvailableProtocols::list_all().len() {
            EngineOperationalMode::Full
        } else {
            EngineOperationalMode::Partial
        };
        
        EngineStatus {
            state,
            operational_mode: operational_mode.clone(),
            protocol_availability,
            stats,
            is_functional: matches!(operational_mode, EngineOperationalMode::Full | EngineOperationalMode::Partial),
        }
    }

    /// Check if the engine can handle a specific URL
    ///
    /// This method provides graceful degradation by checking protocol availability
    /// before attempting to create a download.
    pub async fn can_handle_url(&self, url: &str) -> Result<bool> {
        let parsed_url = url::Url::parse(url)
            .map_err(|e| ZuupError::Config(format!("Invalid URL: {}", e)))?;
            
        let protocol_registry = self.protocol_registry.read().await;
        
        // Graceful degradation: if registry is empty, we can't handle any URLs
        if protocol_registry.is_empty() {
            tracing::debug!("Cannot handle URL '{}': No protocol handlers available", url);
            return Ok(false);
        }
        
        // Check if we have a handler for this URL
        let can_handle = protocol_registry.find_handler(&parsed_url).is_some();
        
        if can_handle {
            tracing::debug!("✅ Can handle URL '{}' with available handlers", url);
        } else {
            tracing::debug!("❌ Cannot handle URL '{}' with current handlers", url);
            
            // Provide helpful debug information about why it can't be handled
            if let Some(required_feature) = crate::protocol::get_required_feature_for_scheme(parsed_url.scheme()) {
                tracing::debug!("   Required feature '{}' may not be enabled", required_feature);
            } else {
                tracing::debug!("   Unknown or unsupported protocol: {}", parsed_url.scheme());
            }
        }
        
        Ok(can_handle)
    }

    /// Get suggestions for enabling support for a URL
    ///
    /// Returns comprehensive guidance for enabling support for the given URL.
    pub fn get_url_support_suggestion(&self, url: &str) -> Option<String> {
        if let Ok(parsed_url) = url::Url::parse(url) {
            use crate::protocol::{get_required_feature_for_scheme, get_protocol_feature_mappings};
            
            if let Some(feature) = get_required_feature_for_scheme(parsed_url.scheme()) {
                // Find the mapping to get more detailed information
                let mappings = get_protocol_feature_mappings();
                if let Some(mapping) = mappings.iter().find(|m| m.feature == feature) {
                    let schemes_text = if mapping.schemes.len() > 1 {
                        format!("{} URLs", mapping.schemes.join("/"))
                    } else {
                        format!("{} URLs", mapping.schemes[0])
                    };
                    
                    return Some(format!(
                        "To support {}, enable the '{}' feature: zuup-core = {{ features = [\"{}\"] }}",
                        schemes_text,
                        feature,
                        feature
                    ));
                } else {
                    return Some(format!(
                        "To support '{}' URLs, enable the '{}' feature: zuup-core = {{ features = [\"{}\"] }}",
                        parsed_url.scheme(),
                        feature,
                        feature
                    ));
                }
            } else {
                return Some(format!(
                    "The '{}' protocol is not supported by this library. Supported protocols require features: http, ftp, sftp, torrent",
                    parsed_url.scheme()
                ));
            }
        }
        
        Some("Invalid URL format - cannot provide protocol support suggestions".to_string())
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

/// Comprehensive engine status information
#[derive(Debug, Clone)]
pub struct EngineStatus {
    /// Current engine state
    pub state: EngineState,
    
    /// Operational mode based on available protocols
    pub operational_mode: EngineOperationalMode,
    
    /// Protocol availability information
    pub protocol_availability: ProtocolAvailabilityInfo,
    
    /// Engine statistics
    pub stats: EngineStats,
    
    /// Whether the engine is functional for downloads
    pub is_functional: bool,
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
