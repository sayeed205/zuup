//! Main application structure integrating the download engine adapter

use std::sync::{Arc, Mutex};
use gpui::*;
use tokio::sync::{mpsc, RwLock};


use crate::{
    config::ConfigManager,
    engine::{DownloadEngineAdapter, UIUpdateEvent, EngineAdapterConfig},
    theme::ThemeManager,
};

/// Main application state that integrates all components
pub struct ZuupApp {
    /// Configuration manager for both GUI and core settings
    config_manager: Arc<Mutex<ConfigManager>>,
    
    /// Theme manager for handling theme switching
    theme_manager: Arc<Mutex<ThemeManager>>,
    
    /// Download engine adapter for core integration
    engine_adapter: Arc<DownloadEngineAdapter>,
    
    /// UI update event receiver
    ui_event_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<UIUpdateEvent>>>>,
    
    /// Application state
    app_state: Arc<RwLock<AppState>>,
}

/// Application state for UI components
#[derive(Debug, Clone)]
pub struct AppState {
    /// Current downloads list
    pub downloads: Vec<zuup_core::types::DownloadInfo>,
    
    /// Selected download IDs
    pub selected_downloads: std::collections::HashSet<zuup_core::types::DownloadId>,
    
    /// Filter state for download list
    pub filter_state: FilterState,
    
    /// Sort state for download list
    pub sort_state: SortState,
    
    /// Whether the add download modal is open
    pub add_download_modal_open: bool,
    
    /// Whether the settings modal is open
    pub settings_modal_open: bool,
    
    /// Last error message (if any)
    pub last_error: Option<String>,
    
    /// Connection status
    pub is_connected: bool,
    
    /// Overall download statistics
    pub total_download_speed: u64,
    pub total_upload_speed: u64,
    pub active_downloads_count: usize,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            downloads: Vec::new(),
            selected_downloads: std::collections::HashSet::new(),
            filter_state: FilterState::All,
            sort_state: SortState::DateAdded,
            add_download_modal_open: false,
            settings_modal_open: false,
            last_error: None,
            is_connected: true,
            total_download_speed: 0,
            total_upload_speed: 0,
            active_downloads_count: 0,
        }
    }
}

impl Global for AppState {}

/// Filter state for the download list
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterState {
    All,
    Active,
    Completed,
    Failed,
    Paused,
}

/// Sort state for the download list
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SortState {
    Name,
    Size,
    Progress,
    Speed,
    DateAdded,
}

impl ZuupApp {
    /// Create a new Zuup application instance
    pub async fn new() -> anyhow::Result<Self> {
        // Initialize configuration manager
        let config_manager = Arc::new(Mutex::new(ConfigManager::new()?));
        
        // Load configuration
        {
            let mut config = config_manager.lock().unwrap();
            config.load()?;
        }
        
        // Get core configuration for engine
        let core_config = {
            let config = config_manager.lock().unwrap();
            config.core_config().clone()
        };
        
        // Create engine adapter with custom configuration
        let adapter_config = EngineAdapterConfig {
            max_ui_buffer_size: 2000,
            progress_batch_interval_ms: 50, // More frequent updates for better UX
            max_batch_size: 100,
            enable_progress_batching: true,
        };
        
        let engine_adapter = Arc::new(
            DownloadEngineAdapter::with_config(core_config, adapter_config).await?
        );
        
        // Initialize the engine adapter
        engine_adapter.initialize().await?;
        
        // Take the UI event receiver
        let ui_event_receiver = Arc::new(RwLock::new(
            engine_adapter.take_ui_receiver().await
        ));
        
        // Create theme manager
        let theme_manager = Arc::new(Mutex::new(
            ThemeManager::new(config_manager.clone())
        ));
        
        let app = Self {
            config_manager,
            theme_manager,
            engine_adapter,
            ui_event_receiver,
            app_state: Arc::new(RwLock::new(AppState::default())),
        };
        
        // Load initial downloads
        app.load_initial_downloads().await?;
        
        Ok(app)
    }
    
    /// Initialize the application with GPUI context
    pub async fn initialize(&self, cx: &mut App) -> anyhow::Result<()> {
        // Initialize theme manager
        {
            let mut theme_manager = self.theme_manager.lock().unwrap();
            theme_manager.initialize(cx)?;
        }
        
        tracing::info!("Zuup application initialized successfully");
        Ok(())
    }
    
    /// Get the UI event receiver for manual processing
    pub async fn take_ui_receiver(&self) -> Option<mpsc::UnboundedReceiver<UIUpdateEvent>> {
        let mut receiver_guard = self.ui_event_receiver.write().await;
        receiver_guard.take()
    }
    
    /// Handle a UI update event
    pub async fn handle_ui_event(
        event: UIUpdateEvent,
        app_state: &Arc<RwLock<AppState>>,
    ) -> anyhow::Result<()> {
        match event {
            UIUpdateEvent::DownloadAdded(info) => {
                let mut state = app_state.write().await;
                state.downloads.push(info);
                state.active_downloads_count = state.downloads.iter()
                    .filter(|d| d.state.is_active())
                    .count();
            }
            
            UIUpdateEvent::ProgressUpdated { id, progress, .. } => {
                let mut state = app_state.write().await;
                if let Some(download) = state.downloads.iter_mut().find(|d| d.id == id) {
                    download.progress = progress;
                }
                
                // Update overall statistics
                state.total_download_speed = state.downloads.iter()
                    .map(|d| d.progress.download_speed)
                    .sum();
                
                state.total_upload_speed = state.downloads.iter()
                    .filter_map(|d| d.progress.upload_speed)
                    .sum();
            }
            
            UIUpdateEvent::StateChanged { id, new_state, .. } => {
                let mut state = app_state.write().await;
                if let Some(download) = state.downloads.iter_mut().find(|d| d.id == id) {
                    download.state = new_state;
                }
                
                state.active_downloads_count = state.downloads.iter()
                    .filter(|d| d.state.is_active())
                    .count();
            }
            
            UIUpdateEvent::DownloadCompleted { id, info } => {
                let mut state = app_state.write().await;
                if let Some(download) = state.downloads.iter_mut().find(|d| d.id == id) {
                    *download = info;
                }
                
                state.active_downloads_count = state.downloads.iter()
                    .filter(|d| d.state.is_active())
                    .count();
            }
            
            UIUpdateEvent::DownloadFailed { id, error } => {
                let mut state = app_state.write().await;
                if let Some(download) = state.downloads.iter_mut().find(|d| d.id == id) {
                    download.state = zuup_core::types::DownloadState::Failed(error.clone());
                    download.error_message = Some(error);
                }
                
                state.active_downloads_count = state.downloads.iter()
                    .filter(|d| d.state.is_active())
                    .count();
            }
            
            UIUpdateEvent::DownloadRemoved(id) => {
                let mut state = app_state.write().await;
                state.downloads.retain(|d| d.id != id);
                state.selected_downloads.remove(&id);
                
                state.active_downloads_count = state.downloads.iter()
                    .filter(|d| d.state.is_active())
                    .count();
            }
            
            UIUpdateEvent::NetworkStatusChanged { connected } => {
                let mut state = app_state.write().await;
                state.is_connected = connected;
            }
            
            UIUpdateEvent::BatchUpdate(events) => {
                // Process batch updates efficiently without recursion
                for event in events {
                    // Handle each event type directly to avoid recursion
                    match event {
                        UIUpdateEvent::ProgressUpdated { id, progress, .. } => {
                            let mut state = app_state.write().await;
                            if let Some(download) = state.downloads.iter_mut().find(|d| d.id == id) {
                                download.progress = progress;
                            }
                            
                            // Update overall statistics
                            state.total_download_speed = state.downloads.iter()
                                .map(|d| d.progress.download_speed)
                                .sum();
                            
                            state.total_upload_speed = state.downloads.iter()
                                .filter_map(|d| d.progress.upload_speed)
                                .sum();
                        }
                        // For non-progress events, we can handle them individually
                        other_event => {
                            // Use Box::pin to handle recursion properly
                            Box::pin(Self::handle_ui_event(other_event, app_state)).await?;
                        }
                    }
                }
            }
            
            UIUpdateEvent::SystemShutdown => {
                tracing::info!("System shutdown event received");
                // Handle graceful shutdown
            }
            
            _ => {
                // Handle other events as needed
                tracing::debug!("Unhandled UI event: {:?}", event);
            }
        }
        
        Ok(())
    }
    
    /// Load initial downloads from the engine
    async fn load_initial_downloads(&self) -> anyhow::Result<()> {
        let downloads = self.engine_adapter.engine().list_downloads().await?;
        
        let mut state = self.app_state.write().await;
        state.downloads = downloads;
        state.active_downloads_count = state.downloads.iter()
            .filter(|d| d.state.is_active())
            .count();
        
        Ok(())
    }
    
    /// Get the engine adapter
    pub fn engine_adapter(&self) -> &Arc<DownloadEngineAdapter> {
        &self.engine_adapter
    }
    
    /// Get the configuration manager
    pub fn config_manager(&self) -> &Arc<Mutex<ConfigManager>> {
        &self.config_manager
    }
    
    /// Get the theme manager
    pub fn theme_manager(&self) -> &Arc<Mutex<ThemeManager>> {
        &self.theme_manager
    }
    
    /// Get the application state
    pub fn app_state(&self) -> &Arc<RwLock<AppState>> {
        &self.app_state
    }
    
    /// Shutdown the application
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        tracing::info!("Shutting down Zuup application");
        
        // Save configuration
        {
            let config = self.config_manager.lock().unwrap();
            config.save()?;
        }
        
        // Shutdown engine adapter
        self.engine_adapter.shutdown().await?;
        
        tracing::info!("Zuup application shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_app_creation() {
        let app = ZuupApp::new().await.unwrap();
        
        // Verify app components are initialized
        assert!(app.engine_adapter.engine().is_running().await);
        
        let state = app.app_state.read().await;
        assert_eq!(state.filter_state, FilterState::All);
        assert_eq!(state.sort_state, SortState::DateAdded);
    }
    
    #[tokio::test]
    async fn test_app_shutdown() {
        let app = ZuupApp::new().await.unwrap();
        
        // Shutdown should complete without errors
        app.shutdown().await.unwrap();
    }
}