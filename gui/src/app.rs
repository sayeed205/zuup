//! Main application structure for the GUI download manager

use std::sync::{Arc, Mutex};
use gpui::*;
use tokio::sync::mpsc;
use zuup_core::types::{DownloadRequest, DownloadInfo};

use crate::{
    config::ConfigManager,
    theme::ThemeManager,
    engine::{DownloadEngineAdapter, UIUpdateEvent},
};

/// Main application state that integrates all components
pub struct ZuupApp {
    /// Configuration manager for both GUI and core settings
    config_manager: Arc<Mutex<ConfigManager>>,
    
    /// Theme manager for handling theme switching
    theme_manager: Arc<Mutex<ThemeManager>>,
    
    /// Application state
    app_state: Arc<std::sync::RwLock<AppState>>,
    
    /// Download engine adapter (optional for simple mode)
    engine_adapter: Option<Arc<DownloadEngineAdapter>>,
    
    /// UI update event receiver
    ui_update_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<UIUpdateEvent>>>>,
}

/// Application state for UI components
#[derive(Debug, Clone)]
pub struct AppState {
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
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            filter_state: FilterState::All,
            sort_state: SortState::DateAdded,
            add_download_modal_open: false,
            settings_modal_open: false,
            last_error: None,
            is_connected: true,
        }
    }
}

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
    /// Create a new Zuup application instance (simplified version for window structure)
    pub fn new_simple() -> anyhow::Result<Self> {
        // Initialize configuration manager
        let config_manager = Arc::new(Mutex::new(ConfigManager::new()?));
        
        // Load configuration
        {
            let mut config = config_manager.lock().unwrap();
            config.load()?;
        }
        
        // Create theme manager
        let theme_manager = Arc::new(Mutex::new(
            ThemeManager::new(config_manager.clone())
        ));
        
        let app = Self {
            config_manager,
            theme_manager,
            app_state: Arc::new(std::sync::RwLock::new(AppState::default())),
            engine_adapter: None,
            ui_update_receiver: Arc::new(Mutex::new(None)),
        };
        
        tracing::info!("Zuup application created in simple mode");
        Ok(app)
    }
    
    /// Initialize the application with GPUI context (synchronous version)
    pub fn initialize_sync(&self, cx: &mut App) -> anyhow::Result<()> {
        // Initialize theme manager
        {
            let mut theme_manager = self.theme_manager.lock().unwrap();
            theme_manager.initialize(cx)?;
        }
        
        tracing::info!("Zuup application initialized successfully");
        Ok(())
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
    pub fn app_state(&self) -> &Arc<std::sync::RwLock<AppState>> {
        &self.app_state
    }
    
    /// Get the engine adapter (if available)
    pub fn engine_adapter(&self) -> Option<&Arc<DownloadEngineAdapter>> {
        self.engine_adapter.as_ref()
    }
    
    /// Initialize the engine adapter with full functionality
    pub async fn initialize_with_engine(&mut self) -> anyhow::Result<()> {
        let core_config = {
            let config = self.config_manager.lock().unwrap();
            config.core_config().clone()
        };
        
        // Create the engine adapter
        let adapter = Arc::new(DownloadEngineAdapter::new(core_config).await?);
        
        // Initialize the adapter
        adapter.initialize().await?;
        
        // Get the UI update receiver
        let receiver = adapter.take_ui_receiver().await;
        *self.ui_update_receiver.lock().unwrap() = receiver;
        
        self.engine_adapter = Some(adapter);
        
        tracing::info!("Engine adapter initialized successfully");
        Ok(())
    }
    
    /// Add a download using the engine adapter
    pub async fn add_download(&self, request: DownloadRequest) -> anyhow::Result<zuup_core::types::DownloadId> {
        if let Some(adapter) = &self.engine_adapter {
            let download_id = adapter.engine().add_download(request).await?;
            tracing::info!("Download added successfully: {:?}", download_id);
            Ok(download_id)
        } else {
            Err(anyhow::anyhow!("Engine adapter not initialized"))
        }
    }
    
    /// Get all downloads from the engine
    pub async fn get_downloads(&self) -> anyhow::Result<Vec<DownloadInfo>> {
        if let Some(adapter) = &self.engine_adapter {
            let downloads = adapter.engine().list_downloads().await?;
            Ok(downloads)
        } else {
            // Return empty list in simple mode
            Ok(Vec::new())
        }
    }
    
    /// Take the UI update receiver for event processing
    pub fn take_ui_receiver(&self) -> Option<mpsc::UnboundedReceiver<UIUpdateEvent>> {
        self.ui_update_receiver.lock().unwrap().take()
    }
    
    /// Shutdown the application
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        tracing::info!("Shutting down Zuup application");
        
        // Shutdown engine adapter if present
        if let Some(adapter) = &self.engine_adapter {
            adapter.shutdown().await?;
        }
        
        // Save configuration
        {
            let config = self.config_manager.lock().unwrap();
            config.save()?;
        }
        
        tracing::info!("Zuup application shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_app_creation() {
        let app = ZuupApp::new_simple().unwrap();
        
        // Verify app components are initialized
        let state = app.app_state.read().unwrap();
        assert_eq!(state.filter_state, FilterState::All);
        assert_eq!(state.sort_state, SortState::DateAdded);
    }
    
    #[test]
    fn test_app_shutdown() {
        let app = ZuupApp::new_simple().unwrap();
        
        // Shutdown should complete without errors
        app.shutdown().unwrap();
    }
}