//! Main application structure for the GUI download manager

use std::sync::{Arc, Mutex};
use gpui::*;

use crate::{
    config::ConfigManager,
    theme::ThemeManager,
};

/// Main application state that integrates all components
pub struct ZuupApp {
    /// Configuration manager for both GUI and core settings
    config_manager: Arc<Mutex<ConfigManager>>,
    
    /// Theme manager for handling theme switching
    theme_manager: Arc<Mutex<ThemeManager>>,
    
    /// Application state
    app_state: Arc<std::sync::RwLock<AppState>>,
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
    
    /// Shutdown the application
    pub fn shutdown(&self) -> anyhow::Result<()> {
        tracing::info!("Shutting down Zuup application");
        
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