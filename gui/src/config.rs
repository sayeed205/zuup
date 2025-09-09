//! Configuration management for the GUI application

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zuup_core::config::ZuupConfig;

/// GUI-specific configuration errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Configuration validation error: {0}")]
    Validation(String),
    #[error("Config directory creation failed: {0}")]
    DirectoryCreation(String),
}

pub type Result<T> = std::result::Result<T, ConfigError>;

/// Theme mode for the application
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ThemeMode {
    /// Follow system theme
    System,
    /// Always use light theme
    Light,
    /// Always use dark theme
    Dark,
}

impl Default for ThemeMode {
    fn default() -> Self {
        ThemeMode::System
    }
}

/// GUI-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuiConfig {
    /// Theme preference
    pub theme: ThemeMode,
    
    /// Whether to show window control buttons (minimize, maximize, close)
    pub window_controls_visible: bool,
    
    /// Window size (width, height)
    pub window_size: (u32, u32),
    
    /// Window position (x, y) - None for centered
    pub window_position: Option<(i32, i32)>,
    
    /// Sidebar width in pixels
    pub sidebar_width: u32,
    
    /// Whether to automatically start downloads when added
    pub auto_start_downloads: bool,
    
    /// Whether to show desktop notifications
    pub show_notifications: bool,
    
    /// Whether to minimize to system tray
    pub minimize_to_tray: bool,
    
    /// Whether to start minimized
    pub start_minimized: bool,
    
    /// Whether to close to tray instead of exit
    pub close_to_tray: bool,
}

impl Default for GuiConfig {
    fn default() -> Self {
        Self {
            theme: ThemeMode::System,
            window_controls_visible: true,
            window_size: (1200, 800),
            window_position: None,
            sidebar_width: 250,
            auto_start_downloads: true,
            show_notifications: true,
            minimize_to_tray: false,
            start_minimized: false,
            close_to_tray: false,
        }
    }
}

impl GuiConfig {
    /// Validate the GUI configuration
    pub fn validate(&self) -> Result<()> {
        if self.window_size.0 < 400 || self.window_size.1 < 300 {
            return Err(ConfigError::Validation(
                "Window size must be at least 400x300".to_string(),
            ));
        }
        
        if self.sidebar_width < 100 || self.sidebar_width > 500 {
            return Err(ConfigError::Validation(
                "Sidebar width must be between 100 and 500 pixels".to_string(),
            ));
        }
        
        Ok(())
    }
}

/// Configuration manager for both GUI and core configurations
pub struct ConfigManager {
    gui_config: GuiConfig,
    core_config: ZuupConfig,
    gui_config_path: PathBuf,
    core_config_path: PathBuf,
}

impl ConfigManager {
    /// Create a new configuration manager with default paths
    pub fn new() -> Result<Self> {
        let config_dir = Self::get_config_dir()?;
        let gui_config_path = config_dir.join("gui.json");
        let core_config_path = config_dir.join("config.json");
        
        Ok(Self {
            gui_config: GuiConfig::default(),
            core_config: ZuupConfig::default(),
            gui_config_path,
            core_config_path,
        })
    }
    
    /// Create a configuration manager with custom paths
    pub fn with_paths<P1: AsRef<Path>, P2: AsRef<Path>>(
        gui_config_path: P1,
        core_config_path: P2,
    ) -> Self {
        Self {
            gui_config: GuiConfig::default(),
            core_config: ZuupConfig::default(),
            gui_config_path: gui_config_path.as_ref().to_path_buf(),
            core_config_path: core_config_path.as_ref().to_path_buf(),
        }
    }
    
    /// Get the configuration directory path (~/.config/zuup/)
    pub fn get_config_dir() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| ConfigError::DirectoryCreation(
                "Could not determine config directory".to_string()
            ))?
            .join("zuup");
        
        // Create the directory if it doesn't exist
        if !config_dir.exists() {
            std::fs::create_dir_all(&config_dir)
                .map_err(|e| ConfigError::DirectoryCreation(
                    format!("Failed to create config directory {}: {}", config_dir.display(), e)
                ))?;
        }
        
        Ok(config_dir)
    }
    
    /// Load both GUI and core configurations from their respective files
    pub fn load(&mut self) -> Result<()> {
        self.load_gui_config()?;
        self.load_core_config()?;
        Ok(())
    }
    
    /// Load GUI configuration from file
    pub fn load_gui_config(&mut self) -> Result<()> {
        if self.gui_config_path.exists() {
            let content = std::fs::read_to_string(&self.gui_config_path)?;
            self.gui_config = serde_json::from_str(&content)?;
            self.gui_config.validate()?;
        } else {
            // Create default config file if it doesn't exist
            self.save_gui_config()?;
        }
        Ok(())
    }
    
    /// Load core configuration from file
    pub fn load_core_config(&mut self) -> Result<()> {
        if self.core_config_path.exists() {
            let content = std::fs::read_to_string(&self.core_config_path)?;
            self.core_config = serde_json::from_str(&content)?;
        } else {
            // Create default config file if it doesn't exist
            self.save_core_config()?;
        }
        Ok(())
    }
    
    /// Save both GUI and core configurations to their respective files
    pub fn save(&self) -> Result<()> {
        self.save_gui_config()?;
        self.save_core_config()?;
        Ok(())
    }
    
    /// Save GUI configuration to file
    pub fn save_gui_config(&self) -> Result<()> {
        // Validate before saving
        self.gui_config.validate()?;
        
        // Ensure parent directory exists
        if let Some(parent) = self.gui_config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let content = serde_json::to_string_pretty(&self.gui_config)?;
        std::fs::write(&self.gui_config_path, content)?;
        Ok(())
    }
    
    /// Save core configuration to file
    pub fn save_core_config(&self) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.core_config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let content = serde_json::to_string_pretty(&self.core_config)?;
        std::fs::write(&self.core_config_path, content)?;
        Ok(())
    }
    
    /// Get a reference to the GUI configuration
    pub fn gui_config(&self) -> &GuiConfig {
        &self.gui_config
    }
    
    /// Get a mutable reference to the GUI configuration
    pub fn gui_config_mut(&mut self) -> &mut GuiConfig {
        &mut self.gui_config
    }
    
    /// Get a reference to the core configuration
    pub fn core_config(&self) -> &ZuupConfig {
        &self.core_config
    }
    
    /// Get a mutable reference to the core configuration
    pub fn core_config_mut(&mut self) -> &mut ZuupConfig {
        &mut self.core_config
    }
    
    /// Get the GUI config file path
    pub fn gui_config_path(&self) -> &Path {
        &self.gui_config_path
    }
    
    /// Get the core config file path
    pub fn core_config_path(&self) -> &Path {
        &self.core_config_path
    }
    
    /// Update GUI configuration and save to file
    pub fn update_gui_config<F>(&mut self, updater: F) -> Result<()>
    where
        F: FnOnce(&mut GuiConfig),
    {
        updater(&mut self.gui_config);
        self.save_gui_config()
    }
    
    /// Update core configuration and save to file
    pub fn update_core_config<F>(&mut self, updater: F) -> Result<()>
    where
        F: FnOnce(&mut ZuupConfig),
    {
        updater(&mut self.core_config);
        self.save_core_config()
    }
    
    /// Reset GUI configuration to defaults
    pub fn reset_gui_config(&mut self) -> Result<()> {
        self.gui_config = GuiConfig::default();
        self.save_gui_config()
    }
    
    /// Reset core configuration to defaults
    pub fn reset_core_config(&mut self) -> Result<()> {
        self.core_config = ZuupConfig::default();
        self.save_core_config()
    }
    
    /// Reset both configurations to defaults
    pub fn reset_all(&mut self) -> Result<()> {
        self.reset_gui_config()?;
        self.reset_core_config()?;
        Ok(())
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new().expect("Failed to create default ConfigManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;
    
    #[test]
    fn test_gui_config_default() {
        let config = GuiConfig::default();
        assert_eq!(config.theme, ThemeMode::System);
        assert_eq!(config.window_controls_visible, true);
        assert_eq!(config.window_size, (1200, 800));
        assert_eq!(config.window_position, None);
        assert_eq!(config.sidebar_width, 250);
        assert_eq!(config.auto_start_downloads, true);
    }
    
    #[test]
    fn test_gui_config_validation() {
        let mut config = GuiConfig::default();
        
        // Valid config should pass
        assert!(config.validate().is_ok());
        
        // Invalid window size should fail
        config.window_size = (300, 200);
        assert!(config.validate().is_err());
        
        // Invalid sidebar width should fail
        config.window_size = (800, 600);
        config.sidebar_width = 50;
        assert!(config.validate().is_err());
        
        config.sidebar_width = 600;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = GuiConfig::default();
        
        // Test serialization
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.is_empty());
        
        // Test deserialization
        let deserialized: GuiConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.theme, config.theme);
        assert_eq!(deserialized.window_size, config.window_size);
    }
    
    #[test]
    fn test_config_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let gui_path = temp_dir.path().join("gui.json");
        let core_path = temp_dir.path().join("core.json");
        
        let manager = ConfigManager::with_paths(&gui_path, &core_path);
        assert_eq!(manager.gui_config_path(), gui_path);
        assert_eq!(manager.core_config_path(), core_path);
    }
    
    #[test]
    fn test_config_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let gui_path = temp_dir.path().join("gui.json");
        let core_path = temp_dir.path().join("core.json");
        
        let mut manager = ConfigManager::with_paths(&gui_path, &core_path);
        
        // Modify config
        manager.gui_config_mut().theme = ThemeMode::Dark;
        manager.gui_config_mut().window_size = (1600, 900);
        
        // Save config
        assert!(manager.save_gui_config().is_ok());
        assert!(gui_path.exists());
        
        // Load config in new manager
        let mut new_manager = ConfigManager::with_paths(&gui_path, &core_path);
        assert!(new_manager.load_gui_config().is_ok());
        
        assert_eq!(new_manager.gui_config().theme, ThemeMode::Dark);
        assert_eq!(new_manager.gui_config().window_size, (1600, 900));
    }
    
    #[test]
    fn test_config_update_methods() {
        let temp_dir = TempDir::new().unwrap();
        let gui_path = temp_dir.path().join("gui.json");
        let core_path = temp_dir.path().join("core.json");
        
        let mut manager = ConfigManager::with_paths(&gui_path, &core_path);
        
        // Test GUI config update
        assert!(manager.update_gui_config(|config| {
            config.theme = ThemeMode::Light;
            config.window_controls_visible = false;
        }).is_ok());
        
        assert_eq!(manager.gui_config().theme, ThemeMode::Light);
        assert_eq!(manager.gui_config().window_controls_visible, false);
        assert!(gui_path.exists());
    }
    
    #[test]
    fn test_config_reset() {
        let temp_dir = TempDir::new().unwrap();
        let gui_path = temp_dir.path().join("gui.json");
        let core_path = temp_dir.path().join("core.json");
        
        let mut manager = ConfigManager::with_paths(&gui_path, &core_path);
        
        // Modify config
        manager.gui_config_mut().theme = ThemeMode::Dark;
        manager.gui_config_mut().window_size = (1600, 900);
        
        // Reset config
        assert!(manager.reset_gui_config().is_ok());
        
        // Should be back to defaults
        assert_eq!(manager.gui_config().theme, ThemeMode::System);
        assert_eq!(manager.gui_config().window_size, (1200, 800));
    }
}