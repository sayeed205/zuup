use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Config directory not found")]
    ConfigDirNotFound,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub general: GeneralConfig,
    pub network: NetworkConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    pub download_directory: PathBuf,
    pub max_concurrent_downloads: u32,
    pub auto_start_downloads: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub max_connections_per_download: u32,
    pub connection_timeout: u64,
    pub read_timeout: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuiConfig {
    pub theme: String,
    pub window_controls_visible: bool,
    pub window_size: (u32, u32),
    pub window_position: Option<(i32, i32)>,
    pub sidebar_width: u32,
    pub auto_start_downloads: bool,
    pub show_notifications: bool,
    pub minimize_to_tray: bool,
    pub start_minimized: bool,
    pub close_to_tray: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            general: GeneralConfig {
                download_directory: dirs::download_dir().unwrap_or_else(|| dirs::home_dir().unwrap_or_default()),
                max_concurrent_downloads: 3,
                auto_start_downloads: true,
            },
            network: NetworkConfig {
                max_connections_per_download: 8,
                connection_timeout: 30,
                read_timeout: 30,
            },
        }
    }
}

impl Default for GuiConfig {
    fn default() -> Self {
        Self {
            theme: "system".to_string(),
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

pub struct ConfigService {
    config_dir: PathBuf,
    app_config: Arc<Mutex<AppConfig>>,
    gui_config: Arc<Mutex<GuiConfig>>,
}

impl ConfigService {
    pub fn new() -> Result<Self, ConfigError> {
        let config_dir = dirs::config_dir()
            .ok_or(ConfigError::ConfigDirNotFound)?
            .join("zuup");

        // Ensure config directory exists
        std::fs::create_dir_all(&config_dir)?;

        let service = Self {
            config_dir,
            app_config: Arc::new(Mutex::new(AppConfig::default())),
            gui_config: Arc::new(Mutex::new(GuiConfig::default())),
        };

        // Load existing configs or create defaults
        service.load_configs()?;

        Ok(service)
    }

    fn load_configs(&self) -> Result<(), ConfigError> {
        // Load app config
        let app_config_path = self.config_dir.join("config.json");
        if app_config_path.exists() {
            let content = std::fs::read_to_string(&app_config_path)?;
            match serde_json::from_str::<AppConfig>(&content) {
                Ok(config) => *self.app_config.lock() = config,
                Err(e) => {
                    tracing::warn!("Failed to parse app config, using defaults: {}", e);
                    self.save_app_config()?;
                }
            }
        } else {
            self.save_app_config()?;
        }

        // Load GUI config
        let gui_config_path = self.config_dir.join("gui.json");
        if gui_config_path.exists() {
            let content = std::fs::read_to_string(&gui_config_path)?;
            match serde_json::from_str::<GuiConfig>(&content) {
                Ok(config) => *self.gui_config.lock() = config,
                Err(e) => {
                    tracing::warn!("Failed to parse GUI config, using defaults: {}", e);
                    self.save_gui_config()?;
                }
            }
        } else {
            self.save_gui_config()?;
        }

        Ok(())
    }

    pub fn get_app_config(&self) -> AppConfig {
        self.app_config.lock().clone()
    }

    pub fn get_gui_config(&self) -> GuiConfig {
        self.gui_config.lock().clone()
    }

    pub fn update_app_config(&self, config: AppConfig) -> Result<(), ConfigError> {
        *self.app_config.lock() = config;
        self.save_app_config()
    }

    pub fn update_gui_config(&self, config: GuiConfig) -> Result<(), ConfigError> {
        *self.gui_config.lock() = config;
        self.save_gui_config()
    }

    fn save_app_config(&self) -> Result<(), ConfigError> {
        let config = self.app_config.lock().clone();
        let content = serde_json::to_string_pretty(&config)?;
        let path = self.config_dir.join("config.json");
        std::fs::write(path, content)?;
        Ok(())
    }

    fn save_gui_config(&self) -> Result<(), ConfigError> {
        let config = self.gui_config.lock().clone();
        let content = serde_json::to_string_pretty(&config)?;
        let path = self.config_dir.join("gui.json");
        std::fs::write(path, content)?;
        Ok(())
    }
}