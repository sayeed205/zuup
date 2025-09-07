//! Configuration management for Zuup

use std::{collections::HashMap, path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

use crate::{
    metrics::MetricsConfig,
    types::{ProxyConfig, TlsConfig},
};

/// Main configuration structure
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct ZuupConfig {
    /// General configuration
    pub general: GeneralConfig,

    /// Network configuration
    pub network: NetworkConfig,

    /// BitTorrent configuration
    pub bittorrent: BitTorrentConfig,

    /// HTTP server configuration (replaces RPC)
    pub server: Option<ServerConfig>,

    /// GUI application configuration
    pub gui: Option<GuiConfig>,

    /// Media download configuration
    pub media: MediaConfig,

    /// Logging configuration
    pub logging: LoggingConfig,

    /// Metrics configuration
    pub metrics: MetricsConfig,
}

/// General application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Maximum number of concurrent downloads
    pub max_concurrent_downloads: u32,

    /// Maximum overall download speed in bytes per second
    pub max_overall_download_speed: Option<u64>,

    /// Maximum overall upload speed in bytes per second
    pub max_overall_upload_speed: Option<u64>,

    /// Default download directory
    pub download_dir: PathBuf,

    /// Session file path
    pub session_file: Option<PathBuf>,

    /// Auto-save interval for session data
    #[serde(with = "duration_serde")]
    pub auto_save_interval: Duration,

    /// Whether to continue downloads on startup
    pub continue_on_startup: bool,

    /// Whether to save session on exit
    pub save_session_on_exit: bool,

    /// Maximum number of download history entries to keep
    pub max_download_history: usize,

    /// Download categories configuration
    pub categories: HashMap<String, CategoryConfig>,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            max_concurrent_downloads: 5,
            max_overall_download_speed: None,
            max_overall_upload_speed: None,
            download_dir: dirs::download_dir().unwrap_or_else(|| PathBuf::from(".")),
            session_file: None,
            auto_save_interval: Duration::from_secs(30),
            continue_on_startup: true,
            save_session_on_exit: true,
            max_download_history: 1000,
            categories: HashMap::new(),
        }
    }
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// User agent string
    pub user_agent: String,

    /// Request timeout
    #[serde(with = "duration_serde")]
    pub timeout: Duration,

    /// Maximum connections per server
    pub max_connections_per_server: u32,

    /// Maximum retry attempts
    pub max_tries: u32,

    /// Wait time between retries
    #[serde(with = "duration_serde")]
    pub retry_wait: Duration,

    /// Global proxy configuration
    pub proxy: Option<ProxyConfig>,

    /// TLS/SSL configuration
    pub tls: TlsConfig,

    /// DNS servers to use
    pub dns_servers: Vec<String>,

    /// Whether to use IPv6
    pub enable_ipv6: bool,

    /// Connection keep-alive timeout
    #[serde(with = "duration_serde")]
    pub keep_alive_timeout: Duration,

    /// Maximum number of redirects to follow
    pub max_redirects: u32,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            user_agent: format!("Zuup/{}", env!("CARGO_PKG_VERSION")),
            timeout: Duration::from_secs(30),
            max_connections_per_server: 4,
            max_tries: 3,
            retry_wait: Duration::from_secs(5),
            proxy: None,
            tls: TlsConfig::default(),
            dns_servers: Vec::new(),
            enable_ipv6: true,
            keep_alive_timeout: Duration::from_secs(90),
            max_redirects: 10,
        }
    }
}

/// BitTorrent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitTorrentConfig {
    /// Whether BitTorrent is enabled
    pub enabled: bool,

    /// DHT port
    pub dht_port: u16,

    /// Peer port range
    pub peer_port_range: (u16, u16),

    /// Maximum number of peers per torrent
    pub max_peers_per_torrent: u32,

    /// Maximum upload speed for seeding (bytes per second)
    pub max_upload_speed: Option<u64>,

    /// Seed ratio limit (stop seeding after this ratio)
    pub seed_ratio_limit: Option<f64>,

    /// Seed time limit (stop seeding after this time)
    #[serde(with = "option_duration_serde")]
    pub seed_time_limit: Option<Duration>,

    /// Whether to enable DHT
    pub enable_dht: bool,

    /// Whether to enable PEX (Peer Exchange)
    pub enable_pex: bool,

    /// Whether to enable encryption
    pub enable_encryption: bool,

    /// Tracker announce interval
    #[serde(with = "duration_serde")]
    pub announce_interval: Duration,
}

impl Default for BitTorrentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dht_port: 6881,
            peer_port_range: (6881, 6889),
            max_peers_per_torrent: 50,
            max_upload_speed: None,
            seed_ratio_limit: Some(2.0),
            seed_time_limit: Some(Duration::from_secs(24 * 3600)), // 24 hours
            enable_dht: true,
            enable_pex: true,
            enable_encryption: true,
            announce_interval: Duration::from_secs(1800), // 30 minutes
        }
    }
}

/// HTTP server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server bind address
    pub bind_address: String,

    /// Server port (default: 5775)
    pub port: u16,

    /// Whether to enable CORS
    pub enable_cors: bool,

    /// Allowed CORS origins
    pub cors_origins: Vec<String>,

    /// Session timeout duration
    #[serde(with = "duration_serde")]
    pub session_timeout: Duration,

    /// Maximum concurrent connections
    pub max_concurrent_connections: u32,

    /// Sync interval for web UI updates (milliseconds)
    pub sync_interval_ms: u64,

    /// API secret (generated automatically if not set)
    pub api_secret: Option<String>,

    /// Whether to enable authentication
    pub enable_auth: bool,

    /// Username for basic auth (if enabled)
    pub username: Option<String>,

    /// Password for basic auth (if enabled)
    pub password: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "127.0.0.1".to_string(),
            port: 5775,
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
            session_timeout: Duration::from_secs(3600), // 1 hour
            max_concurrent_connections: 100,
            sync_interval_ms: 1000, // 1 second
            api_secret: None,
            enable_auth: false,
            username: None,
            password: None,
        }
    }
}

/// GUI application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuiConfig {
    /// UI theme
    pub theme: String,

    /// Window size (width, height)
    pub window_size: (u32, u32),

    /// Window position (x, y) - None for center
    pub window_position: Option<(i32, i32)>,

    /// Whether to show desktop notifications
    pub show_notifications: bool,

    /// Whether to minimize to system tray
    pub minimize_to_tray: bool,

    /// Whether to start minimized
    pub start_minimized: bool,

    /// Port for browser extension communication
    pub extension_port: u16,

    /// Whether to enable browser extension server
    pub enable_extension_server: bool,

    /// Whether to auto-start with system
    pub auto_start: bool,

    /// Whether to close to tray instead of exit
    pub close_to_tray: bool,
}

impl Default for GuiConfig {
    fn default() -> Self {
        Self {
            theme: "dark".to_string(),
            window_size: (1200, 800),
            window_position: None,
            show_notifications: true,
            minimize_to_tray: true,
            start_minimized: false,
            extension_port: 5776,
            enable_extension_server: true,
            auto_start: false,
            close_to_tray: true,
        }
    }
}

/// Media download configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaConfig {
    /// Path to yt-dlp executable
    pub ytdlp_path: Option<PathBuf>,

    /// Default quality preference
    pub default_quality: String,

    /// Whether to prefer audio-only downloads
    pub prefer_audio_only: bool,

    /// Whether to download subtitles by default
    pub download_subtitles: bool,

    /// Default subtitle languages
    pub subtitle_languages: Vec<String>,

    /// Output filename template
    pub output_template: String,

    /// Whether to auto-update yt-dlp
    pub auto_update_ytdlp: bool,

    /// Whether to extract metadata for file naming
    pub extract_metadata: bool,

    /// Whether to download thumbnails
    pub download_thumbnails: bool,

    /// Maximum video resolution
    pub max_resolution: Option<String>,

    /// Preferred video format
    pub preferred_video_format: Option<String>,

    /// Preferred audio format
    pub preferred_audio_format: Option<String>,
}

impl Default for MediaConfig {
    fn default() -> Self {
        Self {
            ytdlp_path: None,
            default_quality: "best".to_string(),
            prefer_audio_only: false,
            download_subtitles: false,
            subtitle_languages: vec!["en".to_string()],
            output_template: "%(title)s.%(ext)s".to_string(),
            auto_update_ytdlp: false,
            extract_metadata: true,
            download_thumbnails: false,
            max_resolution: None,
            preferred_video_format: None,
            preferred_audio_format: Some("mp3".to_string()),
        }
    }
}

/// Category configuration for organizing downloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryConfig {
    /// Download directory for this category
    pub download_dir: PathBuf,

    /// Whether to auto-organize files by type
    pub auto_organize: bool,

    /// File naming template for this category
    pub file_naming_template: Option<String>,

    /// Whether to create subdirectories by date
    pub create_date_subdirs: bool,

    /// Maximum download speed for this category (bytes/sec)
    pub max_download_speed: Option<u64>,

    /// Priority for downloads in this category (1-10, higher = more priority)
    pub priority: u8,

    /// File extensions associated with this category
    pub file_extensions: Vec<String>,
}

impl Default for CategoryConfig {
    fn default() -> Self {
        Self {
            download_dir: dirs::download_dir().unwrap_or_else(|| PathBuf::from(".")),
            auto_organize: false,
            file_naming_template: None,
            create_date_subdirs: false,
            max_download_speed: None,
            priority: 5,
            file_extensions: Vec::new(),
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,

    /// Log format
    pub format: LogFormat,

    /// Whether to log to console
    pub console: bool,

    /// Log file path
    pub file: Option<PathBuf>,

    /// Maximum log file size in bytes
    pub max_file_size: u64,

    /// Number of log files to keep
    pub max_files: u32,

    /// Whether to enable colored output
    pub colored: bool,

    /// Whether to include timestamps
    pub timestamps: bool,

    /// Whether to include thread IDs
    pub thread_ids: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            format: LogFormat::Pretty,
            console: true,
            file: None,
            max_file_size: 10 * 1024 * 1024, // 10MB
            max_files: 5,
            colored: true,
            timestamps: true,
            thread_ids: false,
        }
    }
}

/// Log level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl From<LogLevel> for tracing::Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => tracing::Level::TRACE,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Error => tracing::Level::ERROR,
        }
    }
}

/// Log format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    /// Pretty human-readable format
    Pretty,
    /// JSON format
    Json,
    /// Compact format
    Compact,
}

/// Configuration loader and manager
pub struct ConfigManager {
    config: ZuupConfig,
}

impl ConfigManager {
    /// Create a new config manager with default configuration
    pub fn new() -> Self {
        Self {
            config: ZuupConfig::default(),
        }
    }

    /// Load configuration from file
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(&path)?;
        let config = match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("toml") => toml::from_str(&content)
                .map_err(|e| crate::error::ZuupError::Config(format!("TOML parse error: {}", e)))?,
            Some("yaml") | Some("yml") => serde_yaml::from_str(&content)
                .map_err(|e| crate::error::ZuupError::Config(format!("YAML parse error: {}", e)))?,
            Some("json") => serde_json::from_str(&content)
                .map_err(|e| crate::error::ZuupError::Config(format!("JSON parse error: {}", e)))?,
            _ => {
                return Err(crate::error::ZuupError::Config(
                    "Unsupported config file format".to_string(),
                ));
            }
        };

        Ok(Self { config })
    }

    /// Get the configuration
    pub fn config(&self) -> &ZuupConfig {
        &self.config
    }

    /// Get mutable configuration
    pub fn config_mut(&mut self) -> &mut ZuupConfig {
        &mut self.config
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> crate::error::Result<()> {
        let content = match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("toml") => toml::to_string_pretty(&self.config).map_err(|e| {
                crate::error::ZuupError::Config(format!("TOML serialize error: {}", e))
            })?,
            Some("yaml") | Some("yml") => serde_yaml::to_string(&self.config).map_err(|e| {
                crate::error::ZuupError::Config(format!("YAML serialize error: {}", e))
            })?,
            Some("json") => serde_json::to_string_pretty(&self.config).map_err(|e| {
                crate::error::ZuupError::Config(format!("JSON serialize error: {}", e))
            })?,
            _ => {
                return Err(crate::error::ZuupError::Config(
                    "Unsupported config file format".to_string(),
                ));
            }
        };

        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> crate::error::Result<()> {
        // Validate general config
        if self.config.general.max_concurrent_downloads == 0 {
            return Err(crate::error::ZuupError::Config(
                "max_concurrent_downloads must be greater than 0".to_string(),
            ));
        }

        // Validate network config
        if self.config.network.max_connections_per_server == 0 {
            return Err(crate::error::ZuupError::Config(
                "max_connections_per_server must be greater than 0".to_string(),
            ));
        }

        if self.config.network.max_tries == 0 {
            return Err(crate::error::ZuupError::Config(
                "max_tries must be greater than 0".to_string(),
            ));
        }

        // Validate BitTorrent config
        if self.config.bittorrent.peer_port_range.0 > self.config.bittorrent.peer_port_range.1 {
            return Err(crate::error::ZuupError::Config(
                "Invalid peer port range".to_string(),
            ));
        }

        // Validate server config
        if let Some(server_config) = &self.config.server {
            if server_config.port == 0 {
                return Err(crate::error::ZuupError::Config(
                    "Server port must be greater than 0".to_string(),
                ));
            }

            if server_config.max_concurrent_connections == 0 {
                return Err(crate::error::ZuupError::Config(
                    "max_concurrent_connections must be greater than 0".to_string(),
                ));
            }

            if server_config.enable_auth
                && (server_config.username.is_none() || server_config.password.is_none())
            {
                return Err(crate::error::ZuupError::Config(
                    "Username and password must be provided when authentication is enabled"
                        .to_string(),
                ));
            }
        }

        // Validate GUI config
        if let Some(gui_config) = &self.config.gui {
            if gui_config.window_size.0 == 0 || gui_config.window_size.1 == 0 {
                return Err(crate::error::ZuupError::Config(
                    "Window size must be greater than 0".to_string(),
                ));
            }

            if gui_config.extension_port == 0 {
                return Err(crate::error::ZuupError::Config(
                    "Extension port must be greater than 0".to_string(),
                ));
            }
        }

        // Validate media config
        if let Some(ytdlp_path) = &self.config.media.ytdlp_path
            && !ytdlp_path.exists()
        {
            return Err(crate::error::ZuupError::Config(format!(
                "yt-dlp executable not found at: {}",
                ytdlp_path.display()
            )));
        }

        // Validate categories
        for (name, category) in &self.config.general.categories {
            if name.is_empty() {
                return Err(crate::error::ZuupError::Config(
                    "Category name cannot be empty".to_string(),
                ));
            }

            if category.priority == 0 || category.priority > 10 {
                return Err(crate::error::ZuupError::Config(format!(
                    "Category '{}' priority must be between 1 and 10",
                    name
                )));
            }
        }

        Ok(())
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

// Helper module for Duration serialization
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

// Helper module for Option<Duration> serialization
mod option_duration_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match duration {
            Some(d) => serializer.serialize_some(&d.as_secs()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs_opt: Option<u64> = Option::deserialize(deserializer)?;
        Ok(secs_opt.map(Duration::from_secs))
    }
}
