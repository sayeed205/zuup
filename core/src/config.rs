use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZuupConfig {
    /// General Configuration
    pub general: GeneralConfig,

    /// Network Configuration
    pub network: NetworkConfig,

    /// BitTorrent configuration
    pub bittorrent: BitTorrentConfig,
}

impl Default for ZuupConfig {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            network: NetworkConfig::default(),
            bittorrent: BitTorrentConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Maximum number of concurrent downloads
    pub max_concurrent_downloads: u32,
    // todo)) add more general config
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            max_concurrent_downloads: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// User Agent string
    pub user_agent: String,
    // todo)) add more network config
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            user_agent: format!("Zuup/{}", env!("CARGO_PKG_VERSION")),
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
