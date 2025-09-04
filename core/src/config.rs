use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct ZuupConfig {
    /// General Configuration
    pub general: GeneralConfig,

    /// Network Configuration
    pub network: NetworkConfig,
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
            user_agent: format!("Ruso/{}", env!("CARGO_PKG_VERSION")),
        }
    }
}
