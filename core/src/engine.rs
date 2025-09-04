use std::sync::Arc;

use tokio::sync::RwLock;

use crate::config::ZuupConfig;
use crate::download::DownloadManager;
use crate::event::EventBus;
use crate::session::SessionManager;

pub struct ZuupEngine {
    /// Configuration for the Zuup engine.
    config: Arc<RwLock<ZuupConfig>>,

    /// Session manager for persistence.
    session_manager: Arc<SessionManager>,

    /// Download manager for handling downloads.
    download_manager: Arc<DownloadManager>,

    /// Event bus for notifications.
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
        let config = Arc::new(RwLock::new(config));
        let session_manager = Arc::new(SessionManager::new(config.clone()).await?);
        let download_manager = Arc::new(DownloadManager::new(config.clone()).await?);
        let event_bus = Arc::new(EventBus::new(config.clone()).await?);
        let state = Arc::new(RwLock::new(EngineState::Starting));

        Ok(Self {
            config,
            session_manager,
            download_manager,
            event_bus,
            state,
        })
    }
}
