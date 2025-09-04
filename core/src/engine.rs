use std::sync::Arc;

use tokio::sync::RwLock;

use crate::config::ZuupConfig;
use crate::download::DownloadManager;
use crate::event::EventBus;
use crate::protocol::ProtocolRegistry;
use crate::session::SessionManager;

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

impl ZuupEngine {
    /// Create a new Zuup engine with given configuration
    pub async fn new(config: ZuupConfig) -> Result<Self> {
        let session_manager = Arc::new(SessionManager::new().await?);
        let protocol_registry = Arc::new(RwLock::new(ProtocolRegistry::new()));
        let download_manager = Arc::new(DownloadManager::new(
            config.general.max_concurrent_downloads,
            protocol_registry.clone(),
        ));
        let protocol_registry = protocol_registry;
        let event_bus = Arc::new(EventBus::new(1000));

        let engine = Self {
            config: Arc::new(RwLock::new(config)),
            session_manager,
            download_manager,
            protocol_registry,
            event_bus,
            state: Arc::new(RwLock::new(EngineState::Starting)),
        };

        // Initialize the engine
        engine.initialize().await?;

        Ok(engine)
    }
}
