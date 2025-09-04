use std::sync::Arc;

use tokio::sync::RwLock;

use crate::config::ZuupConfig;
use crate::download::DownloadManager;
use crate::event::EventBus;
use crate::protocol::ProtocolRegistry;
use crate::session::SessionManager;
use crate::types::DownloadRequest;

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

    /// Initialize the engine
    async fn initialize(&self) -> Result<()> {
        // Register protocol handlers
        self.register_protocol_handlers().await?;

        // Load session if configured
        if let Some(_session_file) = &self.config.read().await.general.session_file {
            if let Err(e) = self.session_manager.load().await {
                tracing::warn!(error = %e, "Failed to load session");
            }
        }

        // Start auto-save if enabled
        self.session_manager.start_auto_save().await?;

        // Set state to running
        *self.state.write().await = EngineState::Running;

        tracing::info!("Zuup engine initialized successfully");
        Ok(())
    }

    /// Register all protocol handlers
    async fn register_protocol_handlers(&self) -> Result<()> {
        let mut registry = self.protocol_registry.write().await;

        // todo)) Protocol handlers should be registered externally via register_protocol_handler()
        // This allows for modular protocol support through separate crates like zuup-protocols

        tracing::debug!("Protocol registry initialized, handlers can be registered externally");
        Ok(())
    }
}
