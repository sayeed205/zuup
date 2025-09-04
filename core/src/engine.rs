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
