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
}
