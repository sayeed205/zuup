use std::{collections::HashMap, sync::Arc};

use tokio::sync::RwLock;

use crate::types::DownloadId;

/// Download manager responsible for coordinating downloads.
pub struct DownloadManager {
    /// All downloads(active, pending, completed, failed, stopped)
    downloads: Arc<RwLock<HashMap<DownloadId, Arc<DownloadTask>>>>,
}

#[derive(Clone)]
pub struct DownloadTask {
    /// Unique identifier
    id: DownloadId,
    // todo)) add more info about download task
}

impl DownloadTask {
    /// Create a new download task.
    pub fn new(id: DownloadId) -> Self {
        Self {
            id,
            // todo)) add more info about download task
        }
    }
}
