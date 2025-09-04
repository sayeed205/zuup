//! Session management and persistence using sled embedded database

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::download::DownloadInfo;
use crate::error::ZuupError;
use crate::types::DownloadId;

/// Session format version for backward compatibility
pub const SESSION_VERSION: u32 = 1;

/// Session metadata stored separately from downloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Session format version for compatibility
    pub version: u32,

    /// When the session was created
    pub created_at: DateTime<Utc>,

    /// When the session was last saved
    pub saved_at: DateTime<Utc>,

    /// Session-specific configuration overrides
    pub config_overrides: HashMap<String, serde_json::Value>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl SessionMetadata {
    /// Create new session metadata
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            version: SESSION_VERSION,
            created_at: now,
            saved_at: now,
            config_overrides: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Update the saved timestamp
    pub fn touch(&mut self) {
        self.saved_at = Utc::now();
    }
}

impl Default for SessionMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Session data that can be persisted and restored
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    /// Session metadata
    pub metadata: SessionMetadata,

    /// Downloads in this session
    pub downloads: HashMap<DownloadId, DownloadInfo>,
}

impl SessionData {
    /// Create a new session
    pub fn new() -> Self {
        Self {
            metadata: SessionMetadata::new(),
            downloads: HashMap::new(),
        }
    }

    /// Update the saved timestamp
    pub fn touch(&mut self) {
        self.metadata.touch();
    }

    /// Add a download to the session
    pub fn add_download(&mut self, info: DownloadInfo) {
        self.downloads.insert(info.id.clone(), info);
        self.touch();
    }

    /// Remove a download from the session
    pub fn remove_download(&mut self, id: &DownloadId) -> Option<DownloadInfo> {
        let result = self.downloads.remove(id);
        if result.is_some() {
            self.touch();
        }
        result
    }

    /// Update download information
    pub fn update_download(&mut self, info: DownloadInfo) {
        if self.downloads.contains_key(&info.id) {
            self.downloads.insert(info.id.clone(), info);
            self.touch();
        }
    }

    /// Get downloads that can be resumed
    pub fn resumable_downloads(&self) -> Vec<&DownloadInfo> {
        self.downloads
            .values()
            .filter(|info| info.state.can_resume())
            .collect()
    }

    /// Get active downloads (should be paused on restore)
    pub fn active_downloads(&self) -> Vec<&DownloadInfo> {
        self.downloads
            .values()
            .filter(|info| info.state.is_active())
            .collect()
    }

    /// Clean up completed and failed downloads
    pub fn cleanup_terminal_downloads(&mut self) {
        self.downloads.retain(|_, info| !info.state.is_terminal());
        self.touch();
    }
}

impl Default for SessionData {
    fn default() -> Self {
        Self::new()
    }
}

/// Keys used in the sled database
const METADATA_KEY: &[u8] = b"session_metadata";

/// Session storage backend using sled embedded database - todo))
pub struct SessionStorage {
    // Sled database instance
    db: sled::Db,

    /// Downloads tree for starting download information
    downloads_tree: sled::Tree,
}

impl SessionStorage {
    /// Create a new session storage with the given database path
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let db = sled::open(db_path)
            .map_err(|e| ZuupError::Session(format!("Failed to open session database: {}", e)))?;

        let downloads_tree = db
            .open_tree("downloads")
            .map_err(|e| ZuupError::Session(format!("Failed to open downloads tree: {}", e)))?;

        Ok(Self { db, downloads_tree })
    }

    /// Create an in-memory session storage for testing
    pub fn in_memory() -> Result<Self> {
        let config = sled::Config::new().temporary(true);
        let db = config.open().map_err(|e| {
            ZuupError::Session(format!("Failed to create in-memory database: {}", e))
        })?;

        let downloads_tree = db
            .open_tree("downloads")
            .map_err(|e| ZuupError::Session(format!("Failed to open downloads tree: {}", e)))?;

        Ok(Self { db, downloads_tree })
    }

    /// Save session metadata
    pub fn save_metadata(&self, metadata: &SessionMetadata) -> Result<()> {
        let data = serde_json::to_vec(metadata)
            .map_err(|e| ZuupError::Session(format!("Failed to serialize metadata: {}", e)))?;

        self.db
            .insert(METADATA_KEY, data)
            .map_err(|e| ZuupError::Session(format!("Failed to save metadata: {}", e)))?;

        self.db
            .flush()
            .map_err(|e| ZuupError::Session(format!("Failed to flush database: {}", e)))?;

        Ok(())
    }

    /// Load session metadata
    pub fn load_metadata(&self) -> Result<Option<SessionMetadata>> {
        if let Some(data) = self
            .db
            .get(METADATA_KEY)
            .map_err(|e| ZuupError::Session(format!("Failed to load metadata: {}", e)))?
        {
            let metadata: SessionMetadata = serde_json::from_slice(&data).map_err(|e| {
                ZuupError::Session(format!("Failed to deserialize metadata: {}", e))
            })?;

            // Check version compatibility
            if metadata.version > SESSION_VERSION {
                return Err(ZuupError::Session(format!(
                    "Unsupported session version: {} (current: {})",
                    metadata.version, SESSION_VERSION
                )));
            }

            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }

    /// Save a download to the database
    pub fn save_download(&self, download: &DownloadInfo) -> Result<()> {
        let key = format!("download:{}", download.id);
        let data = serde_json::to_vec(download)
            .map_err(|e| ZuupError::Session(format!("Failed to serialize download: {}", e)))?;

        self.downloads_tree
            .insert(key.as_bytes(), data)
            .map_err(|e| ZuupError::Session(format!("Failed to save download: {}", e)))?;

        Ok(())
    }

    /// Load a download from the database
    pub fn load_download(&self, id: &DownloadId) -> Result<Option<DownloadInfo>> {
        let key = format!("download:{}", id);

        if let Some(data) = self
            .downloads_tree
            .get(key.as_bytes())
            .map_err(|e| ZuupError::Session(format!("Failed to load download: {}", e)))?
        {
            let download: DownloadInfo = serde_json::from_slice(&data).map_err(|e| {
                ZuupError::Session(format!("Failed to deserialize download: {}", e))
            })?;

            Ok(Some(download))
        } else {
            Ok(None)
        }
    }

    /// Remove a download from the database
    pub fn remove_download(&self, id: &DownloadId) -> Result<Option<DownloadInfo>> {
        let key = format!("download:{}", id);

        if let Some(data) = self
            .downloads_tree
            .remove(key.as_bytes())
            .map_err(|e| ZuupError::Session(format!("Failed to remove download: {}", e)))?
        {
            let download: DownloadInfo = serde_json::from_slice(&data).map_err(|e| {
                ZuupError::Session(format!("Failed to deserialize download: {}", e))
            })?;

            Ok(Some(download))
        } else {
            Ok(None)
        }
    }

    /// Load all downloads from the database
    pub fn load_all_downloads(&self) -> Result<HashMap<DownloadId, DownloadInfo>> {
        let mut downloads = HashMap::new();

        for result in self.downloads_tree.iter() {
            let (key, value) = result
                .map_err(|e| ZuupError::Session(format!("Failed to iterate downloads: {}", e)))?;

            // Skip non-download keys
            if !key.starts_with(b"download:") {
                continue;
            }

            let download: DownloadInfo = serde_json::from_slice(&value).map_err(|e| {
                ZuupError::Session(format!("Failed to deserialize download: {}", e))
            })?;

            downloads.insert(download.id.clone(), download);
        }

        Ok(downloads)
    }

    /// Get the number of downloads in the database
    pub fn download_count(&self) -> Result<usize> {
        let count = self
            .downloads_tree
            .iter()
            .filter(|result| {
                if let Ok((key, _)) = result {
                    key.starts_with(b"download:")
                } else {
                    false
                }
            })
            .count();

        Ok(count)
    }

    /// Clean up terminal downloads (completed, failed, cancelled)
    pub fn cleanup_terminal_downloads(&self) -> Result<usize> {
        let mut removed_count = 0;
        let mut to_remove = Vec::new();

        // First, collect downloads to remove
        for result in self.downloads_tree.iter() {
            let (key, value) = result
                .map_err(|e| ZuupError::Session(format!("Failed to iterate downloads: {}", e)))?;

            if !key.starts_with(b"download:") {
                continue;
            }

            let download: DownloadInfo = serde_json::from_slice(&value).map_err(|e| {
                ZuupError::Session(format!("Failed to deserialize download: {}", e))
            })?;

            if download.state.is_terminal() {
                to_remove.push(key.to_vec());
            }
        }

        // Then remove them
        for key in to_remove {
            self.downloads_tree
                .remove(&key)
                .map_err(|e| ZuupError::Session(format!("Failed to remove download: {}", e)))?;
            removed_count += 1;
        }

        if removed_count > 0 {
            self.downloads_tree
                .flush()
                .map_err(|e| ZuupError::Session(format!("Failed to flush database: {}", e)))?;
        }

        Ok(removed_count)
    }

    /// Flush all pending writes to disk
    pub fn flush(&self) -> Result<()> {
        self.db
            .flush()
            .map_err(|e| ZuupError::Session(format!("Failed to flush database: {}", e)))?;
        Ok(())
    }

    /// Get database size information
    pub fn size_on_disk(&self) -> Result<u64> {
        self.db
            .size_on_disk()
            .map_err(|e| ZuupError::Session(format!("Failed to get database size: {}", e)))
    }

    /// Perform database maintenance (compaction)
    pub fn compact(&self) -> Result<()> {
        // Sled doesn't have explicit compaction, but we can trigger a flush
        self.flush()?;
        Ok(())
    }

    /// Check database integrity
    pub fn check_integrity(&self) -> Result<bool> {
        // Try to iterate through all data to check for corruption
        let mut download_count = 0;

        // Check metadata
        if let Some(_) = self.load_metadata()? {
            // Metadata is readable
        }

        // Check all downloads
        for result in self.downloads_tree.iter() {
            let (key, value) = result
                .map_err(|e| ZuupError::Session(format!("Database corruption detected: {}", e)))?;

            if key.starts_with(b"download:") {
                let _: DownloadInfo = serde_json::from_slice(&value).map_err(|e| {
                    ZuupError::Session(format!("Download data corruption detected: {}", e))
                })?;
                download_count += 1;
            }
        }

        tracing::debug!(
            downloads = download_count,
            "Session database integrity check completed"
        );

        Ok(true)
    }
}

pub struct SessionManager {
    /// Session storage backend
    storage: Arc<SessionStorage>,
}

impl SessionManager {
    /// Create a new session manager with given configuration
    pub async fn new() -> Result<Self> {
        let storage = Arc::new(SessionStorage::in_memory()?);
        Ok(Self { storage })
    }
}
