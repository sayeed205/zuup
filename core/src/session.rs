//! Session management and persistence using sled embedded database

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::error::ZuupError;
use crate::types::{DownloadId, DownloadInfo, DownloadState};

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

/// Session manager for handling persistence and recovery
pub struct SessionManager {
    /// Session storage backend
    storage: Arc<SessionStorage>,

    /// Current session metadata
    metadata: Arc<RwLock<SessionMetadata>>,

    /// Auto-save interval in seconds
    auto_save_interval: Option<u64>,

    /// Whether auto-save is enabled
    auto_save_enabled: bool,

    /// Auto-save task handle
    auto_save_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl SessionManager {
    /// Create a new session manager with in-memory storage
    pub fn new() -> Result<Self> {
        let storage = Arc::new(SessionStorage::in_memory()?);
        let metadata = Arc::new(RwLock::new(SessionMetadata::new()));

        Ok(Self {
            storage,
            metadata,
            auto_save_interval: Some(30), // 30 seconds default
            auto_save_enabled: true,
            auto_save_handle: Arc::new(RwLock::new(None)),
        })
    }

    /// Create a session manager with a specific database path
    pub fn with_database<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let storage = Arc::new(SessionStorage::new(db_path)?);
        let metadata = Arc::new(RwLock::new(SessionMetadata::new()));

        Ok(Self {
            storage,
            metadata,
            auto_save_interval: Some(30),
            auto_save_enabled: true,
            auto_save_handle: Arc::new(RwLock::new(None)),
        })
    }

    /// Set auto-save interval
    pub fn set_auto_save_interval(&mut self, seconds: Option<u64>) {
        self.auto_save_interval = seconds;
    }

    /// Enable or disable auto-save
    pub fn set_auto_save_enabled(&mut self, enabled: bool) {
        self.auto_save_enabled = enabled;
    }

    /// Load session from database
    pub async fn load(&self) -> Result<()> {
        // Load metadata
        if let Some(loaded_metadata) = self.storage.load_metadata()? {
            *self.metadata.write().await = loaded_metadata;

            let download_count = self.storage.download_count()?;

            tracing::info!(
                downloads = download_count,
                version = self.metadata.read().await.version,
                "Loaded session from database"
            );
        } else {
            // No existing session, create new metadata
            let metadata = SessionMetadata::new();
            self.storage.save_metadata(&metadata)?;
            *self.metadata.write().await = metadata;

            tracing::info!("Created new session");
        }

        Ok(())
    }

    /// Save session metadata to database
    pub async fn save(&self) -> Result<()> {
        let mut metadata = self.metadata.write().await;
        metadata.touch();

        self.storage.save_metadata(&*metadata)?;
        self.storage.flush()?;

        tracing::debug!("Saved session metadata to database");

        Ok(())
    }

    /// Add a download to the session
    pub async fn add_download(&self, info: DownloadInfo) -> Result<()> {
        self.storage.save_download(&info)?;

        if self.auto_save_enabled {
            self.save().await?;
        }

        tracing::debug!(
            download_id = %info.id,
            filename = %info.filename,
            "Added download to session"
        );

        Ok(())
    }

    /// Remove a download from the session
    pub async fn remove_download(&self, id: &DownloadId) -> Result<Option<DownloadInfo>> {
        let result = self.storage.remove_download(id)?;

        if result.is_some() && self.auto_save_enabled {
            self.save().await?;
        }

        if let Some(ref info) = result {
            tracing::debug!(
                download_id = %id,
                filename = %info.filename,
                "Removed download from session"
            );
        }

        Ok(result)
    }

    /// Update download information in the session
    pub async fn update_download(&self, info: DownloadInfo) -> Result<()> {
        self.storage.save_download(&info)?;

        if self.auto_save_enabled {
            self.save().await?;
        }

        tracing::trace!(
            download_id = %info.id,
            state = ?info.state,
            progress = info.progress.downloaded_size,
            "Updated download in session"
        );

        Ok(())
    }

    /// Get a specific download from the session
    pub async fn get_download(&self, id: &DownloadId) -> Result<Option<DownloadInfo>> {
        self.storage.load_download(id)
    }

    /// Get all downloads in the session
    pub async fn get_downloads(&self) -> Result<HashMap<DownloadId, DownloadInfo>> {
        self.storage.load_all_downloads()
    }

    /// Get downloads that can be resumed
    pub async fn get_resumable_downloads(&self) -> Result<Vec<DownloadInfo>> {
        let downloads = self.storage.load_all_downloads()?;
        let resumable = downloads
            .into_values()
            .filter(|info| info.state.can_resume())
            .collect();

        Ok(resumable)
    }

    /// Get active downloads (should be paused on restore)
    pub async fn get_active_downloads(&self) -> Result<Vec<DownloadInfo>> {
        let downloads = self.storage.load_all_downloads()?;
        let active = downloads
            .into_values()
            .filter(|info| info.state.is_active())
            .collect();

        Ok(active)
    }

    /// Clean up completed and failed downloads
    pub async fn cleanup_terminal_downloads(&self) -> Result<usize> {
        let removed_count = self.storage.cleanup_terminal_downloads()?;

        if removed_count > 0 && self.auto_save_enabled {
            self.save().await?;
        }

        tracing::info!(
            removed_count = removed_count,
            "Cleaned up terminal downloads from session"
        );

        Ok(removed_count)
    }

    /// Get session statistics
    pub async fn stats(&self) -> Result<SessionStats> {
        let downloads = self.storage.load_all_downloads()?;
        let metadata = self.metadata.read().await;
        let mut stats = SessionStats::default();

        for info in downloads.values() {
            match &info.state {
                DownloadState::Pending => stats.pending += 1,
                DownloadState::Active => stats.active += 1,
                DownloadState::Paused => stats.paused += 1,
                DownloadState::Completed => stats.completed += 1,
                DownloadState::Failed(_) => stats.failed += 1,
                DownloadState::Cancelled => stats.cancelled += 1,
                DownloadState::Waiting => stats.pending += 1,
                DownloadState::Preparing => stats.active += 1,
            }

            stats.total_size += info.progress.total_size.unwrap_or(0);
            stats.downloaded_size += info.progress.downloaded_size;
        }

        stats.created_at = metadata.created_at;
        stats.saved_at = metadata.saved_at;

        Ok(stats)
    }

    /// Get database size information
    pub async fn database_size(&self) -> Result<u64> {
        self.storage.size_on_disk()
    }

    /// Perform database maintenance
    pub async fn compact(&self) -> Result<()> {
        self.storage.compact()
    }

    /// Check database integrity
    pub async fn check_integrity(&self) -> Result<bool> {
        self.storage.check_integrity()
    }

    /// Migrate session data from older versions
    pub async fn migrate(&self) -> Result<()> {
        let metadata = self.metadata.read().await;

        if metadata.version < SESSION_VERSION {
            tracing::info!(
                from_version = metadata.version,
                to_version = SESSION_VERSION,
                "Migrating session data"
            );

            // For now, we only have version 1, so no migration needed
            // Future versions would implement migration logic here

            // Update version after successful migration
            drop(metadata);
            let mut metadata = self.metadata.write().await;
            metadata.version = SESSION_VERSION;
            self.storage.save_metadata(&*metadata)?;

            tracing::info!("Session migration completed");
        }

        Ok(())
    }

    /// Recover session after interruption
    pub async fn recover(&self) -> Result<SessionRecoveryInfo> {
        tracing::info!("Starting session recovery");

        // Load session data
        self.load().await?;

        // Migrate if necessary
        self.migrate().await?;

        let downloads = self.storage.load_all_downloads()?;
        let mut recovery_info = SessionRecoveryInfo::new();

        for (id, mut download_info) in downloads {
            match &download_info.state {
                DownloadState::Active => {
                    // Downloads that were active should be paused for manual restart
                    tracing::info!(
                        download_id = %id,
                        filename = %download_info.filename,
                        "Pausing previously active download"
                    );

                    download_info.state = DownloadState::Paused;
                    self.storage.save_download(&download_info)?;
                    recovery_info.paused_downloads.push(download_info);
                }
                DownloadState::Paused => {
                    // Paused downloads can be resumed
                    recovery_info.resumable_downloads.push(download_info);
                }
                DownloadState::Failed(_) => {
                    // Failed downloads can be retried
                    recovery_info.failed_downloads.push(download_info);
                }
                DownloadState::Pending => {
                    // Pending downloads can be started
                    recovery_info.pending_downloads.push(download_info);
                }
                DownloadState::Completed => {
                    recovery_info.completed_downloads.push(download_info);
                }
                DownloadState::Cancelled => {
                    recovery_info.cancelled_downloads.push(download_info);
                }
                DownloadState::Waiting => {
                    // Waiting downloads should be treated as pending
                    recovery_info.pending_downloads.push(download_info);
                }
                DownloadState::Preparing => {
                    // Preparing downloads should be paused for manual restart
                    tracing::info!(
                        download_id = %id,
                        filename = %download_info.filename,
                        "Pausing previously preparing download"
                    );

                    download_info.state = DownloadState::Paused;
                    self.storage.save_download(&download_info)?;
                    recovery_info.paused_downloads.push(download_info);
                }
            }
        }

        // Check for partial downloads that need cleanup
        recovery_info.partial_files = self.find_partial_files().await?;

        tracing::info!(
            resumable = recovery_info.resumable_downloads.len(),
            failed = recovery_info.failed_downloads.len(),
            pending = recovery_info.pending_downloads.len(),
            completed = recovery_info.completed_downloads.len(),
            partial_files = recovery_info.partial_files.len(),
            "Session recovery completed"
        );

        Ok(recovery_info)
    }

    /// Find partial download files that may need cleanup
    async fn find_partial_files(&self) -> Result<Vec<PartialFileInfo>> {
        let downloads = self.storage.load_all_downloads()?;
        let mut partial_files = Vec::new();

        for download_info in downloads.values() {
            // Check if there are partial files for incomplete downloads
            if !download_info.state.is_terminal() {
                let partial_file_path = download_info.output_path.join(&download_info.filename);

                if let Ok(metadata) = tokio::fs::metadata(&partial_file_path).await {
                    let partial_info = PartialFileInfo {
                        download_id: download_info.id.clone(),
                        file_path: partial_file_path,
                        current_size: metadata.len(),
                        expected_size: download_info.progress.total_size,
                        last_modified: metadata
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| chrono::DateTime::from_timestamp(d.as_secs() as i64, 0))
                            .flatten()
                            .unwrap_or_else(|| chrono::Utc::now()),
                    };

                    partial_files.push(partial_info);
                }
            }
        }

        Ok(partial_files)
    }

    /// Resume a specific download
    pub async fn resume_download(&self, download_id: &DownloadId) -> Result<DownloadInfo> {
        if let Some(mut download_info) = self.storage.load_download(download_id)? {
            if download_info.state.can_resume() {
                // Update state to pending so it can be picked up by the download manager
                download_info.state = DownloadState::Pending;
                self.storage.save_download(&download_info)?;

                tracing::info!(
                    download_id = %download_id,
                    filename = %download_info.filename,
                    "Marked download for resume"
                );

                Ok(download_info)
            } else {
                Err(ZuupError::InvalidStateTransition {
                    from: download_info.state.clone(),
                    to: DownloadState::Pending,
                })
            }
        } else {
            Err(ZuupError::DownloadNotFound(download_id.clone()))
        }
    }

    /// Resume all resumable downloads
    pub async fn resume_all(&self) -> Result<Vec<DownloadInfo>> {
        let resumable = self.get_resumable_downloads().await?;
        let mut resumed = Vec::new();

        for download_info in resumable {
            match self.resume_download(&download_info.id).await {
                Ok(info) => resumed.push(info),
                Err(e) => {
                    tracing::warn!(
                        download_id = %download_info.id,
                        error = %e,
                        "Failed to resume download"
                    );
                }
            }
        }

        tracing::info!(resumed_count = resumed.len(), "Resumed downloads");

        Ok(resumed)
    }

    /// Clean up orphaned partial files
    pub async fn cleanup_partial_files(&self, partial_files: &[PartialFileInfo]) -> Result<usize> {
        let mut cleaned_count = 0;

        for partial_file in partial_files {
            // Check if the download still exists and is not completed
            if let Some(download_info) = self.storage.load_download(&partial_file.download_id)? {
                if download_info.state.is_terminal() {
                    // Download is completed/cancelled, safe to remove partial file
                    if let Err(e) = tokio::fs::remove_file(&partial_file.file_path).await {
                        tracing::warn!(
                            file_path = %partial_file.file_path.display(),
                            error = %e,
                            "Failed to remove partial file"
                        );
                    } else {
                        tracing::debug!(
                            file_path = %partial_file.file_path.display(),
                            "Removed orphaned partial file"
                        );
                        cleaned_count += 1;
                    }
                }
            } else {
                // Download doesn't exist, remove the partial file
                if let Err(e) = tokio::fs::remove_file(&partial_file.file_path).await {
                    tracing::warn!(
                        file_path = %partial_file.file_path.display(),
                        error = %e,
                        "Failed to remove orphaned partial file"
                    );
                } else {
                    tracing::debug!(
                        file_path = %partial_file.file_path.display(),
                        "Removed orphaned partial file"
                    );
                    cleaned_count += 1;
                }
            }
        }

        tracing::info!(cleaned_count = cleaned_count, "Cleaned up partial files");

        Ok(cleaned_count)
    }

    /// Perform session maintenance
    pub async fn maintenance(&self) -> Result<MaintenanceReport> {
        tracing::info!("Starting session maintenance");

        let mut report = MaintenanceReport::new();

        // Clean up terminal downloads
        report.terminal_downloads_removed = self.cleanup_terminal_downloads().await?;

        // Find and optionally clean up partial files
        let partial_files = self.find_partial_files().await?;
        report.partial_files_found = partial_files.len();

        // Clean up orphaned partial files (files for completed/cancelled downloads)
        report.partial_files_cleaned = self.cleanup_partial_files(&partial_files).await?;

        // Check database integrity
        report.integrity_check_passed = self.storage.check_integrity()?;

        // Get database size
        report.database_size = self.storage.size_on_disk()?;

        // Compact database
        self.storage.compact()?;
        report.database_size_after_compact = self.storage.size_on_disk()?;

        tracing::info!(
            terminal_removed = report.terminal_downloads_removed,
            partial_files_found = report.partial_files_found,
            partial_files_cleaned = report.partial_files_cleaned,
            integrity_ok = report.integrity_check_passed,
            size_before = report.database_size,
            size_after = report.database_size_after_compact,
            "Session maintenance completed"
        );

        Ok(report)
    }

    /// Start auto-save task
    pub async fn start_auto_save(&self) -> Result<()> {
        // Stop existing auto-save task if running
        self.stop_auto_save().await;

        if let Some(interval) = self.auto_save_interval {
            if self.auto_save_enabled {
                let session_manager = self.clone();

                let handle = tokio::spawn(async move {
                    let mut interval =
                        tokio::time::interval(std::time::Duration::from_secs(interval));

                    loop {
                        interval.tick().await;

                        if let Err(e) = session_manager.save().await {
                            tracing::error!(error = %e, "Auto-save failed");
                        }
                    }
                });

                *self.auto_save_handle.write().await = Some(handle);

                tracing::debug!(interval_seconds = interval, "Started auto-save task");
            }
        }

        Ok(())
    }

    /// Stop auto-save task
    pub async fn stop_auto_save(&self) {
        if let Some(handle) = self.auto_save_handle.write().await.take() {
            handle.abort();
            tracing::debug!("Stopped auto-save task");
        }
    }

    /// Shutdown the session manager
    pub async fn shutdown(&self) -> Result<()> {
        self.stop_auto_save().await;
        self.save().await?;
        self.storage.flush()?;

        tracing::info!("Session manager shutdown completed");

        Ok(())
    }
}

impl Clone for SessionManager {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            metadata: self.metadata.clone(),
            auto_save_interval: self.auto_save_interval,
            auto_save_enabled: self.auto_save_enabled,
            auto_save_handle: Arc::new(RwLock::new(None)), // Don't clone the handle
        }
    }
}

/// Session statistics
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    pub pending: usize,
    pub active: usize,
    pub paused: usize,
    pub completed: usize,
    pub failed: usize,
    pub cancelled: usize,
    pub total_size: u64,
    pub downloaded_size: u64,
    pub created_at: DateTime<Utc>,
    pub saved_at: DateTime<Utc>,
}

/// Information about session recovery
#[derive(Debug, Clone, Default)]
pub struct SessionRecoveryInfo {
    /// Downloads that were paused (previously active)
    pub paused_downloads: Vec<DownloadInfo>,

    /// Downloads that can be resumed
    pub resumable_downloads: Vec<DownloadInfo>,

    /// Downloads that failed and can be retried
    pub failed_downloads: Vec<DownloadInfo>,

    /// Downloads that are pending
    pub pending_downloads: Vec<DownloadInfo>,

    /// Downloads that completed successfully
    pub completed_downloads: Vec<DownloadInfo>,

    /// Downloads that were cancelled
    pub cancelled_downloads: Vec<DownloadInfo>,

    /// Partial files found on disk
    pub partial_files: Vec<PartialFileInfo>,
}

impl SessionRecoveryInfo {
    /// Create a new recovery info instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Get total number of downloads recovered
    pub fn total_downloads(&self) -> usize {
        self.paused_downloads.len()
            + self.resumable_downloads.len()
            + self.failed_downloads.len()
            + self.pending_downloads.len()
            + self.completed_downloads.len()
            + self.cancelled_downloads.len()
    }

    /// Get number of downloads that can be resumed
    pub fn resumable_count(&self) -> usize {
        self.resumable_downloads.len() + self.failed_downloads.len() + self.pending_downloads.len()
    }
}

/// Information about a partial download file
#[derive(Debug, Clone)]
pub struct PartialFileInfo {
    /// Download ID this file belongs to
    pub download_id: DownloadId,

    /// Path to the partial file
    pub file_path: PathBuf,

    /// Current size of the partial file
    pub current_size: u64,

    /// Expected total size (if known)
    pub expected_size: Option<u64>,

    /// When the file was last modified
    pub last_modified: DateTime<Utc>,
}

impl PartialFileInfo {
    /// Check if the partial file appears to be complete
    pub fn is_complete(&self) -> bool {
        if let Some(expected) = self.expected_size {
            self.current_size >= expected
        } else {
            false
        }
    }

    /// Get completion percentage
    pub fn completion_percentage(&self) -> Option<f64> {
        self.expected_size.map(|expected| {
            if expected == 0 {
                100.0
            } else {
                (self.current_size as f64 / expected as f64) * 100.0
            }
        })
    }
}

/// Report from session maintenance operations
#[derive(Debug, Clone, Default)]
pub struct MaintenanceReport {
    /// Number of terminal downloads removed
    pub terminal_downloads_removed: usize,

    /// Number of partial files found
    pub partial_files_found: usize,

    /// Number of partial files cleaned up
    pub partial_files_cleaned: usize,

    /// Whether integrity check passed
    pub integrity_check_passed: bool,

    /// Database size before maintenance
    pub database_size: u64,

    /// Database size after compaction
    pub database_size_after_compact: u64,
}

impl MaintenanceReport {
    /// Create a new maintenance report
    pub fn new() -> Self {
        Self::default()
    }

    /// Get space saved by compaction
    pub fn space_saved(&self) -> u64 {
        self.database_size
            .saturating_sub(self.database_size_after_compact)
    }
}

impl SessionStats {
    /// Get total number of downloads
    pub fn total_downloads(&self) -> usize {
        self.pending + self.active + self.paused + self.completed + self.failed + self.cancelled
    }

    /// Get completion percentage
    pub fn completion_percentage(&self) -> f64 {
        if self.total_size == 0 {
            0.0
        } else {
            (self.downloaded_size as f64 / self.total_size as f64) * 100.0
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::DownloadProgress;
    use crate::types::DownloadOptions;
    use std::path::PathBuf;
    use std::time::Duration;
    use tempfile::TempDir;
    use url::Url;

    fn create_test_download_info(id: DownloadId, state: DownloadState) -> DownloadInfo {
        DownloadInfo {
            id,
            urls: vec![Url::parse("https://example.com/file.txt").unwrap()],
            filename: "file.txt".to_string(),
            output_path: PathBuf::from("/tmp"),
            state,
            progress: DownloadProgress::new(),
            priority: crate::download::DownloadPriority::Normal,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            options: DownloadOptions::default(),
        }
    }

    #[tokio::test]
    async fn test_session_storage_in_memory() {
        let storage = SessionStorage::in_memory().unwrap();

        // Test metadata operations
        let metadata = SessionMetadata::new();
        storage.save_metadata(&metadata).unwrap();

        let loaded_metadata = storage.load_metadata().unwrap().unwrap();
        assert_eq!(loaded_metadata.version, metadata.version);
        assert_eq!(loaded_metadata.created_at, metadata.created_at);

        // Test download operations
        let download_id = DownloadId::new();
        let download_info = create_test_download_info(download_id.clone(), DownloadState::Pending);

        storage.save_download(&download_info).unwrap();

        let loaded_download = storage.load_download(&download_id).unwrap().unwrap();
        assert_eq!(loaded_download.id, download_info.id);
        assert_eq!(loaded_download.filename, download_info.filename);
        assert_eq!(loaded_download.state, download_info.state);

        // Test loading all downloads
        let all_downloads = storage.load_all_downloads().unwrap();
        assert_eq!(all_downloads.len(), 1);
        assert!(all_downloads.contains_key(&download_id));

        // Test removing download
        let removed = storage.remove_download(&download_id).unwrap().unwrap();
        assert_eq!(removed.id, download_id);

        let not_found = storage.load_download(&download_id).unwrap();
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn test_session_storage_persistent() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_session.db");

        let download_id = DownloadId::new();
        let download_info = create_test_download_info(download_id.clone(), DownloadState::Active);

        // Create storage and save data
        {
            let storage = SessionStorage::new(&db_path).unwrap();
            let metadata = SessionMetadata::new();

            storage.save_metadata(&metadata).unwrap();
            storage.save_download(&download_info).unwrap();
            storage.flush().unwrap();
        }

        // Reopen storage and verify data persisted
        {
            let storage = SessionStorage::new(&db_path).unwrap();

            let loaded_metadata = storage.load_metadata().unwrap().unwrap();
            assert_eq!(loaded_metadata.version, SESSION_VERSION);

            let loaded_download = storage.load_download(&download_id).unwrap().unwrap();
            assert_eq!(loaded_download.id, download_info.id);
            assert_eq!(loaded_download.state, download_info.state);
        }
    }

    #[tokio::test]
    async fn test_session_storage_cleanup_terminal() {
        let storage = SessionStorage::in_memory().unwrap();

        // Add downloads in various states
        let pending_id = DownloadId::new();
        let active_id = DownloadId::new();
        let completed_id = DownloadId::new();
        let failed_id = DownloadId::new();
        let cancelled_id = DownloadId::new();

        storage
            .save_download(&create_test_download_info(
                pending_id.clone(),
                DownloadState::Pending,
            ))
            .unwrap();
        storage
            .save_download(&create_test_download_info(
                active_id.clone(),
                DownloadState::Active,
            ))
            .unwrap();
        storage
            .save_download(&create_test_download_info(
                completed_id.clone(),
                DownloadState::Completed,
            ))
            .unwrap();
        storage
            .save_download(&create_test_download_info(
                failed_id.clone(),
                DownloadState::Failed("error".to_string()),
            ))
            .unwrap();
        storage
            .save_download(&create_test_download_info(
                cancelled_id.clone(),
                DownloadState::Cancelled,
            ))
            .unwrap();

        assert_eq!(storage.download_count().unwrap(), 5);

        // Clean up terminal downloads
        let removed_count = storage.cleanup_terminal_downloads().unwrap();
        assert_eq!(removed_count, 3); // completed, failed, cancelled

        // Verify only non-terminal downloads remain
        let remaining = storage.load_all_downloads().unwrap();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.contains_key(&pending_id));
        assert!(remaining.contains_key(&active_id));
        assert!(!remaining.contains_key(&completed_id));
        assert!(!remaining.contains_key(&failed_id));
        assert!(!remaining.contains_key(&cancelled_id));
    }

    #[tokio::test]
    async fn test_session_storage_integrity_check() {
        let storage = SessionStorage::in_memory().unwrap();

        // Add some test data
        let metadata = SessionMetadata::new();
        storage.save_metadata(&metadata).unwrap();

        let download_id = DownloadId::new();
        let download_info = create_test_download_info(download_id, DownloadState::Pending);
        storage.save_download(&download_info).unwrap();

        // Check integrity
        let is_valid = storage.check_integrity().unwrap();
        assert!(is_valid);
    }

    #[tokio::test]
    async fn test_session_manager_basic_operations() {
        let manager = SessionManager::new().unwrap();
        manager.load().await.unwrap();

        let download_id = DownloadId::new();
        let download_info = create_test_download_info(download_id.clone(), DownloadState::Pending);

        // Add download
        manager.add_download(download_info.clone()).await.unwrap();

        // Get download
        let retrieved = manager.get_download(&download_id).await.unwrap().unwrap();
        assert_eq!(retrieved.id, download_info.id);

        // Update download
        let mut updated_info = download_info.clone();
        updated_info.state = DownloadState::Active;
        manager.update_download(updated_info.clone()).await.unwrap();

        let retrieved = manager.get_download(&download_id).await.unwrap().unwrap();
        assert_eq!(retrieved.state, DownloadState::Active);

        // Get all downloads
        let all_downloads = manager.get_downloads().await.unwrap();
        assert_eq!(all_downloads.len(), 1);

        // Remove download
        let removed = manager
            .remove_download(&download_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(removed.id, download_id);

        let not_found = manager.get_download(&download_id).await.unwrap();
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn test_session_manager_resumable_downloads() {
        let manager = SessionManager::new().unwrap();
        manager.load().await.unwrap();

        // Add downloads in various states
        let paused_id = DownloadId::new();
        let failed_id = DownloadId::new();
        let active_id = DownloadId::new();
        let completed_id = DownloadId::new();

        manager
            .add_download(create_test_download_info(
                paused_id.clone(),
                DownloadState::Paused,
            ))
            .await
            .unwrap();
        manager
            .add_download(create_test_download_info(
                failed_id.clone(),
                DownloadState::Failed("error".to_string()),
            ))
            .await
            .unwrap();
        manager
            .add_download(create_test_download_info(
                active_id.clone(),
                DownloadState::Active,
            ))
            .await
            .unwrap();
        manager
            .add_download(create_test_download_info(
                completed_id.clone(),
                DownloadState::Completed,
            ))
            .await
            .unwrap();

        // Get resumable downloads
        let resumable = manager.get_resumable_downloads().await.unwrap();
        assert_eq!(resumable.len(), 2); // paused and failed

        let resumable_ids: Vec<_> = resumable.iter().map(|d| &d.id).collect();
        assert!(resumable_ids.contains(&&paused_id));
        assert!(resumable_ids.contains(&&failed_id));

        // Get active downloads
        let active = manager.get_active_downloads().await.unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, active_id);
    }

    #[tokio::test]
    async fn test_session_manager_cleanup() {
        let manager = SessionManager::new().unwrap();
        manager.load().await.unwrap();

        // Add downloads in various states
        manager
            .add_download(create_test_download_info(
                DownloadId::new(),
                DownloadState::Pending,
            ))
            .await
            .unwrap();
        manager
            .add_download(create_test_download_info(
                DownloadId::new(),
                DownloadState::Completed,
            ))
            .await
            .unwrap();
        manager
            .add_download(create_test_download_info(
                DownloadId::new(),
                DownloadState::Failed("error".to_string()),
            ))
            .await
            .unwrap();
        manager
            .add_download(create_test_download_info(
                DownloadId::new(),
                DownloadState::Cancelled,
            ))
            .await
            .unwrap();

        let all_downloads = manager.get_downloads().await.unwrap();
        assert_eq!(all_downloads.len(), 4);

        // Clean up terminal downloads
        let removed_count = manager.cleanup_terminal_downloads().await.unwrap();
        assert_eq!(removed_count, 3); // completed, failed, cancelled

        let remaining = manager.get_downloads().await.unwrap();
        assert_eq!(remaining.len(), 1); // only pending remains
    }

    #[tokio::test]
    async fn test_session_manager_stats() {
        let manager = SessionManager::new().unwrap();
        manager.load().await.unwrap();

        // Add downloads with different states and sizes
        let mut pending_info = create_test_download_info(DownloadId::new(), DownloadState::Pending);
        pending_info.progress.total_size = Some(1000);
        pending_info.progress.downloaded_size = 100;

        let mut active_info = create_test_download_info(DownloadId::new(), DownloadState::Active);
        active_info.progress.total_size = Some(2000);
        active_info.progress.downloaded_size = 500;

        let mut completed_info =
            create_test_download_info(DownloadId::new(), DownloadState::Completed);
        completed_info.progress.total_size = Some(1500);
        completed_info.progress.downloaded_size = 1500;

        manager.add_download(pending_info).await.unwrap();
        manager.add_download(active_info).await.unwrap();
        manager.add_download(completed_info).await.unwrap();

        let stats = manager.stats().await.unwrap();
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.active, 1);
        assert_eq!(stats.completed, 1);
        assert_eq!(stats.total_downloads(), 3);
        assert_eq!(stats.total_size, 4500);
        assert_eq!(stats.downloaded_size, 2100);
        assert_eq!(stats.completion_percentage(), (2100.0 / 4500.0) * 100.0);
    }

    #[tokio::test]
    async fn test_session_manager_with_database() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_session.db");

        let download_id = DownloadId::new();
        let download_info = create_test_download_info(download_id.clone(), DownloadState::Paused);

        // Create manager and add download
        {
            let manager = SessionManager::with_database(&db_path).unwrap();
            manager.load().await.unwrap();
            manager.add_download(download_info.clone()).await.unwrap();
            manager.shutdown().await.unwrap();
        }

        // Reopen manager and verify data persisted
        {
            let manager = SessionManager::with_database(&db_path).unwrap();
            manager.load().await.unwrap();

            let loaded = manager.get_download(&download_id).await.unwrap().unwrap();
            assert_eq!(loaded.id, download_info.id);
            assert_eq!(loaded.state, download_info.state);

            manager.shutdown().await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_session_version_compatibility() {
        let storage = SessionStorage::in_memory().unwrap();

        // Test loading metadata with current version
        let metadata = SessionMetadata::new();
        storage.save_metadata(&metadata).unwrap();

        let loaded = storage.load_metadata().unwrap().unwrap();
        assert_eq!(loaded.version, SESSION_VERSION);

        // Test that future versions are rejected
        let mut future_metadata = metadata.clone();
        future_metadata.version = SESSION_VERSION + 1;
        storage.save_metadata(&future_metadata).unwrap();

        let result = storage.load_metadata();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported session version")
        );
    }

    #[tokio::test]
    async fn test_session_manager_auto_save() {
        let manager = SessionManager::new().unwrap();
        manager.load().await.unwrap();

        // Test starting auto-save
        manager.start_auto_save().await.unwrap();

        // Add a download (should trigger auto-save)
        let download_info = create_test_download_info(DownloadId::new(), DownloadState::Pending);
        manager.add_download(download_info).await.unwrap();

        // Wait a bit to ensure auto-save has a chance to run
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Stop auto-save
        manager.stop_auto_save().await;

        // Shutdown should work without issues
        manager.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_session_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("recovery_test.db");

        // Create downloads in various states
        let active_id = DownloadId::new();
        let paused_id = DownloadId::new();
        let failed_id = DownloadId::new();
        let pending_id = DownloadId::new();
        let completed_id = DownloadId::new();

        // Set up initial session
        {
            let manager = SessionManager::with_database(&db_path).unwrap();
            manager.load().await.unwrap();

            manager
                .add_download(create_test_download_info(
                    active_id.clone(),
                    DownloadState::Active,
                ))
                .await
                .unwrap();
            manager
                .add_download(create_test_download_info(
                    paused_id.clone(),
                    DownloadState::Paused,
                ))
                .await
                .unwrap();
            manager
                .add_download(create_test_download_info(
                    failed_id.clone(),
                    DownloadState::Failed("network error".to_string()),
                ))
                .await
                .unwrap();
            manager
                .add_download(create_test_download_info(
                    pending_id.clone(),
                    DownloadState::Pending,
                ))
                .await
                .unwrap();
            manager
                .add_download(create_test_download_info(
                    completed_id.clone(),
                    DownloadState::Completed,
                ))
                .await
                .unwrap();

            manager.shutdown().await.unwrap();
        }

        // Recover session
        {
            let manager = SessionManager::with_database(&db_path).unwrap();
            let recovery_info = manager.recover().await.unwrap();

            // Check recovery results
            assert_eq!(recovery_info.total_downloads(), 5);
            assert_eq!(recovery_info.paused_downloads.len(), 1); // active -> paused
            assert_eq!(recovery_info.resumable_downloads.len(), 1); // paused
            assert_eq!(recovery_info.failed_downloads.len(), 1); // failed
            assert_eq!(recovery_info.pending_downloads.len(), 1); // pending
            assert_eq!(recovery_info.completed_downloads.len(), 1); // completed

            // Verify that active download was converted to paused
            let paused_download = &recovery_info.paused_downloads[0];
            assert_eq!(paused_download.id, active_id);
            assert_eq!(paused_download.state, DownloadState::Paused);

            manager.shutdown().await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_resume_download() {
        let manager = SessionManager::new().unwrap();
        manager.load().await.unwrap();

        // Add a paused download
        let download_id = DownloadId::new();
        let download_info = create_test_download_info(download_id.clone(), DownloadState::Paused);
        manager.add_download(download_info).await.unwrap();

        // Resume the download
        let resumed = manager.resume_download(&download_id).await.unwrap();
        assert_eq!(resumed.state, DownloadState::Pending);

        // Verify it was updated in storage
        let stored = manager.get_download(&download_id).await.unwrap().unwrap();
        assert_eq!(stored.state, DownloadState::Pending);

        manager.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_resume_all_downloads() {
        let manager = SessionManager::new().unwrap();
        manager.load().await.unwrap();

        // Add downloads in various states
        let paused_id = DownloadId::new();
        let failed_id = DownloadId::new();
        let completed_id = DownloadId::new();

        manager
            .add_download(create_test_download_info(
                paused_id.clone(),
                DownloadState::Paused,
            ))
            .await
            .unwrap();
        manager
            .add_download(create_test_download_info(
                failed_id.clone(),
                DownloadState::Failed("error".to_string()),
            ))
            .await
            .unwrap();
        manager
            .add_download(create_test_download_info(
                completed_id.clone(),
                DownloadState::Completed,
            ))
            .await
            .unwrap();

        // Resume all downloads
        let resumed = manager.resume_all().await.unwrap();
        assert_eq!(resumed.len(), 2); // paused and failed should be resumed

        // Verify states were updated
        let paused_download = manager.get_download(&paused_id).await.unwrap().unwrap();
        assert_eq!(paused_download.state, DownloadState::Pending);

        let failed_download = manager.get_download(&failed_id).await.unwrap().unwrap();
        assert_eq!(failed_download.state, DownloadState::Pending);

        let completed_download = manager.get_download(&completed_id).await.unwrap().unwrap();
        assert_eq!(completed_download.state, DownloadState::Completed); // Should remain completed

        manager.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_session_maintenance() {
        let manager = SessionManager::new().unwrap();
        manager.load().await.unwrap();

        // Add downloads in various states
        manager
            .add_download(create_test_download_info(
                DownloadId::new(),
                DownloadState::Pending,
            ))
            .await
            .unwrap();
        manager
            .add_download(create_test_download_info(
                DownloadId::new(),
                DownloadState::Completed,
            ))
            .await
            .unwrap();
        manager
            .add_download(create_test_download_info(
                DownloadId::new(),
                DownloadState::Failed("error".to_string()),
            ))
            .await
            .unwrap();
        manager
            .add_download(create_test_download_info(
                DownloadId::new(),
                DownloadState::Cancelled,
            ))
            .await
            .unwrap();

        // Run maintenance
        let report = manager.maintenance().await.unwrap();

        // Check that terminal downloads were removed
        assert_eq!(report.terminal_downloads_removed, 3); // completed, failed, cancelled
        assert!(report.integrity_check_passed);
        assert!(report.database_size > 0);

        // Verify only pending download remains
        let remaining = manager.get_downloads().await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert!(
            remaining
                .values()
                .all(|d| d.state == DownloadState::Pending)
        );

        manager.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_partial_file_info() {
        let partial_file = PartialFileInfo {
            download_id: DownloadId::new(),
            file_path: PathBuf::from("/tmp/test.txt"),
            current_size: 500,
            expected_size: Some(1000),
            last_modified: Utc::now(),
        };

        assert!(!partial_file.is_complete());
        assert_eq!(partial_file.completion_percentage(), Some(50.0));

        let complete_file = PartialFileInfo {
            download_id: DownloadId::new(),
            file_path: PathBuf::from("/tmp/complete.txt"),
            current_size: 1000,
            expected_size: Some(1000),
            last_modified: Utc::now(),
        };

        assert!(complete_file.is_complete());
        assert_eq!(complete_file.completion_percentage(), Some(100.0));
    }

    #[tokio::test]
    async fn test_session_recovery_info() {
        let mut recovery_info = SessionRecoveryInfo::new();

        recovery_info
            .paused_downloads
            .push(create_test_download_info(
                DownloadId::new(),
                DownloadState::Paused,
            ));
        recovery_info
            .resumable_downloads
            .push(create_test_download_info(
                DownloadId::new(),
                DownloadState::Paused,
            ));
        recovery_info
            .failed_downloads
            .push(create_test_download_info(
                DownloadId::new(),
                DownloadState::Failed("error".to_string()),
            ));
        recovery_info
            .completed_downloads
            .push(create_test_download_info(
                DownloadId::new(),
                DownloadState::Completed,
            ));

        assert_eq!(recovery_info.total_downloads(), 4);
        assert_eq!(recovery_info.resumable_count(), 2); // resumable + failed
    }

    #[tokio::test]
    async fn test_maintenance_report() {
        let mut report = MaintenanceReport::new();
        report.database_size = 1000;
        report.database_size_after_compact = 800;

        assert_eq!(report.space_saved(), 200);

        // Test no space saved case
        report.database_size_after_compact = 1200; // Larger after (shouldn't happen but test edge case)
        assert_eq!(report.space_saved(), 0);
    }
}
