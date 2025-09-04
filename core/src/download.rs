use std::path::PathBuf;
use std::sync::Arc;
use std::{collections::HashMap, time::Duration};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::task::JoinHandle;
use url::Url;

use crate::error::ZuupError;
use crate::protocol::{Download as ProtoDownload, ProtocolRegistry};
use crate::types::{DownloadId, DownloadOptions, DownloadSegment, SegmentProgress};

/// Download request containing URLs and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRequest {
    /// List of URLs to download from (for multi-source downloads)
    pub urls: Vec<Url>,

    /// Output directory path
    pub output_path: Option<PathBuf>,

    /// Custom filename (if not specified, derived from URL)
    pub filename: Option<String>,

    /// Download options and configuration
    pub options: DownloadOptions,
}

impl DownloadRequest {
    /// Create a new download request with a single URL
    pub fn new(url: Url) -> Self {
        Self {
            urls: vec![url],
            output_path: None,
            filename: None,
            options: DownloadOptions::default(),
        }
    }

    /// Create a new download request with multiple URLs
    pub fn with_urls(urls: Vec<Url>) -> Self {
        Self {
            urls,
            output_path: None,
            filename: None,
            options: DownloadOptions::default(),
        }
    }

    /// Set the output path
    pub fn output_path(mut self, path: PathBuf) -> Self {
        self.output_path = Some(path);
        self
    }

    /// Set the filename
    pub fn filename(mut self, name: String) -> Self {
        self.filename = Some(name);
        self
    }

    /// Set the download options
    pub fn options(mut self, options: DownloadOptions) -> Self {
        self.options = options;
        self
    }
}

/// Current state of a download
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadState {
    /// Download is queued but not started
    Pending,

    /// Download is actively running
    Active,

    /// Download is paused
    Paused,

    /// Download completed successfully
    Completed,

    /// Download failed with an error
    Failed(String), // Store error as string for serialization

    /// Download was cancelled by user
    Cancelled,

    /// Download is waiting for resources
    Waiting,

    /// Download is being prepared (analyzing, creating segments, etc.)
    Preparing,
}

/// Priority level for downloads
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DownloadPriority {
    /// Lowest priority
    Low = 1,
    /// Normal priority (default)
    Normal = 2,
    /// High priority
    High = 3,
    /// Highest priority
    Critical = 4,
}

impl Default for DownloadPriority {
    fn default() -> Self {
        DownloadPriority::Normal
    }
}

/// Task control commands
#[derive(Debug, Clone)]
pub enum TaskCommand {
    Start,
    Pause,
    Resume,
    Cancel,
    UpdatePriority(DownloadPriority),
}

impl DownloadState {
    /// Check if the download is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            DownloadState::Completed | DownloadState::Failed(_) | DownloadState::Cancelled
        )
    }

    /// Check if the download is active
    pub fn is_active(&self) -> bool {
        matches!(self, DownloadState::Active)
    }

    /// Check if the download can be resumed
    pub fn can_resume(&self) -> bool {
        matches!(self, DownloadState::Paused | DownloadState::Failed(_))
    }

    /// Check if the download can be paused
    pub fn can_pause(&self) -> bool {
        matches!(self, DownloadState::Active | DownloadState::Preparing)
    }

    /// Check if the download can be started
    pub fn can_start(&self) -> bool {
        matches!(self, DownloadState::Pending | DownloadState::Waiting)
    }

    /// Check if the download is waiting for resources
    pub fn is_waiting(&self) -> bool {
        matches!(self, DownloadState::Waiting)
    }

    /// Check if the download is preparing
    pub fn is_preparing(&self) -> bool {
        matches!(self, DownloadState::Preparing)
    }

    /// Validate state transition
    pub fn can_transition_to(&self, new_state: &DownloadState) -> bool {
        use DownloadState::*;

        match (self, new_state) {
            // From Pending
            (Pending, Waiting) => true,
            (Pending, Preparing) => true,
            (Pending, Cancelled) => true,

            // From Waiting
            (Waiting, Preparing) => true,
            (Waiting, Cancelled) => true,

            // From Preparing
            (Preparing, Active) => true,
            (Preparing, Paused) => true,
            (Preparing, Failed(_)) => true,
            (Preparing, Cancelled) => true,

            // From Active
            (Active, Paused) => true,
            (Active, Completed) => true,
            (Active, Failed(_)) => true,
            (Active, Cancelled) => true,

            // From Paused
            (Paused, Active) => true,
            (Paused, Cancelled) => true,

            // From Failed
            (Failed(_), Preparing) => true,
            (Failed(_), Cancelled) => true,

            // Terminal states cannot transition
            (Completed, _) | (Cancelled, _) => false,

            // Same state is always valid
            (a, b) if a == b => true,

            // All other transitions are invalid
            _ => false,
        }
    }
}

/// Progress information for a download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Total size of the download in bytes (if known)
    pub total_size: Option<u64>,

    /// Number of bytes downloaded so far
    pub downloaded_size: u64,

    /// Current download speed in bytes per second
    pub download_speed: u64,

    /// Current upload speed in bytes per second (if available)
    pub upload_speed: Option<u64>,

    /// Estimated time to completion
    pub eta: Option<Duration>,

    /// Number of active connections
    pub connections: u32,

    /// Progress of individual segments
    pub segments: Vec<SegmentProgress>,
}

impl DownloadProgress {
    /// Create a new progress instance
    pub fn new() -> Self {
        Self {
            total_size: None,
            downloaded_size: 0,
            download_speed: 0,
            upload_speed: None,
            eta: None,
            connections: 0,
            segments: Vec::new(),
        }
    }

    /// Calculate completion percentage (0.0 to 1.0)
    pub fn percentage(&self) -> f64 {
        self.total_size.map_or(0.0, |total| {
            if total == 0 {
                1.0
            } else {
                self.downloaded_size as f64 / total as f64
            }
        })
    }

    /// Check if the download is complete
    pub fn is_complete(&self) -> bool {
        if let Some(total) = self.total_size {
            self.downloaded_size >= total
        } else {
            false
        }
    }
}

impl Default for DownloadProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete information about a download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadInfo {
    /// Unique download identifier
    pub id: DownloadId,

    /// List of URLs being downloaded from
    pub urls: Vec<Url>,

    /// Final filename
    pub filename: String,

    /// Output path where file will be saved
    pub output_path: PathBuf,

    /// Current download state
    pub state: DownloadState,

    /// Current progress information
    pub progress: DownloadProgress,

    /// Download priority
    pub priority: DownloadPriority,

    /// When the download was created
    pub created_at: DateTime<Utc>,

    /// When the download was started (if ever)
    pub started_at: Option<DateTime<Utc>>,

    /// When the download was completed (if completed)
    pub completed_at: Option<DateTime<Utc>>,

    /// Download options used
    pub options: DownloadOptions,
}

/// Download task representing an active or queued download
#[derive(Clone)]
pub struct DownloadTask {
    /// Unique identifier
    id: DownloadId,

    /// Protocol registry for resolving handlers
    pub protocol_registry: Arc<RwLock<ProtocolRegistry>>,

    /// Protocol-specific download instance (when running)
    pub protocol_download: Arc<Mutex<Option<Box<dyn ProtoDownload>>>>,

    /// Original download request
    pub request: DownloadRequest,

    /// Current state
    pub state: Arc<RwLock<DownloadState>>,

    /// Download segments
    pub segments: Arc<RwLock<Vec<DownloadSegment>>>,

    /// Progress information
    pub progress: Arc<RwLock<DownloadProgress>>,

    /// Task priority
    pub priority: Arc<RwLock<DownloadPriority>>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Start timestamp
    pub started_at: Arc<RwLock<Option<DateTime<Utc>>>>,

    /// Completion timestamp
    pub completed_at: Arc<RwLock<Option<DateTime<Utc>>>>,

    /// Task control channel sender
    pub control_tx: Arc<Mutex<Option<mpsc::UnboundedSender<TaskCommand>>>>,

    /// Task handle for cancellation
    pub task_handle: Arc<Mutex<Option<JoinHandle<Result<()>>>>>,
}

impl DownloadTask {
    /// Compute output directory and final filename
    fn compute_output_and_filename(&self) -> (PathBuf, String) {
        // Determine filename
        let filename = self.request.filename.clone().unwrap_or_else(|| {
            self.request
                .urls
                .first()
                .and_then(|url| url.path_segments())
                .and_then(|segments| segments.last())
                .filter(|name| !name.is_empty())
                .unwrap_or("download")
                .to_string()
        });
        // Determine output path
        let output_path = self
            .request
            .output_path
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        (output_path, filename)
    }

    /// Path to the sidecar .zuup file for this download
    fn zuup_path(&self) -> PathBuf {
        let (output, filename) = self.compute_output_and_filename();
        output.join(format!("{}.zuup", filename))
    }

    /// Persist current download info into the .zuup sidecar file
    async fn save_zuup(&self) -> Result<()> {
        let info = self.info().await;
        let path = self.zuup_path();
        // Ensure directory exists
        if let Some(parent) = path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        let tmp_path = path.with_extension("zuup.tmp");
        let json = serde_json::to_vec_pretty(&info)
            .map_err(|e| ZuupError::Internal(format!("Failed to serialize .zuup: {}", e)))?;
        tokio::fs::write(&tmp_path, json).await.map_err(|e| {
            ZuupError::Internal(format!(
                "Failed to write temporary .zuup file {}: {}",
                tmp_path.display(),
                e
            ))
        })?;
        tokio::fs::rename(&tmp_path, &path).await.map_err(|e| {
            ZuupError::Internal(format!(
                "Failed to atomically persist .zuup file {}: {}",
                path.display(),
                e
            ))
        })?;
        Ok(())
    }

    /// Remove the .zuup sidecar file if it exists
    async fn remove_zuup(&self) {
        let path = self.zuup_path();
        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            let _ = tokio::fs::remove_file(path).await;
        }
    }

    /// Create a new download task.
    pub fn new(
        id: DownloadId,
        request: DownloadRequest,
        protocol_registry: Arc<RwLock<ProtocolRegistry>>,
        priority: Option<DownloadPriority>,
    ) -> Self {
        Self {
            protocol_registry,
            protocol_download: Arc::new(Mutex::new(None)),
            id,
            request,
            state: Arc::new(RwLock::new(DownloadState::Pending)),
            segments: Arc::new(RwLock::new(Vec::new())),
            progress: Arc::new(RwLock::new(DownloadProgress::new())),
            priority: Arc::new(RwLock::new(priority.unwrap_or_default())),
            created_at: Utc::now(),
            started_at: Arc::new(RwLock::new(None)),
            completed_at: Arc::new(RwLock::new(None)),
            control_tx: Arc::new(Mutex::new(None)),
            task_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Get current download info
    pub async fn info(&self) -> DownloadInfo {
        let state = self.state.read().await.clone();
        let progress = self.progress.read().await.clone();
        let priority = *self.priority.read().await;
        let started_at = *self.started_at.read().await;
        let completed_at = *self.completed_at.read().await;

        // Determine filename
        let filename = self.request.filename.clone().unwrap_or_else(|| {
            self.request
                .urls
                .first()
                .and_then(|url| url.path_segments())
                .and_then(|segments| segments.last())
                .filter(|name| !name.is_empty())
                .unwrap_or("download")
                .to_string()
        });

        // Determine output path
        let output_path = self
            .request
            .output_path
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        DownloadInfo {
            id: self.id.clone(),
            urls: self.request.urls.clone(),
            filename,
            output_path,
            state,
            progress,
            priority,
            created_at: self.created_at,
            started_at,
            completed_at,
            options: self.request.options.clone(),
        }
    }

    /// Start the download task
    pub async fn start(&self) -> Result<()> {
        tracing::debug!("Starting download task for ID: {}", self.id);
        let mut state = self.state.write().await;

        if !state.can_start() && !state.is_active() {
            tracing::error!(
                "Cannot start download - invalid state transition from {:?} to Preparing",
                state
            );
            return Err(ZuupError::InvalidStateTransition {
                from: state.clone(),
                to: DownloadState::Preparing,
            });
        }

        // If already active, nothing to do
        if state.is_active() {
            tracing::debug!("Download task already active for ID: {}", self.id);
            return Ok(());
        }

        tracing::debug!("Setting download state to Preparing for ID: {}", self.id);
        *state = DownloadState::Preparing;
        *self.started_at.write().await = Some(Utc::now());
        drop(state);

        // Spawn the actual download task
        tracing::debug!("Spawning download task for ID: {}", self.id);
        let task_clone = self.clone();
        let handle = tokio::spawn(async move {
            tracing::debug!(
                "Download task spawned, starting run_download for ID: {}",
                task_clone.id
            );
            let result = task_clone.run_download().await;
            if let Err(ref e) = result {
                tracing::error!("Download task failed for ID {}: {}", task_clone.id, e);
            }
            result
        });

        // Store the task handle
        *self.task_handle.lock().await = Some(handle);

        // Set state to active
        tracing::debug!("Setting download state to Active for ID: {}", self.id);
        *self.state.write().await = DownloadState::Active;

        Ok(())
    }

    /// Run the actual download process
    async fn run_download(&self) -> Result<()> {
        tracing::info!("Starting download task for URL: {}", &self.request.urls[0]);
        let url = &self.request.urls[0];

        // Create protocol-specific download via handler from registry
        let mut proto_download = {
            let registry = self.protocol_registry.read().await;
            if let Some(handler) = registry.find_handler(url) {
                handler.create_download(&self.request).await?
            } else {
                let err = ZuupError::Protocol(crate::error::ProtocolError::UnsupportedProtocol(
                    url.scheme().to_string(),
                ));
                *self.state.write().await = DownloadState::Failed(err.to_string());
                return Err(err);
            }
        };

        // Store it for control operations
        {
            let mut guard = self.protocol_download.lock().await;
            *guard = Some(proto_download);
        }

        // Start the protocol download
        {
            let mut guard = self.protocol_download.lock().await;
            if let Some(download) = guard.as_mut() {
                download.start().await?;
            } else {
                return Err(ZuupError::Internal(
                    "Protocol download instance missing".to_string(),
                ));
            }
        }

        // Poll the protocol download for progress/state and mirror to task
        loop {
            // Snapshot state/progress
            let (state_snapshot, progress_snapshot) = {
                let guard = self.protocol_download.lock().await;
                if let Some(ref download) = *guard {
                    (download.state(), download.progress())
                } else {
                    return Err(ZuupError::Internal(
                        "Protocol download instance missing during polling".to_string(),
                    ));
                }
            };

            // Mirror into task
            {
                let mut state_guard = self.state.write().await;
                *state_guard = state_snapshot.clone();
                let mut prog_guard = self.progress.write().await;
                let p = progress_snapshot;
                let mut mapped = DownloadProgress::new();
                mapped.total_size = p.total_size;
                mapped.downloaded_size = p.downloaded_size;
                mapped.download_speed = p.download_speed;
                mapped.upload_speed = p.upload_speed;
                mapped.eta = p.eta;
                mapped.connections = p.connections;
                mapped.segments = p.segments;
                *prog_guard = mapped;
            }

            // Persist snapshot to .zuup sidecar for resume support
            if let Err(e) = self.save_zuup().await {
                tracing::warn!(error = %format!("{}", e), "Failed to save .zuup state");
            }

            if state_snapshot.is_terminal() {
                if matches!(state_snapshot, DownloadState::Completed) {
                    *self.completed_at.write().await = Some(Utc::now());
                    // On successful completion, remove .zuup
                    self.remove_zuup().await;
                } else {
                    // On failure/cancel, persist final state
                    let _ = self.save_zuup().await;
                }
                break;
            }

            // Sleep a bit before next poll
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }

        Ok(())
    }

    /// Pause the download task
    pub async fn pause(&self) -> Result<()> {
        let mut state = self.state.write().await;

        if !state.can_pause() {
            return Err(ZuupError::InvalidStateTransition {
                from: state.clone(),
                to: DownloadState::Paused,
            });
        }

        // Delegate to protocol download if available
        {
            let mut guard = self.protocol_download.lock().await;
            if let Some(download) = guard.as_mut() {
                let _ = download.pause().await; // best-effort
            }
        }
        // Send pause command if task is running (legacy path)
        if let Some(tx) = self.control_tx.lock().await.as_ref() {
            let _ = tx.send(TaskCommand::Pause);
        }

        *state = DownloadState::Paused;
        // Persist paused state for resume
        if let Err(e) = self.save_zuup().await {
            tracing::warn!(error = %format!("{}", e), "Failed to save .zuup on pause");
        }
        Ok(())
    }

    /// Resume the download task
    pub async fn resume(&self) -> Result<()> {
        let mut state = self.state.write().await;

        if !state.can_resume() {
            return Err(ZuupError::InvalidStateTransition {
                from: state.clone(),
                to: DownloadState::Active,
            });
        }

        // Delegate to protocol download if available
        {
            let mut guard = self.protocol_download.lock().await;
            if let Some(download) = guard.as_mut() {
                let _ = download.resume().await; // best-effort
            }
        }
        // Send resume command if task is running (legacy path)
        if let Some(tx) = self.control_tx.lock().await.as_ref() {
            let _ = tx.send(TaskCommand::Resume);
        }

        *state = DownloadState::Active;
        Ok(())
    }

    /// Cancel the download task
    pub async fn cancel(&self) -> Result<()> {
        let mut state = self.state.write().await;

        if state.is_terminal() {
            return Ok(()); // Already terminal, nothing to do
        }

        // Delegate to protocol download if available
        {
            let mut guard = self.protocol_download.lock().await;
            if let Some(download) = guard.as_mut() {
                let _ = download.cancel().await; // best-effort
            }
        }
        // Send cancel command if task is running (legacy path)
        if let Some(tx) = self.control_tx.lock().await.as_ref() {
            let _ = tx.send(TaskCommand::Cancel);
        }

        // Cancel the task handle if it exists
        if let Some(handle) = self.task_handle.lock().await.take() {
            handle.abort();
        }

        *state = DownloadState::Cancelled;
        *self.completed_at.write().await = Some(Utc::now());
        // Remove any resume sidecar on cancel
        self.remove_zuup().await;
        Ok(())
    }

    /// Update task priority
    pub async fn set_priority(&self, priority: DownloadPriority) -> Result<()> {
        *self.priority.write().await = priority;

        // Send priority update command if task is running
        if let Some(tx) = self.control_tx.lock().await.as_ref() {
            let _ = tx.send(TaskCommand::UpdatePriority(priority));
        }

        Ok(())
    }

    /// Get current priority
    pub async fn priority(&self) -> DownloadPriority {
        *self.priority.read().await
    }

    /// Check if task is terminal
    pub async fn is_terminal(&self) -> bool {
        self.state.read().await.is_terminal()
    }

    /// Check if task is active
    pub async fn is_active(&self) -> bool {
        self.state.read().await.is_active()
    }

    /// Set control channel
    pub async fn set_control_channel(&self, tx: mpsc::UnboundedSender<TaskCommand>) {
        *self.control_tx.lock().await = Some(tx);
    }

    /// Set task handle
    pub async fn set_task_handle(&self, handle: JoinHandle<Result<()>>) {
        *self.task_handle.lock().await = Some(handle);
    }
}

/// Download manager responsible for coordinating downloads.
pub struct DownloadManager {
    /// All downloads(active, pending, completed, failed, stopped)
    downloads: Arc<RwLock<HashMap<DownloadId, Arc<DownloadTask>>>>,
}
