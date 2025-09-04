use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::{collections::HashMap, time::Duration};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::task::JoinHandle;
use url::Url;

use crate::bandwidth::BandwidthManager;
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

/// Resource usage information
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Number of active downloads
    pub active_downloads: u32,

    /// Number of pending downloads
    pub pending_downloads: u32,

    /// Total memory usage estimate (bytes)
    pub memory_usage: u64,

    /// Total network connections
    pub network_connections: u32,

    /// Available slots for new downloads
    pub available_slots: u32,
}

/// Queue statistics
#[derive(Debug, Clone)]
pub struct QueueStats {
    /// Total tasks in queue
    pub total_queued: usize,

    /// Tasks by priority
    pub by_priority: HashMap<DownloadPriority, usize>,

    /// Average wait time for tasks
    pub average_wait_time: Option<Duration>,

    /// Longest waiting task
    pub longest_wait_time: Option<Duration>,
}

/// Task scheduler for managing download priorities and queuing
pub struct TaskScheduler {
    /// Priority queue for pending downloads
    pending_queue: Arc<Mutex<VecDeque<Arc<DownloadTask>>>>,

    /// Currently running tasks
    running_tasks: Arc<RwLock<HashMap<DownloadId, Arc<DownloadTask>>>>,

    /// Maximum concurrent downloads
    max_concurrent: u32,

    /// Resource usage tracking
    resource_usage: Arc<RwLock<ResourceUsage>>,
}

impl TaskScheduler {
    /// Create a new task scheduler
    pub fn new(max_concurrent: u32) -> Self {
        Self {
            pending_queue: Arc::new(Mutex::new(VecDeque::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            max_concurrent,
            resource_usage: Arc::new(RwLock::new(ResourceUsage {
                active_downloads: 0,
                pending_downloads: 0,
                memory_usage: 0,
                network_connections: 0,
                available_slots: max_concurrent,
            })),
        }
    }

    /// Add a task to the scheduler
    pub async fn add_task(&self, task: Arc<DownloadTask>) -> Result<()> {
        tracing::debug!("TaskScheduler: Adding task {} to scheduler", task.id);
        let mut queue = self.pending_queue.lock().await;
        tracing::debug!(
            "TaskScheduler: Acquired queue lock, current queue size: {}",
            queue.len()
        );

        // Insert task in priority order (higher priority first)
        tracing::debug!("TaskScheduler: Getting priority for task {}", task.id);
        let task_priority = task.priority().await;
        tracing::debug!(
            "TaskScheduler: Task {} has priority {:?}",
            task.id,
            task_priority
        );
        let mut insert_pos = queue.len();

        for (i, existing_task) in queue.iter().enumerate() {
            tracing::debug!(
                "TaskScheduler: Checking existing task {} at position {}",
                existing_task.id,
                i
            );
            let existing_priority = existing_task.priority().await;
            tracing::debug!(
                "TaskScheduler: Existing task {} has priority {:?}",
                existing_task.id,
                existing_priority
            );
            if task_priority > existing_priority {
                insert_pos = i;
                break;
            }
        }

        tracing::debug!(
            "TaskScheduler: Inserting task {} at position {}",
            task.id,
            insert_pos
        );
        queue.insert(insert_pos, task.clone());
        tracing::debug!(
            "TaskScheduler: Task {} inserted, queue size now: {}",
            task.id,
            queue.len()
        );

        // Release queue lock before updating resource usage
        drop(queue);

        // Update resource usage
        tracing::debug!(
            "TaskScheduler: Updating resource usage after adding task {}",
            task.id
        );
        self.update_resource_usage().await;
        tracing::debug!(
            "TaskScheduler: Successfully added task {} to scheduler",
            task.id
        );

        Ok(())
    }

    /// Try to start the next task if resources are available
    pub async fn try_start_next(&self) -> Result<Option<Arc<DownloadTask>>> {
        tracing::debug!("TaskScheduler: try_start_next called");
        let running_count = self.running_tasks.read().await.len();
        tracing::debug!(
            "TaskScheduler: running_count={}, max_concurrent={}",
            running_count,
            self.max_concurrent
        );

        if running_count >= self.max_concurrent as usize {
            tracing::debug!("TaskScheduler: At max concurrent limit, returning None");
            return Ok(None);
        }

        tracing::debug!("TaskScheduler: Acquiring pending queue lock");
        let mut queue = self.pending_queue.lock().await;
        tracing::debug!("TaskScheduler: Queue has {} tasks", queue.len());

        // Find the highest priority task that can be started
        let mut task_index = None;
        for (i, task) in queue.iter().enumerate() {
            tracing::debug!("TaskScheduler: Checking task {} at index {}", task.id, i);
            let state = task.state.read().await;
            tracing::debug!("TaskScheduler: Task {} state is {:?}", task.id, state);
            if state.can_start() {
                tracing::debug!("TaskScheduler: Task {} can start, selecting it", task.id);
                task_index = Some(i);
                break;
            }
        }

        if let Some(index) = task_index {
            let task = queue.remove(index).unwrap();
            tracing::debug!("TaskScheduler: Removed task {} from queue", task.id);
            drop(queue); // Release queue lock before acquiring running lock

            tracing::debug!("TaskScheduler: Adding task {} to running tasks", task.id);
            let mut running = self.running_tasks.write().await;
            running.insert(task.id.clone(), task.clone());
            drop(running); // Release running lock before updating resource usage

            // Update resource usage
            tracing::debug!(
                "TaskScheduler: Updating resource usage for task {}",
                task.id
            );
            self.update_resource_usage().await;

            tracing::debug!("TaskScheduler: Returning task {}", task.id);
            Ok(Some(task))
        } else {
            tracing::debug!("TaskScheduler: No startable task found in queue");
            Ok(None)
        }
    }

    /// Mark a task as completed and remove from running tasks
    pub async fn complete_task(&self, task_id: &DownloadId) -> Result<()> {
        let mut running = self.running_tasks.write().await;
        running.remove(task_id);
        drop(running); // Release lock before updating resource usage

        // Update resource usage
        self.update_resource_usage().await;

        Ok(())
    }

    /// Get running task count
    pub async fn running_count(&self) -> usize {
        self.running_tasks.read().await.len()
    }

    /// Get pending task count
    pub async fn pending_count(&self) -> usize {
        self.pending_queue.lock().await.len()
    }

    /// Update task priority and reorder queue if necessary
    pub async fn update_task_priority(
        &self,
        task_id: &DownloadId,
        priority: DownloadPriority,
    ) -> Result<()> {
        // Check if task is in running tasks
        if let Some(task) = self.running_tasks.read().await.get(task_id) {
            task.set_priority(priority).await?;
            return Ok(());
        }

        // Check if task is in pending queue
        let mut queue = self.pending_queue.lock().await;
        let mut task_index = None;

        for (i, task) in queue.iter().enumerate() {
            if task.id == *task_id {
                task_index = Some(i);
                break;
            }
        }

        if let Some(index) = task_index {
            let task = queue.remove(index).unwrap();
            task.set_priority(priority).await?;

            // Re-insert in correct priority position
            let mut insert_pos = queue.len();
            for (i, existing_task) in queue.iter().enumerate() {
                let existing_priority = existing_task.priority().await;
                if priority > existing_priority {
                    insert_pos = i;
                    break;
                }
            }

            queue.insert(insert_pos, task);
            return Ok(());
        }

        // Task not found in scheduler
        Err(ZuupError::DownloadNotFound(task_id.clone()))
    }

    /// Get current resource usage
    pub async fn resource_usage(&self) -> ResourceUsage {
        self.resource_usage.read().await.clone()
    }

    /// Get queue statistics
    pub async fn queue_stats(&self) -> QueueStats {
        let queue = self.pending_queue.lock().await;
        let mut by_priority = HashMap::new();
        let mut wait_times = Vec::new();
        let now = Utc::now();

        for task in queue.iter() {
            let priority = task.priority().await;
            *by_priority.entry(priority).or_insert(0) += 1;

            let wait_time = now.signed_duration_since(task.created_at);
            if let Ok(duration) = wait_time.to_std() {
                wait_times.push(duration);
            }
        }

        let average_wait_time = if !wait_times.is_empty() {
            let total: Duration = wait_times.iter().sum();
            Some(total / wait_times.len() as u32)
        } else {
            None
        };

        let longest_wait_time = wait_times.into_iter().max();

        QueueStats {
            total_queued: queue.len(),
            by_priority,
            average_wait_time,
            longest_wait_time,
        }
    }

    /// Check if scheduler can accept more tasks
    pub async fn can_accept_more(&self) -> bool {
        let usage = self.resource_usage.read().await;
        usage.available_slots > 0 || usage.pending_downloads < 1000 // Reasonable queue limit
    }

    /// Get tasks waiting longer than specified duration
    pub async fn get_stale_tasks(&self, max_wait_time: Duration) -> Vec<DownloadId> {
        let queue = self.pending_queue.lock().await;
        let now = Utc::now();
        let mut stale_tasks = Vec::new();

        for task in queue.iter() {
            let wait_time = now.signed_duration_since(task.created_at);
            if let Ok(duration) = wait_time.to_std() {
                if duration > max_wait_time {
                    stale_tasks.push(task.id.clone());
                }
            }
        }

        stale_tasks
    }

    /// Update resource usage statistics
    async fn update_resource_usage(&self) {
        tracing::debug!("TaskScheduler: update_resource_usage called");
        tracing::debug!("TaskScheduler: Getting running tasks count");
        let running_count = self.running_tasks.read().await.len() as u32;
        tracing::debug!("TaskScheduler: Running count: {}", running_count);

        tracing::debug!("TaskScheduler: Getting pending queue count");
        let pending_count = self.pending_queue.lock().await.len() as u32;
        tracing::debug!("TaskScheduler: Pending count: {}", pending_count);

        tracing::debug!("TaskScheduler: Updating resource usage");
        let mut usage = self.resource_usage.write().await;
        usage.active_downloads = running_count;
        usage.pending_downloads = pending_count;
        usage.available_slots = self.max_concurrent.saturating_sub(running_count);

        // Estimate memory usage (rough calculation)
        usage.memory_usage = (running_count as u64 * 1024 * 1024) + (pending_count as u64 * 1024); // 1MB per active, 1KB per pending

        // Estimate network connections (assume 4 connections per active download on average)
        usage.network_connections = running_count * 4;
        tracing::debug!("TaskScheduler: Resource usage updated successfully");
    }

    /// Set maximum concurrent downloads
    pub async fn set_max_concurrent(&mut self, max_concurrent: u32) {
        self.max_concurrent = max_concurrent;
        self.update_resource_usage().await;
    }

    /// Get maximum concurrent downloads
    pub fn max_concurrent(&self) -> u32 {
        self.max_concurrent
    }
}

/// Download manager responsible for coordinating downloads
pub struct DownloadManager {
    /// Protocol registry reference
    protocol_registry: Arc<RwLock<ProtocolRegistry>>,
    /// All downloads (active, pending, completed)
    downloads: Arc<RwLock<HashMap<DownloadId, Arc<DownloadTask>>>>,

    /// Task scheduler
    scheduler: Arc<TaskScheduler>,

    /// Bandwidth manager
    bandwidth_manager: Arc<BandwidthManager>,
}

impl DownloadManager {
    /// Create a new download manager
    pub fn new(max_concurrent: u32, protocol_registry: Arc<RwLock<ProtocolRegistry>>) -> Self {
        Self {
            protocol_registry,
            downloads: Arc::new(RwLock::new(HashMap::new())),
            scheduler: Arc::new(TaskScheduler::new(max_concurrent)),
            bandwidth_manager: Arc::new(BandwidthManager::new()),
        }
    }
}
