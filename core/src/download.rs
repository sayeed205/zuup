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
use crate::error::{Result, ZuupError};
use crate::metalink::{MetalinkFile, MetalinkUrl};
use crate::protocol::{Download as ProtoDownload, ProtocolRegistry};
use crate::types::{
    DownloadId, DownloadInfo, DownloadPriority, DownloadProgress, DownloadRequest, DownloadSegment,
    DownloadState,
};

/// Task control commands
#[derive(Debug, Clone)]
pub enum TaskCommand {
    Start,
    Pause,
    Resume,
    Cancel,
    UpdatePriority(DownloadPriority),
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
            download_type: crate::types::DownloadType::Standard,
            category: self.request.category.clone(),
            created_at: self.created_at,
            started_at,
            completed_at,
            options: self.request.options.clone(),
            error_message: None,
            file_size: None,
            content_type: None,
            last_modified: None,
            referrer: self.request.referrer.clone(),
            cookies: self.request.cookies.clone(),
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
        let proto_download = {
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
            if let Ok(duration) = wait_time.to_std()
                && duration > max_wait_time
            {
                stale_tasks.push(task.id.clone());
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

    /// Create a new download manager with custom bandwidth manager
    pub fn with_bandwidth_manager(
        max_concurrent: u32,
        bandwidth_manager: BandwidthManager,
        protocol_registry: Arc<RwLock<ProtocolRegistry>>,
    ) -> Self {
        Self {
            protocol_registry,
            downloads: Arc::new(RwLock::new(HashMap::new())),
            scheduler: Arc::new(TaskScheduler::new(max_concurrent)),
            bandwidth_manager: Arc::new(bandwidth_manager),
        }
    }

    /// Add a new download
    pub async fn add_download(
        &self,
        request: DownloadRequest,
        priority: Option<DownloadPriority>,
    ) -> Result<DownloadId> {
        let id = DownloadId::new();
        let task = Arc::new(DownloadTask::new(
            id.clone(),
            request,
            self.protocol_registry.clone(),
            priority,
        ));

        // Add to downloads collection
        let mut downloads = self.downloads.write().await;
        downloads.insert(id.clone(), task.clone());

        // Don't automatically add to scheduler or start - let user control this

        Ok(id)
    }

    /// Start a download
    pub async fn start_download(&self, id: &DownloadId) -> Result<()> {
        tracing::debug!("DownloadManager: Starting download {}", id);
        let downloads = self.downloads.read().await;
        let task = downloads
            .get(id)
            .ok_or_else(|| ZuupError::DownloadNotFound(id.clone()))?;

        let state = task.state.read().await;
        tracing::debug!("DownloadManager: Current state for {} is {:?}", id, state);
        if state.is_active() {
            tracing::debug!("DownloadManager: Download {} is already active", id);
            return Ok(()); // Already active
        }

        if !state.can_start() {
            tracing::error!(
                "DownloadManager: Cannot start download {} - invalid state {:?}",
                id,
                state
            );
            return Err(ZuupError::InvalidStateTransition {
                from: state.clone(),
                to: DownloadState::Active,
            });
        }
        drop(state);

        // Add to scheduler if not already there
        tracing::debug!("DownloadManager: Adding download {} to scheduler", id);
        self.scheduler.add_task(task.clone()).await?;

        // Try to start more tasks
        tracing::debug!("DownloadManager: Trying to start next task for {}", id);
        self.try_start_next_task().await?;

        tracing::debug!("DownloadManager: Successfully started download {}", id);
        Ok(())
    }

    /// Pause a download
    pub async fn pause_download(&self, id: &DownloadId) -> Result<()> {
        let downloads = self.downloads.read().await;
        let task = downloads
            .get(id)
            .ok_or_else(|| ZuupError::DownloadNotFound(id.clone()))?;

        task.pause().await?;

        // Mark task as completed in scheduler to free up resources
        self.scheduler.complete_task(id).await?;

        // Try to start the next task
        self.try_start_next_task().await?;

        Ok(())
    }

    /// Resume a download
    pub async fn resume_download(&self, id: &DownloadId) -> Result<()> {
        let downloads = self.downloads.read().await;
        let task = downloads
            .get(id)
            .ok_or_else(|| ZuupError::DownloadNotFound(id.clone()))?;

        // Check if we can resume
        let state = task.state.read().await;
        if !state.can_resume() {
            return Ok(()); // Already active or not resumable
        }
        drop(state);

        // Add back to scheduler first (while still in resumable state)
        self.scheduler.add_task(task.clone()).await?;

        // Try to start the next task (which will call task.start() and change state to Active)
        self.try_start_next_task().await?;

        Ok(())
    }

    /// Cancel a download
    pub async fn cancel_download(&self, id: &DownloadId) -> Result<()> {
        let downloads = self.downloads.read().await;
        let task = downloads
            .get(id)
            .ok_or_else(|| ZuupError::DownloadNotFound(id.clone()))?;

        task.cancel().await?;

        // Remove from scheduler
        self.scheduler.complete_task(id).await?;

        // Try to start the next task
        self.try_start_next_task().await?;

        Ok(())
    }

    /// Update download priority
    pub async fn set_download_priority(
        &self,
        id: &DownloadId,
        priority: DownloadPriority,
    ) -> Result<()> {
        let downloads = self.downloads.read().await;
        let task = downloads
            .get(id)
            .ok_or_else(|| ZuupError::DownloadNotFound(id.clone()))?;

        // Always update the task's priority
        task.set_priority(priority).await?;

        // Try to update in scheduler (may fail if not in scheduler, which is OK)
        let _ = self.scheduler.update_task_priority(id, priority).await;

        Ok(())
    }

    /// Get download information
    pub async fn get_download(&self, id: &DownloadId) -> Result<DownloadInfo> {
        let downloads = self.downloads.read().await;
        let task = downloads
            .get(id)
            .ok_or_else(|| ZuupError::DownloadNotFound(id.clone()))?;

        Ok(task.info().await)
    }

    /// List all downloads
    pub async fn list_downloads(&self) -> Result<Vec<DownloadInfo>> {
        let downloads = self.downloads.read().await;
        let mut infos = Vec::new();

        for task in downloads.values() {
            infos.push(task.info().await);
        }

        // Sort by priority and creation time
        infos.sort_by(|a, b| {
            // First by state (active first, then by priority)
            match (&a.state, &b.state) {
                (DownloadState::Active, DownloadState::Active) => a.created_at.cmp(&b.created_at),
                (DownloadState::Active, _) => std::cmp::Ordering::Less,
                (_, DownloadState::Active) => std::cmp::Ordering::Greater,
                _ => a.created_at.cmp(&b.created_at),
            }
        });

        Ok(infos)
    }

    /// Remove a download
    pub async fn remove_download(&self, id: &DownloadId, force: bool) -> Result<()> {
        let downloads = self.downloads.read().await;

        if let Some(task) = downloads.get(id) {
            let state = task.state.read().await;

            // Check if we can remove the download
            if !force && state.is_active() {
                return Err(ZuupError::InvalidStateTransition {
                    from: state.clone(),
                    to: DownloadState::Cancelled,
                });
            }

            // Cancel the task if it's not terminal
            if !state.is_terminal() {
                drop(state);
                task.cancel().await?;
            }
        }

        drop(downloads);
        let mut downloads = self.downloads.write().await;
        downloads
            .remove(id)
            .ok_or_else(|| ZuupError::DownloadNotFound(id.clone()))?;

        // Remove from scheduler
        self.scheduler.complete_task(id).await?;

        // Try to start the next task
        self.try_start_next_task().await?;

        Ok(())
    }

    /// Get the number of active downloads
    pub async fn active_count(&self) -> usize {
        self.scheduler.running_count().await
    }

    /// Get the number of pending downloads
    pub async fn pending_count(&self) -> usize {
        self.scheduler.pending_count().await
    }

    /// Get total download count
    pub async fn total_count(&self) -> usize {
        self.downloads.read().await.len()
    }

    /// Check if we can start more downloads
    pub async fn can_start_more(&self) -> bool {
        self.scheduler.running_count().await < self.scheduler.max_concurrent as usize
    }

    /// Try to start the next task from the queue
    async fn try_start_next_task(&self) -> Result<()> {
        tracing::debug!("DownloadManager: Trying to start next task from scheduler");
        if let Some(task) = self.scheduler.try_start_next().await? {
            tracing::debug!(
                "DownloadManager: Got task {} from scheduler, starting it",
                task.id
            );
            // Start the task
            if let Err(e) = task.start().await {
                tracing::error!("DownloadManager: Failed to start task {}: {}", task.id, e);
                // If we can't start the task, remove it from running tasks
                self.scheduler.complete_task(&task.id).await?;
                return Err(e);
            }
            tracing::debug!("DownloadManager: Successfully started task {}", task.id);
        } else {
            tracing::debug!("DownloadManager: No task available to start from scheduler");
        }
        Ok(())
    }

    /// Get scheduler statistics
    pub async fn scheduler_stats(&self) -> (usize, usize) {
        (
            self.scheduler.running_count().await,
            self.scheduler.pending_count().await,
        )
    }

    /// Get detailed resource usage information
    pub async fn resource_usage(&self) -> ResourceUsage {
        self.scheduler.resource_usage().await
    }

    /// Get queue statistics
    pub async fn queue_stats(&self) -> QueueStats {
        self.scheduler.queue_stats().await
    }

    /// Start multiple downloads up to the concurrent limit
    pub async fn start_downloads(&self, ids: Vec<DownloadId>) -> Result<Vec<DownloadId>> {
        let mut started = Vec::new();

        for id in ids {
            if !self.can_start_more().await {
                break;
            }

            if self.start_download(&id).await.is_ok() {
                started.push(id);
            }
        }

        Ok(started)
    }

    /// Pause all active downloads
    pub async fn pause_all_downloads(&self) -> Result<Vec<DownloadId>> {
        let downloads = self.downloads.read().await;
        let mut active_ids = Vec::new();

        // First collect active download IDs
        for (id, task) in downloads.iter() {
            if task.is_active().await {
                active_ids.push(id.clone());
            }
        }
        drop(downloads); // Release the lock before calling pause_download

        let mut paused = Vec::new();
        for id in active_ids {
            if self.pause_download(&id).await.is_ok() {
                paused.push(id);
            }
        }

        Ok(paused)
    }

    /// Resume all paused downloads (up to concurrent limit)
    pub async fn resume_all_downloads(&self) -> Result<Vec<DownloadId>> {
        let downloads = self.downloads.read().await;
        let mut resumable_ids = Vec::new();

        // First collect resumable download IDs
        for (id, task) in downloads.iter() {
            let state = task.state.read().await;
            if state.can_resume() {
                resumable_ids.push(id.clone());
            }
        }
        drop(downloads); // Release the lock before calling resume_download

        let mut resumed = Vec::new();
        for id in resumable_ids {
            if !self.can_start_more().await {
                break;
            }

            if self.resume_download(&id).await.is_ok() {
                resumed.push(id);
            }
        }

        Ok(resumed)
    }

    /// Cancel all downloads
    pub async fn cancel_all_downloads(&self) -> Result<Vec<DownloadId>> {
        let downloads = self.downloads.read().await;
        let mut cancellable_ids = Vec::new();

        // First collect non-terminal download IDs
        for (id, task) in downloads.iter() {
            if !task.is_terminal().await {
                cancellable_ids.push(id.clone());
            }
        }
        drop(downloads); // Release the lock before calling cancel_download

        let mut cancelled = Vec::new();
        for id in cancellable_ids {
            if self.cancel_download(&id).await.is_ok() {
                cancelled.push(id);
            }
        }

        Ok(cancelled)
    }

    /// Get downloads by state
    pub async fn get_downloads_by_state(&self, state: DownloadState) -> Result<Vec<DownloadInfo>> {
        let downloads = self.downloads.read().await;
        let mut matching_tasks = Vec::new();

        // First collect matching tasks
        for task in downloads.values() {
            let task_state = task.state.read().await;
            if *task_state == state {
                matching_tasks.push(task.clone());
            }
        }
        drop(downloads); // Release the lock before calling info()

        let mut matching = Vec::new();
        for task in matching_tasks {
            matching.push(task.info().await);
        }

        Ok(matching)
    }

    /// Get downloads by priority
    pub async fn get_downloads_by_priority(
        &self,
        priority: DownloadPriority,
    ) -> Result<Vec<DownloadInfo>> {
        let downloads = self.downloads.read().await;
        let mut matching_tasks = Vec::new();

        // First collect matching tasks
        for task in downloads.values() {
            if task.priority().await == priority {
                matching_tasks.push(task.clone());
            }
        }
        drop(downloads); // Release the lock before calling info()

        let mut matching = Vec::new();
        for task in matching_tasks {
            matching.push(task.info().await);
        }

        Ok(matching)
    }

    /// Set maximum concurrent downloads
    pub async fn set_max_concurrent(&self, _max_concurrent: u32) -> Result<()> {
        // todo)) This would require making scheduler mutable, which would require Arc<Mutex<TaskScheduler>>
        // For now, we'll return an error indicating this operation is not supported
        // In a real implementation, we'd need to restructure the scheduler to be mutable
        Err(ZuupError::Config(
            "Cannot change max concurrent downloads after creation".to_string(),
        ))
    }

    /// Get maximum concurrent downloads
    pub fn max_concurrent(&self) -> u32 {
        self.scheduler.max_concurrent()
    }

    /// Check if any downloads are stale (waiting too long)
    pub async fn get_stale_downloads(&self, max_wait_time: Duration) -> Vec<DownloadId> {
        self.scheduler.get_stale_tasks(max_wait_time).await
    }

    /// Get bandwidth manager reference
    pub fn bandwidth_manager(&self) -> &Arc<BandwidthManager> {
        &self.bandwidth_manager
    }
}

/// Multi-source download coordinator for handling downloads from multiple URLs
pub struct MultiSourceCoordinator {
    /// Available sources for the download
    sources: Vec<DownloadSource>,
    /// Current active sources
    active_sources: HashMap<Url, SourceStatus>,
    /// Source health monitoring
    health_monitor: SourceHealthMonitor,
    /// Failover configuration
    failover_config: FailoverConfig,
}

/// Information about a download source
#[derive(Debug, Clone)]
pub struct DownloadSource {
    /// Source URL
    pub url: Url,

    /// Priority (higher = preferred)
    pub priority: u32,

    /// Location hint (country code, etc.)
    pub location: Option<String>,

    /// Maximum connections allowed
    pub max_connections: Option<u32>,

    /// Source reliability score (0.0 - 1.0)
    pub reliability_score: f64,

    /// Last known speed (bytes per second)
    pub last_speed: Option<u64>,

    /// Connection latency
    pub latency: Option<Duration>,
}

/// Status of a download source
#[derive(Debug, Clone)]
pub struct SourceStatus {
    /// Current state
    pub state: SourceState,

    /// Number of active connections
    pub active_connections: u32,

    /// Bytes downloaded from this source
    pub bytes_downloaded: u64,

    /// Current download speed
    pub current_speed: u64,

    /// Error count
    pub error_count: u32,

    /// Last error time
    pub last_error: Option<DateTime<Utc>>,

    /// Last successful activity
    pub last_activity: DateTime<Utc>,
}

/// State of a download source
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceState {
    /// Source is available and ready
    Available,

    /// Source is currently being used
    Active,

    /// Source is temporarily unavailable
    Unavailable,

    /// Source has failed and should not be retried
    Failed,

    /// Source is being tested for availability
    Testing,
}

/// Source health monitoring
pub struct SourceHealthMonitor {
    /// Health check interval
    check_interval: Duration,

    /// Timeout for health checks
    check_timeout: Duration,

    /// Minimum reliability threshold
    min_reliability: f64,

    /// Maximum error rate before marking as failed
    max_error_rate: f64,
}

/// Failover configuration
#[derive(Debug, Clone)]
pub struct FailoverConfig {
    /// Maximum time to wait before switching sources
    max_wait_time: Duration,

    /// Minimum speed threshold before considering failover
    min_speed_threshold: u64,

    /// Enable automatic source switching
    auto_switch: bool,

    /// Prefer sources by location
    prefer_location: Option<String>,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            max_wait_time: Duration::from_secs(30),
            min_speed_threshold: 1024, // 1 KB/s
            auto_switch: true,
            prefer_location: None,
        }
    }
}

impl MultiSourceCoordinator {
    /// Create a new multi-source coordinator
    pub fn new(sources: Vec<DownloadSource>, failover_config: FailoverConfig) -> Self {
        let active_sources = HashMap::new();
        let health_monitor = SourceHealthMonitor {
            check_interval: Duration::from_secs(30),
            check_timeout: Duration::from_secs(10),
            min_reliability: 0.7,
            max_error_rate: 0.3,
        };

        Self {
            sources,
            active_sources,
            health_monitor,
            failover_config,
        }
    }

    /// Create from Metalink file
    pub fn from_metalink(metalink_file: &MetalinkFile, failover_config: FailoverConfig) -> Self {
        let sources = metalink_file
            .urls
            .iter()
            .map(|metalink_url| {
                DownloadSource {
                    url: metalink_url.url.clone(),
                    priority: metalink_url.priority.unwrap_or(0),
                    location: metalink_url.location.clone(),
                    max_connections: metalink_url.max_connections,
                    reliability_score: 1.0, // Start with full reliability
                    last_speed: None,
                    latency: None,
                }
            })
            .collect();

        Self::new(sources, failover_config)
    }

    /// Get the best sources for downloading, sorted by preference
    pub fn get_best_sources(&self, max_sources: usize) -> Vec<&DownloadSource> {
        let mut available_sources: Vec<&DownloadSource> = self
            .sources
            .iter()
            .filter(|source| {
                self.active_sources
                    .get(&source.url)
                    .map(|status| status.state != SourceState::Failed)
                    .unwrap_or(true)
            })
            .collect();

        // Sort by priority, reliability, and speed
        available_sources.sort_by(|a, b| {
            // First by priority (higher is better)
            let priority_cmp = b.priority.cmp(&a.priority);
            if priority_cmp != std::cmp::Ordering::Equal {
                return priority_cmp;
            }

            // Then by reliability score (higher is better)
            let reliability_cmp = b
                .reliability_score
                .partial_cmp(&a.reliability_score)
                .unwrap_or(std::cmp::Ordering::Equal);
            if reliability_cmp != std::cmp::Ordering::Equal {
                return reliability_cmp;
            }

            // Finally by last known speed (higher is better)
            match (b.last_speed, a.last_speed) {
                (Some(b_speed), Some(a_speed)) => b_speed.cmp(&a_speed),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
        });

        // Apply location preference if configured
        if let Some(preferred_location) = &self.failover_config.prefer_location {
            available_sources.sort_by(|a, b| {
                let a_matches = a
                    .location
                    .as_ref()
                    .map(|loc| loc == preferred_location)
                    .unwrap_or(false);
                let b_matches = b
                    .location
                    .as_ref()
                    .map(|loc| loc == preferred_location)
                    .unwrap_or(false);

                match (a_matches, b_matches) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => std::cmp::Ordering::Equal,
                }
            });
        }

        available_sources.into_iter().take(max_sources).collect()
    }

    /// Update source status based on download performance
    pub fn update_source_status(
        &mut self,
        url: &Url,
        bytes_downloaded: u64,
        speed: u64,
        error: Option<&ZuupError>,
    ) {
        let status = self
            .active_sources
            .entry(url.clone())
            .or_insert_with(|| SourceStatus {
                state: SourceState::Available,
                active_connections: 0,
                bytes_downloaded: 0,
                current_speed: 0,
                error_count: 0,
                last_error: None,
                last_activity: Utc::now(),
            });

        status.bytes_downloaded += bytes_downloaded;
        status.current_speed = speed;
        status.last_activity = Utc::now();

        if let Some(_error) = error {
            status.error_count += 1;
            status.last_error = Some(Utc::now());

            // Update source reliability based on error rate
            if let Some(source) = self.sources.iter_mut().find(|s| s.url == *url) {
                let error_rate = status.error_count as f64 / (status.error_count + 1) as f64;
                source.reliability_score = (1.0 - error_rate).max(0.0);

                // Mark as failed if error rate is too high
                if error_rate > self.health_monitor.max_error_rate {
                    status.state = SourceState::Failed;
                }
            }
        } else {
            // Update source speed information on successful activity
            if let Some(source) = self.sources.iter_mut().find(|s| s.url == *url) {
                source.last_speed = Some(speed);
            }
        }
    }

    /// Check if failover should be triggered
    pub fn should_failover(&self, current_sources: &[Url]) -> bool {
        if !self.failover_config.auto_switch {
            return false;
        }

        // Check if any current source is performing poorly
        for url in current_sources {
            if let Some(status) = self.active_sources.get(url) {
                // Failover if speed is below threshold
                if status.current_speed < self.failover_config.min_speed_threshold {
                    return true;
                }

                // Failover if source has been inactive for too long
                let inactive_duration = Utc::now().signed_duration_since(status.last_activity);
                if inactive_duration.to_std().unwrap_or(Duration::ZERO)
                    > self.failover_config.max_wait_time
                {
                    return true;
                }
            }
        }

        false
    }

    /// Get alternative sources for failover
    pub fn get_failover_sources(
        &self,
        failed_sources: &[Url],
        max_sources: usize,
    ) -> Vec<&DownloadSource> {
        self.sources
            .iter()
            .filter(|source| {
                // Exclude failed sources
                !failed_sources.contains(&source.url) &&
                // Only include available sources
                self.active_sources.get(&source.url)
                    .map(|status| status.state == SourceState::Available)
                    .unwrap_or(true)
            })
            .take(max_sources)
            .collect()
    }

    /// Mark a source as active
    pub fn mark_source_active(&mut self, url: &Url) {
        let status = self
            .active_sources
            .entry(url.clone())
            .or_insert_with(|| SourceStatus {
                state: SourceState::Available,
                active_connections: 0,
                bytes_downloaded: 0,
                current_speed: 0,
                error_count: 0,
                last_error: None,
                last_activity: Utc::now(),
            });

        status.state = SourceState::Active;
        status.active_connections += 1;
    }

    /// Mark a source as inactive
    pub fn mark_source_inactive(&mut self, url: &Url) {
        if let Some(status) = self.active_sources.get_mut(url) {
            status.active_connections = status.active_connections.saturating_sub(1);
            if status.active_connections == 0 {
                status.state = SourceState::Available;
            }
        }
    }

    /// Get statistics for all sources
    pub fn get_source_statistics(&self) -> Vec<SourceStatistics> {
        self.sources
            .iter()
            .map(|source| {
                let status = self.active_sources.get(&source.url);
                SourceStatistics {
                    url: source.url.clone(),
                    priority: source.priority,
                    reliability_score: source.reliability_score,
                    state: status
                        .map(|s| s.state.clone())
                        .unwrap_or(SourceState::Available),
                    bytes_downloaded: status.map(|s| s.bytes_downloaded).unwrap_or(0),
                    current_speed: status.map(|s| s.current_speed).unwrap_or(0),
                    error_count: status.map(|s| s.error_count).unwrap_or(0),
                    last_activity: status.map(|s| s.last_activity),
                }
            })
            .collect()
    }
}

/// Statistics for a download source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceStatistics {
    pub url: Url,
    pub priority: u32,
    pub reliability_score: f64,
    pub state: SourceState,
    pub bytes_downloaded: u64,
    pub current_speed: u64,
    pub error_count: u32,
    pub last_activity: Option<DateTime<Utc>>,
}

impl DownloadSource {
    /// Create a new download source
    pub fn new(url: Url) -> Self {
        Self {
            url,
            priority: 0,
            location: None,
            max_connections: None,
            reliability_score: 1.0,
            last_speed: None,
            latency: None,
        }
    }

    /// Create from MetalinkUrl
    pub fn from_metalink_url(metalink_url: &MetalinkUrl) -> Self {
        Self {
            url: metalink_url.url.clone(),
            priority: metalink_url.priority.unwrap_or(0),
            location: metalink_url.location.clone(),
            max_connections: metalink_url.max_connections,
            reliability_score: 1.0,
            last_speed: None,
            latency: None,
        }
    }
}
