//! Callback system for download events and progress monitoring
//!
//! This module provides various callback mechanisms for monitoring download
//! progress, handling events, and implementing custom recovery strategies.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{Result, ZuupError};
use crate::event::{Event, EventSubscriber, EventType};
use crate::types::{DownloadId, DownloadOptions};
use crate::types::{DownloadProgress, DownloadState};

/// Progress callback function type
pub type ProgressCallback = Arc<dyn Fn(DownloadId, DownloadProgress) -> Result<()> + Send + Sync>;

/// Async progress callback function type
pub type AsyncProgressCallback = Arc<
    dyn Fn(
            DownloadId,
            DownloadProgress,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>
        + Send
        + Sync,
>;

/// Completion callback function type
pub type CompletionCallback = Arc<dyn Fn(DownloadId, DownloadResult) -> Result<()> + Send + Sync>;

/// Async completion callback function type
pub type AsyncCompletionCallback = Arc<
    dyn Fn(
            DownloadId,
            DownloadResult,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>
        + Send
        + Sync,
>;

/// Error callback function type with recovery decision
pub type ErrorCallback =
    Arc<dyn Fn(DownloadId, ZuupError) -> Result<ErrorRecoveryAction> + Send + Sync>;

/// Async error callback function type with recovery decision
pub type AsyncErrorCallback = Arc<
    dyn Fn(
            DownloadId,
            ZuupError,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<ErrorRecoveryAction>> + Send>,
        > + Send
        + Sync,
>;

/// State change callback function type
pub type StateChangeCallback =
    Arc<dyn Fn(DownloadId, DownloadState, DownloadState) -> Result<()> + Send + Sync>;

/// Async state change callback function type
pub type AsyncStateChangeCallback = Arc<
    dyn Fn(
            DownloadId,
            DownloadState,
            DownloadState,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>
        + Send
        + Sync,
>;

/// Bandwidth callback function type
pub type BandwidthCallback = Arc<dyn Fn(BandwidthStats) -> Result<()> + Send + Sync>;

/// Async bandwidth callback function type
pub type AsyncBandwidthCallback = Arc<
    dyn Fn(
            BandwidthStats,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>
        + Send
        + Sync,
>;

/// Result of a download operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadResult {
    /// Download ID
    pub id: DownloadId,

    /// Whether the download was successful
    pub success: bool,

    /// Final download state
    pub final_state: DownloadState,

    /// Total bytes downloaded
    pub bytes_downloaded: u64,

    /// Total time taken
    pub duration: Duration,

    /// Average download speed
    pub average_speed: u64,

    /// Error message (if failed)
    pub error: Option<String>,

    /// Checksum verification result
    pub checksum_verified: Option<bool>,

    /// Number of retry attempts made
    pub retry_attempts: u32,

    /// Final file path (if successful)
    pub file_path: Option<std::path::PathBuf>,
}

/// Action to take when an error occurs
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorRecoveryAction {
    /// Retry the download immediately
    Retry,

    /// Retry after a delay
    RetryAfter(Duration),

    /// Retry with different options
    RetryWithOptions(Box<DownloadOptions>),

    /// Fail the download
    Fail,

    /// Pause the download for manual intervention
    Pause,

    /// Try alternative URL (if available)
    TryAlternativeUrl,
}

/// Bandwidth statistics
#[derive(Debug, Clone)]
pub struct BandwidthStats {
    /// Current download speed across all downloads
    pub download_speed: u64,

    /// Current upload speed across all downloads
    pub upload_speed: u64,

    /// Peak download speed in this session
    pub peak_download_speed: u64,

    /// Peak upload speed in this session
    pub peak_upload_speed: u64,

    /// Total bytes downloaded in this session
    pub total_downloaded: u64,

    /// Total bytes uploaded in this session
    pub total_uploaded: u64,

    /// Number of active connections
    pub active_connections: u32,

    /// Timestamp of these statistics
    pub timestamp: std::time::Instant,
}

/// Progress monitoring configuration
#[derive(Debug, Clone)]
pub struct ProgressMonitorConfig {
    /// Minimum interval between progress updates
    pub update_interval: Duration,

    /// Whether to report progress for completed segments
    pub report_segment_progress: bool,

    /// Whether to calculate ETA
    pub calculate_eta: bool,

    /// Whether to track speed history for smoothing
    pub track_speed_history: bool,

    /// Number of speed samples to keep for averaging
    pub speed_history_size: usize,
}

impl Default for ProgressMonitorConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_millis(500),
            report_segment_progress: true,
            calculate_eta: true,
            track_speed_history: true,
            speed_history_size: 10,
        }
    }
}

/// Comprehensive callback manager for download events
pub struct CallbackManager {
    /// Progress callbacks
    progress_callbacks: Vec<ProgressCallback>,

    /// Async progress callbacks
    async_progress_callbacks: Vec<AsyncProgressCallback>,

    /// Completion callbacks
    completion_callbacks: Vec<CompletionCallback>,

    /// Async completion callbacks
    async_completion_callbacks: Vec<AsyncCompletionCallback>,

    /// Error callbacks
    error_callbacks: Vec<ErrorCallback>,

    /// Async error callbacks
    async_error_callbacks: Vec<AsyncErrorCallback>,

    /// State change callbacks
    state_change_callbacks: Vec<StateChangeCallback>,

    /// Async state change callbacks
    async_state_change_callbacks: Vec<AsyncStateChangeCallback>,

    /// Bandwidth callbacks
    bandwidth_callbacks: Vec<BandwidthCallback>,

    /// Async bandwidth callbacks
    async_bandwidth_callbacks: Vec<AsyncBandwidthCallback>,

    /// Progress monitoring configuration
    progress_config: ProgressMonitorConfig,

    /// Last progress update times (to throttle updates)
    last_progress_updates: std::collections::HashMap<DownloadId, Instant>,
}

impl CallbackManager {
    /// Create a new callback manager
    pub fn new() -> Self {
        Self {
            progress_callbacks: Vec::new(),
            async_progress_callbacks: Vec::new(),
            completion_callbacks: Vec::new(),
            async_completion_callbacks: Vec::new(),
            error_callbacks: Vec::new(),
            async_error_callbacks: Vec::new(),
            state_change_callbacks: Vec::new(),
            async_state_change_callbacks: Vec::new(),
            bandwidth_callbacks: Vec::new(),
            async_bandwidth_callbacks: Vec::new(),
            progress_config: ProgressMonitorConfig::default(),
            last_progress_updates: std::collections::HashMap::new(),
        }
    }

    /// Add a progress callback
    pub fn on_progress<F>(&mut self, callback: F)
    where
        F: Fn(DownloadId, DownloadProgress) -> Result<()> + Send + Sync + 'static,
    {
        self.progress_callbacks.push(Arc::new(callback));
    }

    /// Add an async progress callback
    pub fn on_progress_async<F, Fut>(&mut self, callback: F)
    where
        F: Fn(DownloadId, DownloadProgress) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        let wrapped = Arc::new(move |id: DownloadId, progress: DownloadProgress| {
            Box::pin(callback(id, progress))
                as std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>
        });
        self.async_progress_callbacks.push(wrapped);
    }

    /// Add a completion callback
    pub fn on_completion<F>(&mut self, callback: F)
    where
        F: Fn(DownloadId, DownloadResult) -> Result<()> + Send + Sync + 'static,
    {
        self.completion_callbacks.push(Arc::new(callback));
    }

    /// Add an async completion callback
    pub fn on_completion_async<F, Fut>(&mut self, callback: F)
    where
        F: Fn(DownloadId, DownloadResult) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        let wrapped = Arc::new(move |id: DownloadId, result: DownloadResult| {
            Box::pin(callback(id, result))
                as std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>
        });
        self.async_completion_callbacks.push(wrapped);
    }

    /// Add an error callback with recovery decision
    pub fn on_error<F>(&mut self, callback: F)
    where
        F: Fn(DownloadId, ZuupError) -> Result<ErrorRecoveryAction> + Send + Sync + 'static,
    {
        self.error_callbacks.push(Arc::new(callback));
    }

    /// Add an async error callback with recovery decision
    pub fn on_error_async<F, Fut>(&mut self, callback: F)
    where
        F: Fn(DownloadId, ZuupError) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<ErrorRecoveryAction>> + Send + 'static,
    {
        let wrapped = Arc::new(move |id: DownloadId, error: ZuupError| {
            Box::pin(callback(id, error))
                as std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<ErrorRecoveryAction>> + Send>,
                >
        });
        self.async_error_callbacks.push(wrapped);
    }

    /// Add a state change callback
    pub fn on_state_change<F>(&mut self, callback: F)
    where
        F: Fn(DownloadId, DownloadState, DownloadState) -> Result<()> + Send + Sync + 'static,
    {
        self.state_change_callbacks.push(Arc::new(callback));
    }

    /// Add an async state change callback
    pub fn on_state_change_async<F, Fut>(&mut self, callback: F)
    where
        F: Fn(DownloadId, DownloadState, DownloadState) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        let wrapped = Arc::new(
            move |id: DownloadId, old_state: DownloadState, new_state: DownloadState| {
                Box::pin(callback(id, old_state, new_state))
                    as std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>
            },
        );
        self.async_state_change_callbacks.push(wrapped);
    }

    /// Add a bandwidth monitoring callback
    pub fn on_bandwidth_change<F>(&mut self, callback: F)
    where
        F: Fn(BandwidthStats) -> Result<()> + Send + Sync + 'static,
    {
        self.bandwidth_callbacks.push(Arc::new(callback));
    }

    /// Add an async bandwidth monitoring callback
    pub fn on_bandwidth_change_async<F, Fut>(&mut self, callback: F)
    where
        F: Fn(BandwidthStats) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        let wrapped = Arc::new(move |stats: BandwidthStats| {
            Box::pin(callback(stats))
                as std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>
        });
        self.async_bandwidth_callbacks.push(wrapped);
    }

    /// Set progress monitoring configuration
    pub fn set_progress_config(&mut self, config: ProgressMonitorConfig) {
        self.progress_config = config;
    }

    /// Handle progress update
    pub fn handle_progress(&mut self, id: DownloadId, progress: DownloadProgress) -> Result<()> {
        // Check if we should throttle this update
        let now = Instant::now();
        if let Some(last_update) = self.last_progress_updates.get(&id)
            && now.duration_since(*last_update) < self.progress_config.update_interval
        {
            return Ok(());
        }

        self.last_progress_updates.insert(id.clone(), now);

        // Call all progress callbacks
        for callback in &self.progress_callbacks {
            if let Err(e) = callback(id.clone(), progress.clone()) {
                tracing::warn!(
                    download_id = %id,
                    error = %e,
                    "Progress callback failed"
                );
            }
        }

        Ok(())
    }

    /// Handle progress update (async version)
    pub async fn handle_progress_async(
        &mut self,
        id: DownloadId,
        progress: DownloadProgress,
    ) -> Result<()> {
        // Check if we should throttle this update
        let now = Instant::now();
        if let Some(last_update) = self.last_progress_updates.get(&id)
            && now.duration_since(*last_update) < self.progress_config.update_interval
        {
            return Ok(());
        }

        self.last_progress_updates.insert(id.clone(), now);

        // Call all sync progress callbacks
        for callback in &self.progress_callbacks {
            if let Err(e) = callback(id.clone(), progress.clone()) {
                tracing::warn!(
                    download_id = %id,
                    error = %e,
                    "Progress callback failed"
                );
            }
        }

        // Call all async progress callbacks
        for callback in &self.async_progress_callbacks {
            if let Err(e) = callback(id.clone(), progress.clone()).await {
                tracing::warn!(
                    download_id = %id,
                    error = %e,
                    "Async progress callback failed"
                );
            }
        }

        Ok(())
    }

    /// Handle download completion
    pub fn handle_completion(&self, id: DownloadId, result: DownloadResult) -> Result<()> {
        for callback in &self.completion_callbacks {
            if let Err(e) = callback(id.clone(), result.clone()) {
                tracing::warn!(
                    download_id = %id,
                    error = %e,
                    "Completion callback failed"
                );
            }
        }

        Ok(())
    }

    /// Handle download completion (async version)
    pub async fn handle_completion_async(
        &self,
        id: DownloadId,
        result: DownloadResult,
    ) -> Result<()> {
        // Call all sync completion callbacks
        for callback in &self.completion_callbacks {
            if let Err(e) = callback(id.clone(), result.clone()) {
                tracing::warn!(
                    download_id = %id,
                    error = %e,
                    "Completion callback failed"
                );
            }
        }

        // Call all async completion callbacks
        for callback in &self.async_completion_callbacks {
            if let Err(e) = callback(id.clone(), result.clone()).await {
                tracing::warn!(
                    download_id = %id,
                    error = %e,
                    "Async completion callback failed"
                );
            }
        }

        Ok(())
    }

    /// Handle download error and get recovery action
    pub fn handle_error(&self, id: DownloadId, error: ZuupError) -> ErrorRecoveryAction {
        for callback in &self.error_callbacks {
            match callback(id.clone(), error.clone()) {
                Ok(action) => return action,
                Err(e) => {
                    tracing::warn!(
                        download_id = %id,
                        error = %e,
                        "Error callback failed"
                    );
                }
            }
        }

        // Default recovery action
        ErrorRecoveryAction::Fail
    }

    /// Handle download error and get recovery action (async version)
    pub async fn handle_error_async(
        &self,
        id: DownloadId,
        error: ZuupError,
    ) -> ErrorRecoveryAction {
        // Try sync callbacks first
        for callback in &self.error_callbacks {
            match callback(id.clone(), error.clone()) {
                Ok(action) => return action,
                Err(e) => {
                    tracing::warn!(
                        download_id = %id,
                        error = %e,
                        "Error callback failed"
                    );
                }
            }
        }

        // Try async callbacks
        for callback in &self.async_error_callbacks {
            match callback(id.clone(), error.clone()).await {
                Ok(action) => return action,
                Err(e) => {
                    tracing::warn!(
                        download_id = %id,
                        error = %e,
                        "Async error callback failed"
                    );
                }
            }
        }

        // Default recovery action
        ErrorRecoveryAction::Fail
    }

    /// Handle state change
    pub fn handle_state_change(
        &self,
        id: DownloadId,
        old_state: DownloadState,
        new_state: DownloadState,
    ) -> Result<()> {
        for callback in &self.state_change_callbacks {
            if let Err(e) = callback(id.clone(), old_state.clone(), new_state.clone()) {
                tracing::warn!(
                    download_id = %id,
                    error = %e,
                    "State change callback failed"
                );
            }
        }

        Ok(())
    }

    /// Handle bandwidth statistics update
    pub fn handle_bandwidth_stats(&self, stats: BandwidthStats) -> Result<()> {
        for callback in &self.bandwidth_callbacks {
            if let Err(e) = callback(stats.clone()) {
                tracing::warn!(
                    error = %e,
                    "Bandwidth callback failed"
                );
            }
        }

        Ok(())
    }

    /// Clear all callbacks
    pub fn clear_all(&mut self) {
        self.progress_callbacks.clear();
        self.async_progress_callbacks.clear();
        self.completion_callbacks.clear();
        self.async_completion_callbacks.clear();
        self.error_callbacks.clear();
        self.async_error_callbacks.clear();
        self.state_change_callbacks.clear();
        self.async_state_change_callbacks.clear();
        self.bandwidth_callbacks.clear();
        self.async_bandwidth_callbacks.clear();
        self.last_progress_updates.clear();
    }

    /// Get callback counts for debugging
    pub fn callback_counts(&self) -> CallbackCounts {
        CallbackCounts {
            progress: self.progress_callbacks.len() + self.async_progress_callbacks.len(),
            completion: self.completion_callbacks.len() + self.async_completion_callbacks.len(),
            error: self.error_callbacks.len() + self.async_error_callbacks.len(),
            state_change: self.state_change_callbacks.len()
                + self.async_state_change_callbacks.len(),
            bandwidth: self.bandwidth_callbacks.len() + self.async_bandwidth_callbacks.len(),
        }
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Callback counts for debugging
#[derive(Debug, Clone)]
pub struct CallbackCounts {
    pub progress: usize,
    pub completion: usize,
    pub error: usize,
    pub state_change: usize,
    pub bandwidth: usize,
}

/// Event subscriber that bridges events to callbacks
pub struct CallbackEventSubscriber {
    callback_manager: Arc<tokio::sync::Mutex<CallbackManager>>,
    name: String,
}

impl CallbackEventSubscriber {
    /// Create a new callback event subscriber
    pub fn new(callback_manager: Arc<tokio::sync::Mutex<CallbackManager>>, name: String) -> Self {
        Self {
            callback_manager,
            name,
        }
    }
}

#[async_trait]
impl EventSubscriber for CallbackEventSubscriber {
    async fn handle_event(&self, event: Event) -> Result<()> {
        let mut manager = self.callback_manager.lock().await;

        match event {
            Event::DownloadProgress { id, progress } => {
                manager.handle_progress(id, progress)?;
            }
            Event::DownloadCompleted { id, info } => {
                let result = DownloadResult {
                    id: id.clone(),
                    success: true,
                    final_state: info.state.clone(),
                    bytes_downloaded: info.progress.downloaded_size,
                    duration: info
                        .started_at
                        .and_then(|start| {
                            info.completed_at
                                .map(|end| (end - start).to_std().unwrap_or_default())
                        })
                        .unwrap_or_default(),
                    average_speed: if let (Some(start), Some(end)) =
                        (info.started_at, info.completed_at)
                    {
                        let duration_secs = (end - start).num_seconds() as u64;
                        if duration_secs > 0 {
                            info.progress.downloaded_size / duration_secs
                        } else {
                            0
                        }
                    } else {
                        0
                    },
                    error: None,
                    checksum_verified: None, // TODO: Add checksum tracking
                    retry_attempts: 0,       // TODO: Add retry tracking
                    file_path: Some(info.output_path.join(&info.filename)),
                };
                manager.handle_completion(id, result)?;
            }
            Event::DownloadFailed { id, error } => {
                let ruso_error = ZuupError::Internal(error.clone());
                let _recovery_action = manager.handle_error(id.clone(), ruso_error);

                // TODO: Apply recovery action

                let result = DownloadResult {
                    id: id.clone(),
                    success: false,
                    final_state: DownloadState::Failed(error.clone()),
                    bytes_downloaded: 0, // TODO: Get actual progress
                    duration: Duration::default(),
                    average_speed: 0,
                    error: Some(error),
                    checksum_verified: None,
                    retry_attempts: 0, // TODO: Add retry tracking
                    file_path: None,
                };
                manager.handle_completion(id, result)?;
            }
            Event::BandwidthLimitChanged {
                download_limit: _,
                upload_limit: _,
            } => {
                let stats = BandwidthStats {
                    download_speed: 0, // TODO: Get current speeds
                    upload_speed: 0,
                    peak_download_speed: 0,
                    peak_upload_speed: 0,
                    total_downloaded: 0,
                    total_uploaded: 0,
                    active_connections: 0,
                    timestamp: Instant::now(),
                };
                manager.handle_bandwidth_stats(stats)?;
            }
            _ => {
                // Handle other events as needed
            }
        }

        Ok(())
    }

    fn event_types(&self) -> Vec<EventType> {
        vec![
            EventType::DownloadProgress,
            EventType::DownloadCompleted,
            EventType::DownloadFailed,
            EventType::BandwidthLimitChanged,
        ]
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Builder for creating callback configurations
pub struct CallbackBuilder {
    manager: CallbackManager,
}

impl CallbackBuilder {
    /// Create a new callback builder
    pub fn new() -> Self {
        Self {
            manager: CallbackManager::new(),
        }
    }

    /// Add an error callback with automatic retry
    pub fn with_auto_retry(mut self, _max_retries: u32) -> Self {
        // todo)) add max retries
        self.manager.on_error(move |_id, error| {
            match error {
                ZuupError::Network(_) => {
                    // Retry network errors after a delay
                    Ok(ErrorRecoveryAction::RetryAfter(Duration::from_secs(5)))
                }
                ZuupError::Io(_) => {
                    // Retry I/O errors immediately
                    Ok(ErrorRecoveryAction::Retry)
                }
                _ => {
                    // Fail on other errors
                    Ok(ErrorRecoveryAction::Fail)
                }
            }
        });
        self
    }

    /// Add a state change callback that logs state transitions
    pub fn with_state_logging(mut self) -> Self {
        self.manager.on_state_change(|id, old_state, new_state| {
            println!("Download {} state: {:?} -> {:?}", id, old_state, new_state);
            Ok(())
        });
        self
    }

    /// Add a bandwidth monitoring callback
    pub fn with_bandwidth_monitoring(mut self) -> Self {
        self.manager.on_bandwidth_change(|stats| {
            println!(
                "Bandwidth: ↓ {} bytes/sec, ↑ {} bytes/sec ({} connections)",
                stats.download_speed, stats.upload_speed, stats.active_connections
            );
            Ok(())
        });
        self
    }

    /// Set progress monitoring configuration
    pub fn with_progress_config(mut self, config: ProgressMonitorConfig) -> Self {
        self.manager.set_progress_config(config);
        self
    }

    /// Build the callback manager
    pub fn build(self) -> CallbackManager {
        self.manager
    }
}

impl Default for CallbackBuilder {
    fn default() -> Self {
        Self::new()
    }
}
