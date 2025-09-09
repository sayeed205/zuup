//! Core engine integration layer for the GUI application
//!
//! This module provides the bridge between the GPUI interface and the core download engine,
//! handling event processing, UI state synchronization, and async communication.

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use zuup_core::{
    event::{Event, EventBus, EventSubscriber, EventType},
    types::{DownloadId, DownloadInfo, DownloadProgress, DownloadState},
    Zuup, ZuupConfig, Result as CoreResult,
};
use async_trait::async_trait;
use gpui::*;

/// Events for UI state synchronization
#[derive(Debug, Clone)]
pub enum UIUpdateEvent {
    /// A download was added to the queue
    DownloadAdded(DownloadInfo),
    
    /// Download progress update with batched information
    ProgressUpdated {
        id: DownloadId,
        progress: DownloadProgress,
        timestamp: std::time::Instant,
    },
    
    /// Download state changed
    StateChanged {
        id: DownloadId,
        old_state: DownloadState,
        new_state: DownloadState,
    },
    
    /// Download completed successfully
    DownloadCompleted {
        id: DownloadId,
        info: DownloadInfo,
    },
    
    /// Download failed with error
    DownloadFailed {
        id: DownloadId,
        error: String,
    },
    
    /// Download was paused
    DownloadPaused(DownloadId),
    
    /// Download was resumed
    DownloadResumed(DownloadId),
    
    /// Download was cancelled
    DownloadCancelled(DownloadId),
    
    /// Download was removed
    DownloadRemoved(DownloadId),
    
    /// System-wide events
    SystemShutdown,
    
    /// Configuration changed
    ConfigChanged,
    
    /// Network status changed
    NetworkStatusChanged { connected: bool },
    
    /// Bandwidth limit changed
    BandwidthLimitChanged {
        download_limit: Option<u64>,
        upload_limit: Option<u64>,
    },
    
    /// Batch update containing multiple progress updates
    BatchUpdate(Vec<UIUpdateEvent>),
}

impl UIUpdateEvent {
    /// Get the download ID associated with this event (if any)
    pub fn download_id(&self) -> Option<&DownloadId> {
        match self {
            UIUpdateEvent::DownloadAdded(info) => Some(&info.id),
            UIUpdateEvent::ProgressUpdated { id, .. } => Some(id),
            UIUpdateEvent::StateChanged { id, .. } => Some(id),
            UIUpdateEvent::DownloadCompleted { id, .. } => Some(id),
            UIUpdateEvent::DownloadFailed { id, .. } => Some(id),
            UIUpdateEvent::DownloadPaused(id) => Some(id),
            UIUpdateEvent::DownloadResumed(id) => Some(id),
            UIUpdateEvent::DownloadCancelled(id) => Some(id),
            UIUpdateEvent::DownloadRemoved(id) => Some(id),
            _ => None,
        }
    }
    
    /// Check if this is a progress update event
    pub fn is_progress_update(&self) -> bool {
        matches!(self, UIUpdateEvent::ProgressUpdated { .. })
    }
    
    /// Check if this is a state change event
    pub fn is_state_change(&self) -> bool {
        matches!(self, UIUpdateEvent::StateChanged { .. })
    }
}

/// Configuration for the download engine adapter
#[derive(Debug, Clone)]
pub struct EngineAdapterConfig {
    /// Maximum number of UI update events to buffer
    pub max_ui_buffer_size: usize,
    
    /// Interval for batching progress updates (milliseconds)
    pub progress_batch_interval_ms: u64,
    
    /// Maximum number of events in a batch update
    pub max_batch_size: usize,
    
    /// Whether to enable progress update batching
    pub enable_progress_batching: bool,
}

impl Default for EngineAdapterConfig {
    fn default() -> Self {
        Self {
            max_ui_buffer_size: 1000,
            progress_batch_interval_ms: 100, // 100ms batching interval
            max_batch_size: 50,
            enable_progress_batching: true,
        }
    }
}

/// Bridge between UI and core download engine
pub struct DownloadEngineAdapter {
    /// Core download engine instance
    engine: Arc<Zuup>,
    
    /// Event bus from the core engine
    event_bus: Arc<EventBus>,
    
    /// Channel for sending UI update events
    ui_update_sender: mpsc::UnboundedSender<UIUpdateEvent>,
    
    /// Channel for receiving UI update events
    ui_update_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<UIUpdateEvent>>>>,
    
    /// Configuration for the adapter
    config: EngineAdapterConfig,
    
    /// Progress update batcher
    progress_batcher: Arc<RwLock<ProgressUpdateBatcher>>,
    
    /// Task handle for the event processing loop
    event_processor_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    
    /// Task handle for the progress batching loop
    batch_processor_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl DownloadEngineAdapter {
    /// Create a new download engine adapter
    pub async fn new(core_config: ZuupConfig) -> CoreResult<Self> {
        Self::with_config(core_config, EngineAdapterConfig::default()).await
    }
    
    /// Create a new download engine adapter with custom configuration
    pub async fn with_config(
        core_config: ZuupConfig,
        adapter_config: EngineAdapterConfig,
    ) -> CoreResult<Self> {
        // Initialize the core engine
        let engine = Arc::new(Zuup::new(Some(core_config)).await?);
        let event_bus = engine.event_bus();
        
        // Create UI update channel
        let (ui_update_sender, ui_update_receiver) = mpsc::unbounded_channel();
        
        // Create progress batcher
        let progress_batcher = Arc::new(RwLock::new(ProgressUpdateBatcher::new(
            adapter_config.max_batch_size,
            adapter_config.progress_batch_interval_ms,
        )));
        
        let adapter = Self {
            engine,
            event_bus,
            ui_update_sender,
            ui_update_receiver: Arc::new(RwLock::new(Some(ui_update_receiver))),
            config: adapter_config,
            progress_batcher,
            event_processor_handle: Arc::new(RwLock::new(None)),
            batch_processor_handle: Arc::new(RwLock::new(None)),
        };
        
        Ok(adapter)
    }
    
    /// Initialize the adapter and start event processing
    pub async fn initialize(&self) -> CoreResult<()> {
        // Subscribe to core engine events
        let subscriber = Arc::new(EngineEventSubscriber::new(
            self.ui_update_sender.clone(),
            self.progress_batcher.clone(),
            self.config.enable_progress_batching,
        ));
        
        self.event_bus.subscribe(subscriber).await;
        
        // Start the progress batching processor if enabled
        if self.config.enable_progress_batching {
            self.start_batch_processor().await;
        }
        
        tracing::info!("Download engine adapter initialized successfully");
        Ok(())
    }
    
    /// Start the progress batch processor
    async fn start_batch_processor(&self) {
        let batcher = self.progress_batcher.clone();
        let sender = self.ui_update_sender.clone();
        let interval_ms = self.config.progress_batch_interval_ms;
        
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_millis(interval_ms)
            );
            
            loop {
                interval.tick().await;
                
                let batch = {
                    let mut batcher_guard = batcher.write().await;
                    batcher_guard.flush_batch()
                };
                
                if !batch.is_empty() {
                    let batch_event = UIUpdateEvent::BatchUpdate(batch);
                    if sender.send(batch_event).is_err() {
                        tracing::warn!("UI update receiver dropped, stopping batch processor");
                        break;
                    }
                }
            }
        });
        
        *self.batch_processor_handle.write().await = Some(handle);
    }
    
    /// Get the core engine instance
    pub fn engine(&self) -> &Arc<Zuup> {
        &self.engine
    }
    
    /// Get a receiver for UI update events
    pub async fn take_ui_receiver(&self) -> Option<mpsc::UnboundedReceiver<UIUpdateEvent>> {
        self.ui_update_receiver.write().await.take()
    }
    
    /// Send a UI update event (for testing or manual events)
    pub fn send_ui_update(&self, event: UIUpdateEvent) -> Result<(), mpsc::error::SendError<UIUpdateEvent>> {
        self.ui_update_sender.send(event)
    }
    
    /// Get adapter statistics
    pub async fn stats(&self) -> EngineAdapterStats {
        let batcher_stats = {
            let batcher = self.progress_batcher.read().await;
            batcher.stats()
        };
        
        EngineAdapterStats {
            ui_buffer_size: 0, // UnboundedSender doesn't have len() method
            max_ui_buffer_size: self.config.max_ui_buffer_size,
            progress_batching_enabled: self.config.enable_progress_batching,
            batch_interval_ms: self.config.progress_batch_interval_ms,
            pending_progress_updates: batcher_stats.pending_updates,
            total_batches_sent: batcher_stats.total_batches_sent,
            total_events_batched: batcher_stats.total_events_batched,
        }
    }
    
    /// Shutdown the adapter and cleanup resources
    pub async fn shutdown(&self) -> CoreResult<()> {
        tracing::info!("Shutting down download engine adapter");
        
        // Stop batch processor
        if let Some(handle) = self.batch_processor_handle.write().await.take() {
            handle.abort();
        }
        
        // Stop event processor
        if let Some(handle) = self.event_processor_handle.write().await.take() {
            handle.abort();
        }
        
        // Flush any remaining batched updates
        if self.config.enable_progress_batching {
            let batch = {
                let mut batcher = self.progress_batcher.write().await;
                batcher.flush_batch()
            };
            
            if !batch.is_empty() {
                let _ = self.ui_update_sender.send(UIUpdateEvent::BatchUpdate(batch));
            }
        }
        
        // Shutdown the core engine
        self.engine.shutdown(false).await?;
        
        tracing::info!("Download engine adapter shutdown complete");
        Ok(())
    }
}

/// Event subscriber that converts core events to UI update events
struct EngineEventSubscriber {
    ui_sender: mpsc::UnboundedSender<UIUpdateEvent>,
    progress_batcher: Arc<RwLock<ProgressUpdateBatcher>>,
    enable_batching: bool,
}

impl EngineEventSubscriber {
    fn new(
        ui_sender: mpsc::UnboundedSender<UIUpdateEvent>,
        progress_batcher: Arc<RwLock<ProgressUpdateBatcher>>,
        enable_batching: bool,
    ) -> Self {
        Self {
            ui_sender,
            progress_batcher,
            enable_batching,
        }
    }
    
    async fn send_ui_event(&self, event: UIUpdateEvent) {
        if let Err(_) = self.ui_sender.send(event) {
            tracing::warn!("Failed to send UI update event: receiver dropped");
        }
    }
}

#[async_trait]
impl EventSubscriber for EngineEventSubscriber {
    async fn handle_event(&self, event: Event) -> CoreResult<()> {
        let ui_event = match event {
            Event::DownloadAdded { id: _, info } => {
                UIUpdateEvent::DownloadAdded(info)
            }
            
            Event::DownloadProgress { id, progress } => {
                let progress_event = UIUpdateEvent::ProgressUpdated {
                    id,
                    progress,
                    timestamp: std::time::Instant::now(),
                };
                
                // Handle progress batching
                if self.enable_batching {
                    let mut batcher = self.progress_batcher.write().await;
                    batcher.add_progress_update(progress_event);
                    return Ok(());
                } else {
                    progress_event
                }
            }
            
            Event::DownloadCompleted { id, info } => {
                UIUpdateEvent::DownloadCompleted { id, info }
            }
            
            Event::DownloadFailed { id, error } => {
                UIUpdateEvent::DownloadFailed { id, error }
            }
            
            Event::DownloadPaused { id } => {
                UIUpdateEvent::DownloadPaused(id)
            }
            
            Event::DownloadResumed { id } => {
                UIUpdateEvent::DownloadResumed(id)
            }
            
            Event::DownloadCancelled { id } => {
                UIUpdateEvent::DownloadCancelled(id)
            }
            
            Event::DownloadRemoved { id } => {
                UIUpdateEvent::DownloadRemoved(id)
            }
            
            Event::SystemShutdown => {
                UIUpdateEvent::SystemShutdown
            }
            
            Event::ConfigChanged => {
                UIUpdateEvent::ConfigChanged
            }
            
            Event::NetworkStatusChanged { connected } => {
                UIUpdateEvent::NetworkStatusChanged { connected }
            }
            
            Event::BandwidthLimitChanged {
                download_limit,
                upload_limit,
            } => {
                UIUpdateEvent::BandwidthLimitChanged {
                    download_limit,
                    upload_limit,
                }
            }
            
            // Handle state changes by detecting them from other events
            Event::DownloadStarted { id: _, info: _ } => {
                // We could track state changes here, but for now just pass through
                // In a full implementation, we'd maintain state tracking
                return Ok(());
            }
        };
        
        self.send_ui_event(ui_event).await;
        Ok(())
    }
    
    fn event_types(&self) -> Vec<EventType> {
        vec![EventType::All] // Subscribe to all events
    }
    
    fn name(&self) -> &str {
        "EngineEventSubscriber"
    }
}

/// Progress update batcher for efficient UI updates
struct ProgressUpdateBatcher {
    pending_updates: std::collections::HashMap<DownloadId, UIUpdateEvent>,
    max_batch_size: usize,
    batch_interval_ms: u64,
    stats: BatcherStats,
}

impl ProgressUpdateBatcher {
    fn new(max_batch_size: usize, batch_interval_ms: u64) -> Self {
        Self {
            pending_updates: std::collections::HashMap::new(),
            max_batch_size,
            batch_interval_ms,
            stats: BatcherStats::default(),
        }
    }
    
    fn add_progress_update(&mut self, event: UIUpdateEvent) {
        if let Some(id) = event.download_id() {
            self.pending_updates.insert(id.clone(), event);
            
            // Flush if we've reached the maximum batch size
            if self.pending_updates.len() >= self.max_batch_size {
                let _ = self.flush_batch();
            }
        }
    }
    
    fn flush_batch(&mut self) -> Vec<UIUpdateEvent> {
        if self.pending_updates.is_empty() {
            return Vec::new();
        }
        
        let batch: Vec<UIUpdateEvent> = self.pending_updates.drain().map(|(_, event)| event).collect();
        
        // Update statistics
        self.stats.total_batches_sent += 1;
        self.stats.total_events_batched += batch.len();
        
        batch
    }
    
    fn stats(&self) -> BatcherStats {
        BatcherStats {
            pending_updates: self.pending_updates.len(),
            total_batches_sent: self.stats.total_batches_sent,
            total_events_batched: self.stats.total_events_batched,
        }
    }
}

/// Statistics for the progress batcher
#[derive(Debug, Clone, Default)]
struct BatcherStats {
    pending_updates: usize,
    total_batches_sent: usize,
    total_events_batched: usize,
}

/// Statistics for the engine adapter
#[derive(Debug, Clone)]
pub struct EngineAdapterStats {
    pub ui_buffer_size: usize,
    pub max_ui_buffer_size: usize,
    pub progress_batching_enabled: bool,
    pub batch_interval_ms: u64,
    pub pending_progress_updates: usize,
    pub total_batches_sent: usize,
    pub total_events_batched: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    use zuup_core::types::DownloadProgress;
    
    #[tokio::test]
    async fn test_engine_adapter_creation() {
        let config = ZuupConfig::default();
        let adapter = DownloadEngineAdapter::new(config).await.unwrap();
        
        // Verify adapter is created successfully
        assert!(adapter.engine().is_running().await);
    }
    
    #[tokio::test]
    async fn test_ui_event_conversion() {
        let config = ZuupConfig::default();
        let adapter = DownloadEngineAdapter::new(config).await.unwrap();
        
        // Initialize the adapter
        adapter.initialize().await.unwrap();
        
        // Take the receiver
        let mut receiver = adapter.take_ui_receiver().await.unwrap();
        
        // Send a test event
        let test_event = UIUpdateEvent::SystemShutdown;
        adapter.send_ui_update(test_event.clone()).unwrap();
        
        // Receive the event
        let received = receiver.recv().await.unwrap();
        assert!(matches!(received, UIUpdateEvent::SystemShutdown));
    }
    
    #[tokio::test]
    async fn test_progress_batching() {
        let mut batcher = ProgressUpdateBatcher::new(3, 100);
        
        let id1 = DownloadId::new();
        let id2 = DownloadId::new();
        
        // Add some progress updates
        batcher.add_progress_update(UIUpdateEvent::ProgressUpdated {
            id: id1.clone(),
            progress: DownloadProgress::new(),
            timestamp: std::time::Instant::now(),
        });
        
        batcher.add_progress_update(UIUpdateEvent::ProgressUpdated {
            id: id2.clone(),
            progress: DownloadProgress::new(),
            timestamp: std::time::Instant::now(),
        });
        
        // Should have 2 pending updates
        assert_eq!(batcher.stats().pending_updates, 2);
        
        // Flush the batch
        let batch = batcher.flush_batch();
        assert_eq!(batch.len(), 2);
        assert_eq!(batcher.stats().pending_updates, 0);
    }
    
    #[tokio::test]
    async fn test_adapter_shutdown() {
        let config = ZuupConfig::default();
        let adapter = DownloadEngineAdapter::new(config).await.unwrap();
        
        adapter.initialize().await.unwrap();
        
        // Shutdown should complete without errors
        adapter.shutdown().await.unwrap();
    }
}