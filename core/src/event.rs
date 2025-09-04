use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};

use crate::types::{DownloadId, DownloadInfo, DownloadProgress};
use crate::error::Result;

/// Events that can be emitted by the download system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    /// A download was added to the queue
    DownloadAdded { id: DownloadId, info: DownloadInfo },

    /// A download was started
    DownloadStarted { id: DownloadId, info: DownloadInfo },

    /// Download progress update
    DownloadProgress {
        id: DownloadId,
        progress: DownloadProgress,
    },

    /// A download was completed successfully
    DownloadCompleted { id: DownloadId, info: DownloadInfo },

    /// A download failed
    DownloadFailed {
        id: DownloadId,
        error: String, // Serialized error
    },

    /// A download was paused
    DownloadPaused { id: DownloadId },

    /// A download was resumed
    DownloadResumed { id: DownloadId },

    /// A download was cancelled
    DownloadCancelled { id: DownloadId },

    /// A download was removed
    DownloadRemoved { id: DownloadId },

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
}

impl Event {
    /// Get the download ID associated with this event (if any)
    pub fn download_id(&self) -> Option<&DownloadId> {
        match self {
            Event::DownloadAdded { id, .. }
            | Event::DownloadStarted { id, .. }
            | Event::DownloadProgress { id, .. }
            | Event::DownloadCompleted { id, .. }
            | Event::DownloadFailed { id, .. }
            | Event::DownloadPaused { id }
            | Event::DownloadResumed { id }
            | Event::DownloadCancelled { id }
            | Event::DownloadRemoved { id } => Some(id),
            _ => None,
        }
    }

    /// Get the event type as a string
    pub fn event_type(&self) -> &'static str {
        match self {
            Event::DownloadAdded { .. } => "download_added",
            Event::DownloadStarted { .. } => "download_started",
            Event::DownloadProgress { .. } => "download_progress",
            Event::DownloadCompleted { .. } => "download_completed",
            Event::DownloadFailed { .. } => "download_failed",
            Event::DownloadPaused { .. } => "download_paused",
            Event::DownloadResumed { .. } => "download_resumed",
            Event::DownloadCancelled { .. } => "download_cancelled",
            Event::DownloadRemoved { .. } => "download_removed",
            Event::SystemShutdown => "system_shutdown",
            Event::ConfigChanged => "config_changed",
            Event::NetworkStatusChanged { .. } => "network_status_changed",
            Event::BandwidthLimitChanged { .. } => "bandwidth_limit_changed",
        }
    }
}

/// Event types for filtering subscriptions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventType {
    DownloadAdded,
    DownloadStarted,
    DownloadProgress,
    DownloadCompleted,
    DownloadFailed,
    DownloadPaused,
    DownloadResumed,
    DownloadCancelled,
    DownloadRemoved,
    SystemShutdown,
    ConfigChanged,
    NetworkStatusChanged,
    BandwidthLimitChanged,
    All, // Subscribe to all events
}

impl From<&Event> for EventType {
    fn from(event: &Event) -> Self {
        match event {
            Event::DownloadAdded { .. } => EventType::DownloadAdded,
            Event::DownloadStarted { .. } => EventType::DownloadStarted,
            Event::DownloadProgress { .. } => EventType::DownloadProgress,
            Event::DownloadCompleted { .. } => EventType::DownloadCompleted,
            Event::DownloadFailed { .. } => EventType::DownloadFailed,
            Event::DownloadPaused { .. } => EventType::DownloadPaused,
            Event::DownloadResumed { .. } => EventType::DownloadResumed,
            Event::DownloadCancelled { .. } => EventType::DownloadCancelled,
            Event::DownloadRemoved { .. } => EventType::DownloadRemoved,
            Event::SystemShutdown => EventType::SystemShutdown,
            Event::ConfigChanged => EventType::ConfigChanged,
            Event::NetworkStatusChanged { .. } => EventType::NetworkStatusChanged,
            Event::BandwidthLimitChanged { .. } => EventType::BandwidthLimitChanged,
        }
    }
}

/// Trait for event subscribers
#[async_trait]
pub trait EventSubscriber: Send + Sync {
    /// Handle an event
    async fn handle_event(&self, event: Event) -> Result<()>;

    /// Get the event types this subscriber is interested in
    fn event_types(&self) -> Vec<EventType> {
        vec![EventType::All]
    }

    /// Get subscriber name for debugging
    fn name(&self) -> &str {
        "anonymous"
    }
}

/// Event bus for managing event subscriptions and publishing
pub struct EventBus {
    /// Broadcast sender for events
    sender: broadcast::Sender<Event>,

    /// Typed subscribers
    subscribers: Arc<RwLock<HashMap<EventType, Vec<Arc<dyn EventSubscriber>>>>>,

    /// Event history (for debugging and replay)
    history: Arc<RwLock<Vec<Event>>>,

    /// Maximum history size
    max_history: usize,
}

impl EventBus {
    /// Create a new event bus
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);

        Self {
            sender,
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            max_history: 1000,
        }
    }

    /// Subscribe to events with a typed subscriber
    pub async fn subscribe(&self, subscriber: Arc<dyn EventSubscriber>) {
        let mut subscribers = self.subscribers.write().await;

        for event_type in subscriber.event_types() {
            subscribers
                .entry(event_type)
                .or_insert_with(Vec::new)
                .push(subscriber.clone());
        }
    }

    /// Subscribe to events with a broadcast receiver
    pub fn subscribe_broadcast(&self) -> broadcast::Receiver<Event> {
        self.sender.subscribe()
    }

    /// Publish an event
    pub async fn publish(&self, event: Event) -> Result<()> {
        // Add to history
        {
            let mut history = self.history.write().await;
            history.push(event.clone());

            // Trim history if it gets too large
            if history.len() > self.max_history {
                let excess = history.len() - self.max_history;
                history.drain(0..excess);
            }
        }

        // Send to broadcast subscribers
        let _ = self.sender.send(event.clone());

        // Send to typed subscribers
        let subscribers = self.subscribers.read().await;
        let event_type = EventType::from(&event);

        // Send to specific event type subscribers
        if let Some(subs) = subscribers.get(&event_type) {
            for subscriber in subs {
                if let Err(e) = subscriber.handle_event(event.clone()).await {
                    tracing::warn!(
                        subscriber = subscriber.name(),
                        error = %e,
                        "Event subscriber failed to handle event"
                    );
                }
            }
        }

        // Send to "All" event subscribers
        if let Some(subs) = subscribers.get(&EventType::All) {
            for subscriber in subs {
                if let Err(e) = subscriber.handle_event(event.clone()).await {
                    tracing::warn!(
                        subscriber = subscriber.name(),
                        error = %e,
                        "Event subscriber failed to handle event"
                    );
                }
            }
        }

        Ok(())
    }

    /// Get event history
    pub async fn history(&self) -> Vec<Event> {
        self.history.read().await.clone()
    }

    /// Clear event history
    pub async fn clear_history(&self) {
        self.history.write().await.clear();
    }

    /// Get the number of active subscribers
    pub async fn subscriber_count(&self) -> usize {
        let subscribers = self.subscribers.read().await;
        subscribers.values().map(|v| v.len()).sum()
    }

    /// Set maximum history size
    pub fn set_max_history(&mut self, max_size: usize) {
        self.max_history = max_size;
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Simple event subscriber that logs events
pub struct LoggingEventSubscriber {
    name: String,
    event_types: Vec<EventType>,
}

impl LoggingEventSubscriber {
    pub fn new(name: String) -> Self {
        Self {
            name,
            event_types: vec![EventType::All],
        }
    }

    pub fn with_event_types(mut self, event_types: Vec<EventType>) -> Self {
        self.event_types = event_types;
        self
    }
}

#[async_trait]
impl EventSubscriber for LoggingEventSubscriber {
    async fn handle_event(&self, event: Event) -> Result<()> {
        tracing::info!(
            subscriber = %self.name,
            event_type = event.event_type(),
            download_id = ?event.download_id(),
            "Received event"
        );
        Ok(())
    }

    fn event_types(&self) -> Vec<EventType> {
        self.event_types.clone()
    }

    fn name(&self) -> &str {
        &self.name
    }
}
