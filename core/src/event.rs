use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::types::DownloadId;

/// Event bus for managing event subscriptions and publishing.
pub struct EventBus {
    /// Broadcast sender for events
    sender: broadcast::Sender<Event>,
}

/// Events that can be emitted by the download system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    /// A download was added to the queue
    DownloadAdded { id: DownloadId },

    /// A download was started
    DownloadStarted { id: DownloadId },
    // todo)) add more events
}

impl Event {
    /// Get the download ID associated with this event (if any)
    pub fn download_id(&self) -> Option<&DownloadId> {
        match self {
            Event::DownloadAdded { id, .. } | Event::DownloadStarted { id, .. } => Some(id),
            _ => None,
        }
    }

    /// Get the event type as a string
    pub fn event_type(&self) -> &'static str {
        match self {
            Event::DownloadAdded { .. } => "download_added",
            Event::DownloadStarted { .. } => "download_started",
        }
    }
}

/// Event types for filtering subscriptions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventType {
    DownloadAdded,
    DownloadStarted,
    All, // Subscribe to all events
}

impl From<&Event> for EventType {
    fn from(event: &Event) -> Self {
        match event {
            Event::DownloadAdded { .. } => EventType::DownloadAdded,
            Event::DownloadStarted { .. } => EventType::DownloadStarted,
        }
    }
}
